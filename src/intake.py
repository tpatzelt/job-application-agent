from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from .document_text import DocumentExtractionError, extract_text
from .models import IntakeExtraction
from .telegram_api import IncomingMessage, TelegramError
from .user_store import (
    STATE_ACTIVE,
    STATE_AWAITING_ANSWER,
    STATE_AWAITING_CV,
    STATE_AWAITING_JOB_PREFS,
    STATE_AWAITING_MOTIVATION,
    STATE_NEW,
    UserRecord,
    UserStore,
)

# Deterministic fallback questions so missing essentials are always asked
# even when the LLM forgets to ask (or extraction fails entirely).
QUESTION_LOCATIONS = (
    "Which country or cities should I search for jobs in? "
    "(e.g. 'Berlin and Hamburg, Germany' or 'remote, EU')"
)
QUESTION_ROLES = "Which job titles or roles should I look for?"

# Stop asking follow-ups after this many answered questions.
MAX_ANSWERED_QUESTIONS = 6
# Pasted text this long is accepted in place of an uploaded document.
MIN_PASTED_DOC_CHARS = 120

WELCOME = (
    "\U0001f44b Hi{name}! I'm your job search agent.\n\n"
    "I'll continuously scan the web for jobs that match your profile and "
    "message you when I find good ones.\n\n"
    "To get started, please upload your CV as a PDF, DOCX, or text file "
    "(you can also paste it as a message)."
)

HELP_TEXT = (
    "Commands:\n"
    "/start - begin or show setup\n"
    "/status - your profile and search parameters\n"
    "/run - scan for jobs now\n"
    "/reset - restart setup (re-upload documents)\n"
    "/skip - skip the current optional step\n"
    "/help - this message"
)


class DocumentSource(Protocol):
    def download_document(self, document: Any) -> bytes: ...


class IntakeManager:
    """Drives the per-user onboarding conversation.

    States: new -> awaiting_cv -> awaiting_motivation -> awaiting_job_prefs
    -> awaiting_answer (repeat per question) -> active.

    Returns the reply text for each incoming message; the caller sends it
    and handles commands that need the scheduler (e.g. /run).
    """

    def __init__(
        self,
        store: UserStore,
        downloader: DocumentSource,
        extract: Callable[[str, str, str, list[dict[str, str]]], IntakeExtraction],
        scan_interval_hours: float = 6.0,
    ) -> None:
        self._store = store
        self._downloader = downloader
        self._extract = extract
        self._scan_interval_hours = scan_interval_hours
        self._logger = logging.getLogger(self.__class__.__name__)

    def handle_message(self, message: IncomingMessage) -> str:
        record = self._store.load(message.chat_id)
        if not record.name and message.from_name:
            record.name = message.from_name
        text = message.text.strip()

        if text.startswith("/help"):
            return HELP_TEXT
        if text.startswith("/reset"):
            record = self._store.reset(message.chat_id)
            record.name = message.from_name
            record.state = STATE_AWAITING_CV
            self._store.save(record)
            return (
                "Setup restarted.\n\n"
                "Please upload your CV (PDF, DOCX, or text)."
            )
        if text.startswith("/status"):
            return self._status_text(record)
        if text.startswith("/start"):
            return self._handle_start(record)

        handlers = {
            STATE_NEW: self._handle_start_needed,
            STATE_AWAITING_CV: self._handle_cv,
            STATE_AWAITING_MOTIVATION: self._handle_motivation,
            STATE_AWAITING_JOB_PREFS: self._handle_job_prefs,
            STATE_AWAITING_ANSWER: self._handle_answer,
            STATE_ACTIVE: self._handle_active_chat,
        }
        handler = handlers.get(record.state, self._handle_start_needed)
        reply = handler(record, message)
        self._store.save(record)
        return reply

    # ------------------------------------------------------------------
    # State handlers

    def _handle_start(self, record: UserRecord) -> str:
        if record.state == STATE_ACTIVE:
            return (
                "You're already set up. " + self._status_text(record) +
                "\n\nUse /run to scan now or /reset to start over."
            )
        if record.state in (STATE_NEW, ""):
            record.state = STATE_AWAITING_CV
            self._store.save(record)
            name = f" {record.name}" if record.name else ""
            return WELCOME.format(name=name)
        self._store.save(record)
        return self._prompt_for_state(record)

    def _handle_start_needed(self, record: UserRecord, message: IncomingMessage) -> str:
        record.state = STATE_AWAITING_CV
        name = f" {record.name}" if record.name else ""
        return WELCOME.format(name=name)

    def _handle_cv(self, record: UserRecord, message: IncomingMessage) -> str:
        text = self._document_or_text(message, min_chars=MIN_PASTED_DOC_CHARS)
        if isinstance(text, str):
            self._store.save_document(record.chat_id, "cv", text)
            record.state = STATE_AWAITING_MOTIVATION
            return (
                f"✅ Got your CV ({len(text)} characters).\n\n"
                "Next: upload or paste your motivation letter. It helps me "
                "understand what you're looking for. Send /skip if you don't "
                "have one."
            )
        return text.reply  # extraction problem or no usable input

    def _handle_motivation(self, record: UserRecord, message: IncomingMessage) -> str:
        if message.text.strip().startswith("/skip"):
            record.state = STATE_AWAITING_JOB_PREFS
            return self._job_prefs_prompt()
        text = self._document_or_text(message, min_chars=MIN_PASTED_DOC_CHARS)
        if isinstance(text, str):
            self._store.save_document(record.chat_id, "motivation", text)
            record.state = STATE_AWAITING_JOB_PREFS
            return "✅ Motivation letter received.\n\n" + self._job_prefs_prompt()
        return text.reply

    def _handle_job_prefs(self, record: UserRecord, message: IncomingMessage) -> str:
        if message.text.strip().startswith("/skip"):
            self._store.save_document(record.chat_id, "job_prefs", "")
            return self._run_extraction(record)
        text = self._document_or_text(message, min_chars=10)
        if isinstance(text, str):
            self._store.save_document(record.chat_id, "job_prefs", text)
            return self._run_extraction(record)
        return text.reply

    def _handle_answer(self, record: UserRecord, message: IncomingMessage) -> str:
        answer = message.text.strip()
        if not answer:
            return "Please answer in a short text message."
        if not record.pending_questions:
            return self._run_extraction(record)
        question = record.pending_questions.pop(0)
        record.answers.append({"question": question, "answer": answer})
        if record.pending_questions:
            return self._ask_next_question(record)
        return self._run_extraction(record)

    def _handle_active_chat(self, record: UserRecord, message: IncomingMessage) -> str:
        return (
            "You're all set up - I'm scanning for jobs regularly.\n" + HELP_TEXT
        )

    # ------------------------------------------------------------------
    # Extraction and finalization

    def _run_extraction(self, record: UserRecord) -> str:
        cv = self._store.load_document(record.chat_id, "cv")
        motivation = self._store.load_document(record.chat_id, "motivation")
        prefs_text = self._store.load_document(record.chat_id, "job_prefs")
        try:
            extraction = self._extract(cv, motivation, prefs_text, record.answers)
        except Exception as exc:
            self._logger.warning(
                "Intake extraction failed for %s: %s", record.chat_id, exc
            )
            extraction = self._fallback_extraction(record, prefs_text)

        questions = self._missing_info_questions(record, extraction)
        if questions and len(record.answers) < MAX_ANSWERED_QUESTIONS:
            record.pending_questions = questions
            record.state = STATE_AWAITING_ANSWER
            return (
                "I need a bit more information.\n\n"
                + self._ask_next_question(record)
            )
        return self._finalize(record, extraction)

    def _missing_info_questions(
        self, record: UserRecord, extraction: IntakeExtraction
    ) -> list[str]:
        questions = list(extraction.questions[:3])
        answered = {entry["question"] for entry in record.answers}
        if not extraction.locations and QUESTION_LOCATIONS not in answered:
            questions.append(QUESTION_LOCATIONS)
        if not extraction.job_titles and QUESTION_ROLES not in answered:
            questions.append(QUESTION_ROLES)
        deduped: list[str] = []
        for question in questions:
            if question not in answered and question not in deduped:
                deduped.append(question)
        return deduped[:3]

    def _fallback_extraction(
        self, record: UserRecord, prefs_text: str
    ) -> IntakeExtraction:
        """Build a minimal extraction from raw answers when the LLM fails."""
        locations: list[str] = []
        job_titles: list[str] = []
        for entry in record.answers:
            if entry["question"] == QUESTION_LOCATIONS:
                locations = [entry["answer"]]
            elif entry["question"] == QUESTION_ROLES:
                job_titles = [
                    part.strip()
                    for part in entry["answer"].replace(";", ",").split(",")
                    if part.strip()
                ]
        keywords = [
            word for word in prefs_text.replace(",", " ").split() if len(word) > 3
        ][:8]
        return IntakeExtraction(
            job_titles=job_titles, keywords=keywords, locations=locations
        )

    def _finalize(self, record: UserRecord, extraction: IntakeExtraction) -> str:
        location = extraction.locations[0] if extraction.locations else ""
        record.preferences = {
            "location": location,
            "locations": extraction.locations,
            "job_titles": extraction.job_titles,
            "job_description_keywords": extraction.keywords,
        }
        record.pending_questions = []
        record.state = STATE_ACTIVE
        summary = self._format_parameters(record.preferences)
        return (
            "\U0001f389 You're all set!\n\n"
            f"{summary}\n\n"
            f"I'll scan for matching jobs about every "
            f"{self._format_interval()} and message you when I find new "
            "ones. Use /run to start a scan right now, /status to check "
            "your setup, or /reset to change your documents."
        )

    # ------------------------------------------------------------------
    # Helpers

    class _NoText:
        def __init__(self, reply: str) -> None:
            self.reply = reply

    def _document_or_text(
        self, message: IncomingMessage, min_chars: int
    ) -> str | IntakeManager._NoText:
        if message.document is not None:
            try:
                data = self._downloader.download_document(message.document)
                return extract_text(
                    data, message.document.file_name, message.document.mime_type
                )
            except (DocumentExtractionError, TelegramError) as exc:
                return self._NoText(f"⚠️ {exc}\n\nPlease try another file.")
        text = message.text.strip()
        if text and not text.startswith("/") and len(text) >= min_chars:
            return text
        return self._NoText(
            "Please upload a document (PDF, DOCX, or text file) or paste "
            "the content as a message."
        )

    def _ask_next_question(self, record: UserRecord) -> str:
        remaining = len(record.pending_questions)
        prefix = f"❓ ({remaining} question{'s' if remaining > 1 else ''} left) "
        return prefix + record.pending_questions[0]

    def _job_prefs_prompt(self) -> str:
        return (
            "Now describe the jobs you're looking for: roles, industries, "
            "seniority, remote/on-site, and where you want to work "
            "(country and cities). You can also upload a document. "
            "Send /skip to let me infer everything from your CV."
        )

    def _prompt_for_state(self, record: UserRecord) -> str:
        prompts = {
            STATE_AWAITING_CV: "Please upload your CV (PDF, DOCX, or text).",
            STATE_AWAITING_MOTIVATION: (
                "Please upload your motivation letter, or send /skip."
            ),
            STATE_AWAITING_JOB_PREFS: self._job_prefs_prompt(),
        }
        if record.state == STATE_AWAITING_ANSWER and record.pending_questions:
            return self._ask_next_question(record)
        return prompts.get(record.state, "Send /start to begin.")

    def _status_text(self, record: UserRecord) -> str:
        if record.state != STATE_ACTIVE:
            return (
                f"Setup in progress (step: {record.state}).\n"
                + self._prompt_for_state(record)
            )
        return "Your search parameters:\n" + self._format_parameters(
            record.preferences
        )

    def _format_parameters(self, preferences: dict[str, Any]) -> str:
        titles = ", ".join(preferences.get("job_titles", [])) or "-"
        locations = ", ".join(preferences.get("locations", [])) or "-"
        keywords = ", ".join(preferences.get("job_description_keywords", [])) or "-"
        return (
            f"\U0001f3af Roles: {titles}\n"
            f"\U0001f4cd Locations: {locations}\n"
            f"\U0001f511 Keywords: {keywords}"
        )

    def _format_interval(self) -> str:
        hours = self._scan_interval_hours
        if hours < 1:
            return f"{int(hours * 60)} minutes"
        if hours == int(hours):
            return f"{int(hours)} hour{'s' if hours != 1 else ''}"
        return f"{hours:g} hours"
