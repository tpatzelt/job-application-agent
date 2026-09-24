from __future__ import annotations

from pathlib import Path

import pytest

from src.intake import (
    MIN_PASTED_DOC_CHARS,
    QUESTION_LANGUAGE,
    QUESTION_LOCATIONS,
    QUESTION_ROLES,
    IntakeManager,
)
from src.models import IntakeExtraction
from src.telegram_api import IncomingDocument, IncomingMessage, TelegramError
from src.user_store import (
    STATE_ACTIVE,
    STATE_AWAITING_ANSWER,
    STATE_AWAITING_CV,
    STATE_AWAITING_JOB_PREFS,
    STATE_AWAITING_MOTIVATION,
    UserStore,
)

CV_TEXT = "Experienced project manager. " * 20

# The exact replies _document_or_text produces (src/intake.py:312-329) for a
# too-short pasted answer and for a document that fails extraction.
SHORT_TEXT_REPLY = (
    "Please upload a document (PDF, DOCX, or text file) or paste "
    "the content as a message."
)
UNREADABLE_DOC_REPLY = (
    "⚠️ Could not extract readable text from the document. If it is a "
    "scanned PDF, please send a text-based PDF, DOCX, or plain text.\n\n"
    "Please try another file."
)


class FakeDownloader:
    def __init__(self, payload: bytes = CV_TEXT.encode()) -> None:
        self.payload = payload

    def download_document(self, document: IncomingDocument) -> bytes:
        return self.payload


class FailingDownloader:
    """download_document always raises, simulating a failed Telegram fetch."""

    def download_document(self, document: IncomingDocument) -> bytes:
        raise TelegramError("download failed")


class ScriptedExtractor:
    """Returns queued extractions; records the calls it receives."""

    def __init__(self, extractions: list[IntakeExtraction]) -> None:
        self.extractions = list(extractions)
        self.calls: list[dict] = []

    def __call__(self, cv, motivation, prefs, answers) -> IntakeExtraction:
        self.calls.append(
            {"cv": cv, "motivation": motivation, "prefs": prefs, "answers": answers}
        )
        if not self.extractions:
            raise RuntimeError("LLM unavailable")
        return self.extractions.pop(0)


def _msg(text: str = "", document: IncomingDocument | None = None) -> IncomingMessage:
    return IncomingMessage(chat_id="42", text=text, document=document, from_name="Tim")


def _doc() -> IncomingDocument:
    return IncomingDocument(
        file_id="f1", file_name="cv.txt", mime_type="text/plain", file_size=100
    )


def _manager(tmp_path: Path, extractor) -> tuple[IntakeManager, UserStore]:
    store = UserStore(tmp_path)
    manager = IntakeManager(store, FakeDownloader(), extractor)
    return manager, store


COMPLETE = IntakeExtraction(
    job_titles=["Project Manager"],
    keywords=["digital transformation"],
    industries=["public sector"],
    locations=["Berlin, Germany"],
    language="English",
)


def test_full_intake_happy_path(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)

    reply = manager.handle_message(_msg("/start"))
    assert "upload your CV" in reply
    assert store.load("42").state == STATE_AWAITING_CV

    reply = manager.handle_message(_msg(document=_doc()))
    assert "Got your CV" in reply
    assert store.load("42").state == STATE_AWAITING_MOTIVATION
    assert "project manager" in store.load_document("42", "cv").lower()

    reply = manager.handle_message(_msg("/skip"))
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS

    reply = manager.handle_message(_msg("PM roles in Berlin, public sector"))
    record = store.load("42")
    assert record.state == STATE_ACTIVE
    assert "all set" in reply
    assert record.preferences["job_titles"] == ["Project Manager"]
    assert record.preferences["location"] == "Berlin, Germany"
    assert record.preferences["locations"] == ["Berlin, Germany"]
    assert record.preferences["industries"] == ["public sector"]
    assert record.preferences["language"] == "english"
    assert extractor.calls[0]["prefs"] == "PM roles in Berlin, public sector"


def test_missing_location_triggers_question_then_finalizes(tmp_path: Path) -> None:
    no_location = IntakeExtraction(
        job_titles=["Engineer"], keywords=["python"], language="English"
    )
    extractor = ScriptedExtractor([no_location, COMPLETE])
    manager, store = _manager(tmp_path, extractor)

    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    reply = manager.handle_message(_msg("backend jobs"))

    record = store.load("42")
    assert record.state == STATE_AWAITING_ANSWER
    assert QUESTION_LOCATIONS in reply

    reply = manager.handle_message(_msg("Berlin, Germany"))
    record = store.load("42")
    assert record.state == STATE_ACTIVE
    # The answer was passed back into the second extraction call.
    assert extractor.calls[1]["answers"] == [
        {"question": QUESTION_LOCATIONS, "answer": "Berlin, Germany"}
    ]
    assert record.preferences["location"] == "Berlin, Germany"


def test_extraction_failure_falls_back_to_answers(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([])  # every call raises
    manager, store = _manager(tmp_path, extractor)

    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    reply = manager.handle_message(_msg("data science jobs please"))
    # LLM failed, so the deterministic questions are asked.
    assert store.load("42").state == STATE_AWAITING_ANSWER

    manager.handle_message(_msg("Munich, Germany"))
    manager.handle_message(_msg("Data Scientist, ML Engineer"))
    reply = manager.handle_message(_msg("German"))
    record = store.load("42")
    assert record.state == STATE_ACTIVE
    assert record.preferences["location"] == "Munich, Germany"
    assert record.preferences["job_titles"] == ["Data Scientist", "ML Engineer"]
    assert record.preferences["language"] == "german"
    assert "all set" in reply


def test_pasted_cv_text_accepted(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    reply = manager.handle_message(_msg(CV_TEXT))
    assert "Got your CV" in reply
    assert store.load("42").state == STATE_AWAITING_MOTIVATION


def test_short_text_in_cv_state_is_rejected(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    assert len("hello") < MIN_PASTED_DOC_CHARS
    reply = manager.handle_message(_msg("hello"))
    assert "upload a document" in reply.lower()
    assert store.load("42").state == STATE_AWAITING_CV


def test_reset_restarts_intake(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    manager.handle_message(_msg("jobs in Berlin"))
    assert store.load("42").state == STATE_ACTIVE

    reply = manager.handle_message(_msg("/reset"))
    assert "restarted" in reply.lower()
    record = store.load("42")
    assert record.state == STATE_AWAITING_CV
    assert record.preferences == {}
    assert store.load_document("42", "cv") == ""


def test_status_during_setup_and_when_active(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    reply = manager.handle_message(_msg("/status"))
    assert "Setup in progress" in reply

    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    manager.handle_message(_msg("jobs in Berlin"))
    reply = manager.handle_message(_msg("/status"))
    assert "Project Manager" in reply
    assert "Berlin, Germany" in reply


def test_bad_document_reports_error_and_keeps_state(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    downloader = FakeDownloader(payload=b"x")  # too short to be a real CV
    manager = IntakeManager(store, downloader, ScriptedExtractor([COMPLETE]))
    manager.handle_message(_msg("/start"))
    reply = manager.handle_message(_msg(document=_doc()))
    assert "⚠" in reply
    assert store.load("42").state == STATE_AWAITING_CV


def test_cv_state_short_text_and_bad_document_exact_replies(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    manager = IntakeManager(
        store, FakeDownloader(payload=b"x"), ScriptedExtractor([COMPLETE])
    )
    manager.handle_message(_msg("/start"))

    reply = manager.handle_message(_msg("hello"))
    assert reply == SHORT_TEXT_REPLY
    assert store.load("42").state == STATE_AWAITING_CV

    reply = manager.handle_message(_msg(document=_doc()))
    assert reply == UNREADABLE_DOC_REPLY
    assert store.load("42").state == STATE_AWAITING_CV


def test_cv_state_failed_download_exact_reply(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    manager = IntakeManager(store, FailingDownloader(), ScriptedExtractor([COMPLETE]))
    manager.handle_message(_msg("/start"))

    reply = manager.handle_message(_msg(document=_doc()))
    assert reply == "⚠️ download failed\n\nPlease try another file."
    assert store.load("42").state == STATE_AWAITING_CV


def test_motivation_state_short_text_and_bad_document_exact_replies(
    tmp_path: Path,
) -> None:
    store = UserStore(tmp_path)
    manager = IntakeManager(
        store, FakeDownloader(payload=b"x"), ScriptedExtractor([COMPLETE])
    )
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(CV_TEXT))  # pasted CV, avoids the bad downloader
    assert store.load("42").state == STATE_AWAITING_MOTIVATION

    reply = manager.handle_message(_msg("hi"))
    assert reply == SHORT_TEXT_REPLY
    assert store.load("42").state == STATE_AWAITING_MOTIVATION

    reply = manager.handle_message(_msg(document=_doc()))
    assert reply == UNREADABLE_DOC_REPLY
    assert store.load("42").state == STATE_AWAITING_MOTIVATION


def test_motivation_state_failed_download_exact_reply(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    manager = IntakeManager(store, FailingDownloader(), ScriptedExtractor([COMPLETE]))
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(CV_TEXT))
    assert store.load("42").state == STATE_AWAITING_MOTIVATION

    reply = manager.handle_message(_msg(document=_doc()))
    assert reply == "⚠️ download failed\n\nPlease try another file."
    assert store.load("42").state == STATE_AWAITING_MOTIVATION


def test_job_prefs_state_short_text_and_bad_document_exact_replies(
    tmp_path: Path,
) -> None:
    store = UserStore(tmp_path)
    manager = IntakeManager(
        store, FakeDownloader(payload=b"x"), ScriptedExtractor([COMPLETE])
    )
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(CV_TEXT))
    manager.handle_message(_msg("/skip"))
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS

    reply = manager.handle_message(_msg("hi"))
    assert reply == SHORT_TEXT_REPLY
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS

    reply = manager.handle_message(_msg(document=_doc()))
    assert reply == UNREADABLE_DOC_REPLY
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS


def test_job_prefs_state_failed_download_exact_reply(tmp_path: Path) -> None:
    store = UserStore(tmp_path)
    manager = IntakeManager(store, FailingDownloader(), ScriptedExtractor([COMPLETE]))
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(CV_TEXT))
    manager.handle_message(_msg("/skip"))
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS

    reply = manager.handle_message(_msg(document=_doc()))
    assert reply == "⚠️ download failed\n\nPlease try another file."
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS


NO_LANGUAGE = IntakeExtraction(
    job_titles=["Project Manager"],
    keywords=["digital transformation"],
    locations=["Berlin, Germany"],
)


def test_missing_language_triggers_question_then_finalizes(tmp_path: Path) -> None:
    complete_de = NO_LANGUAGE.model_copy(update={"language": "Deutsch"})
    extractor = ScriptedExtractor([NO_LANGUAGE, complete_de])
    manager, store = _manager(tmp_path, extractor)

    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    reply = manager.handle_message(_msg("PM roles in Berlin"))

    assert store.load("42").state == STATE_AWAITING_ANSWER
    assert QUESTION_LANGUAGE in reply

    manager.handle_message(_msg("German"))
    record = store.load("42")
    assert record.state == STATE_ACTIVE
    # The answer was passed back into the second extraction call.
    assert extractor.calls[1]["answers"] == [
        {"question": QUESTION_LANGUAGE, "answer": "German"}
    ]
    # "Deutsch" from the extraction normalizes to the canonical name.
    assert record.preferences["language"] == "german"


def test_language_skip_falls_back_to_input_language(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([NO_LANGUAGE, NO_LANGUAGE])
    manager, store = _manager(tmp_path, extractor)

    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    reply = manager.handle_message(
        _msg(
            "I am looking for a role in the public sector with a focus on "
            "digital transformation and the chance to lead projects."
        )
    )
    assert QUESTION_LANGUAGE in reply

    manager.handle_message(_msg("/skip"))
    record = store.load("42")
    assert record.state == STATE_ACTIVE
    # No stated preference: the language of the user's own (English) input.
    assert record.preferences["language"] == "english"


@pytest.mark.parametrize("command", ["/help", "/start"])
def test_commands_always_reply(tmp_path: Path, command: str) -> None:
    manager, _ = _manager(tmp_path, ScriptedExtractor([COMPLETE]))
    assert manager.handle_message(_msg(command))


def test_uploaded_motivation_letter_reply_and_job_prefs_prompt(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))

    reply = manager.handle_message(_msg(document=_doc()))
    assert "✅ Motivation letter received." in reply
    assert "Now describe the jobs you're looking for" in reply
    assert "Send /skip to let me infer everything from your CV." in reply
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS


def test_job_prefs_prompt_after_motivation_skip(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))

    reply = manager.handle_message(_msg("/skip"))
    assert "Now describe the jobs you're looking for" in reply
    assert "Send /skip to let me infer everything from your CV." in reply
    assert "Motivation letter received" not in reply
    assert store.load("42").state == STATE_AWAITING_JOB_PREFS


def test_empty_answer_is_rejected(tmp_path: Path) -> None:
    no_location = IntakeExtraction(
        job_titles=["Engineer"], keywords=["python"], language="English"
    )
    extractor = ScriptedExtractor([no_location, COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    manager.handle_message(_msg("backend jobs"))
    assert store.load("42").state == STATE_AWAITING_ANSWER

    reply = manager.handle_message(_msg("   "))
    assert reply == "Please answer in a short text message."
    assert store.load("42").state == STATE_AWAITING_ANSWER


def test_active_state_chat_reply(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    manager.handle_message(_msg("jobs in Berlin"))
    assert store.load("42").state == STATE_ACTIVE

    reply = manager.handle_message(_msg("what's up"))
    assert "You're all set up - I'm scanning for jobs regularly." in reply
    assert "/status - your profile and search parameters" in reply
    assert store.load("42").state == STATE_ACTIVE


def test_start_reply_when_already_active(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))
    manager.handle_message(_msg("jobs in Berlin"))
    assert store.load("42").state == STATE_ACTIVE

    reply = manager.handle_message(_msg("/start"))
    assert "You're already set up." in reply
    assert "Your search parameters:" in reply
    assert "Use /run to scan now or /reset to start over." in reply
    assert store.load("42").state == STATE_ACTIVE


def test_need_more_info_lead_in(tmp_path: Path) -> None:
    no_location = IntakeExtraction(
        job_titles=["Engineer"], keywords=["python"], language="English"
    )
    extractor = ScriptedExtractor([no_location, COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))

    reply = manager.handle_message(_msg("backend jobs"))
    assert "I need a bit more information.\n\n" in reply
    assert store.load("42").state == STATE_AWAITING_ANSWER


def test_question_counter_plural_then_singular(tmp_path: Path) -> None:
    missing_both = IntakeExtraction(keywords=["python"], language="English")
    extractor = ScriptedExtractor([missing_both, COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    manager.handle_message(_msg("/skip"))

    reply = manager.handle_message(_msg("open to anything"))
    assert store.load("42").state == STATE_AWAITING_ANSWER
    assert record_pending_count(store, "42") == 2
    assert reply == (
        "I need a bit more information.\n\n"
        f"❓ (2 questions left) {QUESTION_LOCATIONS}"
    )

    reply = manager.handle_message(_msg("Berlin, Germany"))
    assert store.load("42").state == STATE_AWAITING_ANSWER
    assert record_pending_count(store, "42") == 1
    assert f"❓ (1 question left) {QUESTION_ROLES}" == reply


def record_pending_count(store: UserStore, chat_id: str) -> int:
    return len(store.load(chat_id).pending_questions)


def test_prompt_for_state_via_status_and_start_while_awaiting_motivation(
    tmp_path: Path,
) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    manager.handle_message(_msg(document=_doc()))
    assert store.load("42").state == STATE_AWAITING_MOTIVATION

    status_reply = manager.handle_message(_msg("/status"))
    assert "Setup in progress (step: awaiting_motivation)." in status_reply
    assert "Please upload your motivation letter, or send /skip." in status_reply

    start_reply = manager.handle_message(_msg("/start"))
    assert start_reply == "Please upload your motivation letter, or send /skip."
    assert store.load("42").state == STATE_AWAITING_MOTIVATION


def test_prompt_for_state_fallback_for_unknown_state(tmp_path: Path) -> None:
    extractor = ScriptedExtractor([COMPLETE])
    manager, store = _manager(tmp_path, extractor)
    manager.handle_message(_msg("/start"))
    record = store.load("42")
    record.state = "some_unknown_state"
    store.save(record)

    status_reply = manager.handle_message(_msg("/status"))
    assert "Setup in progress (step: some_unknown_state)." in status_reply
    assert "Send /start to begin." in status_reply

    start_reply = manager.handle_message(_msg("/start"))
    assert start_reply == "Send /start to begin."
    assert store.load("42").state == "some_unknown_state"
