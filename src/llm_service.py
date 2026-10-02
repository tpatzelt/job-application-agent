from __future__ import annotations

import json
import logging
import random
import re
import time
from typing import Any

from litellm import completion
from litellm.exceptions import (
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
)

from .config_manager import Config, EffortBudget
from .models import (
    IntakeExtraction,
    JobEvaluation,
    Reflection,
    SearchPlan,
    SearchQueries,
)
from .rate_limiter import DailyQuotaExceeded, estimate_tokens, shared_limiter
from pydantic import ValidationError

REPAIR_ATTEMPTS = 2
# Pace to 90% of the input-token quota: the provider counts a few tokens
# (chat template, system prompt) that the estimate does not see.
TPM_HEADROOM = 0.9
# Wait used when a 429 does not say how long to back off.
DEFAULT_RATE_LIMIT_DELAY = 60.0
_RETRY_DELAY_RE = re.compile(r'retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)s')
# Backoff between retries of a provider-side error (5xx). Gemini's
# "500 Internal error encountered." comes in bursts: three retries a second
# apart all failed on 2026-10-02 and ended a scan, so wait longer each time.
SERVER_ERROR_DELAYS = (10.0, 30.0, 60.0)


class LLMUnavailableError(RuntimeError):
    """The provider never answered the call, so nothing was judged."""


class RateLimitedError(LLMUnavailableError):
    """The provider kept rejecting a call for quota even after waiting."""


class ProviderUnavailableError(LLMUnavailableError):
    """The provider kept failing on its side (5xx) even after backing off."""


def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, RateLimitError) or getattr(exc, "status_code", None) == 429


def _is_server_error(exc: Exception) -> bool:
    if isinstance(exc, (InternalServerError, ServiceUnavailableError, BadGatewayError)):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and 500 <= status < 600


def _retry_delay(exc: Exception) -> float:
    """The backoff a 429 asks for (Gemini sends RetryInfo.retryDelay)."""
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) if match else DEFAULT_RATE_LIMIT_DELAY


def _prompt_tokens(response: Any) -> int | None:
    try:
        tokens = response["usage"]["prompt_tokens"]
    except (KeyError, IndexError, TypeError):
        tokens = getattr(getattr(response, "usage", None), "prompt_tokens", None)
    return tokens if isinstance(tokens, int) and tokens > 0 else None


def _response_content(response: Any) -> str | None:
    """The assistant message text, or None when the model returned nothing.

    litellm mirrors the provider payload, and providers do answer with
    `content: null` (empty completion, reasoning-only output, truncated
    stream), so this can never be assumed to be a string.
    """
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None
    return content if isinstance(content, str) else None


class LLMService:
    def __init__(self, config: Config, budget: EffortBudget, api_key: str | None):
        self._config = config
        self._budget = budget
        self._api_key = api_key
        self._logger = logging.getLogger(self.__class__.__name__)
        tpm = int(config.llm_input_tpm_limit * TPM_HEADROOM)
        # Largest prompt that still fits one minute's input-token quota.
        self._max_prompt_tokens = tpm or None
        self._limiter = (
            shared_limiter(
                config.llm_model, config.llm_rpm_limit, tpm, config.llm_rpd_limit
            )
            if config.llm_rpm_limit or tpm or config.llm_rpd_limit
            else None
        )

    def generate_search_queries(
        self, context: dict[str, Any], history: list[dict[str, Any]]
    ) -> SearchQueries:
        prompt = self._build_query_prompt(context, history)
        response_text = self._call_llm(prompt)
        payload = self._parse_json_payload(response_text, prompt)
        return SearchQueries.model_validate(payload)

    def plan_search(self, context: dict[str, Any]) -> SearchPlan:
        prompt = self._build_plan_prompt(context)
        response_text = self._call_llm(prompt)
        payload = self._parse_json_payload(response_text, prompt)
        try:
            return SearchPlan.model_validate(payload)
        except ValidationError as exc:
            self._logger.warning(
                "Search plan payload failed validation, returning default. payload=%s error=%s",
                payload,
                exc,
            )
            return SearchPlan()

    def reflect(
        self,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        tool_stats: dict[str, Any],
    ) -> Reflection:
        prompt = self._build_reflection_prompt(context, history, tool_stats)
        response_text = self._call_llm(prompt)
        payload = self._parse_json_payload(response_text, prompt)
        try:
            return Reflection.model_validate(payload)
        except ValidationError as exc:
            self._logger.warning(
                "Reflection payload failed validation, returning default. payload=%s error=%s",
                payload,
                exc,
            )
            return Reflection()

    def extract_search_profile(
        self,
        cv_text: str,
        motivation_text: str,
        job_prefs_text: str,
        answers: list[dict[str, str]] | None = None,
    ) -> IntakeExtraction:
        prompt = self._build_intake_prompt(
            cv_text, motivation_text, job_prefs_text, answers or []
        )
        response_text = self._call_llm(prompt)
        payload = self._parse_json_payload(response_text, prompt)
        try:
            return IntakeExtraction.model_validate(payload)
        except ValidationError as exc:
            self._logger.warning(
                "Intake extraction payload failed validation, returning default. "
                "payload=%s error=%s",
                payload,
                exc,
            )
            return IntakeExtraction()

    def evaluate_job(
        self,
        cv: str,
        job_description: str,
        preferences: dict[str, Any] | None = None,
    ) -> JobEvaluation:
        prompt = self._build_evaluation_prompt(cv, job_description, preferences)
        response_text = self._call_llm(prompt)
        payload = self._parse_json_payload(response_text, prompt)
        # Ensure required evaluation fields exist to avoid Pydantic
        # validation errors when the LLM returns incomplete JSON.
        if not isinstance(payload, dict):
            payload = {}
        if "score" not in payload or "reason" not in payload:
            self._logger.warning(
                "LLM returned incomplete evaluation payload, filling defaults: %s",
                payload,
            )
            payload.setdefault("score", 0)
            payload.setdefault(
                "reason",
                "LLM did not return a valid evaluation; defaulting to score 0",
            )
        try:
            return JobEvaluation.model_validate(payload)
        except ValidationError as exc:
            # Defensive: if validation fails, return a safe default so the
            # orchestrator can continue. Log the issue with payload details.
            self._logger.warning(
                "Job evaluation payload failed validation, returning default. payload=%s error=%s",
                payload,
                exc,
            )
            return JobEvaluation.model_validate(
                {"score": 0, "reason": "Invalid evaluation returned by LLM"}
            )

    def _call_llm(self, prompt: str) -> str:
        if not self._budget.can_call_llm():
            raise RuntimeError("Effort budget exceeded: LLM calls")
        self._budget.record_llm_call()

        self._logger.info("Calling LLM model %s", self._config.llm_model)
        return self._complete(prompt, self._config.llm_temperature)

    def _complete(self, prompt: str, temperature: float) -> str:
        """Non-empty assistant text, retrying empty answers and call errors.

        Quota rejections (429) have their own, longer retry budget: each one
        waits the delay the provider asks for, since retrying a second later
        lands in the same exhausted minute.
        """
        last_error: Exception | None = None
        attempt = 0
        rate_limited = 0
        server_errors = 0
        while attempt < self._config.llm_max_retries:
            reservation = (
                self._limiter.acquire(estimate_tokens(prompt))
                if self._limiter
                else None
            )
            try:
                response = completion(
                    model=self._config.llm_model,
                    messages=[
                        {"role": "system", "content": "Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    api_key=self._api_key,
                    # One free-tier call once hung for eight minutes.
                    timeout=self._config.llm_timeout_seconds,
                )
            except Exception as exc:
                if _is_rate_limit(exc):
                    rate_limited += 1
                    self._on_rate_limit(exc, rate_limited)
                    continue
                if _is_server_error(exc):
                    server_errors += 1
                    self._on_server_error(exc, server_errors)
                    continue
                last_error = exc
                attempt += 1
                self._logger.warning(
                    "LLM call failed on attempt %s/%s: %s",
                    attempt,
                    self._config.llm_max_retries,
                    exc,
                )
            else:
                used = _prompt_tokens(response)
                if used is not None and reservation is not None:
                    self._limiter.settle(reservation, used)
                    self._logger.info(
                        "LLM call used %s input tokens (estimated %s)",
                        used,
                        estimate_tokens(prompt),
                    )
                content = _response_content(response)
                if content:
                    return content
                # Free/small models sometimes answer with null content.
                # Returning it would blow up in json.loads far from here.
                last_error = RuntimeError("LLM returned an empty response")
                attempt += 1
                self._logger.warning(
                    "LLM returned empty content on attempt %s/%s",
                    attempt,
                    self._config.llm_max_retries,
                )
            if attempt < self._config.llm_max_retries:
                time.sleep(self._config.llm_min_delay_seconds)
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _on_server_error(self, exc: Exception, count: int) -> None:
        """Back off after a 5xx, or give up once the retries are spent."""
        if count > self._config.llm_server_error_retries:
            raise ProviderUnavailableError(f"LLM provider error: {exc}") from exc
        delay = SERVER_ERROR_DELAYS[min(count, len(SERVER_ERROR_DELAYS)) - 1]
        delay += random.uniform(0.0, 3.0)
        self._logger.warning(
            "LLM provider error (%s/%s), retrying in %.0fs: %s",
            count,
            self._config.llm_server_error_retries,
            delay,
            exc,
        )
        time.sleep(delay)

    def _on_rate_limit(self, exc: Exception, count: int) -> None:
        """Back off after a 429, or give up once the retries are spent."""
        if "PerDay" in str(exc):
            raise DailyQuotaExceeded(f"LLM daily quota exhausted: {exc}") from exc
        if count > self._config.llm_rate_limit_retries:
            raise RateLimitedError(f"LLM rate limited: {exc}") from exc
        delay = _retry_delay(exc) + random.uniform(1.0, 5.0)
        self._logger.warning(
            "LLM rate limited (%s/%s), retrying in %.0fs",
            count,
            self._config.llm_rate_limit_retries,
            delay,
        )
        if self._limiter:
            # Hold back every caller, not only this one: they share the quota.
            self._limiter.block_for(delay)
        else:
            time.sleep(delay)

    def _parse_json_payload(
        self, response_text: str | None, prompt: str
    ) -> dict[str, Any]:
        if not response_text:
            return self._retry_json_response(prompt, response_text or "")
        try:
            payload = json.loads(response_text)
            payload = self._normalize_payload(payload)
            # If the prompt requested an evaluation schema but the
            # returned payload is missing required keys, attempt a repair
            # call to force the model to emit the expected JSON structure.
            if ('"score"' in prompt and '"reason"' in prompt) and (
                "score" not in payload or "reason" not in payload
            ):
                return self._retry_json_response(prompt, response_text)
            return payload
        except json.JSONDecodeError:
            repaired = self._extract_json_object(response_text)
            if repaired is None:
                return self._retry_json_response(prompt, response_text)
            try:
                payload = json.loads(repaired)
                payload = self._normalize_payload(payload)
                # If the prompt requested an evaluation schema but the
                # returned payload is missing required keys (e.g. LLM
                # echoed the prompt instead of producing the fields),
                # attempt a repair call to force the model to emit the
                # expected JSON structure.
                if (
                    '"score"' in prompt
                    and '"reason"' in prompt
                    and ("score" not in payload or "reason" not in payload)
                ):
                    return self._retry_json_response(prompt, response_text)
                return payload
            except json.JSONDecodeError:
                return self._retry_json_response(prompt, response_text)

    def _retry_json_response(self, prompt: str, response_text: str) -> dict[str, Any]:
        # Free models (and openrouter/free, which routes each call to a
        # different model) botch repairs too, so one that comes back empty
        # or as prose is worth one more try before the
        # page (or the run's query generation) is given up on.
        last_error: Exception | None = None
        for attempt in range(1, REPAIR_ATTEMPTS + 1):
            if not self._budget.can_call_llm():
                raise RuntimeError("Effort budget exceeded: LLM calls")
            self._budget.record_llm_call()

            repair_prompt = prompt
            if response_text:
                with_response = (
                    "You must output ONLY valid JSON that matches the output schema. "
                    "Do not include extra text. Fix this response and return JSON only.\n\n"
                    f"Response: {response_text}\n\n"
                    f"Original prompt: {prompt}"
                )
                # Echoing the bad answer back roughly doubles a long prompt;
                # past one minute's token quota it could never be sent, so
                # just ask the original prompt again.
                if self._fits(with_response):
                    repair_prompt = with_response
            try:
                fixed = self._complete(repair_prompt, 0.0)
            except (LLMUnavailableError, DailyQuotaExceeded):
                raise
            except RuntimeError as exc:
                last_error = exc
                continue
            payload = self._load_json(fixed)
            if payload is not None:
                return payload
            last_error = ValueError(f"not JSON: {fixed[:200]!r}")
            self._logger.warning(
                "LLM repair attempt %s/%s returned no JSON: %r",
                attempt,
                REPAIR_ATTEMPTS,
                fixed[:200],
            )
        raise RuntimeError(f"LLM repair failed: {last_error}")

    def _fits(self, prompt: str) -> bool:
        """Whether `prompt` fits one minute of the input-token quota."""
        return (
            self._max_prompt_tokens is None
            or estimate_tokens(prompt) <= self._max_prompt_tokens
        )

    def _load_json(self, text: str) -> dict[str, Any] | None:
        for candidate in (text, self._extract_json_object(text)):
            if candidate is None:
                continue
            try:
                return self._normalize_payload(json.loads(candidate))
            except json.JSONDecodeError:
                continue
        return None

    def _normalize_payload(self, payload: Any) -> dict[str, Any]:
        if isinstance(payload, list):
            payload = {"queries": payload}
        if not isinstance(payload, dict):
            return {"queries": []}
        if "score" in payload:
            try:
                payload["score"] = int(round(float(payload["score"])))
            except (TypeError, ValueError):
                pass
        if "reason" in payload and isinstance(payload["reason"], (list, dict)):
            payload["reason"] = json.dumps(payload["reason"], ensure_ascii=True)
        for key in ("location_match", "domain_match"):
            if key in payload and isinstance(payload[key], str):
                payload[key] = payload[key].strip().lower() in (
                    "true",
                    "yes",
                    "1",
                )
        return payload

    def _extract_json_object(self, text: str) -> str | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return cleaned[start : end + 1]

    def _build_query_prompt(
        self, context: dict[str, Any], history: list[dict[str, Any]]
    ) -> str:
        payload = {
            "task": "Generate search queries for job hunting.",
            "context": context,
            "history": history,
            "output_schema": {"queries": ["string"]},
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "Only include the keys in the output_schema.",
                (
                    "Prefer queries that surface individual job postings on "
                    "employer career sites and applicant tracking systems, "
                    "not job board search pages."
                ),
                (
                    "Mix in queries using the site: operator against applicant "
                    "tracking systems (e.g. site:boards.greenhouse.io, "
                    "site:jobs.lever.co, site:jobs.ashbyhq.com, "
                    "site:apply.workable.com, site:jobs.personio.de) to "
                    "surface individual job postings."
                ),
                (
                    "Mix in company-targeted queries like "
                    "'\"<company>\" careers <role>' for companies in "
                    "context.plan.target_companies that match the profile."
                ),
                (
                    "Avoid queries aimed at aggregators such as LinkedIn, "
                    "Indeed, StepStone, Glassdoor, or XING."
                ),
                (
                    "Every query must name one of the preferred locations "
                    "from context.preferences (city or country); if the "
                    "user wants remote work, use 'remote' plus the country."
                ),
                (
                    "When context.preferences.industries is set, every "
                    "query must pair the role with one of those industry "
                    "or domain terms (e.g. 'product manager food Berlin', "
                    "not just 'product manager Berlin')."
                ),
                (
                    "Write every query in the language named by "
                    "context.preferences.language when it is set (e.g. "
                    "German job titles and keywords for 'german'); "
                    "otherwise match the language of the CV and "
                    "preferences."
                ),
            ],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_plan_prompt(self, context: dict[str, Any]) -> str:
        payload = {
            "task": (
                "Create a job search plan from the CV and preferences: "
                "which roles to target, which skills to emphasize, "
                "which locations to search, which specific companies to "
                "check directly, and the overall strategy."
            ),
            "context": context,
            "output_schema": {
                "target_roles": ["string"],
                "key_skills": ["string"],
                "locations": ["string"],
                "target_companies": ["string"],
                "strategy": "string",
            },
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "Only include the keys in the output_schema.",
                (
                    "target_companies: list 5-15 real companies in the "
                    "target locations that are likely to hire for these "
                    "roles, so their career pages can be searched directly. "
                    "Do not list job boards or staffing agencies. When "
                    "context.preferences.industries is set, only list "
                    "companies in those industries, not generic tech "
                    "companies."
                ),
                (
                    "target_roles must carry the industry/domain qualifier "
                    "when context.preferences.industries is set or the CV "
                    "shows a clear industry (e.g. 'Product Manager Food', "
                    "not just 'Product Manager')."
                ),
                (
                    "If context.preferences.language is set, express "
                    "target_roles and key_skills in that language."
                ),
            ],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_reflection_prompt(
        self,
        context: dict[str, Any],
        history: list[dict[str, Any]],
        tool_stats: dict[str, Any],
    ) -> str:
        payload = {
            "task": (
                "Critique the job search performance so far. Identify which "
                "queries worked, which did not, and suggest concrete "
                "adjustments for the next round of queries."
            ),
            "context": context,
            "history": history,
            "tool_stats": tool_stats,
            "output_schema": {
                "assessment": "string",
                "effective_queries": ["string"],
                "ineffective_queries": ["string"],
                "adjustments": ["string"],
            },
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "Only include the keys in the output_schema.",
            ],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_intake_prompt(
        self,
        cv_text: str,
        motivation_text: str,
        job_prefs_text: str,
        answers: list[dict[str, str]],
    ) -> str:
        payload = {
            "task": (
                "Extract job search parameters from this user's documents. "
                "Derive concrete job titles to search for, keywords that a "
                "matching job description would contain, and the locations "
                "(city and country) where the user wants to work. If any "
                "essential information is missing or ambiguous (especially "
                "the country or cities, or what kind of role is wanted), "
                "add a short clarification question for the user instead of "
                "guessing."
            ),
            "cv": cv_text[:6000],
            "motivation_letter": motivation_text[:3000],
            "desired_jobs_description": job_prefs_text[:3000],
            "user_answers_to_previous_questions": answers,
            "output_schema": {
                "job_titles": ["string"],
                "keywords": ["string"],
                "industries": ["string"],
                "locations": ["string"],
                "language": "string",
                "questions": ["string"],
            },
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "Only include the keys in the output_schema.",
                "locations entries should look like 'Berlin, Germany'.",
                (
                    "industries: the industries or product domains the user "
                    "has worked in or wants to work in (e.g. 'food & "
                    "beverage', 'FMCG', 'public sector software')."
                ),
                (
                    "job_titles must carry the domain qualifier when the "
                    "documents show a clear industry: e.g. 'Product Manager "
                    "Food' rather than just 'Product Manager'."
                ),
                (
                    "If the documents span several industries and the "
                    "target industry is ambiguous, ask a clarification "
                    "question about it."
                ),
                (
                    "language: the language the user wants job postings "
                    "written in (e.g. 'German'), but only when the documents "
                    "or answers state one; otherwise leave it empty."
                ),
                "Ask at most 3 questions, only about missing essentials.",
                "If locations are known, do not ask about locations.",
                "If user_answers_to_previous_questions covers a topic, "
                "use those answers and do not re-ask.",
            ],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_evaluation_prompt(
        self,
        cv: str,
        job_description: str,
        preferences: dict[str, Any] | None = None,
    ) -> str:
        prompt = self._evaluation_prompt(cv, job_description, preferences)
        if self._fits(prompt):
            return prompt
        # Only a page bigger than a whole minute's token quota gets here
        # (around 40k characters of text, ten times a typical posting): such
        # a call is rejected outright however long it waits. The posting
        # itself sits near the top of the page text, and what overflows is
        # trailing boilerplate (related-jobs lists, footers), so keep the
        # head and say what was cut.
        full = len(job_description)

        def trimmed(keep: int) -> str:
            return self._evaluation_prompt(
                cv,
                job_description[:keep]
                + f" [... page text truncated: {full - keep} of {full} characters omitted]",
                preferences,
            )

        # Largest prefix that fits: JSON escaping makes the size per kept
        # character uneven (umlauts), so search rather than compute it.
        low, high = 0, full
        while low < high:
            mid = (low + high + 1) // 2
            if self._fits(trimmed(mid)):
                low = mid
            else:
                high = mid - 1
        keep = low
        prompt = trimmed(keep)
        self._logger.warning(
            "Job page text too long for one call's token quota: kept %s of %s characters",
            keep,
            full,
        )
        return prompt

    def _evaluation_prompt(
        self,
        cv: str,
        job_description: str,
        preferences: dict[str, Any] | None,
    ) -> str:
        preferences = preferences or {}
        locations = preferences.get("locations") or []
        if not locations and preferences.get("location"):
            locations = [preferences["location"]]
        payload = {
            "task": "Evaluate job relevance to the CV and preferences.",
            "cv": cv,
            "preferred_locations": locations,
            "preferred_job_titles": preferences.get("job_titles") or [],
            "preferred_keywords": preferences.get("job_description_keywords")
            or [],
            "industries": preferences.get("industries") or [],
            "job_description": job_description,
            "output_schema": {
                "score": 0,
                "reason": "string",
                "location_match": True,
                "domain_match": True,
            },
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "score must be an integer 0-100.",
                (
                    "location_match: true only if the job is located in (or "
                    "is remote work available from) one of the "
                    "preferred_locations. Locations may appear in another "
                    "language (e.g. Muenchen for Munich). If "
                    "preferred_locations is empty, set true."
                ),
                (
                    "If location_match is false, score must be 25 or lower "
                    "and the reason must mention the location mismatch."
                ),
                (
                    "domain_match: true only if the job's industry and "
                    "product domain match the candidate's actual experience "
                    "and the stated industries. A shared job title is NOT "
                    "enough: e.g. a software/AI product manager posting "
                    "does not match a CV of a product manager for food "
                    "products. If industries is empty, judge the domain "
                    "from the CV."
                ),
                (
                    "If domain_match is false, score must be 25 or lower "
                    "and the reason must name both the job's industry and "
                    "the candidate's industry."
                ),
                (
                    "If the page describes an expired, closed, or already "
                    "filled position, score must be 10 or lower."
                ),
            ],
        }
        return json.dumps(payload, ensure_ascii=True)
