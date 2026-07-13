from __future__ import annotations

import json
import logging
import time
from typing import Any

from litellm import completion

from .config_manager import Config, EffortBudget
from .models import (
    IntakeExtraction,
    JobEvaluation,
    Reflection,
    SearchPlan,
    SearchQueries,
)
from pydantic import ValidationError


class LLMService:
    def __init__(self, config: Config, budget: EffortBudget, api_key: str | None):
        self._config = config
        self._budget = budget
        self._api_key = api_key
        self._logger = logging.getLogger(self.__class__.__name__)

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

    def evaluate_job(self, cv: str, job_description: str) -> JobEvaluation:
        prompt = self._build_evaluation_prompt(cv, job_description)
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

        last_error: Exception | None = None
        for attempt in range(1, self._config.llm_max_retries + 1):
            try:
                response = completion(
                    model=self._config.llm_model,
                    messages=[
                        {"role": "system", "content": "Respond only with valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self._config.llm_temperature,
                    api_key=self._api_key,
                )
                return response["choices"][0]["message"]["content"]
            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    "LLM call failed on attempt %s/%s: %s",
                    attempt,
                    self._config.llm_max_retries,
                    exc,
                )
                if attempt < self._config.llm_max_retries:
                    time.sleep(self._config.llm_min_delay_seconds)
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _parse_json_payload(self, response_text: str, prompt: str) -> dict[str, Any]:
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
        if not self._budget.can_call_llm():
            raise RuntimeError("Effort budget exceeded: LLM calls")
        self._budget.record_llm_call()

        repair_prompt = (
            "You must output ONLY valid JSON that matches the output schema. "
            "Do not include extra text. Fix this response and return JSON only.\n\n"
            f"Response: {response_text}\n\n"
            f"Original prompt: {prompt}"
        )

        response = completion(
            model=self._config.llm_model,
            messages=[
                {"role": "system", "content": "Respond only with valid JSON."},
                {"role": "user", "content": repair_prompt},
            ],
            temperature=0.0,
            api_key=self._api_key,
        )
        fixed = response["choices"][0]["message"]["content"]
        try:
            payload = json.loads(fixed)
            return self._normalize_payload(payload)
        except json.JSONDecodeError:
            repaired = self._extract_json_object(fixed)
            if repaired is None:
                raise
            payload = json.loads(repaired)
            return self._normalize_payload(payload)

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
                    "Do not list job boards or staffing agencies."
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
                "locations": ["string"],
                "questions": ["string"],
            },
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "Only include the keys in the output_schema.",
                "locations entries should look like 'Berlin, Germany'.",
                "Ask at most 3 questions, only about missing essentials.",
                "If locations are known, do not ask about locations.",
                "If user_answers_to_previous_questions covers a topic, "
                "use those answers and do not re-ask.",
            ],
        }
        return json.dumps(payload, ensure_ascii=True)

    def _build_evaluation_prompt(self, cv: str, job_description: str) -> str:
        payload = {
            "task": "Evaluate job relevance to the CV.",
            "cv": cv,
            "job_description": job_description,
            "output_schema": {"score": 0, "reason": "string"},
            "rules": [
                "Return ONLY JSON.",
                "Do not include explanations.",
                "score must be an integer 0-100.",
            ],
        }
        return json.dumps(payload, ensure_ascii=True)
