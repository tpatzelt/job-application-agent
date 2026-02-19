from __future__ import annotations

import json
import logging
import time
from typing import Any

from litellm import completion

from .config_manager import Config, EffortBudget
from .models import JobEvaluation, SearchQueries


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

    def evaluate_job(self, cv: str, job_description: str) -> JobEvaluation:
        prompt = self._build_evaluation_prompt(cv, job_description)
        response_text = self._call_llm(prompt)
        payload = self._parse_json_payload(response_text, prompt)
        return JobEvaluation.model_validate(payload)

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
            return self._normalize_payload(payload)
        except json.JSONDecodeError:
            repaired = self._extract_json_object(response_text)
            if repaired is None:
                return self._retry_json_response(prompt, response_text)
            try:
                payload = json.loads(repaired)
                return self._normalize_payload(payload)
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
