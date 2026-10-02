import json

import pytest
from litellm.exceptions import InternalServerError, RateLimitError

from src.config_manager import Config, EffortBudget
from src.llm_service import LLMService, ProviderUnavailableError, RateLimitedError
from src.rate_limiter import DailyQuotaExceeded, RateLimiter, estimate_tokens


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _limiter(clock: FakeClock, rpm=0, tpm=0, rpd=0, wall=lambda: 1_790_000_000.0):
    return RateLimiter(rpm, tpm, rpd, clock=clock, sleep=clock.sleep, wall_clock=wall)


def test_requests_per_minute_waits_for_oldest_to_age_out():
    clock = FakeClock()
    limiter = _limiter(clock, rpm=2)
    limiter.acquire(1)
    clock.now += 10
    limiter.acquire(1)
    limiter.acquire(1)
    # Third call must wait until the first (60s old) leaves the window.
    assert clock.sleeps == [50.0]


def test_input_tokens_per_minute_paces_large_calls():
    clock = FakeClock()
    limiter = _limiter(clock, tpm=10_000)
    limiter.acquire(6000)
    clock.now += 5
    limiter.acquire(3000)
    limiter.acquire(4000)
    # 6000+3000+4000 > 10k: wait for the 6000 call to age out.
    assert clock.sleeps == [55.0]


def test_settle_replaces_estimate_with_actual_usage():
    clock = FakeClock()
    limiter = _limiter(clock, tpm=10_000)
    reservation = limiter.acquire(9000)
    limiter.settle(reservation, 2000)
    limiter.acquire(7000)
    assert clock.sleeps == []


def test_oversized_call_waits_for_an_empty_window_then_goes():
    clock = FakeClock()
    limiter = _limiter(clock, tpm=10_000)
    limiter.acquire(500)
    limiter.acquire(20_000)
    assert clock.sleeps == [60.0]


def test_block_for_holds_back_callers():
    clock = FakeClock()
    limiter = _limiter(clock, rpm=100)
    limiter.block_for(23)
    limiter.acquire(1)
    assert clock.sleeps == [23.0]


def test_requests_per_day_raises_and_resets_next_pacific_day():
    clock = FakeClock()
    wall = [1_790_000_000.0]
    limiter = _limiter(clock, rpd=2, wall=lambda: wall[0])
    limiter.acquire(1)
    limiter.acquire(1)
    with pytest.raises(DailyQuotaExceeded):
        limiter.acquire(1)
    wall[0] += 86_400
    limiter.acquire(1)


# ----------------------------------------------------------------------
# LLMService integration


def _config(**overrides) -> Config:
    params = dict(
        max_results=1,
        min_score=50,
        results_json="unused.json",
        results_csv="unused.csv",
        cache_path="unused_cache.json",
        llm_model="test-model",
        llm_temperature=0.0,
        llm_max_retries=2,
        llm_min_delay_seconds=0,
        brave_endpoint="mock",
        results_per_query=1,
        request_timeout_seconds=1,
        search_min_delay_seconds=0,
        max_queries_per_iteration=1,
        budget=EffortBudget(max_llm_calls=10, max_search_iterations=10),
    )
    params.update(overrides)
    return Config(**params)


def _rate_limit_error(delay: str = "17s", quota: str = "PerModelPerMinute") -> Exception:
    body = json.dumps(
        {
            "error": {
                "code": 429,
                "details": [
                    {"quotaId": f"GenerateContentInputTokens{quota}-FreeTier"},
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": delay,
                    },
                ],
            }
        },
        indent=4,
    )
    return RateLimitError(message=body, llm_provider="gemini", model="test-model")


def _service(config: Config, limiter: RateLimiter | None = None) -> LLMService:
    service = LLMService(config, config.budget, api_key=None)
    service._limiter = limiter
    return service


def _ok(content: str = '{"score": 80, "reason": "fits"}', prompt_tokens: int = 1234):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": prompt_tokens},
    }


def test_429_waits_retry_delay_and_does_not_use_up_normal_retries(monkeypatch):
    clock = FakeClock()
    limiter = _limiter(clock, rpm=15)
    config = _config(llm_max_retries=1, llm_rate_limit_retries=3)
    service = _service(config, limiter)
    outcomes = [_rate_limit_error("17s"), _rate_limit_error("4s"), _ok()]

    def fake_completion(**kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    monkeypatch.setattr("src.llm_service.random.uniform", lambda a, b: 1.0)
    evaluation = service.evaluate_job("cv", "job")
    assert evaluation.score == 80
    # Each retry waited the provider's retryDelay (+1s jitter).
    assert clock.sleeps == [18.0, 5.0]


def test_429_gives_up_with_rate_limited_error(monkeypatch):
    clock = FakeClock()
    config = _config(llm_rate_limit_retries=2)
    service = _service(config, _limiter(clock, rpm=15))

    def fake_completion(**kwargs):
        raise _rate_limit_error("1s")

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    with pytest.raises(RateLimitedError):
        service.evaluate_job("cv", "job")


def test_daily_quota_429_is_not_retried(monkeypatch):
    calls = []
    service = _service(_config(), _limiter(FakeClock(), rpm=15))

    def fake_completion(**kwargs):
        calls.append(1)
        raise _rate_limit_error("1s", quota="PerModelPerDay")

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    with pytest.raises(DailyQuotaExceeded):
        service.evaluate_job("cv", "job")
    assert len(calls) == 1


def test_actual_prompt_tokens_settle_the_reservation(monkeypatch):
    clock = FakeClock()
    limiter = _limiter(clock, tpm=10_000)
    service = _service(_config(), limiter)
    monkeypatch.setattr(
        "src.llm_service.completion", lambda **kwargs: _ok(prompt_tokens=321)
    )
    service.evaluate_job("cv", "job " * 2000)
    assert [item.tokens for item in limiter._window] == [321]


def test_typical_job_page_is_sent_in_full():
    service = _service(_config(llm_input_tpm_limit=16_000))
    page = "Wir suchen eine Produktmanagerin (m/w/d) in Berlin. " * 200  # ~10k chars
    prompt = json.loads(service._build_evaluation_prompt("cv", page))
    assert prompt["job_description"] == page


def test_oversized_job_page_is_trimmed_to_fit_one_minute_quota():
    service = _service(_config(llm_input_tpm_limit=16_000))
    head = "Produktmanager Lebensmittel in München, Aufgaben und Profil. "
    page = head + "Ähnliche Jobs: Lagerlogistik Plaidt. " * 3000  # ~110k chars
    prompt_text = service._build_evaluation_prompt("cv " * 500, page)
    prompt = json.loads(prompt_text)
    assert estimate_tokens(prompt_text) <= service._max_prompt_tokens
    assert prompt["job_description"].startswith(head)
    assert "page text truncated" in prompt["job_description"]
    # Trimmed only as far as needed: most of the quota is still used.
    assert estimate_tokens(prompt_text) > 0.9 * service._max_prompt_tokens


def test_limits_off_by_default_without_config():
    service = LLMService(_config(), _config().budget, api_key=None)
    assert service._limiter is None


def _server_error() -> Exception:
    return InternalServerError(
        message='{"error": {"code": 500, "message": "Internal error encountered.", '
        '"status": "INTERNAL"}}',
        llm_provider="gemini",
        model="test-model",
    )


def test_server_errors_back_off_and_do_not_use_up_normal_retries(monkeypatch):
    sleeps: list[float] = []
    config = _config(llm_max_retries=1, llm_server_error_retries=3)
    service = _service(config)
    outcomes = [_server_error(), _server_error(), _server_error(), _ok()]

    def fake_completion(**kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    monkeypatch.setattr("src.llm_service.time.sleep", sleeps.append)
    monkeypatch.setattr("src.llm_service.random.uniform", lambda a, b: 0.0)
    assert service.evaluate_job("cv", "job").score == 80
    # The burst that ended a scan on 2026-10-02 was three 500s a second apart.
    assert sleeps == [10.0, 30.0, 60.0]


def test_persistent_server_errors_raise_provider_unavailable(monkeypatch):
    calls = []
    service = _service(_config(llm_server_error_retries=2))

    def fake_completion(**kwargs):
        calls.append(1)
        raise _server_error()

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    monkeypatch.setattr("src.llm_service.time.sleep", lambda seconds: None)
    with pytest.raises(ProviderUnavailableError):
        service.evaluate_job("cv", "job")
    assert len(calls) == 3


def test_client_errors_keep_the_fast_retry(monkeypatch):
    sleeps: list[float] = []
    service = _service(_config(llm_max_retries=2, llm_min_delay_seconds=1))

    def fake_completion(**kwargs):
        raise ValueError("bad request")

    monkeypatch.setattr("src.llm_service.completion", fake_completion)
    monkeypatch.setattr("src.llm_service.time.sleep", sleeps.append)
    with pytest.raises(RuntimeError, match="LLM request failed"):
        service.evaluate_job("cv", "job")
    assert sleeps == [1]
