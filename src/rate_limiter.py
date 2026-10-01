"""Client-side pacing for the LLM provider's quotas.

Gemma on the Gemini API free tier enforces, per project and model:
requests per minute, input tokens per minute, and requests per day. A scan
used to fire its calls back to back and blow the 16k input-token minute
budget within seconds; the provider's 429s then failed whole scans.
`RateLimiter` makes callers wait until a call fits the trailing 60-second
window instead, so the quota is never hit in the first place.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Callable
from zoneinfo import ZoneInfo

WINDOW_SECONDS = 60.0
# Google resets the per-day quotas at midnight Pacific time.
QUOTA_DAY_TZ = ZoneInfo("America/Los_Angeles")
# Prompts here are JSON-escaped (umlauts become ü), so characters per
# token run high; 3 overestimates on purpose. Real usage replaces the
# estimate once the response arrives.
CHARS_PER_TOKEN = 3.0


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


class DailyQuotaExceeded(RuntimeError):
    """The requests-per-day quota is spent; waiting minutes will not help."""


class Reservation:
    """One admitted call's slot in the window; `settle` corrects its size."""

    def __init__(self, at: float, tokens: int) -> None:
        self.at = at
        self.tokens = tokens


class RateLimiter:
    """Blocks callers until a request fits the RPM, input-TPM and RPD limits.

    A limit of 0 disables that check. Thread-safe: the bot's intake (main
    thread) and its scan worker share one provider quota.
    """

    def __init__(
        self,
        rpm: int,
        input_tpm: int,
        rpd: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        wall_clock: Callable[[], float] = time.time,
    ) -> None:
        self.rpm = rpm
        self.input_tpm = input_tpm
        self.rpd = rpd
        self._clock = clock
        self._sleep = sleep
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._window: deque[Reservation] = deque()
        self._blocked_until = 0.0
        self._day = ""
        self._day_count = 0
        self._logger = logging.getLogger(self.__class__.__name__)

    def acquire(self, tokens: int) -> Reservation:
        """Wait until a call of `tokens` input tokens fits, then admit it."""
        while True:
            with self._lock:
                now = self._clock()
                self._check_day()
                wait = self._wait_needed(now, tokens)
                if wait <= 0:
                    reservation = Reservation(now, tokens)
                    self._window.append(reservation)
                    self._day_count += 1
                    return reservation
            self._logger.info(
                "Pacing LLM call (~%s input tokens): waiting %.1fs for quota",
                tokens,
                wait,
            )
            self._sleep(wait)

    def settle(self, reservation: Reservation, actual_tokens: int) -> None:
        """Replace a reservation's estimate with the provider's count."""
        with self._lock:
            reservation.tokens = actual_tokens

    def block_for(self, seconds: float) -> None:
        """Hold every caller back, e.g. for a 429's retryDelay."""
        with self._lock:
            self._blocked_until = max(self._blocked_until, self._clock() + seconds)

    def _wait_needed(self, now: float, tokens: int) -> float:
        while self._window and now - self._window[0].at >= WINDOW_SECONDS:
            self._window.popleft()
        waits = [self._blocked_until - now]
        if self.rpm and len(self._window) >= self.rpm:
            waits.append(self._window[-self.rpm].at + WINDOW_SECONDS - now)
        if self.input_tpm:
            # A call bigger than the whole budget can only go into an empty
            # window; callers keep prompts under the budget so this is rare.
            budget = self.input_tpm - tokens
            used = sum(item.tokens for item in self._window)
            for item in self._window:
                if used <= budget:
                    break
                used -= item.tokens
                waits.append(item.at + WINDOW_SECONDS - now)
        return max(waits)

    def _check_day(self) -> None:
        day = datetime.fromtimestamp(self._wall_clock(), QUOTA_DAY_TZ).date()
        if day.isoformat() != self._day:
            self._day = day.isoformat()
            self._day_count = 0
        if self.rpd and self._day_count >= self.rpd:
            reset = datetime.combine(day + timedelta(days=1), datetime.min.time())
            raise DailyQuotaExceeded(
                f"LLM requests-per-day limit ({self.rpd}) reached; resets at "
                f"{reset.replace(tzinfo=QUOTA_DAY_TZ).isoformat()}"
            )


_limiters: dict[str, RateLimiter] = {}
_limiters_lock = threading.Lock()


def shared_limiter(model: str, rpm: int, input_tpm: int, rpd: int) -> RateLimiter:
    """The process-wide limiter for `model`: every LLMService shares it."""
    with _limiters_lock:
        limiter = _limiters.get(model)
        if limiter is None:
            limiter = _limiters[model] = RateLimiter(rpm, input_tpm, rpd)
        return limiter
