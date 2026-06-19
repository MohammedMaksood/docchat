"""Bounded retry with exponential backoff for transient LLM API errors.

Retries only clearly-transient failures (503 overload, 429 rate-limit) — never client
errors like bad keys or bad requests — so it never wastes calls or money (Rule 3).
"""
from __future__ import annotations

import time

_TRANSIENT = ("503", "unavailable", "overloaded", "high demand", "429", "resource_exhausted")


def retry_transient(fn, attempts: int = 3, base: float = 1.5):
    """Call fn(); on a transient error, back off (base * 2**i) and retry, up to `attempts`."""
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if i == attempts - 1 or not any(t in str(exc).lower() for t in _TRANSIENT):
                raise
            time.sleep(base * (2 ** i))
