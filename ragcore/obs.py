"""LLM call observability: one structured record per call, with token cost (Rules 3 & 6).

Records go to logs/llm_calls.jsonl (gitignored). Each record carries model id,
prompt version, token counts, latency, cost, and a request id so spend and
regressions are auditable.
"""
from __future__ import annotations

import json
import pathlib
import time
import uuid

from .config import PRICING

LOG_PATH = pathlib.Path(__file__).resolve().parent.parent / "logs" / "llm_calls.jsonl"


def estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    p = PRICING.get(model)
    if not p:
        return 0.0
    return round(in_tok / 1e6 * p["input"] + out_tok / 1e6 * p["output"], 6)


def log_call(*, provider: str, model: str, prompt_version: str,
             in_tok: int, out_tok: int, latency_ms: int) -> dict:
    rec = {
        "ts": round(time.time(), 3),
        "request_id": uuid.uuid4().hex[:12],
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        "input_tokens": int(in_tok),
        "output_tokens": int(out_tok),
        "latency_ms": int(latency_ms),
        "cost_usd": estimate_cost(model, in_tok, out_tok) if provider == "gemini" else 0.0,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
