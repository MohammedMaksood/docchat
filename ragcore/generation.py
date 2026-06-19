"""Generation backends with schema-validated output (Rule 6).

Both providers use native structured output (Gemini `response_schema`, Ollama
`format`=JSON schema) so the model returns JSON matching AnswerModel. This is the
cheaper, more reliable alternative to a parse-and-re-call retry loop (Rule 3); a
lightweight salvage handles the rare malformed string. Every call is logged.
"""
from __future__ import annotations

import re
import time

import requests
from pydantic import BaseModel, Field, ValidationError

from .config import GEMINI_CHAT_MODEL, OLLAMA_CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_VISION_MODEL
from .obs import log_call
from .prompts import PROMPT_VERSION
from .retry import retry_transient


class AnswerModel(BaseModel):
    answer: str = Field(description="Concise answer with inline [n] citations.")
    citations: list[int] = Field(default_factory=list)


def _validate(text: str) -> AnswerModel:
    try:
        return AnswerModel.model_validate_json(text)
    except ValidationError:
        match = re.search(r"\{.*\}", text, re.DOTALL)  # salvage JSON embedded in prose
        if match:
            return AnswerModel.model_validate_json(match.group(0))
        raise ValueError("Model did not return valid JSON for the answer.")


class GeminiGenerator:
    def __init__(self, client, model: str = GEMINI_CHAT_MODEL):
        from google.genai import types
        self._client = client
        self._types = types
        self.model = model
        self.last_call: dict | None = None

    def generate(self, system: str, user: str, images: list[bytes] | None = None) -> AnswerModel:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=AnswerModel,
            temperature=0.2,
        )
        contents = user if not images else (
            [self._types.Part.from_bytes(data=im, mime_type="image/png") for im in images] + [user]
        )
        t0 = time.time()
        resp = retry_transient(
            lambda: self._client.models.generate_content(model=self.model, contents=contents, config=cfg)
        )
        latency = int((time.time() - t0) * 1000)

        um = getattr(resp, "usage_metadata", None)
        self.last_call = log_call(
            provider="gemini", model=self.model, prompt_version=PROMPT_VERSION,
            in_tok=getattr(um, "prompt_token_count", 0) or 0,
            out_tok=getattr(um, "candidates_token_count", 0) or 0,
            latency_ms=latency,
        )
        parsed = getattr(resp, "parsed", None)
        return parsed if isinstance(parsed, AnswerModel) else _validate(resp.text)


class OllamaGenerator:
    def __init__(self, model: str = OLLAMA_CHAT_MODEL, base_url: str = OLLAMA_BASE_URL,
                 vision_model: str = OLLAMA_VISION_MODEL):
        self.model = model
        self.vision_model = vision_model
        self.base_url = base_url.rstrip("/")
        self.last_call: dict | None = None

    def generate(self, system: str, user: str, images: list[bytes] | None = None) -> AnswerModel:
        model = self.vision_model if images else self.model
        payload = {
            "model": model,
            "system": system,
            "prompt": user,
            "stream": False,
            "think": False,                          # suppress reasoning models' <think> blocks
            "format": AnswerModel.model_json_schema(),  # constrain output to the schema
            "options": {"temperature": 0.2},
        }
        if images:
            import base64
            payload["images"] = [base64.b64encode(im).decode() for im in images]
        t0 = time.time()
        r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        latency = int((time.time() - t0) * 1000)

        self.last_call = log_call(
            provider="ollama", model=model, prompt_version=PROMPT_VERSION,
            in_tok=data.get("prompt_eval_count", 0), out_tok=data.get("eval_count", 0),
            latency_ms=latency,
        )
        return _validate(data["response"])
