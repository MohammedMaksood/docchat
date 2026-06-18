"""Embedding backends: Gemini (cloud) and Ollama (local).

Document embedding is batched into as few calls as possible (Rule 3).
Both backends expose the same tiny interface: embed_documents / embed_query.
"""
from __future__ import annotations

import requests

from .config import (
    GEMINI_EMBED_MODEL, EMBED_DIM, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL,
)

_GEMINI_BATCH = 100  # stay under per-request embedding batch limits


class GeminiEmbedder:
    def __init__(self, client, model: str = GEMINI_EMBED_MODEL, dim: int = EMBED_DIM):
        from google.genai import types
        self._client = client
        self._types = types
        self.model = model
        self.dim = dim

    def _embed(self, contents: list[str], task_type: str) -> list[list[float]]:
        cfg = self._types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        res = self._client.models.embed_content(model=self.model, contents=contents, config=cfg)
        return [e.values for e in res.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), _GEMINI_BATCH):
            out.extend(self._embed(texts[i:i + _GEMINI_BATCH], "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]


class OllamaEmbedder:
    def __init__(self, model: str = OLLAMA_EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed(self, inputs) -> list[list[float]]:
        r = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": inputs},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)        # one batched call

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)[0]
