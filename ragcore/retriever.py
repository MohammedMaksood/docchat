"""Hybrid retrieval: dense vector search + sparse BM25, fused with Reciprocal Rank
Fusion (RRF).

Dense search uses **ChromaDB** (cosine) when it's importable. On constrained hosts
where Chroma's native dependency stack won't load, it transparently falls back to an
in-memory **NumPy** cosine index — same interface, same RRF fusion. RRF combines the
two rankings by position, so the score scales never need normalizing.
"""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Okapi

from .config import RRF_POOL, RRF_K, CHROMA_COLLECTION
from .ingest import Chunk

try:                       # ChromaDB pulls a heavy native stack that fails to import on some runtimes
    import chromadb
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False


def _normalize(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return v / n if n else v


class HybridRetriever:
    """Hybrid BM25 + dense retriever. Dense backend is ChromaDB if available, else NumPy."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.backend = "chromadb" if CHROMA_AVAILABLE else "numpy"
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._collection = None          # chromadb path
        self._matrix: np.ndarray | None = None  # numpy path

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        vectors = self.embedder.embed_documents([c.text for c in chunks])
        if self.backend == "chromadb":
            client = chromadb.EphemeralClient()
            try:
                client.delete_collection(CHROMA_COLLECTION)
            except Exception:
                pass
            self._collection = client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
            self._collection.add(
                ids=[str(i) for i in range(len(chunks))],
                embeddings=vectors,
                documents=[c.text for c in chunks],
                metadatas=[{"source": c.source, "page": c.page} for c in chunks],
            )
        else:
            self._matrix = np.vstack([_normalize(v) for v in vectors])
        self._bm25 = BM25Okapi([c.text.lower().split() for c in chunks])

    def _dense(self, query: str, pool: int) -> list[int]:
        q = self.embedder.embed_query(query)
        if self.backend == "chromadb":
            res = self._collection.query(query_embeddings=[q], n_results=min(pool, len(self.chunks)))
            return [int(i) for i in res["ids"][0]]
        sims = self._matrix @ _normalize(q)
        return [int(i) for i in np.argsort(-sims)[:pool]]

    def _sparse(self, query: str, pool: int) -> list[int]:
        scores = self._bm25.get_scores(query.lower().split())
        return [int(i) for i in np.argsort(-scores)[:pool]]

    def retrieve(self, query: str, k: int, pool: int = RRF_POOL) -> list[Chunk]:
        if not self.chunks:
            return []
        fused: dict[int, float] = {}
        for ranking in (self._dense(query, pool), self._sparse(query, pool)):
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=lambda i: -fused[i])[:k]
        return [self.chunks[i] for i in top]
