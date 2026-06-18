"""Hybrid retrieval: dense vector search (ChromaDB) + sparse (BM25), fused with
Reciprocal Rank Fusion (RRF).

RRF combines the two rankings by rank position, so the cosine distances from Chroma
and the BM25 scores don't need to be normalized to the same scale.
"""
from __future__ import annotations

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from .config import RRF_POOL, RRF_K, CHROMA_COLLECTION
from .ingest import Chunk


class HybridRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self._client = chromadb.EphemeralClient()
        self._col = None
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        try:
            self._client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        self._col = self._client.create_collection(CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
        self._col.add(
            ids=[str(i) for i in range(len(chunks))],
            embeddings=self.embedder.embed_documents([c.text for c in chunks]),
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        self._bm25 = BM25Okapi([c.text.lower().split() for c in chunks])

    def _dense(self, query: str, pool: int) -> list[int]:
        res = self._col.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(pool, len(self.chunks)),
        )
        return [int(i) for i in res["ids"][0]]

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
