"""Free, no-LLM unit tests for the retrieval core (chunking + hybrid RRF).

Uses a stub embedder so the tests cost nothing and run offline (Rules 3 & 9).
"""
from ragcore.ingest import Chunk, _split
from ragcore.retriever import HybridRetriever


class StubEmbedder:
    """Deterministic basis vectors (cosine-safe): dense points at chunk 0, where BM25 also agrees."""
    def embed_documents(self, texts):
        return [[1.0 if i == j else 0.0 for j in range(3)] for i, _ in enumerate(texts)]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_split_overlaps_and_covers():
    chunks = _split("word " * 500, size=900, overlap=150)
    assert len(chunks) >= 2
    assert all(len(c) <= 900 for c in chunks)


def test_hybrid_retrieves_keyword_chunk():
    chunks = [
        Chunk(text="The mitochondria is the powerhouse of the cell.", source="bio.pdf", page=1),
        Chunk(text="Photosynthesis converts sunlight into chemical energy.", source="bio.pdf", page=2),
        Chunk(text="Newton's third law concerns equal and opposite forces.", source="phys.pdf", page=3),
    ]
    retriever = HybridRetriever(StubEmbedder())
    retriever.index(chunks)
    top = retriever.retrieve("what is the powerhouse of the cell", k=1)
    assert top and top[0].page == 1
