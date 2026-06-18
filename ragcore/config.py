"""Central config: pinned model IDs, pricing, and tunables.

Single source of truth so model versions are pinned and diffable (Rule 6), never
referenced as `-latest` aliases. Verified against ai.google.dev (June 2026):
Gemini 2.0 Flash was shut down 2026-06-01, so it is intentionally absent.
"""

# --- Gemini (cloud) — GA, pinned ---
GEMINI_CHAT_MODEL = "gemini-2.5-flash"      # cheapest capable chat model (Rule 3)
GEMINI_EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 768                             # gemini-embedding-001 supports output_dimensionality

# --- Ollama (local) — free, used for development/offline ---
OLLAMA_CHAT_MODEL = "qwen3:8b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"     # dedicated embedder; run `ollama pull nomic-embed-text`
OLLAMA_VISION_MODEL = "llava"               # local vision; run `ollama pull llava` (tiny models are too weak)
OLLAMA_BASE_URL = "http://localhost:11434"

# --- On-demand vision (bounds cost per Rule 3) ---
MAX_VISION_PAGES = 2     # max page images sent in a single question
VISION_DPI = 150

# --- Pricing, USD per 1M tokens (paid Standard tier, ai.google.dev/gemini-api/docs/pricing, June 2026) ---
PRICING = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-embedding-001": {"input": 0.15, "output": 0.0},
}

# --- Vector store (ChromaDB) ---
CHROMA_COLLECTION = "docchat"

# --- Retrieval tunables ---
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
RETRIEVE_K = 5
RRF_POOL = 20
RRF_K = 60
