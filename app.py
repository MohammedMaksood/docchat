"""DocChat — Streamlit UI.

Chat with your PDFs using hybrid retrieval and grounded, cited answers.
Run: `streamlit run app.py`
"""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from ragcore.config import (
    GEMINI_CHAT_MODEL, OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_EMBED_MODEL, OLLAMA_VISION_MODEL,
)
from ragcore.embeddings import GeminiEmbedder, OllamaEmbedder
from ragcore.generation import GeminiGenerator, OllamaGenerator
from ragcore.rag import DocChat

load_dotenv()

st.set_page_config(page_title="DocChat — Advanced RAG", page_icon="📄", layout="wide")
st.title("📄 DocChat")
st.caption("Chat with your PDFs · hybrid retrieval (BM25 + embeddings, fused with RRF) · grounded answers with citations")


def build_engine() -> DocChat:
    """Construct a DocChat engine for the selected provider (no network call here)."""
    if provider.startswith("Gemini"):
        if not api_key:
            raise ValueError("Enter your Gemini API key (free at aistudio.google.com/apikey).")
        from google import genai
        client = genai.Client(api_key=api_key)
        return DocChat(GeminiEmbedder(client), GeminiGenerator(client))
    return DocChat(
        OllamaEmbedder(model=embed_model, base_url=base_url),
        OllamaGenerator(model=gen_model, base_url=base_url, vision_model=vision_model),
    )


with st.sidebar:
    st.header("Settings")
    provider = st.radio("Model provider", ["Gemini (cloud)", "Ollama (local)"])
    if provider.startswith("Gemini"):
        api_key = st.text_input("Gemini API key", type="password",
                                value=os.getenv("GEMINI_API_KEY", ""),
                                help="Free key: aistudio.google.com/apikey. Never stored or logged.")
        st.caption(f"Model: `{GEMINI_CHAT_MODEL}`")
    else:
        base_url = st.text_input("Ollama base URL", OLLAMA_BASE_URL)
        gen_model = st.text_input("Chat model", OLLAMA_CHAT_MODEL)
        embed_model = st.text_input("Embedding model", OLLAMA_EMBED_MODEL)
        vision_model = st.text_input("Vision model", OLLAMA_VISION_MODEL, help="Used only when 🖼️ vision is on (e.g. moondream, llava)")
    vision = st.toggle("🖼️ Read page images (vision)", value=False,
                       help="Sends up to 2 cited PDF page images to the model for figures/charts/scanned visuals. Adds image tokens (cost).")
    if vision:
        st.caption("Best with Gemini. Local needs a capable vision model: `ollama pull llava`.")
    files = st.file_uploader("Upload PDF(s) — scanned PDFs are OCR'd", type="pdf", accept_multiple_files=True)
    url_text = st.text_area("…or paste web page URL(s), one per line", height=80, placeholder="https://example.com/article")
    if st.button("Build index", type="primary"):
        try:
            urls = [u.strip() for u in url_text.splitlines() if u.strip()]
            if not files and not urls:
                raise ValueError("Upload at least one PDF or paste a URL.")
            engine = build_engine()
            with st.spinner("Reading and indexing…"):
                n = engine.ingest(pdfs=[(f.name, f) for f in files], urls=urls)
            st.session_state.engine = engine
            st.session_state.messages = []
            st.success(f"Indexed {n} chunks from {len(files)} file(s) + {len(urls)} URL(s).")
        except Exception as exc:
            st.error(f"Setup failed: {exc}")

engine: DocChat | None = st.session_state.get("engine")
st.session_state.setdefault("messages", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask about your documents" if engine else "Build an index first (sidebar) →")
if question:
    if not engine:
        st.warning("Upload PDFs and click **Build index** first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Reading page images…" if vision else "Thinking…"):
                    result = engine.ask(question, vision=vision)
                st.markdown(result.answer)
                nums = result.cited or list(range(1, len(result.retrieved) + 1))
                with st.expander(f"Sources ({len(nums)})"):
                    for n in nums:
                        c = result.retrieved[n - 1]
                        loc = c.source if c.source.startswith("http") else f"{c.source} — page {c.page}"
                        st.markdown(f"**[{n}] {loc}**")
                        st.caption(c.text[:400] + ("…" if len(c.text) > 400 else ""))
                call = engine.generator.last_call
                if call:
                    st.caption(
                        f"⚙️ {call['model']} · {call['input_tokens']}→{call['output_tokens']} tok · "
                        f"{call['latency_ms']} ms · ${call['cost_usd']:.5f}"
                    )
                st.session_state.messages.append({"role": "assistant", "content": result.answer})
            except Exception as exc:
                st.error(f"Error: {exc}")
