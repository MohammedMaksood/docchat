"""DocChat orchestration: ingest -> hybrid retrieve -> grounded, cited answer.

When `vision=True`, the cited PDF pages are rendered to images and sent with the
prompt in a single multimodal call (bounded to MAX_VISION_PAGES).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import RETRIEVE_K, MAX_VISION_PAGES
from .ingest import Chunk, load_pdf, load_url, read_bytes
from .prompts import SYSTEM, USER_TEMPLATE, VISION_HINT
from .retriever import HybridRetriever
from .vision import render_page_png


@dataclass
class Result:
    answer: str
    cited: list[int]         # 1-based context-block numbers the model cited (match the inline [n])
    retrieved: list[Chunk]   # all retrieved blocks, indexable by (n - 1)


class DocChat:
    def __init__(self, embedder, generator):
        self.retriever = HybridRetriever(embedder)
        self.generator = generator
        self.ready = False
        self._pdf_bytes: dict[str, bytes] = {}   # kept so cited pages can be rendered for vision

    def ingest(self, pdfs: list[tuple[str, object]] | None = None,
               urls: list[str] | None = None) -> int:
        chunks: list[Chunk] = []
        for name, f in (pdfs or []):
            data = read_bytes(f)
            self._pdf_bytes[name] = data
            chunks.extend(load_pdf(data, name))
        for url in (urls or []):
            chunks.extend(load_url(url))
        if not chunks:
            raise ValueError("No extractable text found in the provided PDF(s)/URL(s).")
        self.retriever.index(chunks)
        self.ready = True
        return len(chunks)

    def _page_images(self, hits: list[Chunk]) -> list[bytes]:
        """Render up to MAX_VISION_PAGES distinct cited PDF pages to PNG bytes."""
        images, seen = [], set()
        for c in hits:
            key = (c.source, c.page)
            if c.source in self._pdf_bytes and key not in seen:
                seen.add(key)
                images.append(render_page_png(self._pdf_bytes[c.source], c.page - 1))
                if len(images) >= MAX_VISION_PAGES:
                    break
        return images

    def ask(self, question: str, k: int = RETRIEVE_K, vision: bool = False) -> Result:
        hits = self.retriever.retrieve(question, k=k)
        context = "\n\n".join(
            f"[{i + 1}] (p.{c.page}, {c.source})\n{c.text}" for i, c in enumerate(hits)
        )
        user = USER_TEMPLATE.format(context=context, question=question)
        images = self._page_images(hits) if vision else None
        if images:
            user += VISION_HINT
        answer = self.generator.generate(SYSTEM, user, images=images)
        cited = [n for n in answer.citations if 1 <= n <= len(hits)]
        return Result(answer=answer.answer, cited=cited, retrieved=hits)
