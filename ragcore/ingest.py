"""Loaders: PDFs (with OCR fallback for scanned pages) and web URLs.

Both produce the same `Chunk` list, so the rest of the pipeline is source-agnostic.
Heavy deps (PyMuPDF, Tesseract, trafilatura) are imported lazily so the core and
the unit tests stay light.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader

from .config import CHUNK_SIZE, CHUNK_OVERLAP

_OCR_MIN_CHARS = 20  # a page with less extractable text than this is treated as scanned


@dataclass
class Chunk:
    text: str
    source: str   # file name or URL
    page: int     # PDF page number; 1 for web pages


def _split(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    step = max(size - overlap, 1)
    return [text[i:i + size] for i in range(0, len(text), step)]


def read_bytes(source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "read"):
        return source.read()
    with open(source, "rb") as fh:
        return fh.read()


def _ocr_page(doc, index: int) -> str:
    import pytesseract
    from PIL import Image
    pix = doc[index].get_pixmap(dpi=200)
    return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))))


def load_pdf(source, source_name: str, ocr: bool = True) -> list[Chunk]:
    """Extract text per page (pypdf); fall back to Tesseract OCR on scanned pages."""
    data = read_bytes(source)
    reader = PdfReader(io.BytesIO(data))
    doc = None
    chunks: list[Chunk] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if ocr and len(text.strip()) < _OCR_MIN_CHARS:
            import fitz
            doc = doc or fitz.open(stream=data, filetype="pdf")
            text = _ocr_page(doc, page_no - 1) or text
        chunks.extend(Chunk(text=p, source=source_name, page=page_no) for p in _split(text))
    if doc is not None:
        doc.close()
    return chunks


def load_url(url: str, source_name: str | None = None) -> list[Chunk]:
    """Fetch a web page and extract its main text (navigation/ads stripped)."""
    import trafilatura
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Could not fetch {url}")
    text = trafilatura.extract(downloaded) or ""
    if not text.strip():
        raise ValueError(f"No readable main text found at {url}")
    return [Chunk(text=p, source=source_name or url, page=1) for p in _split(text)]
