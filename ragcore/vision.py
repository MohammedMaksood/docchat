"""Render PDF pages to PNG images for on-demand multimodal answers.

Only called when the user enables vision, so the (paid) image tokens are opt-in.
"""
from __future__ import annotations

from .config import VISION_DPI


def render_page_png(pdf_bytes: bytes, page_index: int, dpi: int = VISION_DPI) -> bytes:
    """Render a single 0-indexed PDF page to PNG bytes."""
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc[page_index].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()
