"""
ingest.py — turn PDFs into small, searchable text chunks.

Why chunk? An embedding is a summary of meaning. A whole 20-page paper squeezed
into one vector would blur everything together. Small chunks (~200 words) give
precise matches, and each chunk remembers which paper + page it came from, so
answers can cite their source.
"""

from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader

CHUNK_WORDS = 180     # size of each chunk
OVERLAP_WORDS = 40    # overlap so a sentence cut at a boundary still appears whole somewhere


def clean(text: str) -> str:
    text = text.replace("­", "")                 # soft hyphens
    text = re.sub(r"-\n(\w)", r"\1", text)             # re-join words hyphenated across lines
    text = re.sub(r"\s+", " ", text)                   # collapse whitespace
    return text.strip()


def pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    return [clean(page.extract_text() or "") for page in reader.pages]


def chunk_pdf(path: Path) -> list[dict]:
    """Return a list of {doc, page, text} dicts for one PDF."""
    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for page_no, text in enumerate(pdf_pages(path), start=1):
        words = text.split()
        if len(words) < 15:                            # skip near-empty pages
            continue
        for start in range(0, len(words), step):
            piece = words[start:start + CHUNK_WORDS]
            if len(piece) < 15 and start > 0:
                break
            chunks.append({"doc": path.name, "page": page_no, "text": " ".join(piece)})
            if start + CHUNK_WORDS >= len(words):
                break
    return chunks
