"""
highlight.py - open a cited passage in the PDF with it highlighted.

the chunk text came from pypdf, but to draw highlights we need the position of
every word on the page, so we use PyMuPDF here. the two libraries don't extract
text exactly the same way (hyphens, ligatures, odd spacing), so instead of an
exact search we line up the two word lists with difflib and take the matching
runs.

  yellow = the whole retrieved passage
  orange = the one sentence in it that is closest to the question
"""

from __future__ import annotations

import difflib
import re

import numpy as np
import pymupdf

from .models import embed

YELLOW = (1.0, 0.92, 0.35)
ORANGE = (1.0, 0.6, 0.2)


def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _match(page_words: list[str], target: list[str], min_run: int = 3) -> list[int]:
    """indexes of page words that line up with the target word sequence"""
    sm = difflib.SequenceMatcher(None, page_words, target, autojunk=False)
    hits = []
    for a, _, size in sm.get_matching_blocks():
        if size >= min_run:
            hits.extend(range(a, a + size))
    if not hits:
        return []
    # fill small gaps (hyphenated words, ligatures) as long as the span
    # isn't much longer than the target, which happens with 2-column layouts
    lo, hi = min(hits), max(hits)
    if hi - lo + 1 <= len(target) * 1.3:
        return list(range(lo, hi + 1))
    return sorted(set(hits))


def best_sentence(text: str, question: str) -> str | None:
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text) if len(s.split()) >= 5]
    if not sents or not question:
        return None
    q = embed([question], kind="query")[0]
    s = embed(sents, kind="document")
    return sents[int(np.argmax(s @ q))]


def highlighted_pdf(path: str, page_no: int, text: str, question: str = "") -> tuple[bytes, int]:
    doc = pymupdf.open(path)
    page = doc[page_no - 1]
    words = page.get_text("words")          # (x0, y0, x1, y1, word, block, line, word_no)
    keep = [i for i, w in enumerate(words) if _norm(w[4])]
    page_words = [_norm(words[i][4]) for i in keep]

    def mark(word_ids, color):
        lines: dict[tuple, pymupdf.Rect] = {}
        for k in word_ids:
            x0, y0, x1, y1, _, block, line, _ = words[keep[k]]
            r = pymupdf.Rect(x0, y0, x1, y1)
            key = (block, line)
            lines[key] = lines[key] | r if key in lines else r
        for r in lines.values():
            a = page.add_highlight_annot(r)
            a.set_colors(stroke=color)
            a.update()
        return len(lines)

    passage = _match(page_words, [w for w in map(_norm, text.split()) if w])
    n = mark(passage, YELLOW)

    sent = best_sentence(text, question)
    if sent and passage:
        sub = [page_words[k] for k in passage]
        hit = _match(sub, [w for w in map(_norm, sent.split()) if w], min_run=2)
        mark([passage[k] for k in hit], ORANGE)

    out = doc.tobytes()
    doc.close()
    return out, n
