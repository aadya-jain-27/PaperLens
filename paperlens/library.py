"""
library.py — the user's paper collection: PDFs on disk + their chunk vectors.

Saved files (in data/):
  papers/*.pdf     the uploaded papers
  vectors.npy      (num_chunks, 768) float32 embeddings
  chunks.json      text + paper name + page for each row of vectors.npy

"Scale test": a few papers give only a few hundred chunks, a size where the CPU
is actually faster (a GPU launch costs ~0.2 ms of fixed overhead). To show what
happens at search-engine scale, we can append N synthetic random vectors behind
the real ones. Random 768-d unit vectors are almost orthogonal to everything
(cosine ≈ 0 ± 0.04), so they never outrank real matches, and we filter them out
anyway. They only make the GPU do realistic amounts of work.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from .gpu_search import VectorIndex
from .ingest import chunk_pdf
from .models import embed

DATA = Path(__file__).resolve().parent.parent / "data"
PAPERS = DATA / "papers"
DIM = 768
MAX_PADDING = 150_000      # ~460 MB of fp32 vectors; safe on an 8 GB Mac


class Library:
    def __init__(self):
        PAPERS.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.chunks: list[dict] = []
        self.vectors = np.zeros((0, DIM), np.float32)
        if (DATA / "vectors.npy").exists() and (DATA / "chunks.json").exists():
            self.vectors = np.load(DATA / "vectors.npy")
            self.chunks = json.loads((DATA / "chunks.json").read_text())
        self.padding = 0
        self.index = VectorIndex()
        self._rebuild()
        self.index.warmup()

    # ---- internal helpers -------------------------------------------------
    def _rebuild(self) -> None:
        # Build [real vectors | synthetic padding] in ONE array (no extra copies;
        # this laptop has 8 GB of RAM shared by the CPU, the GPU and the AI models).
        n_real = len(self.vectors)
        allv = np.empty((n_real + self.padding, DIM), np.float32)
        allv[:n_real] = self.vectors
        if self.padding:
            rng = np.random.default_rng(42)
            pad = allv[n_real:]
            for s in range(0, self.padding, 50_000):          # fill in slices to limit temp memory
                block = rng.standard_normal((min(50_000, self.padding - s), DIM), dtype=np.float32)
                block /= np.linalg.norm(block, axis=1, keepdims=True)
                pad[s:s + len(block)] = block
        self.index.set(allv)

    def _save(self) -> None:
        np.save(DATA / "vectors.npy", self.vectors)
        (DATA / "chunks.json").write_text(json.dumps(self.chunks))

    def _remove(self, name: str) -> bool:
        keep = [i for i, c in enumerate(self.chunks) if c["doc"] != name]
        if len(keep) == len(self.chunks):
            return False
        self.chunks = [self.chunks[i] for i in keep]
        self.vectors = self.vectors[keep]
        return True

    # ---- public API -------------------------------------------------------
    def docs(self) -> list[dict]:
        counts: dict[str, int] = {}
        pages: dict[str, int] = {}
        for c in self.chunks:
            counts[c["doc"]] = counts.get(c["doc"], 0) + 1
            pages[c["doc"]] = max(pages.get(c["doc"], 0), c["page"])
        return [{"name": d, "chunks": n, "pages": pages[d]} for d, n in sorted(counts.items())]

    def set_padding(self, n: int) -> None:
        n = max(0, min(int(n), MAX_PADDING))
        with self.lock:
            self.padding = n
            self.index.set(self.vectors)                      # free the old big arrays first
            self._rebuild()
            q = np.zeros(DIM, np.float32)                     # touch the new memory once so the
            for m in ("gpu_fp16", "gpu_simd", "cpu_numpy"):   # first live search isn't slow
                self.index.search(q, 5, m)

    def add_pdf(self, path: Path) -> dict:
        t0 = time.perf_counter()
        new = chunk_pdf(path)
        t1 = time.perf_counter()
        if not new:
            raise ValueError("No extractable text found (is it a scanned image PDF?)")
        vecs = embed([c["text"] for c in new], kind="document")
        t2 = time.perf_counter()
        with self.lock:
            self._remove(path.name)                  # re-uploading replaces the old copy
            self.chunks.extend(new)
            self.vectors = np.vstack([self.vectors, vecs])
            self._rebuild()
            self._save()
        return {"doc": path.name, "chunks": len(new),
                "parse_ms": round((t1 - t0) * 1000), "embed_ms": round((t2 - t1) * 1000)}

    def remove(self, name: str) -> None:
        with self.lock:
            if self._remove(name):
                self._rebuild()
                self._save()
        (PAPERS / name).unlink(missing_ok=True)

    def search(self, question: str, k: int = 5, method: str = "gpu_fp16") -> dict:
        t0 = time.perf_counter()
        q = embed([question], kind="query")[0]
        embed_ms = (time.perf_counter() - t0) * 1000
        with self.lock:
            real = len(self.chunks)
            res = self.index.search(q, k + 10, method)
            cpu = self.index.search(q, k + 10, "cpu_numpy")   # same search on CPU, for comparison
            passages = [dict(self.chunks[i], id=int(i), score=round(float(s), 4))
                        for i, s in zip(res.indices, res.scores) if i < real][:k]
        return {"passages": passages, "embed_ms": round(embed_ms, 1),
                "search_ms": round(res.ms, 3), "cpu_search_ms": round(cpu.ms, 3),
                "method": method, "num_chunks": len(self.index), "real_chunks": real}
