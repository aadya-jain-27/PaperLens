"""
server.py — the web API (FastAPI). Run with:   python run.py

Endpoints
  GET    /                      the web app (static/index.html)
  GET    /api/status            papers, chunk counts, models, GPU device
  POST   /api/upload            upload one or more PDFs -> chunk -> embed -> index
  DELETE /api/docs/{name}       remove a paper
  GET    /api/pdf/{name}        serve a PDF
  GET    /api/cite/{id}         PDF with a cited passage highlighted
  POST   /api/ask               question -> GPU search -> streamed answer
  POST   /api/scale             add N synthetic chunks for the scale test
  POST   /api/benchmark         run the CPU-vs-GPU kernel benchmark
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import models
from .highlight import highlighted_pdf
from .gpu_search import METHOD_LABELS, run_benchmark
from .library import PAPERS, Library

STATIC = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="PaperLens")
lib = Library()


class AskBody(BaseModel):
    question: str
    k: int = 5
    method: str = "gpu_fp16"
    use_papers: bool = True


class ScaleBody(BaseModel):
    n: int


class BenchBody(BaseModel):
    sizes: list[int] = [1_000, 10_000, 50_000, 100_000, 150_000]


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/status")
def status():
    have = models.available_models()
    return {
        "docs": lib.docs(),
        "real_chunks": len(lib.chunks),
        "padding": lib.padding,
        "device": str(mx.default_device()),
        "embed_model": models.EMBED_MODEL,
        "chat_model": models.CHAT_MODEL,
        "ollama_ok": bool(have),
        "models_ok": any(m.startswith(models.EMBED_MODEL) for m in have)
                     and any(m.startswith(models.CHAT_MODEL) for m in have),
        "methods": {k: v for k, v in METHOD_LABELS.items() if k != "cpu_python"},
    }


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        name = Path(f.filename or "paper.pdf").name
        if not name.lower().endswith(".pdf"):
            results.append({"doc": name, "error": "Not a PDF"})
            continue
        dest = PAPERS / name
        dest.write_bytes(f.file.read())
        try:
            results.append(lib.add_pdf(dest))
        except Exception as e:  # noqa: BLE001 - report any parsing/embedding error to the UI
            dest.unlink(missing_ok=True)
            results.append({"doc": name, "error": str(e)})
    return {"results": results, **status()}


@app.delete("/api/docs/{name}")
def delete_doc(name: str):
    lib.remove(Path(name).name)
    return status()


@app.get("/api/pdf/{name}")
def get_pdf(name: str):
    path = PAPERS / Path(name).name
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="application/pdf")


@app.get("/api/cite/{chunk_id}")
def cite(chunk_id: int, q: str = ""):
    """the cited page with the passage in yellow and the best sentence in orange"""
    if not 0 <= chunk_id < len(lib.chunks):
        raise HTTPException(404, "passage not found, ask the question again")
    ch = lib.chunks[chunk_id]
    path = PAPERS / ch["doc"]
    if not path.exists():
        raise HTTPException(404)
    pdf, _ = highlighted_pdf(str(path), ch["page"], ch["text"], q)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="{ch["doc"]}"'})


@app.post("/api/scale")
def scale(body: ScaleBody):
    lib.set_padding(body.n)
    return status()


@app.post("/api/ask")
def ask(body: AskBody):
    """Streams newline-delimited JSON events:
       {"type":"sources", ...}  then many {"type":"token","text":...}  then {"type":"done"}"""
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Empty question")

    def events():
        passages = None
        if body.use_papers:
            if not lib.chunks:
                yield json.dumps({"type": "error", "text": "Upload a paper first."}) + "\n"
                return
            found = lib.search(q, body.k, body.method)
            passages = found["passages"]
            yield json.dumps({"type": "sources", **found}) + "\n"
        t0 = time.perf_counter()
        first = None
        n = 0
        try:
            for tok in models.stream_answer(q, passages):
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
                yield json.dumps({"type": "token", "text": tok}) + "\n"
        except Exception as e:  # noqa: BLE001
            yield json.dumps({"type": "error", "text": f"Model error: {e}"}) + "\n"
            return
        total = time.perf_counter() - t0
        yield json.dumps({"type": "done", "gen_ms": round(total * 1000),
                          "first_token_ms": round((first or 0) * 1000),
                          "tokens_per_s": round(n / total, 1) if total else 0}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson")


@app.post("/api/benchmark")
def benchmark(body: BenchBody):
    sizes = sorted({max(256, min(int(s), 200_000)) for s in body.sizes})
    return run_benchmark(sizes=tuple(sizes))


app.mount("/static", StaticFiles(directory=STATIC), name="static")
