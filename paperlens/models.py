"""
models.py — talk to the two local AI models through Ollama (http://localhost:11434).

  * EMBED_MODEL  turns text into a 768-number vector ("embedding").
  * CHAT_MODEL   writes the final answer.

Both run on the Mac's GPU via Ollama's Metal backend, and nothing leaves the laptop.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import httpx
import numpy as np

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.environ.get("PAPERLENS_EMBED", "nomic-embed-text")
CHAT_MODEL = os.environ.get("PAPERLENS_CHAT", "qwen2.5:3b-instruct")

_client = httpx.Client(timeout=300)


def embed(texts: list[str], kind: str = "document", batch: int = 32) -> np.ndarray:
    """Embed texts -> (len(texts), 768) float32 array, each row normalised to length 1.

    nomic-embed-text expects a task prefix so it knows whether the text is a
    document being stored or a question being asked.
    """
    prefix = "search_document: " if kind == "document" else "search_query: "
    out = []
    for i in range(0, len(texts), batch):
        r = _client.post(f"{OLLAMA}/api/embed",
                         json={"model": EMBED_MODEL, "input": [prefix + t for t in texts[i:i + batch]]})
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    vecs = np.asarray(out, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs


SYSTEM_RAG = """You are PaperLens, a careful research assistant.
Answer the question using ONLY the numbered context passages from the user's papers.
Rules:
- After every claim, cite the source label(s) in square brackets, e.g. [S1] or [S2][S4].
- Only cite labels S1, S2, ... that appear below. Never copy reference numbers like [12] from the papers.
- Example of the required style: "The model is tuned with LoRA [S2], which freezes the original weights [S2][S4]." 
- If the passages do not contain the answer, say exactly: "I couldn't find this in your papers."
- Do not use outside knowledge. Be concise and clear (3–6 sentences or a short list)."""

SYSTEM_PLAIN = "You are a helpful assistant. Answer the question concisely."


def build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[S{i}] (from \"{p['doc']}\", page {p['page']})\n{p['text']}"
        for i, p in enumerate(passages, start=1)
    )
    return f"Context passages:\n\n{ctx}\n\nQuestion: {question}"


def stream_answer(question: str, passages: list[dict] | None) -> Iterator[str]:
    """Yield answer tokens as the model generates them.
    passages=None -> plain chatbot mode (no retrieval) for comparison."""
    if passages is None:
        messages = [{"role": "system", "content": SYSTEM_PLAIN},
                    {"role": "user", "content": question}]
    else:
        messages = [{"role": "system", "content": SYSTEM_RAG},
                    {"role": "user", "content": build_prompt(question, passages)}]
    body = {"model": CHAT_MODEL, "messages": messages, "stream": True,
            "options": {"temperature": 0.2, "num_ctx": 4096}}
    with _client.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("message", {}).get("content"):
                yield msg["message"]["content"]
            if msg.get("done"):
                break


def available_models() -> list[str]:
    try:
        r = _client.get(f"{OLLAMA}/api/tags", timeout=3)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []
