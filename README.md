# PaperLens: GPU-Accelerated Private Research Assistant

**Ask questions about your research papers and get answers with page-level citations. It runs 100% on your laptop's GPU, with custom Metal kernels for vector search.**

PaperLens is a Retrieval-Augmented Generation (RAG) system built for Apple Silicon. Its core, the similarity search over every chunk of every paper, is a set of **hand-written GPU kernels in the Metal Shading Language**. Each kernel version was profiled and optimised in turn; the final one is **3.7× faster than optimised multi-core CPU code (NumPy + Apple Accelerate)** and about 2× faster than Apple's general-purpose MLX matrix routine.

## Features
- **Private and offline:** PDFs, embeddings and the language model all stay on the machine.
- **Cited answers:** every claim links to the exact paper and page. Citations the model invents are flagged.
- **Says "I couldn't find this"** instead of making up an answer.
- **Side-by-side comparison** with a plain chatbot to show why grounding matters.
- **Live GPU telemetry:** embed, search and generate times, with GPU vs. CPU search time on every query.
- **Scale test:** adds 150k background chunks to show GPU behaviour at search-engine scale.
- **Benchmark dashboard:** CPU vs. three GPU kernels at sizes from 1k to 150k chunks.

## Architecture
```
PDF ─► pypdf text ─► 180-word chunks (40 overlap, page kept) ─► nomic-embed-text (GPU) ─► 768-d unit vectors
                                                                                             │
Question ─► embed (GPU) ─► ★ custom Metal kernel: score all N chunks in parallel ◄────────────┘
                              └► GPU top-k (argpartition) ─► top 5 passages
                                                                 └► Qwen2.5-3B (GPU via Ollama) ─► answer + [n] citations
```

| Layer | Tech |
|---|---|
| GPU kernels | Metal Shading Language via `mlx.core.fast.metal_kernel` |
| Embeddings / LLM | `nomic-embed-text`, `qwen2.5:3b-instruct` via Ollama (Metal backend) |
| Backend | Python, FastAPI, NumPy, pypdf |
| Frontend | Plain HTML/CSS/JS (no CDN, works offline) |

## The GPU kernels (`paperlens/gpu_search.py`)
| Version | Idea | Why it is faster |
|---|---|---|
| v1 naive | 1 thread per chunk, loops over 768 floats | baseline; uncoalesced memory reads |
| v2 SIMD | 32 threads (one SIMD-group, like a CUDA warp) per chunk, `float4` loads, `simd_sum` reduction | coalesced 16-byte reads; hardware reduction; reaches ~85 GB/s (M2 peak ≈ 100 GB/s) |
| v3 fp16 | v2 but vectors stored as `half`, accumulated in `float` | the kernel is **memory-bound**, so halving the bytes nearly halves the time; top-10 unchanged |

Every kernel is checked against NumPy (scores within tolerance and an identical top-10) before it is timed.

## Benchmark (Apple M2, 8-core GPU, 8 GB; median of 20 runs)
| Chunks | CPU Python | CPU NumPy | GPU v1 | GPU v2 | MLX matmul | **GPU v3** | v3 vs NumPy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 155 ms | **0.03 ms** | 0.25 ms | 0.20 ms | 0.24 ms | 0.22 ms | 0.1× |
| 10,000 | 1,069 ms | 1.18 ms | 0.73 ms | 0.59 ms | 0.57 ms | **0.43 ms** | 2.7× |
| 50,000 | – | 3.26 ms | 1.94 ms | 1.93 ms | 1.96 ms | **1.15 ms** | 2.8× |
| 100,000 | – | 5.96 ms | 3.71 ms | 4.00 ms | 3.66 ms | **2.02 ms** | 3.0× |
| 150,000 | – | 10.83 ms | 5.63 ms | 5.43 ms | 6.12 ms | **2.97 ms** | **3.7×** |

Batched (64 questions × 150k chunks): CPU 40.3 ms vs GPU 8.4 ms (**4.8×**).

**Takeaways:** the CPU wins below ~5k chunks because a GPU launch has a fixed overhead of about 0.2 ms. Single-question search is memory-bandwidth-bound, so the gain came from reading fewer bytes, not from doing more maths. Batching makes the work compute-bound, which is where GPUs gain the most.

## Run it
Requirements: an Apple Silicon Mac, Python 3.10+, and [Ollama](https://ollama.com).
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b-instruct
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py          # open http://localhost:8000
.venv/bin/python benchmark.py    # optional: terminal benchmark, refreshes the dashboard
```

## Project layout
```
paperlens/gpu_search.py   Metal kernels, CPU baselines, top-k, VectorIndex, benchmark
paperlens/ingest.py       PDF → cleaned text → overlapping chunks with page numbers
paperlens/models.py       Ollama calls: embeddings + streaming answer, RAG prompt
paperlens/library.py      paper collection, saving/loading vectors, scale test
paperlens/server.py       FastAPI endpoints (upload, ask [streaming], benchmark…)
static/                   web UI (index.html, style.css, app.js)
benchmark.py              CLI benchmark → static/benchmark_results.json
```

## Limitations and future work
- Scanned (image-only) PDFs need OCR.
- A 3B model is less fluent than cloud LLMs; swap in a larger model on machines with more RAM.
- Exact (brute-force) search is used. Approximate indexes (IVF, HNSW) would be needed beyond ~10M chunks.
- Future work: a custom batched kernel using threadgroup memory tiling, a fused top-k kernel, a CUDA port for NVIDIA GPUs, and int8 quantised vectors.
