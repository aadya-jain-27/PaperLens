"""
gpu_search.py — the GPU heart of PaperLens.

Problem: we have N text chunks, each turned into a vector of D numbers (D = 768).
A question is also turned into a vector q. To find the most relevant chunks we
compute a similarity score for EVERY chunk:

        score[i] = dot(docs[i], q) = sum over d of docs[i][d] * q[d]

Because all vectors are normalised to length 1, the dot product equals the
cosine similarity (1.0 = same meaning, 0 = unrelated).

Every score[i] is independent of every other one, so this is "embarrassingly
parallel": perfect for a GPU, which runs thousands of threads at once.

This file has 4 ways of computing the scores, from slowest to fastest:

  1. cpu_python   - plain Python loops (one CPU core, no tricks). Teaching baseline.
  2. cpu_numpy    - NumPy, which uses Apple's Accelerate BLAS (multi-core + SIMD).
                    This is a STRONG, honest CPU baseline.
  3. gpu_naive    - OUR Metal kernel v1: one GPU thread per document.
  4. gpu_simd     - OUR Metal kernel v2: 32 GPU threads (one SIMD-group)
                    cooperate on each document, reading memory in a coalesced
                    pattern with float4 vector loads, then combine their
                    partial sums with the hardware instruction simd_sum().

We also compare against mlx_matmul (Apple's own hand-tuned GPU library) to see
how close our hand-written kernel gets to a professional implementation.

The kernels are written in the Metal Shading Language (Apple's equivalent of
CUDA C++) and compiled at runtime through MLX's `mx.fast.metal_kernel`.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

# --------------------------------------------------------------------------
# One dedicated "GPU thread". MLX binds GPU command streams to the thread that
# created them, while the web server answers requests from many threads. So
# every piece of GPU work is queued onto this single worker, which also stops
# two requests from fighting over the GPU at the same time.
# --------------------------------------------------------------------------
_GPU_THREAD_NAME = "paperlens-gpu"
_gpu_worker = ThreadPoolExecutor(
    max_workers=1, thread_name_prefix=_GPU_THREAD_NAME,
    initializer=lambda: mx.set_default_stream(mx.new_stream(mx.gpu)),
)


def on_gpu(fn, *args, **kwargs):
    """Run fn on the GPU worker thread and wait for its result."""
    if threading.current_thread().name.startswith(_GPU_THREAD_NAME):
        return fn(*args, **kwargs)
    return _gpu_worker.submit(fn, *args, **kwargs).result()

# --------------------------------------------------------------------------
# Kernel v1 — NAIVE: one thread computes one document's full dot product.
# --------------------------------------------------------------------------
# MLX generates the function signature for us. Our inputs become
#   const device float* docs, const device float* query
# our output becomes
#   device float* scores
# D (vector length) is a compile-time constant passed through `template=`, and
# because we use `docs_shape`, MLX also passes the array's shape so N is known
# at run time (one compiled kernel works for any library size).
#
# Why it is slow-ish: thread 0 reads docs[0..767], thread 1 reads docs[768..],
# thread 2 reads docs[1536..] ... neighbouring threads touch memory addresses
# that are 3 KB apart, so the GPU cannot merge their reads into one wide
# memory transaction (the reads are NOT "coalesced").
NAIVE_SOURCE = """
    uint N   = docs_shape[0];                       // number of documents (runtime value)
    uint row = thread_position_in_grid.x;          // which document am I?
    if (row >= N) return;                          // guard: extra threads do nothing

    float acc = 0.0f;
    for (uint d = 0; d < D; d++) {                 // walk the whole vector alone
        acc += docs[row * D + d] * query[d];
    }
    scores[row] = acc;
"""

# --------------------------------------------------------------------------
# Kernel v2 — SIMD-GROUP COOPERATIVE: 32 threads share one document.
# --------------------------------------------------------------------------
# Apple GPUs execute threads in groups of 32 called SIMD-groups (NVIDIA calls
# them "warps"). Here each SIMD-group owns one document:
#   lane 0 reads float4 #0, lane 1 reads float4 #1, ... lane 31 reads float4 #31,
#   then they all jump ahead by 32 and repeat.
# Neighbouring lanes read neighbouring addresses -> coalesced memory access,
# and float4 means every load fetches 16 bytes instead of 4.
# Finally simd_sum() adds the 32 partial results together in hardware,
# with no shared memory and no locks.
SIMD_SOURCE = """
    uint N    = docs_shape[0];                                     // number of documents
    uint lane = thread_index_in_simdgroup;                         // 0..31
    uint row  = threadgroup_position_in_grid.x * ROWS_PER_TG
              + simdgroup_index_in_threadgroup;                     // my document
    if (row >= N) return;          // whole SIMD-group shares `row`, so it exits together

    const device float4* doc4 = (const device float4*)(docs + row * D);
    const device float4* q4   = (const device float4*)(query);

    float acc = 0.0f;
    for (uint i = lane; i < D / 4; i += 32) {      // 768/4 = 192 float4s -> 6 per lane
        acc += dot(doc4[i], q4[i]);                // 4 multiply-adds in one instruction
    }
    acc = simd_sum(acc);                           // hardware reduction across 32 lanes
    if (lane == 0) {
        scores[row] = acc;                         // one lane writes the answer
    }
"""

# --------------------------------------------------------------------------
# Kernel v3 — SIMD + HALF PRECISION: same as v2 but documents are stored as
# 16-bit floats instead of 32-bit.
# --------------------------------------------------------------------------
# Profiling v2 shows it already reaches ~85% of the M2's peak memory bandwidth
# (~100 GB/s). That means the kernel is MEMORY-BOUND: the GPU's math units are
# mostly waiting for data to arrive. Faster math won't help, so we move fewer
# bytes. Storing vectors in float16 halves the data to read, so it nearly
# halves the time. We still ACCUMULATE in float32 so rounding errors don't pile up.
# Search quality is unchanged (top results identical; checked in the benchmark).
FP16_SOURCE = """
    uint N    = docs_shape[0];
    uint lane = thread_index_in_simdgroup;
    uint row  = threadgroup_position_in_grid.x * ROWS_PER_TG
              + simdgroup_index_in_threadgroup;
    if (row >= N) return;

    const device half4*  doc4 = (const device half4*)(docs + row * D);
    const device float4* q4   = (const device float4*)(query);

    float acc = 0.0f;
    for (uint i = lane; i < D / 4; i += 32) {
        acc += dot(float4(doc4[i]), q4[i]);        // load 8 bytes, compute in fp32
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        scores[row] = acc;
    }
"""

THREADGROUP_SIZE = 256                 # threads per threadgroup ("block" in CUDA)
SIMD_WIDTH = 32                         # threads per SIMD-group ("warp" in CUDA)
ROWS_PER_TG = THREADGROUP_SIZE // SIMD_WIDTH   # 8 documents per threadgroup

_naive_kernel = mx.fast.metal_kernel(
    name="paperlens_dot_naive",
    input_names=["docs", "query"],
    output_names=["scores"],
    source=NAIVE_SOURCE,
)

_simd_kernel = mx.fast.metal_kernel(
    name="paperlens_dot_simd",
    input_names=["docs", "query"],
    output_names=["scores"],
    source=SIMD_SOURCE,
)

_fp16_kernel = mx.fast.metal_kernel(
    name="paperlens_dot_fp16",
    input_names=["docs", "query"],
    output_names=["scores"],
    source=FP16_SOURCE,
)


def gpu_scores_naive(docs: mx.array, query: mx.array) -> mx.array:
    n, d = docs.shape
    (out,) = _naive_kernel(
        inputs=[docs, query],
        template=[("D", d)],
        grid=(n, 1, 1),                           # one thread per document
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    return out


def gpu_scores_simd(docs: mx.array, query: mx.array) -> mx.array:
    n, d = docs.shape
    assert d % 4 == 0, "vector length must be a multiple of 4 for float4 loads"
    n_groups = (n + ROWS_PER_TG - 1) // ROWS_PER_TG
    (out,) = _simd_kernel(
        inputs=[docs, query],
        template=[("D", d), ("ROWS_PER_TG", ROWS_PER_TG)],
        grid=(n_groups * THREADGROUP_SIZE, 1, 1),  # 32 threads per document
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    return out


def gpu_scores_fp16(docs16: mx.array, query: mx.array) -> mx.array:
    """docs16 must be float16; query stays float32."""
    n, d = docs16.shape
    assert docs16.dtype == mx.float16 and d % 4 == 0
    n_groups = (n + ROWS_PER_TG - 1) // ROWS_PER_TG
    (out,) = _fp16_kernel(
        inputs=[docs16, query],
        template=[("D", d), ("ROWS_PER_TG", ROWS_PER_TG)],
        grid=(n_groups * THREADGROUP_SIZE, 1, 1),
        threadgroup=(THREADGROUP_SIZE, 1, 1),
        output_shapes=[(n,)],
        output_dtypes=[mx.float32],
    )
    return out


def gpu_scores_mlx(docs: mx.array, query: mx.array) -> mx.array:
    """Apple's professionally tuned GPU matrix-vector product (reference)."""
    return docs @ query


# --------------------------------------------------------------------------
# CPU baselines
# --------------------------------------------------------------------------
def cpu_scores_python(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Pure Python: what a beginner would write. One core, one number at a time."""
    rows = docs.tolist()
    q = query.tolist()
    return np.array([sum(a * b for a, b in zip(row, q)) for row in rows], dtype=np.float32)


def cpu_scores_numpy(docs: np.ndarray, query: np.ndarray) -> np.ndarray:
    """NumPy -> Apple Accelerate BLAS: multi-threaded + CPU SIMD. Strong baseline."""
    return docs @ query


# --------------------------------------------------------------------------
# Top-k: keep only the k best scores.
# --------------------------------------------------------------------------
def gpu_topk(scores: mx.array, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[0])
    # argpartition finds the k largest in O(N) without fully sorting, on the GPU.
    idx = mx.argpartition(-scores, kth=k - 1)[:k]
    top = scores[idx]
    order = mx.argsort(-top)
    idx, top = idx[order], top[order]
    mx.eval(idx, top)                             # force the GPU work to finish
    return np.array(idx), np.array(top)


def cpu_topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, scores.shape[0])
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]


# Each GPU method receives (docs_fp32, docs_fp16, query) and picks what it needs.
GPU_METHODS = {
    "gpu_fp16": lambda d32, d16, q: gpu_scores_fp16(d16, q),
    "gpu_simd": lambda d32, d16, q: gpu_scores_simd(d32, q),
    "gpu_naive": lambda d32, d16, q: gpu_scores_naive(d32, q),
    "mlx_matmul": lambda d32, d16, q: gpu_scores_mlx(d32, q),
}
CPU_METHODS = {
    "cpu_numpy": cpu_scores_numpy,
    "cpu_python": cpu_scores_python,
}

METHOD_LABELS = {
    "cpu_python": "CPU · pure Python",
    "cpu_numpy": "CPU · NumPy (Accelerate BLAS)",
    "gpu_naive": "GPU · our kernel v1 (naive)",
    "gpu_simd": "GPU · our kernel v2 (SIMD + float4)",
    "gpu_fp16": "GPU · our kernel v3 (SIMD + fp16)",
    "mlx_matmul": "GPU · Apple MLX matmul (reference)",
}


# --------------------------------------------------------------------------
# A small vector index that keeps one copy of the data for the CPU (NumPy) and
# one for the GPU (MLX). On Apple Silicon both live in the same "unified"
# memory, so there is no slow PCIe copy like on a desktop NVIDIA card.
# --------------------------------------------------------------------------
@dataclass
class SearchResult:
    indices: np.ndarray
    scores: np.ndarray
    ms: float
    method: str


class VectorIndex:
    def __init__(self, vectors: np.ndarray | None = None):
        self.set(vectors if vectors is not None else np.zeros((0, 768), np.float32))

    def set(self, vectors: np.ndarray) -> None:
        self.cpu = np.ascontiguousarray(vectors, dtype=np.float32)
        on_gpu(self._upload)

    def _upload(self) -> None:
        self.gpu = mx.array(self.cpu)
        self.gpu16 = self.gpu.astype(mx.float16)
        mx.eval(self.gpu, self.gpu16)

    def warmup(self) -> None:
        """Compile every kernel once at start-up so the first real search is fast."""
        def work():
            d = mx.zeros((64, 768)); q = mx.zeros((768,))
            for fn in GPU_METHODS.values():
                gpu_topk(fn(d, d.astype(mx.float16), q), 5)
        on_gpu(work)

    def __len__(self) -> int:
        return self.cpu.shape[0]

    def search(self, query: np.ndarray, k: int = 5, method: str = "gpu_fp16") -> SearchResult:
        if len(self) == 0:
            return SearchResult(np.array([], int), np.array([], np.float32), 0.0, method)
        q = np.ascontiguousarray(query, dtype=np.float32)
        if method in GPU_METHODS:
            def work():
                qg = mx.array(q)
                mx.eval(qg)
                t0 = time.perf_counter()
                scores = GPU_METHODS[method](self.gpu, self.gpu16, qg)
                idx, top = gpu_topk(scores, k)    # includes mx.eval -> waits for GPU
                return idx, top, (time.perf_counter() - t0) * 1000
            idx, top, ms = on_gpu(work)
        else:
            t0 = time.perf_counter()
            scores = CPU_METHODS[method](self.cpu, q)
            idx, top = cpu_topk(scores, k)
            ms = (time.perf_counter() - t0) * 1000
        return SearchResult(idx, top, ms, method)


# --------------------------------------------------------------------------
# Benchmark: how long does one search take as the library grows?
# --------------------------------------------------------------------------
def _time(fn, reps: int) -> float:
    fn()                                          # warm-up (compiles the kernel)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def run_benchmark(*args, **kwargs) -> dict:
    return on_gpu(_run_benchmark, *args, **kwargs)


def _run_benchmark(sizes=(1_000, 10_000, 50_000, 100_000, 200_000), dim=768,
                   reps=20, python_max=10_000, progress=None) -> dict:
    rng = np.random.default_rng(0)
    rows = []
    for n in sizes:
        docs = rng.standard_normal((n, dim), dtype=np.float32)
        docs /= np.linalg.norm(docs, axis=1, keepdims=True)
        q = docs[rng.integers(n)] + 0.05 * rng.standard_normal(dim, dtype=np.float32)
        q /= np.linalg.norm(q)
        dg, qg = mx.array(docs), mx.array(q)
        dg16 = dg.astype(mx.float16)
        mx.eval(dg, dg16, qg)

        # Correctness first: every GPU kernel must agree with NumPy.
        ref = docs @ q
        ref_top = set(cpu_topk(ref, 10)[0].tolist())
        for name, fn in GPU_METHODS.items():
            got = np.array(fn(dg, dg16, qg))
            tol = 2e-3 if name == "gpu_fp16" else 1e-4
            if not np.allclose(got, ref, atol=tol):
                raise RuntimeError(f"{name} produced wrong scores at N={n}")
            if set(cpu_topk(got, 10)[0].tolist()) != ref_top:
                raise RuntimeError(f"{name} returned different top-10 at N={n}")

        row = {"n": n, "mb": round(docs.nbytes / 1e6, 1)}
        row["cpu_numpy"] = _time(lambda: docs @ q, reps)
        row["cpu_python"] = _time(lambda: cpu_scores_python(docs, q), 1) if n <= python_max else None
        for name, fn in GPU_METHODS.items():
            row[name] = _time(lambda fn=fn: mx.eval(fn(dg, dg16, qg)), reps)
        # Effective memory bandwidth = bytes read / time. M2 peak is ~100 GB/s.
        row["simd_gbps"] = round(docs.nbytes / (row["gpu_simd"] / 1000) / 1e9, 1)
        row["fp16_gbps"] = round(docs.nbytes / 2 / (row["gpu_fp16"] / 1000) / 1e9, 1)
        rows.append(row)
        if progress:
            progress(row)
        del docs, dg, dg16

    # Batched search: 64 questions at once (matrix x matrix). This turns a
    # memory-bound problem into a compute-bound one, where GPUs shine most.
    n, b = sizes[-1], 64
    docs = rng.standard_normal((n, dim), dtype=np.float32)
    Q = rng.standard_normal((b, dim), dtype=np.float32)
    dg, Qg = mx.array(docs), mx.array(Q)
    mx.eval(dg, Qg)
    batched = {
        "n": n, "batch": b,
        "cpu_numpy": _time(lambda: docs @ Q.T, 5),
        "mlx_matmul": _time(lambda: mx.eval(dg @ Qg.T), 5),
    }
    return {"dim": dim, "rows": rows, "batched": batched, "labels": METHOD_LABELS,
            "device": str(mx.default_device())}
