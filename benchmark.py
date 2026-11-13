"""
Run the CPU-vs-GPU benchmark from the terminal and save the results for the web app.

    .venv/bin/python benchmark.py
"""
import datetime
import json
from pathlib import Path

from paperlens.gpu_search import METHOD_LABELS, run_benchmark

ORDER = ["cpu_python", "cpu_numpy", "gpu_naive", "gpu_simd", "mlx_matmul", "gpu_fp16"]


def show(row):
    cells = "  ".join(f"{(row[k] if row[k] is not None else float('nan')):>10.3f}" for k in ORDER)
    print(f"{row['n']:>8,}  {cells}   v3 = {row['cpu_numpy'] / row['gpu_fp16']:.2f}x NumPy")


if __name__ == "__main__":
    print("\nTime per search in milliseconds (lower is better). All GPU kernels verified against NumPy.\n")
    print(f"{'N':>8}  " + "  ".join(f"{k:>10}" for k in ORDER))
    r = run_benchmark(sizes=(1_000, 10_000, 50_000, 100_000, 150_000), progress=show)
    b = r["batched"]
    print(f"\nBatched {b['batch']} questions x {b['n']:,} chunks: CPU {b['cpu_numpy']:.1f} ms  vs  GPU {b['mlx_matmul']:.1f} ms "
          f"({b['cpu_numpy'] / b['mlx_matmul']:.1f}x)")
    print("\nLegend:", ", ".join(f"{k} = {METHOD_LABELS[k]}" for k in ORDER))
    r["recorded"] = datetime.date.today().isoformat()
    out = Path(__file__).parent / "static" / "benchmark_results.json"
    out.write_text(json.dumps(r, indent=1))
    print(f"\nSaved -> {out.relative_to(Path(__file__).parent)}")
