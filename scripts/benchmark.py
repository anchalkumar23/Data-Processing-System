"""
scripts/benchmark.py
─────────────────────
Throughput and latency benchmarks for the inference engine.

Measures:
  - Single-image latency  (avg, p50, p95, p99, max)
  - Batch throughput      (images/second at various batch sizes)
  - Memory usage          (resident set size during inference)

Results are printed as a formatted table and saved to logs/benchmark.json.

Run:
    python -m scripts.benchmark
    python -m scripts.benchmark --num-images 500 --warmup 20
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import psutil
import yaml

log = logging.getLogger(__name__)


def get_memory_mb() -> float:
    """Returns the current process RSS memory in megabytes."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def run_latency_benchmark(
    engine,
    num_images: int = 200,
    img_size:   int = 224,
) -> dict:
    """
    Measures single-image inference latency over `num_images` runs.
    Each run uses a fresh random image to avoid caching effects.
    """
    import cv2

    latencies = []
    log.info(f"Running single-image latency benchmark ({num_images} images)...")

    for i in range(num_images):
        # Random BGR image — represents a worst-case (no caching)
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        engine.predict(img)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {
        "count":  num_images,
        "mean":   round(float(arr.mean()), 2),
        "std":    round(float(arr.std()),  2),
        "min":    round(float(arr.min()),  2),
        "p50":    round(float(np.percentile(arr, 50)), 2),
        "p95":    round(float(np.percentile(arr, 95)), 2),
        "p99":    round(float(np.percentile(arr, 99)), 2),
        "max":    round(float(arr.max()),  2),
    }


def print_table(results: dict) -> None:
    """Prints a nicely formatted latency table."""
    print("\n" + "─" * 45)
    print("  Inference Latency Benchmark (ms)")
    print("─" * 45)
    for key, val in results.items():
        if key == "count":
            print(f"  {'Images tested':<22} {val}")
        else:
            print(f"  {key.upper():<22} {val}")
    print("─" * 45)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Benchmark the inference engine")
    parser.add_argument("--config",     default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/model.onnx")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--warmup",     type=int, default=10)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    img_size = config["data_gen"]["image_size"][0]

    from inference.engine import InferenceEngine
    from inference.warmup import warmup as do_warmup

    engine = InferenceEngine(onnx_path=args.checkpoint, config_path=args.config)
    do_warmup(engine, num_runs=args.warmup, img_size=img_size)

    # Clear warmup metrics
    engine.metrics._buffer.clear()

    mem_before = get_memory_mb()
    results = run_latency_benchmark(engine, args.num_images, img_size)
    mem_after  = get_memory_mb()

    results["memory_mb"] = round(mem_after - mem_before, 1)
    results["throughput_per_sec"] = round(1000 / results["mean"], 1)

    print_table(results)
    print(f"  {'Throughput':<22} {results['throughput_per_sec']} img/s")
    print(f"  {'Memory delta':<22} {results['memory_mb']} MB")
    print()

    # Save results
    output_path = Path(config["paths"]["logs"]) / "benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
