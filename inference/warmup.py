"""
inference/warmup.py
────────────────────
Warms up the ONNX Runtime session before the API starts accepting requests.

Why warm up?
  The first inference call is always slower because:
    • ONNX Runtime compiles and optimises the execution plan on first run
    • CUDA kernels are JIT-compiled on first GPU use
    • The OS may need to page in the model weights from disk

  Without warmup, the first real request from a client would experience
  ~500ms latency instead of the typical <20ms.  We run a few dummy
  inferences at startup so the session is "hot" by the time real traffic
  arrives.

  This is standard practice in production ML serving systems.
"""

import logging
import time

import numpy as np

from inference.engine import InferenceEngine

log = logging.getLogger(__name__)


def warmup(
    engine:    InferenceEngine,
    num_runs:  int = 5,
    img_size:  int = 224,
) -> None:
    """
    Runs `num_runs` dummy inferences to warm up the ONNX session.

    Args:
        engine:   The InferenceEngine instance to warm up.
        num_runs: How many warmup passes to run.
        img_size: Image dimensions (must match model's expected input).
    """
    log.info(f"Warming up inference engine ({num_runs} runs)...")

    latencies = []
    for i in range(num_runs):
        # Create a random BGR uint8 image — no need for it to be realistic
        dummy_image = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

        t0 = time.perf_counter()
        engine.predict(dummy_image)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

    # Clear the dummy results so they don't pollute the real metrics
    engine.metrics._buffer.clear()

    avg = sum(latencies) / len(latencies)
    log.info(
        f"Warmup complete — avg latency: {avg:.1f}ms "
        f"(runs: {[f'{l:.1f}' for l in latencies]})"
    )
