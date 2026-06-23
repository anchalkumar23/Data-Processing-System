"""
inference/engine.py
────────────────────
Production inference engine built on ONNX Runtime.

Design goals:
  1. Low latency — ONNX Runtime is typically 2-4× faster than PyTorch
     eager mode, especially on CPU.
  2. Thread-safe — multiple FastAPI worker threads can call `predict()`
     concurrently because ONNX Runtime sessions are thread-safe.
  3. Observable — every inference call records latency, confidence, and
     prediction so the dashboard can display live metrics.
  4. Resilient — handles malformed inputs gracefully without crashing
     the server.

The engine wraps a single ONNX Runtime InferenceSession and exposes
a clean `predict(image)` → `PredictionResult` interface that the API
layer calls directly.
"""

import logging
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from preprocessing.pipeline import ImagePreprocessor, load_image

log = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "defective"]


@dataclass
class PredictionResult:
    """Structured output of a single inference call."""
    label:       str    # "normal" or "defective"
    class_idx:   int    # 0 or 1
    confidence:  float  # probability of the predicted class
    latency_ms:  float  # wall-clock inference time in milliseconds
    # Probabilities for all classes: {"normal": 0.12, "defective": 0.88}
    probabilities: Dict[str, float] = field(default_factory=dict)


class MetricsBuffer:
    """
    Thread-safe ring buffer that stores recent PredictionResults.

    The dashboard polls /metrics which reads from this buffer — so we
    need a lock to prevent race conditions between the inference threads
    and the metrics-reading thread.
    """

    def __init__(self, maxlen: int = 500) -> None:
        self._buffer: Deque[PredictionResult] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, result: PredictionResult) -> None:
        with self._lock:
            self._buffer.append(result)

    def snapshot(self) -> List[PredictionResult]:
        """Returns a copy of the current buffer contents (safe to iterate)."""
        with self._lock:
            return list(self._buffer)

    def summary(self) -> Dict:
        """
        Computes aggregate stats over all buffered predictions.
        Called by the /metrics endpoint every second.
        """
        data = self.snapshot()
        if not data:
            return {
                "total_predictions":   0,
                "defect_rate":         0.0,
                "avg_latency_ms":      0.0,
                "p95_latency_ms":      0.0,
                "throughput_per_min":  0.0,
            }

        latencies  = np.array([r.latency_ms for r in data])
        defective  = sum(1 for r in data if r.class_idx == 1)

        return {
            "total_predictions":   len(data),
            "defect_rate":         round(defective / len(data), 4),
            "avg_latency_ms":      round(float(latencies.mean()), 2),
            "p95_latency_ms":      round(float(np.percentile(latencies, 95)), 2),
            "max_latency_ms":      round(float(latencies.max()), 2),
            # Approximate throughput based on buffered window
            "throughput_per_min":  round(len(data) / max(latencies.sum() / 60000, 1e-6), 1),
        }


class InferenceEngine:
    """
    ONNX Runtime-based inference engine for the defect classifier.

    Thread-safe, lazy-loaded singleton pattern: the session is created
    once on first use (or during server startup via warmup.py).

    Args:
        onnx_path:    Path to the exported .onnx model file.
        config_path:  Path to config.yaml.
        providers:    ONNX Runtime execution providers in priority order.
    """

    def __init__(
        self,
        onnx_path:   str = "checkpoints/model.onnx",
        config_path: str = "configs/config.yaml",
        providers:   Optional[List[str]] = None,
    ) -> None:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg             = cfg
        self.inf_cfg         = cfg["inference"]
        self.conf_threshold  = self.inf_cfg["confidence_threshold"]

        # Execution providers — CUDA first, CPU fallback
        self.providers = providers or self.inf_cfg.get(
            "providers", ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        # Load ONNX session
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {onnx_path}. "
                f"Run `python -m training.export_onnx` first."
            )

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.intra_op_num_threads = 2

        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_opts,
            providers=self.providers,
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.img_size    = cfg["data_gen"]["image_size"][0]

        # Preprocessing pipeline (val mode — no augmentation during inference)
        self.preprocessor = ImagePreprocessor(config_path)

        # Metrics ring buffer
        self.metrics = MetricsBuffer(
            maxlen=cfg["api"]["metrics_buffer_size"]
        )

        actual_providers = self.session.get_providers()
        log.info(f"InferenceEngine ready | model: {onnx_path} | providers: {actual_providers}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Runs the preprocessing pipeline and adds a batch dimension.

        Returns float32 array of shape (1, 3, H, W).
        """
        tensor = self.preprocessor.process(image, mode="val")   # (C, H, W)
        return tensor[np.newaxis, :].astype(np.float32)          # (1, C, H, W)

    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Runs one inference pass on a single BGR image.

        Args:
            image: BGR uint8 numpy array from cv2.imread() or camera.

        Returns:
            PredictionResult with label, confidence, latency, and class probs.
        """
        if image is None or image.size == 0:
            raise ValueError("Empty image passed to InferenceEngine.predict()")

        # Preprocess
        input_batch = self.preprocess(image)

        # Inference — measure wall-clock latency
        t_start = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: input_batch})
        latency_ms = (time.perf_counter() - t_start) * 1000

        # Logits → softmax probabilities
        logits = outputs[0][0]                    # shape: (num_classes,)
        exp    = np.exp(logits - logits.max())    # numerically stable softmax
        probs  = exp / exp.sum()

        class_idx   = int(np.argmax(probs))
        confidence  = float(probs[class_idx])
        label       = CLASS_NAMES[class_idx]

        # Warn if latency exceeds budget
        warn_ms  = self.inf_cfg.get("latency_warn_ms", 50)
        error_ms = self.inf_cfg.get("latency_error_ms", 200)
        if latency_ms > error_ms:
            log.error(f"Inference latency {latency_ms:.1f}ms exceeds error threshold {error_ms}ms")
        elif latency_ms > warn_ms:
            log.warning(f"Inference latency {latency_ms:.1f}ms exceeds warning threshold {warn_ms}ms")

        result = PredictionResult(
            label        = label,
            class_idx    = class_idx,
            confidence   = confidence,
            latency_ms   = round(latency_ms, 2),
            probabilities = {
                CLASS_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(CLASS_NAMES))
            },
        )

        self.metrics.append(result)
        return result

    def predict_from_path(self, image_path: str) -> PredictionResult:
        """Convenience method — loads an image from disk and runs predict()."""
        image = load_image(image_path)
        return self.predict(image)

    def predict_batch(self, images: List[np.ndarray]) -> List[PredictionResult]:
        """
        Runs inference on a list of images.
        Currently processes them sequentially; could be batched for throughput.
        """
        return [self.predict(img) for img in images]
