"""
api/schemas.py
───────────────
Pydantic models for API request/response validation.

FastAPI uses these schemas to:
  1. Validate incoming request data (wrong types → 422 Unprocessable Entity)
  2. Serialize outgoing responses to JSON automatically
  3. Generate the OpenAPI / Swagger documentation at /docs

Keeping schemas in a separate file makes them easy to version and
import without circular dependencies.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Response models ───────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """
    Response body returned by POST /predict.

    Example JSON:
        {
          "label": "defective",
          "class_idx": 1,
          "confidence": 0.9231,
          "latency_ms": 12.4,
          "probabilities": {"normal": 0.0769, "defective": 0.9231}
        }
    """
    label:         str   = Field(..., description="Predicted class name", example="defective")
    class_idx:     int   = Field(..., description="Predicted class index (0=normal, 1=defective)")
    confidence:    float = Field(..., description="Probability of the predicted class", ge=0.0, le=1.0)
    latency_ms:    float = Field(..., description="Inference wall-clock time in milliseconds")
    probabilities: Dict[str, float] = Field(
        ..., description="Softmax probabilities for all classes"
    )


class MetricsSummary(BaseModel):
    """
    Aggregate metrics returned by GET /metrics.
    Used by the dashboard to update live charts.

    Example JSON:
        {
          "total_predictions": 142,
          "defect_rate": 0.4366,
          "avg_latency_ms": 14.2,
          "p95_latency_ms": 28.1,
          "max_latency_ms": 45.3,
          "throughput_per_min": 601.0
        }
    """
    total_predictions:  int   = Field(0, description="Total predictions in the current buffer")
    defect_rate:        float = Field(0.0, description="Fraction of images classified as defective")
    avg_latency_ms:     float = Field(0.0, description="Mean inference latency (ms)")
    p95_latency_ms:     float = Field(0.0, description="95th percentile inference latency (ms)")
    max_latency_ms:     float = Field(0.0, description="Maximum inference latency (ms)")
    throughput_per_min: float = Field(0.0, description="Estimated inferences per minute")


class HealthResponse(BaseModel):
    """
    Response body for GET /health.
    Returns system health info used by Docker / Kubernetes health checks.
    """
    status:        str  = Field(..., example="ok")
    model_loaded:  bool = Field(..., description="Whether the ONNX model is loaded")
    gpu_available: bool = Field(..., description="Whether a CUDA device is detected")
    version:       str  = Field(..., description="API version string")


class RecentPrediction(BaseModel):
    """A single recent prediction entry returned by GET /predictions."""
    label:      str
    confidence: float
    latency_ms: float


class PredictionsHistory(BaseModel):
    """Last N predictions returned by GET /predictions."""
    predictions: List[RecentPrediction]
    count:       int
