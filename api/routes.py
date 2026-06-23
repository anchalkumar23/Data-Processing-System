"""
api/routes.py
──────────────
FastAPI route handlers for the defect detection API.

Endpoints:
  POST /predict      — upload an image, get a defect prediction
  GET  /metrics      — aggregate stats for the dashboard
  GET  /predictions  — last N predictions (for the dashboard timeline)
  GET  /health       — liveness / readiness check

We keep route logic thin: routes validate inputs, call the engine,
and format the response.  Business logic lives in inference/engine.py.
"""

import io
import logging
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    MetricsSummary,
    PredictionResponse,
    PredictionsHistory,
    RecentPrediction,
)
from inference.engine import InferenceEngine

log = logging.getLogger(__name__)

router = APIRouter()

# The engine is injected via FastAPI's dependency injection system.
# This makes routes easy to test with a mock engine.
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Dependency that returns the global engine instance."""
    if _engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not initialised. Check server startup logs.",
        )
    return _engine


def set_engine(engine: InferenceEngine) -> None:
    """Called from main.py during server startup."""
    global _engine
    _engine = engine


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Classify an image as normal or defective",
    description=(
        "Upload a surface image (JPEG/PNG) and receive a defect classification. "
        "The model was trained on synthetic industrial surface data."
    ),
)
async def predict(
    file:   UploadFile = File(..., description="Image file (JPEG or PNG)"),
    engine: InferenceEngine = Depends(get_engine),
) -> PredictionResponse:
    """
    Accepts an image upload and returns a defect prediction.

    The image is decoded from bytes → OpenCV BGR array, preprocessed
    through the same pipeline used during training, and passed to the
    ONNX Runtime session.
    """
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image format: {file.content_type}. Use JPEG or PNG.",
        )

    # Read file bytes and decode to OpenCV array
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image. File may be corrupted.",
        )

    # Run inference
    try:
        result = engine.predict(image)
    except Exception as e:
        log.exception("Inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )

    return PredictionResponse(
        label         = result.label,
        class_idx     = result.class_idx,
        confidence    = result.confidence,
        latency_ms    = result.latency_ms,
        probabilities = result.probabilities,
    )


@router.get(
    "/metrics",
    response_model=MetricsSummary,
    summary="Get aggregate inference metrics",
    description="Returns live stats: defect rate, latency percentiles, throughput.",
)
async def get_metrics(
    engine: InferenceEngine = Depends(get_engine),
) -> MetricsSummary:
    """Polls the metrics ring buffer and returns aggregate statistics."""
    summary = engine.metrics.summary()
    return MetricsSummary(**summary)


@router.get(
    "/predictions",
    response_model=PredictionsHistory,
    summary="Get recent prediction history",
    description="Returns the last N predictions for the dashboard timeline chart.",
)
async def get_predictions(
    limit:  int = Query(50, ge=1, le=500, description="Number of recent predictions to return"),
    engine: InferenceEngine = Depends(get_engine),
) -> PredictionsHistory:
    """Returns recent prediction history from the ring buffer."""
    recent = engine.metrics.snapshot()[-limit:]
    predictions = [
        RecentPrediction(
            label      = r.label,
            confidence = r.confidence,
            latency_ms = r.latency_ms,
        )
        for r in recent
    ]
    return PredictionsHistory(predictions=predictions, count=len(predictions))


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    description="Used by Docker and load balancers to verify the service is alive.",
)
async def health_check(
    engine: Optional[InferenceEngine] = Depends(get_engine),
) -> HealthResponse:
    """Liveness probe — returns 200 if the model is loaded and ready."""
    return HealthResponse(
        status        = "ok",
        model_loaded  = engine is not None,
        gpu_available = torch.cuda.is_available(),
        version       = "1.0.0",
    )
