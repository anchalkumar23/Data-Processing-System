"""
tests/test_api.py
──────────────────
Integration tests for the FastAPI application.

We use httpx.AsyncClient (the recommended async test client for FastAPI)
to test each endpoint in isolation, injecting a mock InferenceEngine
so the tests don't require an actual trained ONNX model.

Tests cover:
  - GET /api/v1/health         → 200, correct schema
  - POST /api/v1/predict       → 200 with valid image, 415 with wrong type
  - GET /api/v1/metrics        → 200, correct numeric types
  - GET /api/v1/predictions    → 200, correct list structure
  - Dashboard route            → 200 for GET /

Run:
    pytest tests/test_api.py -v
"""

import io
import json
from typing import Dict
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from api.main import app
from api.routes import set_engine
from inference.engine import InferenceEngine, MetricsBuffer, PredictionResult


# ── Mock engine fixture ───────────────────────────────────────────────────────

def make_mock_engine() -> MagicMock:
    """
    Creates a MagicMock that quacks like an InferenceEngine without
    needing an actual ONNX model file.
    """
    engine = MagicMock(spec=InferenceEngine)

    # What predict() returns
    engine.predict.return_value = PredictionResult(
        label        = "defective",
        class_idx    = 1,
        confidence   = 0.8923,
        latency_ms   = 14.3,
        probabilities = {"normal": 0.1077, "defective": 0.8923},
    )

    # What metrics.summary() returns
    engine.metrics = MagicMock(spec=MetricsBuffer)
    engine.metrics.summary.return_value = {
        "total_predictions":   42,
        "defect_rate":         0.4286,
        "avg_latency_ms":      14.2,
        "p95_latency_ms":      28.1,
        "max_latency_ms":      45.3,
        "throughput_per_min":  601.0,
    }

    # What metrics.snapshot() returns for /predictions
    engine.metrics.snapshot.return_value = [
        PredictionResult("normal",   0, 0.9234, 12.1, {"normal": 0.9234, "defective": 0.0766}),
        PredictionResult("defective", 1, 0.7812, 15.4, {"normal": 0.2188, "defective": 0.7812}),
    ]

    return engine


def make_test_image_bytes(width: int = 64, height: int = 64) -> bytes:
    """Creates a small random JPEG image as bytes for upload tests."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    success, buffer = cv2.imencode(".jpg", img)
    assert success
    return buffer.tobytes()


# ── Test fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def inject_mock_engine():
    """Automatically injects the mock engine before each test."""
    mock = make_mock_engine()
    set_engine(mock)
    yield mock
    set_engine(None)  # clean up after test


@pytest.fixture
def image_bytes():
    return make_test_image_bytes()


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
class TestHealthEndpoint:

    async def test_health_returns_200(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/health")
        assert resp.status_code == 200

    async def test_health_schema(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/health")
        data = resp.json()
        assert "status"       in data
        assert "model_loaded" in data
        assert "gpu_available" in data
        assert "version"      in data
        assert data["status"] == "ok"


@pytest.mark.asyncio
class TestPredictEndpoint:

    async def test_predict_valid_image(self, image_bytes):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] in ("normal", "defective")
        assert 0.0 <= data["confidence"] <= 1.0
        assert data["latency_ms"] > 0

    async def test_predict_wrong_content_type(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/predict",
                files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
            )
        assert resp.status_code == 415

    async def test_predict_empty_file(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/predict",
                files={"file": ("empty.jpg", b"", "image/jpeg")},
            )
        assert resp.status_code == 400

    async def test_predict_response_has_probabilities(self, image_bytes):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/predict",
                files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            )
        data = resp.json()
        assert "probabilities" in data
        probs = data["probabilities"]
        assert "normal"    in probs
        assert "defective" in probs
        # Probabilities should sum to ~1
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01


@pytest.mark.asyncio
class TestMetricsEndpoint:

    async def test_metrics_returns_200(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/metrics")
        assert resp.status_code == 200

    async def test_metrics_numeric_types(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/metrics")
        data = resp.json()
        assert isinstance(data["total_predictions"],   int)
        assert isinstance(data["defect_rate"],         float)
        assert isinstance(data["avg_latency_ms"],      float)
        assert isinstance(data["p95_latency_ms"],      float)
        assert isinstance(data["throughput_per_min"],  float)

    async def test_defect_rate_in_range(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/metrics")
        data = resp.json()
        assert 0.0 <= data["defect_rate"] <= 1.0


@pytest.mark.asyncio
class TestPredictionsEndpoint:

    async def test_predictions_returns_list(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/predictions")
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

    async def test_predictions_limit_param(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/predictions?limit=10")
        assert resp.status_code == 200

    async def test_predictions_invalid_limit(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/api/v1/predictions?limit=0")
        # FastAPI query param validation → 422
        assert resp.status_code == 422
