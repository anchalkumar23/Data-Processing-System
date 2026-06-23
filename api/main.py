"""
api/main.py
────────────
FastAPI application entry point.

Responsibilities:
  - Creates the FastAPI app with metadata (title, version, docs URL)
  - Manages the inference engine lifecycle (startup / shutdown)
  - Mounts the static dashboard files
  - Wires in the API router
  - Configures CORS so the dashboard JS can call the API

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

In Docker:
    CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import router, set_engine
from inference.engine import InferenceEngine
from inference.warmup import warmup

log = logging.getLogger(__name__)

CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/config.yaml")


# ── App lifespan ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs setup code before the server starts accepting requests, and
    teardown code after it stops.

    FastAPI's lifespan context manager is the modern replacement for
    @app.on_event("startup") / @app.on_event("shutdown").
    """
    # ── Startup ──────────────────────────────────────────────
    log.info("Starting up defect detection API...")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    onnx_path = config["paths"]["onnx_model"]

    try:
        engine = InferenceEngine(
            onnx_path   = onnx_path,
            config_path = CONFIG_PATH,
        )
        warmup(engine, num_runs=5, img_size=config["data_gen"]["image_size"][0])
        set_engine(engine)
        log.info("Inference engine ready — API accepting requests")
    except FileNotFoundError as e:
        # Model not exported yet — server still starts but /predict returns 503
        log.warning(f"ONNX model not found: {e}")
        log.warning("Run `python -m training.export_onnx` to generate the model.")

    yield  # ← server is running here

    # ── Shutdown ─────────────────────────────────────────────
    log.info("Shutting down API...")
    # ONNX Runtime sessions clean up automatically; no explicit close needed


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""

    app = FastAPI(
        title       = "Defect Detection API",
        description = (
            "Real-time industrial surface defect classification using a ResNet-18 "
            "CNN trained on synthetic data and exported to ONNX for production serving."
        ),
        version     = "1.0.0",
        docs_url    = "/docs",
        redoc_url   = "/redoc",
        lifespan    = lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────
    # Allow the dashboard (served from the same origin) and local dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["*"],   # tighten this in production
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── API routes ────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1", tags=["inference"])

    # ── Static dashboard ─────────────────────────────────────
    static_dir = Path("dashboard/static")
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_dashboard():
            """Serve the monitoring dashboard at the root URL."""
            return FileResponse(str(static_dir / "index.html"))
    else:
        log.warning("Dashboard static files not found at dashboard/static/")

    return app


app = create_app()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    api_cfg = cfg["api"]
    uvicorn.run(
        "api.main:app",
        host    = api_cfg["host"],
        port    = api_cfg["port"],
        reload  = api_cfg.get("reload", False),
        workers = api_cfg.get("workers", 1),
        log_level = "info",
    )
