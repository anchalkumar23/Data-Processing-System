# ============================================================
# Dockerfile — Automated Visual Data Processing System
#
# Multi-stage build:
#   Stage 1 (builder): Installs Python dependencies
#   Stage 2 (runtime): Copies only what's needed to run the API
#
# The multi-stage approach keeps the final image lean:
# we don't ship build tools, compilers, or test dependencies
# into the production container.
#
# Build:
#   docker build -t defect-detection:latest .
#
# Run (CPU):
#   docker run -p 8000:8000 defect-detection:latest
#
# Run (GPU):
#   docker run --gpus all -p 8000:8000 defect-detection:latest
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

# Install system dependencies needed to build OpenCV and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only the requirements first — Docker layer cache means
# pip install only reruns when requirements.txt changes
COPY requirements.txt .

# Install to a dedicated prefix so we can copy just the packages later
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Anchal"
LABEL description="Automated Visual Data Processing System — Defect Detection API"
LABEL version="1.0.0"

# Runtime system libraries (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Create a non-root user for security best practices
RUN groupadd --gid 1001 appgroup \
 && useradd  --uid 1001 --gid 1001 --no-create-home appuser

WORKDIR /app

# Copy application source
COPY api/          api/
COPY inference/    inference/
COPY preprocessing/ preprocessing/
COPY dashboard/    dashboard/
COPY configs/      configs/
COPY checkpoints/  checkpoints/

# Create directories the app writes to at runtime
RUN mkdir -p logs data && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose the API port
EXPOSE 8000

# Health check — Docker will mark the container as unhealthy if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

# Start the API server
# --workers 1 because ONNX Runtime sessions are not fork-safe
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
