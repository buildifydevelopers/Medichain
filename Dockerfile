# ────────────────────────────────────────────────────────────────
# MediChain Face Verification API — Railway-optimised Dockerfile
#
# Strategy: python:3.11-slim + CPU-only torch + baked model weights
# Final image: ~700 MB  (was ~6 GB with CUDA torch + full base)
#
# Two-stage build:
#   Stage 1 (builder): install all packages, download model weights
#   Stage 2 (runtime): copy only what's needed — no build tools
# ────────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

# Install only what's needed to compile C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy requirements first (layer caching — only re-runs when requirements change)
COPY requirements.txt .

# ── KEY FIX: Install CPU-only PyTorch from the dedicated CPU index ──
# This fetches torch+cpu (~178 MB) instead of torch+cu121 (~2.4 GB)
# Must be done BEFORE installing other packages so pip doesn't
# pull the CUDA version as a transitive dependency.
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

# Install remaining packages from PyPI (torch is already satisfied above)
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download FaceNet weights into the image ─────────────────
# This bakes the 107 MB VGGFace2 weights into the image layer.
# Without this, Railway startup takes 30-60 sec on first request
# while the model downloads. With this, startup is instant.
RUN python3 -c "
from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
# Downloads and caches to ~/.cache/torch/checkpoints/
print('Downloading MTCNN weights...')
MTCNN(device='cpu')
print('Downloading FaceNet (VGGFace2) weights...')
InceptionResnetV1(pretrained='vggface2').eval()
print('Weights baked into image successfully.')
"

# ── Stage 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Runtime system libs only — no compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded model weights from builder's cache
COPY --from=builder /root/.cache/torch /root/.cache/torch

# Copy application source
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 medichain && chown -R medichain:medichain /app /root/.cache/torch
USER medichain

# Railway injects PORT env var
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE $PORT

# Healthcheck so Railway knows when the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health')" || exit 1

CMD uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 75
