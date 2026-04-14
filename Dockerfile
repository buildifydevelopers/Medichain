# ── STAGE 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

# Install system dependencies for building C extensions
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

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install CPU-only PyTorch first to prevent CUDA bloat
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image (Fixed multi-line syntax)
RUN python3 -c "\
from facenet_pytorch import InceptionResnetV1, MTCNN; \
import torch; \
print('Downloading MTCNN weights...'); \
MTCNN(device='cpu'); \
print('Downloading FaceNet (VGGFace2) weights...'); \
InceptionResnetV1(pretrained='vggface2').eval(); \
print('Weights baked successfully.'); \
"

# ── STAGE 2: Runtime ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

# Runtime system libs (specifically libgl1 for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a non-root user immediately
RUN useradd -m -u 1000 medichain

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy model weights to the NEW user's home directory
COPY --from=builder /root/.cache/torch /home/medichain/.cache/torch

# Copy application source code
COPY . .

# Fix permissions for the app directory and the model cache
RUN chown -R medichain:medichain /app /home/medichain/.cache/torch

USER medichain

# Environment Variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Force Torch to look in the medichain user's home
ENV TORCH_HOME=/home/medichain/.cache/torch

EXPOSE $PORT

# Healthcheck to verify the API is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:$PORT/health')" || exit 1

# Run with a single worker to save RAM on Railway's free tier
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
