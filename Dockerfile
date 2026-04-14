# ── STAGE 1: Builder ─────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

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

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu

RUN pip install --no-cache-dir -r requirements.txt

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 medichain

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /root/.cache/torch /home/medichain/.cache/torch

COPY . .

RUN chown -R medichain:medichain /app /home/medichain/.cache/torch

USER medichain

# Railway will override this PORT variable automatically
ENV PORT=8000
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/home/medichain/.cache/torch
ENV PYTHONPATH=/app

EXPOSE $PORT

# Healthcheck updated to use the dynamic PORT variable
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request, os; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://localhost:{port}/health')" || exit 1

# FINAL FIX: Use the environment variable for the port
# The shell form (no brackets) allows variable substitution
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --workers 1
