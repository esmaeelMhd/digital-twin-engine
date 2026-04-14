# Digital Twin Engine – multi-stage Docker build
#
# Stage 1 (builder): installs all dependencies into a virtual environment.
# Stage 2 (api):     lean inference image with only the runtime dependencies.
# Stage 3 (train):   full training image (includes CUDA + JAX GPU extras).
#
# Build the lean API image:
#   docker build --target api -t dte-api:latest .
#
# Build the training image:
#   docker build --target train -t dte-train:latest .

# ============================================================
# Stage 1: base + dependency installer
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (cache-friendly)
COPY pyproject.toml ./
COPY dte/__init__.py dte/__init__.py

# Install runtime deps into a virtualenv
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install CPU-only JAX for the API image (no CUDA)
RUN pip install --no-cache-dir "jax[cpu]>=0.4.20"

# Install remaining deps from pyproject.toml (excluding GPU-specific ones)
RUN pip install --no-cache-dir \
    equinox>=0.11.0 \
    diffrax>=0.4.0 \
    optax>=0.1.7 \
    jaxtyping>=0.2.20 \
    pyyaml>=6.0 \
    h5py>=3.9.0 \
    numpy>=1.24.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.66.0 \
    fastapi>=0.111.0 \
    uvicorn[standard]>=0.29.0 \
    pydantic>=2.0.0 \
    httpx>=0.27.0 \
    pandas>=2.0.0

# ============================================================
# Stage 2: lean API inference image
# ============================================================
FROM python:3.12-slim AS api

WORKDIR /app

# Copy virtualenv from builder
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy source
COPY dte/ ./dte/
COPY configs/ ./configs/
COPY scripts/ingest_real_data.py ./scripts/

# Environment defaults (override at runtime)
ENV DTE_SYSTEM_CONFIG="configs/cstr_default.yaml"
ENV DTE_MODEL_PATH="outputs/best_model.eqx"
ENV DTE_TRAINING_CONFIG="configs/training_default.yaml"
# Set DTE_API_KEY to enable API key authentication
# ENV DTE_API_KEY="changeme"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["uvicorn", "dte.api.service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================================
# Stage 3: full training image (CPU, add GPU support via CUDA base)
# ============================================================
FROM python:3.12-slim AS train

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Install additional training-only dependencies
RUN pip install --no-cache-dir \
    streamlit>=1.28.0 \
    plotly>=5.17.0 \
    scikit-learn>=1.3.0

# Copy full source
COPY . .

# Default entrypoint: show help
CMD ["python", "scripts/train.py", "--help"]

# ============================================================
# Stage 4: frontend build
# ============================================================
FROM node:24-alpine AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

ARG VITE_API_BASE_URL=http://localhost:8000
ARG VITE_DTE_API_KEY=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
ENV VITE_DTE_API_KEY=${VITE_DTE_API_KEY}

RUN npm run build

# ============================================================
# Stage 5: static frontend runtime
# ============================================================
FROM nginx:1.29-alpine AS frontend

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=frontend-builder /frontend/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
