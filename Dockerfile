# GUARD Platform — Multi-stage Docker build (optimized for Railway 4GB limit)
# Single container: FastAPI serves both API and built frontend

# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /build
COPY guard-ui/package*.json ./
RUN npm ci
COPY guard-ui/ ./
RUN npm run build

# Stage 2: Build Python deps with compilers
FROM python:3.11-slim AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY guard/ ./guard/

# CPU-only PyTorch (~200MB vs ~2.5GB with CUDA)
RUN pip install --no-cache-dir --target=/deps torch --index-url https://download.pytorch.org/whl/cpu
# All other deps
RUN pip install --no-cache-dir --target=/deps -e ".[all]"

# Stage 3: Lean runtime (no compilers)
FROM python:3.11-slim AS runtime
WORKDIR /app

# Bowtie2 for off-target screening (M4)
RUN apt-get update && apt-get install -y --no-install-recommends \
    bowtie2 && \
    rm -rf /var/lib/apt/lists/*

# Python packages from builder
COPY --from=builder /deps /usr/local/lib/python3.11/site-packages

# Application code
COPY pyproject.toml README.md ./
COPY guard/ ./guard/
COPY api/ ./api/
COPY configs/ ./configs/
COPY data/ ./data/

# Editable install (just .egg-link, no downloads)
RUN pip install --no-cache-dir --no-deps -e .

# Build Bowtie2 index from reference FASTA
RUN bowtie2-build data/references/H37Rv.fasta data/references/H37Rv

# Frontend
COPY --from=frontend /build/dist ./guard-ui/dist

RUN mkdir -p results/api results/panels results/validation

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
