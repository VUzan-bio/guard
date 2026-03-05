# GUARD Platform — Slim Docker build for Railway (4GB image limit)
# Single container: FastAPI serves both API and built frontend

# Stage 1: Build frontend
FROM node:20-alpine AS frontend
WORKDIR /build
COPY guard-ui/package*.json ./
RUN npm ci
COPY guard-ui/ ./
RUN npm run build

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app

# Bowtie2 for off-target screening + minimal build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libffi-dev bowtie2 && \
    rm -rf /var/lib/apt/lists/*

# Install only the deps needed for deployment (no torch — heuristic fallback)
COPY pyproject.toml README.md ./
COPY guard/ ./guard/
RUN pip install --no-cache-dir -e ".[primers,api]" && \
    pip cache purge 2>/dev/null; true

# Remove build deps to save space
RUN apt-get purge -y gcc libffi-dev && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# API + configs + data
COPY api/ ./api/
COPY configs/ ./configs/
COPY data/ ./data/

# Build Bowtie2 index from reference FASTA
RUN bowtie2-build data/references/H37Rv.fasta data/references/H37Rv

# Frontend
COPY --from=frontend /build/dist ./guard-ui/dist

RUN mkdir -p results/api results/panels results/validation

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
