# =============================================================================
# CodKing Inference API - Dockerfile
# HRM+H-Net Hybrid Model for Cybersecurity Data Processing
# =============================================================================
#
# NOTE: PyTorch is installed with CPU-only wheels to keep the image size
# manageable (~1.5 GB vs ~6 GB with full CUDA). For GPU inference, build
# with: docker build --build-arg TORCH_INDEX=cu126 .
#
# torchvision and torchaudio are intentionally excluded from the Docker
# build. No codking source code imports them; they were only pulled in
# transitively by torchmetrics (a pytorch-lightning dep). Installing
# torchvision from PyPI while torch comes from the CPU-only index causes
# a circular-import crash (torchvision.extension AttributeError).
# Without torchvision installed, torchmetrics gracefully degrades
# (its _TORCHVISION_AVAILABLE guard prevents the import).
# =============================================================================

ARG TORCH_INDEX=cpu

# ---------------------------------------------------------------------------
# Stage 1: Builder - install dependencies in a throwaway layer
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ARG TORCH_INDEX

WORKDIR /build

# Install build-time system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch first with the appropriate index for CPU-only builds,
# then install the rest of the dependencies.
# flash-attn and mamba-ssm are excluded (GPU-only, require CUDA at build).
# numpy is also excluded here and installed at the end to avoid version
# conflicts between torch (needs numpy 2.x) and stale pins.
RUN pip install --no-cache-dir --prefix=/install \
    torch==2.7.0 \
    --extra-index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
    && pip install --no-cache-dir --prefix=/install \
    $(sed 's/#.*$//' requirements.txt | grep -v -E '^\s*$' | grep -v -E '^(torch==|torchvision==|torchaudio==|flash-attn|mamba-ssm|numpy)')

# Ensure numpy is at a version compatible with torch and scipy.
# torch 2.7.0 ships against numpy 2.x; downgrading breaks multiarray.
RUN pip install --no-cache-dir --prefix=/install --upgrade numpy

# ---------------------------------------------------------------------------
# Stage 2: Runtime - lean production image
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Install runtime-only system dependencies
#   curl  - health check probe
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /srv

# Copy application source into a proper 'codking' package directory
# so relative imports (from ..training) resolve correctly
COPY models/ ./codking/models/
COPY training/ ./codking/training/
COPY utils/ ./codking/utils/
COPY api/ ./codking/api/
COPY configs/ ./codking/configs/
COPY integrations/ ./codking/integrations/
COPY __init__.py ./codking/

# Create directories for checkpoints and data
RUN mkdir -p codking/checkpoints codking/data

# Create non-root user
RUN useradd -r -s /bin/false appuser \
    && chown -R appuser:appuser /srv

USER appuser

# Avoid matplotlib permission errors for non-root user
ENV MPLCONFIGDIR=/tmp/matplotlib

# CodKing inference server listens on port 8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run from /srv so 'codking' is a top-level package
CMD ["python", "-m", "uvicorn", "codking.api.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
