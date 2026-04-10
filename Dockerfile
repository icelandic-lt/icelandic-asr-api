# CUDA 12.4 matches Ubuntu 24.04 LTS default nvidia-driver-550
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create venv with access to system site-packages (pre-installed PyTorch)
RUN uv venv --python $(which python) --system-site-packages /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies first (cached unless pyproject.toml changes)
COPY pyproject.toml .
RUN uv pip install --no-cache \
        fastapi "uvicorn[standard]" transformers faster-whisper \
        omnivad huggingface_hub numpy

# Copy source code and install package (no-deps, deps already installed)
COPY .flake8 .
COPY icelandic_asr/ icelandic_asr/
RUN uv pip install --no-cache --no-deps .

# Models are downloaded on first start and cached here
ENV HF_HOME=/app/models
VOLUME /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

ENTRYPOINT ["icelandic-asr"]
CMD ["--host", "0.0.0.0", "--port", "8000"]