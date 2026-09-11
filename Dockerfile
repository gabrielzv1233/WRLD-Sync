FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python --no-cache -r requirements.txt


FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WRLD_SYNC_STATE_DIR=/app/data \
    XDG_CACHE_HOME=/app/cache \
    HF_HOME=/app/cache/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/cache/huggingface/hub \
    TORCH_HOME=/app/cache/torch

WORKDIR /app

# FFmpeg is required for audio decoding; libsndfile supports common audio I/O.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 appuser
COPY --chown=appuser:appuser . .
RUN mkdir -p data models cache \
    && chown appuser:appuser /app \
    && chown -R appuser:appuser data models cache

USER appuser

RUN python -m compileall -q /app

VOLUME ["/app/data", "/app/models", "/app/cache"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4)" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--log-level", "warning"]
