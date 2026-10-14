# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.13.0 AS uv
FROM python:3.11-slim-bookworm AS runtime
COPY --from=uv /uv /uvx /usr/local/bin/
# MuseScore 3 provides native .mscz export on both amd64 and arm64.
# Keep Debian copyright notices and corresponding-source package references.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git ca-certificates ffmpeg libsndfile1 musescore3 \
    fonts-dejavu-core libgl1 libegl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# cpu, or cu126 for NVIDIA GPUs (driver >= 525; CUDA libraries ship in the wheels).
ARG TORCH_VARIANT=cpu
# Plain copies: reflink/clone from the uv cache fails on some Docker storage drivers.
ENV UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --extra ${TORCH_VARIANT} --python /usr/local/bin/python \
    && uv cache clean \
    && useradd --create-home --uid 10001 brass \
    && mkdir -p /data /models \
    && chown -R brass:brass /data /models /app
ENV PATH="/app/.venv/bin:$PATH" \
    BRASS_HOST=0.0.0.0 BRASS_DATA=/data BRASS_CACHE=/models \
    QT_QPA_PLATFORM=offscreen MUSESCORE_BIN=/usr/bin/musescore3 MUSESCORE_MAJOR=3 \
    MPLBACKEND=Agg PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
USER brass
RUN python -m brass_transcriber.container_check
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)" || exit 1
CMD ["brass-web"]

# Optional self-contained image; explicitly downloads ~1.67 GB, plus prepared PS copy.
FROM runtime AS bundled
RUN brass-transcribe download-model --cache /models --variant nops \
    && brass-transcribe download-model --cache /models --variant ps \
    && brass-transcribe download-model --cache /models --variant beats \
    && brass-transcribe download-model --cache /models --variant separation \
    && python -c "from pathlib import Path; from brass_transcriber.backend import prepare_ps_checkpoint; prepare_ps_checkpoint(Path('/models'))"

FROM runtime AS app
