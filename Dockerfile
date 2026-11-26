FROM python:3.11.17-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --no-cache-dir uv==0.13.1

WORKDIR /app
COPY .python-version pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv sync --locked --no-dev --no-cache

ENV PFS_DATA_DIR=/data/source \
    PFS_STATE_DIR=/data/state \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "pfs.service:create_service", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
