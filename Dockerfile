# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy the dependency definition AND the lock: the image installs the locked set.
COPY pyproject.toml uv.lock ./

# Install exactly what uv.lock pins -- the same set CI tests with (`uv sync`). The image used
# to re-resolve pyproject.toml's ranges at every build, so a dependency released between two
# deploys reached production without ever passing through CI. On 2026-09-25 that was
# SQLAlchemy 2.1.0 (lock: 2.0.52), whose resolution dropped greenlet; the async engine then
# raised at import and the API crash-looped behind nginx 502s. `--frozen` refuses to touch
# the lock, so a build can never drift from it.
RUN uv export --frozen --no-dev --no-hashes -o requirements.txt && \
    uv pip install --system --no-cache -r requirements.txt

# Final runtime image
FROM python:3.12-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
