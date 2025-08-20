FROM python:3.13-slim AS build

# Environment configuration
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy only the package directory explicitly (resolves previous 'does not contain any element' issue)
# Using absolute target path per request
COPY src/tie_mcp /app/src/tie_mcp

# Copy dependency metadata + README first (layer caching + build requirements)
COPY pyproject.toml poetry.lock README.md ./

WORKDIR /app

# System dependencies (minimal set for building scientific deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry (specified version)
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# (Optional) Normalize lock to current Poetry version to avoid compatibility warning
# (Will be a no-op if already compatible)
RUN poetry lock --no-update

# Install only main runtime dependencies (faster; no dev)
RUN poetry install --only main --no-root

# Build wheel (produces dist/*.whl)
RUN poetry build -f wheel

# -------- Runtime image --------
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user (suppress uid range warning by choosing uid < 1000)
RUN useradd -r -u 900 tie && chown tie:tie /app

# Copy built wheel from build stage and install (only runtime artifacts)
COPY --from=build /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-warn-script-location /tmp/tie_mcp_server-*.whl && rm /tmp/*.whl

# (Optional) Copy source package only for better stack traces (omit tests/examples)
COPY --from=build src/tie_mcp /app/src/tie_mcp

USER tie
EXPOSE 8000

# Entrypoint provided by poetry-installed console script
ENTRYPOINT ["tie-mcp-server"]