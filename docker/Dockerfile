# Unified multi-stage Dockerfile (merged root + previous docker version)
# - Python 3.13 (align with root improvements)
# - Wheel build in dedicated build stage (smaller production image)
# - Explicit src/tie_mcp copy (earlier empty dir safeguard)
# - Lock normalization (avoids Poetry version warning)
# - Healthcheck retained
# - Development stage preserved (with Jupyter) but upgraded

############################
# Build stage (produce wheel)
############################
FROM python:3.13-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1

# Workdir early
WORKDIR /app

# System build dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata + README (needed for wheel long_description)
COPY pyproject.toml poetry.lock README.md ./

# Install Poetry (official installer for requested version)
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Normalize / verify lock without updating versions (no-op if already current)
RUN poetry lock --no-update

# Install only main (runtime) dependencies (faster; no dev)
RUN poetry install --only main --no-root

# Copy ONLY the package (explicit path to avoid stray files & retain cache above)
COPY src/tie_mcp /app/src/tie_mcp

# Fail fast if package is empty (safeguard from earlier incident)
RUN [ "$(ls -A /app/src/tie_mcp)" ] || (echo 'src/tie_mcp is empty!' && exit 1)

# Build wheel (ends up in dist/)
RUN poetry build -f wheel

############################
# Production stage (minimal runtime)
############################
FROM python:3.13-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime OS deps only (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (fixed uid to avoid warning)
RUN useradd -r -u 900 tie && chown tie:tie /app

# Copy and install wheel only (no build system baggage)
COPY --from=build /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir --no-warn-script-location /tmp/tie_mcp_server-*.whl && rm /tmp/*.whl

# (Optional) copy source for richer tracebacks (still small; excludes tests by design)
COPY --from=build /app/src/tie_mcp /app/src/tie_mcp

USER tie

EXPOSE 8000

# Healthcheck (server expected to expose /health)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Console entrypoint installed by wheel
ENTRYPOINT ["tie-mcp-server"]

############################
# Development stage
############################
FROM python:3.13-slim AS development

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

# Dev system packages (git + build toolchain + curl)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata + README
COPY pyproject.toml poetry.lock README.md ./

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Full install incl. dev (tests, linters, etc.)
RUN poetry install

# Jupyter & optional interactive tooling
RUN poetry run pip install jupyter jupyterlab ipywidgets

# Copy rest of repo (dev needs tests, configs, etc.)
COPY . .

# Create dev user
RUN useradd -r -u 901 dev && chown -R dev:dev /app
USER dev

# Expose app + jupyter
EXPOSE 8000 8888

# Default dev command
CMD ["poetry", "run", "python", "-m", "tie_mcp.server"]