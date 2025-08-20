FROM python:3.13-slim AS build

# Environment configuration
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.8.3 \
    POETRY_NO_INTERACTION=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

# Copy only dependency metadata + README (required by poetry build) for better layer caching
COPY pyproject.toml poetry.lock README.md ./

# Install only main (runtime) dependencies (no dev) without installing the package itself yet
RUN poetry install --only main --no-root

# Now copy full source tree (ensures package discovery works; previous layout copied only subdir)
COPY src ./src

# Build wheel (produces dist/*.whl)
RUN poetry build -f wheel

# -------- Runtime image --------
FROM python:3.13-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

WORKDIR /app

# Create non-root user
RUN useradd -r -u 1001 tie && chown tie:tie /app

# Copy built wheel from build stage and install
COPY --from=build /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/tie_mcp_server-*.whl && rm /tmp/*.whl

# (Optional) Copy source tree for better stack traces / introspection (remove to slim image)
COPY --from=build /app/src ./src

USER tie
EXPOSE 8000

# Entrypoint provided by poetry-installed console script
ENTRYPOINT ["tie-mcp-server"]