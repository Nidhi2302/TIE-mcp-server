# Multi-stage Docker build for TIE MCP Server
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Install Poetry
RUN pip install poetry==1.6.1

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Copy poetry files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --only=main && rm -rf $POETRY_CACHE_DIR

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Copy virtual environment from base stage
COPY --from=base /app/.venv /app/.venv

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY notebooks/ ./notebooks/

# Create necessary directories
RUN mkdir -p logs data/models data/datasets data/configs

# Set ownership
RUN chown -R app:app /app

# Switch to app user
USER app

# Set PATH to include virtual environment
ENV PATH="/app/.venv/bin:$PATH"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "tie_mcp.server"]

# Development stage
FROM base as development

# Install development dependencies
RUN poetry install && rm -rf $POETRY_CACHE_DIR

# Install Jupyter and development tools
RUN poetry run pip install jupyter jupyterlab ipywidgets

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs data/models data/datasets data/configs

# Set ownership
RUN chown -R app:app /app

# Switch to app user
USER app

# Set PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose ports (app + jupyter)
EXPOSE 8000 8888

# Default command for development
CMD ["poetry", "run", "python", "-m", "tie_mcp.server"]