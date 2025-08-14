# Getting Started

This guide will help you set up and run the TIE MCP Server.

## Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)
- Git

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nidhi2302/TIE-mcp-server.git
   cd TIE-mcp-server
   ```

2. **Install dependencies with Poetry:**
   ```bash
   poetry install
   ```

3. **Set up environment variables:**
   ```bash
   cp mcp_config.example.json mcp_config.json
   # Edit mcp_config.json with your configuration
   ```

## Configuration

The server can be configured using environment variables or a configuration file:

- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis connection string (optional)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Running the Server

### Development Mode

```bash
poetry run tie-mcp-server
```

### Production Mode

```bash
poetry run tie-mcp-server --environment production
```

### Using Docker

```bash
docker-compose up -d
```

## First Steps

1. **Test the connection:**
   ```bash
   # The server will be available for MCP clients to connect to
   ```

2. **Load sample data:**
   ```bash
   poetry run python -m tie_mcp.cli load-sample-data
   ```

3. **Train a model:**
   ```bash
   poetry run python -m tie_mcp.cli train-model --dataset sample_data.json
   ```

## Next Steps

- See the [API Reference](api-reference.md) for detailed API documentation
- Check out [Examples](examples.md) for usage examples
- Read about [Contributing](CONTRIBUTING.md) to the project