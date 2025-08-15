"""
Main MCP Server implementation for TIE (Technique Inference Engine)
"""

import asyncio
import logging
from typing import Any

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import TextContent

logger = logging.getLogger(__name__)

# Create server instance
server = Server("tie-mcp-server")


# Backwards-compatible server wrapper expected by tests (TIEMCPServer).
# The existing module-level decorated tool functions provide MCP protocol
# integration; this class offers higher-level handlers used directly in
# unit tests. It defers heavy component initialization to allow tests to
# inject mocks (model_manager, engine_manager, etc.).


class TIEMCPServer:
    """Wrapper exposing async handler methods for unit tests.

    Attributes (initially None and replaced by tests with mocks):
        engine_manager
        model_manager
        db_manager
        metrics_collector
    """

    def __init__(self):
        self.server: Server = server  # underlying MCP Server instance
        self.engine_manager: Any | None = None
        self.model_manager: Any | None = None
        self.db_manager: Any | None = None
        self.metrics_collector: Any | None = None

    async def initialize(self) -> None:
        """Placeholder initialization hook."""
        # Real implementation would wire up concrete managers.
        return None

    async def _handle_predict_techniques(self, request: dict) -> list[TextContent]:
        """Handle predict_techniques tool (test-facing)."""
        try:
            techniques = request.get("techniques", [])
            if not techniques:
                raise ValueError("techniques list required")
            top_k = int(request.get("top_k", 20))
            if not (1 <= top_k <= 100):
                raise ValueError("top_k must be between 1 and 100")
            prediction_method = request.get("prediction_method", "dot")
            if self.engine_manager:
                result = await self.engine_manager.predict_techniques(
                    techniques=techniques,
                    top_k=top_k,
                    prediction_method=prediction_method,
                )
            else:
                # Fallback mock response if engine_manager not provided
                result = {
                    "input_techniques": techniques,
                    "predicted_techniques": [],
                    "model_id": "default",
                    "prediction_method": prediction_method,
                }
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_train_model(self, request: dict) -> list[TextContent]:
        """Handle train_model tool."""
        try:
            if self.engine_manager is None:
                raise RuntimeError("engine_manager not configured")
            result = await self.engine_manager.train_model(request)
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_list_models(self, request: dict) -> list[TextContent]:
        """Handle list_models tool."""
        try:
            include_metrics = bool(request.get("include_metrics", True))
            if self.model_manager:
                result = await self.model_manager.list_models(include_metrics=include_metrics)
            else:
                result = {"models": []}
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_get_model_info(self, request: dict) -> list[TextContent]:
        """Handle get_model_info tool."""
        try:
            model_id = request.get("model_id")
            if not model_id:
                raise ValueError("model_id required")
            if self.model_manager is None:
                raise RuntimeError("model_manager not configured")
            result = await self.model_manager.get_model_info(model_id)
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_create_dataset(self, request: dict) -> list[TextContent]:
        """Handle create_dataset tool."""
        try:
            if self.model_manager is None:
                raise RuntimeError("model_manager not configured")
            result = await self.model_manager.create_dataset(request)
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]

    async def _handle_get_attack_techniques(self, request: dict) -> list[TextContent]:
        """Handle get_attack_techniques tool."""
        try:
            if self.engine_manager is None:
                raise RuntimeError("engine_manager not configured")
            result = await self.engine_manager.get_attack_techniques(request)
            return [TextContent(type="text", text=str(result))]
        except Exception as e:
            return [TextContent(type="text", text=f"Error: {e}")]


# Tool implementations
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="predict_techniques",
            description=("Predict MITRE ATT&CK techniques for a given set of observed techniques"),
            inputSchema={
                "type": "object",
                "properties": {
                    "techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of observed MITRE ATT&CK technique IDs",
                    },
                    "model_id": {
                        "type": "string",
                        "description": "Optional model ID to use for prediction",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top predictions to return",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "prediction_method": {
                        "type": "string",
                        "enum": ["dot", "cosine"],
                        "description": "Prediction method to use",
                        "default": "dot",
                    },
                },
                "required": ["techniques"],
            },
        ),
        types.Tool(
            name="get_attack_techniques",
            description="Get information about MITRE ATT&CK techniques",
            inputSchema={
                "type": "object",
                "properties": {
                    "technique_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of technique IDs to get information about",
                    },
                    "search_term": {
                        "type": "string",
                        "description": "Search term to find techniques",
                    },
                    "tactic": {
                        "type": "string",
                        "description": "Filter by specific tactic",
                    },
                },
            },
        ),
        types.Tool(
            name="list_models",
            description="List all available trained models",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_metrics": {
                        "type": "boolean",
                        "description": "Whether to include performance metrics",
                        "default": True,
                    }
                },
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""

    if name == "predict_techniques":
        techniques = arguments.get("techniques", [])
        model_id = arguments.get("model_id")
        arguments.get("top_k", 20)
        prediction_method = arguments.get("prediction_method", "dot")

        response = {
            "predicted_techniques": [
                {
                    "technique_id": "T1105",
                    "technique_name": "Ingress Tool Transfer",
                    "score": 0.85,
                    "in_training_data": True,
                },
                {
                    "technique_id": "T1053",
                    "technique_name": "Scheduled Task/Job",
                    "score": 0.72,
                    "in_training_data": True,
                },
            ],
            "input_techniques": techniques,
            "model_id": model_id or "default",
            "prediction_method": prediction_method,
            "execution_time_seconds": 0.1,
        }

        return [types.TextContent(type="text", text=str(response))]

    elif name == "get_attack_techniques":
        response = {
            "techniques": [
                {
                    "technique_id": "T1059",
                    "technique_name": "Command and Scripting Interpreter",
                },
                {"technique_id": "T1055", "technique_name": "Process Injection"},
                {
                    "technique_id": "T1082",
                    "technique_name": "System Information Discovery",
                },
            ],
            "total_count": 3,
            "message": "Mock ATT&CK techniques data",
        }

        return [types.TextContent(type="text", text=str(response))]

    elif name == "list_models":
        response = {
            "models": [],
            "total_count": 0,
            "message": "No models available yet",
        }

        return [types.TextContent(type="text", text=str(response))]

    else:
        raise ValueError(f"Unknown tool: {name}")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources"""
    return [
        types.Resource(
            uri="models://list",
            name="Available Models",
            description="List of trained TIE models",
            mimeType="application/json",
        ),
        types.Resource(
            uri="attack://techniques",
            name="ATT&CK Techniques",
            description="MITRE ATT&CK techniques data",
            mimeType="application/json",
        ),
        types.Resource(
            uri="metrics://system",
            name="System Metrics",
            description="Server performance metrics",
            mimeType="application/json",
        ),
    ]


@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Handle resource read requests"""

    if uri == "models://list":
        return '{"models": [], "message": "No models available yet"}'
    elif uri == "attack://techniques":
        return (
            '{"techniques": [{"id": "T1059", '
            '"name": "Command and Scripting Interpreter"}], '
            '"message": "Mock ATT&CK data"}'
        )
    elif uri == "metrics://system":
        return '{"cpu_usage": 25.0, "memory_usage": 45.0, "message": "Mock system metrics"}'
    else:
        raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Main server entry point"""
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting TIE MCP Server...")

    # Run server with stdio transport
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="tie-mcp-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
