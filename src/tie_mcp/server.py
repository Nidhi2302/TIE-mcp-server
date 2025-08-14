"""
Main MCP Server implementation for TIE (Technique Inference Engine)
"""

import asyncio
import logging

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

logger = logging.getLogger(__name__)

# Create server instance
server = Server("tie-mcp-server")


# Tool implementations
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="predict_techniques",
            description=(
                "Predict MITRE ATT&CK techniques for a given set of observed techniques"
            ),
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
        return (
            '{"cpu_usage": 25.0, "memory_usage": 45.0, '
            '"message": "Mock system metrics"}'
        )
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
