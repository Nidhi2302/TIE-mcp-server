#!/usr/bin/env python3
"""
Comprehensive test suite for TIE MCP Server
"""

import json

import pytest
from fastmcp.testing import create_test_client


class TestTIEMCPServer:
    """Test cases for TIE MCP Server"""

    @pytest.fixture
    async def client(self):
        """Fixture to create a test client"""
        from src.tie_mcp.server import TIEMCPServer

        server = TIEMCPServer()
        await server.initialize()

        async with create_test_client(server.server) as client:
            yield client

        await server.cleanup()

    async def test_server_tools(self, client):
        """Test that the server provides the expected tools"""
        tools = await client.list_tools()
        assert len(tools) > 0, "No tools available"

        tool_names = {t.name for t in tools}
        assert "predict_techniques" in tool_names
        assert "train_model" in tool_names
        assert "list_models" in tool_names
        assert "get_attack_techniques" in tool_names

    async def test_server_resources(self, client):
        """Test that the server provides the expected resources"""
        resources = await client.list_resources()
        assert len(resources) > 0, "No resources available"

        resource_uris = {r.uri for r in resources}
        assert "models://" in resource_uris
        assert "datasets://" in resource_uris
        assert "attack://techniques" in resource_uris
        assert "metrics://system" in resource_uris

    async def test_technique_prediction(self, client):
        """Test the technique prediction functionality"""
        result = await client.call_tool(
            "predict_techniques",
            {
                "techniques": ["T1059", "T1055"]
            }
        )
        assert len(result) > 0

        response_data = json.loads(result[0].text)
        assert "input_techniques" in response_data
        assert "predicted_techniques" in response_data
        assert len(response_data["predicted_techniques"]) > 0

        prediction = response_data["predicted_techniques"][0]
        assert "technique_id" in prediction
        assert "technique_name" in prediction
        assert "score" in prediction

    async def test_resource_access(self, client):
        """Test reading server resources"""
        # Test models resource
        models_data = await client.read_resource("models://")
        assert models_data is not None
        models_json = json.loads(models_data)
        assert "models" in models_json

        # Test attack data resource
        attack_data = await client.read_resource("attack://techniques")
        assert attack_data is not None
        attack_json = json.loads(attack_data)
        assert "techniques" in attack_json

        # Test metrics resource
        metrics_data = await client.read_resource("metrics://system")
        assert metrics_data is not None
        metrics_json = json.loads(metrics_data)
        assert "cpu_usage" in metrics_json
        assert "memory_usage" in metrics_json
