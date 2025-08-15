#!/usr/bin/env python3
"""
Simplified test suite for TIE MCP Server
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tie_mcp.server import TIEMCPServer


@pytest.mark.unit
@pytest.mark.asyncio
class TestTIEMCPServer:
    """Test cases for TIE MCP Server"""

    async def test_server_initialization(self):
        """Test basic server initialization"""
        server = TIEMCPServer()
        assert server is not None
        assert hasattr(server, "server")
        assert hasattr(server, "engine_manager")
        assert hasattr(server, "model_manager")

    @patch("tie_mcp.server.TIEMCPServer.initialize")
    async def test_server_initialize(self, mock_initialize):
        """Test server initialization process"""
        server = TIEMCPServer()
        mock_initialize.return_value = None

        # Test that initialize can be called
        await server.initialize()
        mock_initialize.assert_called_once()

    async def test_prediction_handler_structure(self):
        """Test that prediction handler exists and has correct structure"""
        server = TIEMCPServer()
        assert hasattr(server, "_handle_predict_techniques")

        # Test with mocked engine manager
        server.engine_manager = MagicMock()
        server.engine_manager.predict_techniques = AsyncMock(
            return_value={
                "input_techniques": ["T1059", "T1055"],
                "predicted_techniques": [
                    {
                        "technique_id": "T1003",
                        "technique_name": "OS Credential Dumping",
                        "score": 0.95,
                    }
                ],
            }
        )

        result = await server._handle_predict_techniques({"techniques": ["T1059", "T1055"]})
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0

    async def test_model_listing_handler(self):
        """Test model listing handler"""
        server = TIEMCPServer()
        assert hasattr(server, "_handle_list_models")

        # Test with mocked model manager
        server.model_manager = MagicMock()
        server.model_manager.list_models = AsyncMock(
            return_value={"models": [{"id": "default", "name": "Default Model"}]}
        )

        result = await server._handle_list_models({"include_metrics": False})
        assert result is not None
        assert isinstance(result, list)

    async def test_error_handling(self):
        """Test error handling in handlers"""
        server = TIEMCPServer()
        server.engine_manager = MagicMock()
        server.engine_manager.predict_techniques = AsyncMock(side_effect=Exception("Test error"))

        # Should handle error gracefully
        result = await server._handle_predict_techniques({"techniques": ["T1059"]})
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0
        # Should contain error message
        assert "Error" in str(result[0])
