"""
Unit tests for TIE MCP Server
"""


import pytest
from mcp.types import TextContent

from tie_mcp.server import TIEMCPServer


@pytest.mark.unit
@pytest.mark.asyncio
class TestTIEMCPServer:
    """Test suite for TIE MCP Server"""

    async def test_server_initialization(self):
        """Test server initialization"""
        server = TIEMCPServer()
        assert server.server is not None
        assert server.engine_manager is None
        assert server.model_manager is None

    async def test_predict_techniques_tool(
        self, tie_mcp_server, prediction_request
    ):
        """Test predict_techniques tool handler"""
        # Test the tool handler directly
        result = await tie_mcp_server._handle_predict_techniques(
            prediction_request
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)

        # Verify engine manager was called
        tie_mcp_server.engine_manager.predict_techniques.assert_called_once()

    async def test_train_model_tool(self, tie_mcp_server, training_request):
        """Test train_model tool handler"""
        result = await tie_mcp_server._handle_train_model(training_request)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)

        # Verify engine manager was called
        tie_mcp_server.engine_manager.train_model.assert_called_once()

    async def test_list_models_tool(self, tie_mcp_server):
        """Test list_models tool handler"""
        result = await tie_mcp_server._handle_list_models(
            {"include_metrics": True}
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)

        # Verify model manager was called
        tie_mcp_server.model_manager.list_models.assert_called_once()

    async def test_get_model_info_tool(self, tie_mcp_server):
        """Test get_model_info tool handler"""
        result = await tie_mcp_server._handle_get_model_info(
            {"model_id": "test-model"}
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)

        # Verify model manager was called
        tie_mcp_server.model_manager.get_model_info.assert_called_once_with(
            "test-model"
        )

    async def test_create_dataset_tool(self, tie_mcp_server):
        """Test create_dataset tool handler"""
        dataset_request = {
            "reports": [
                {
                    "id": "test_report",
                    "techniques": ["T1059", "T1055"],
                    "metadata": {"source": "test"}
                }
            ],
            "dataset_name": "test_dataset",
            "description": "Test dataset"
        }

        result = await tie_mcp_server._handle_create_dataset(dataset_request)

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)

        # Verify model manager was called
        tie_mcp_server.model_manager.create_dataset.assert_called_once()

    async def test_get_attack_techniques_tool(self, tie_mcp_server):
        """Test get_attack_techniques tool handler"""
        result = await tie_mcp_server._handle_get_attack_techniques({
            "technique_ids": ["T1059", "T1055"]
        })

        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], TextContent)

        # Verify engine manager was called
        tie_mcp_server.engine_manager.get_attack_techniques.assert_called_once()

    async def test_error_handling(self, tie_mcp_server):
        """Test error handling in tool handlers"""
        # Make engine manager raise an exception
        tie_mcp_server.engine_manager.predict_techniques.side_effect = (
            Exception("Test error")
        )

        result = await tie_mcp_server._handle_predict_techniques(
            {"techniques": ["T1059"]}
        )

        assert isinstance(result, list)
        assert len(result) > 0
        assert "Error" in result[0].text

    @pytest.mark.parametrize("invalid_request", [
        {},  # Missing required fields
        {"techniques": []},  # Empty techniques list
        {"techniques": ["T1059"], "top_k": -1},  # Invalid top_k
        {"techniques": ["T1059"], "top_k": 101},  # top_k too large
    ])
    async def test_invalid_prediction_requests(
        self, tie_mcp_server, invalid_request
    ):
        """Test handling of invalid prediction requests"""
        # The validation should happen at the MCP protocol level,
        # but we can test graceful handling of edge cases
        try:
            result = await tie_mcp_server._handle_predict_techniques(
                invalid_request
            )
            # Should either handle gracefully or raise appropriate error
            assert isinstance(result, list)
        except (ValueError, KeyError):
            # These exceptions are acceptable for invalid input
            pass

    async def test_concurrent_requests(
        self, tie_mcp_server, prediction_request
    ):
        """Test handling of concurrent requests"""
        import asyncio

        # Create multiple concurrent requests
        tasks = [
            tie_mcp_server._handle_predict_techniques(prediction_request)
            for _ in range(5)
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, list)
            assert len(result) > 0
