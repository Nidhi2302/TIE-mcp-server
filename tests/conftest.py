"""
Test configuration and fixtures for TIE MCP Server
"""

# Standard library imports
import asyncio
import json
import random
import tempfile
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Third-party imports
import pytest
import pytest_asyncio

# Local imports
from tie_mcp.config.settings import Settings
from tie_mcp.core.engine_manager import TIEEngineManager
from tie_mcp.models.model_manager import ModelManager
from tie_mcp.monitoring.metrics import MetricsCollector
from tie_mcp.server import TIEMCPServer
from tie_mcp.storage.database import DatabaseManager

# Type aliases
TechniqueDict = dict[str, float]
PredictionResponse = dict[str, list[dict[str, str]]]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Test settings configuration"""
    return Settings(
        environment="testing",
        debug=True,
        log_level="DEBUG",
        database={"url": "sqlite+aiosqlite:///:memory:"},
        redis={"url": "redis://localhost:6379/15"},  # Use test database
        model={
            "models_directory": Path(tempfile.mkdtemp()) / "models",
            "datasets_directory": Path(tempfile.mkdtemp()) / "datasets",
            "configs_directory": Path(tempfile.mkdtemp()) / "configs",
        }
    )


@pytest.fixture
def sample_dataset() -> dict:
    """Sample TIE dataset for testing"""
    return {
        "reports": [
            {
                "id": "report_1",
                "mitre_techniques": {
                    "T1059": 1,
                    "T1055": 1,
                    "T1082": 1
                },
                "metadata": {"source": "test"}
            },
            {
                "id": "report_2",
                "mitre_techniques": {
                    "T1059": 1,
                    "T1053": 1,
                    "T1105": 1
                },
                "metadata": {"source": "test"}
            },
            {
                "id": "report_3",
                "mitre_techniques": {
                    "T1055": 1,
                    "T1053": 1,
                    "T1082": 1,
                    "T1105": 1
                },
                "metadata": {"source": "test"}
            }
        ]
    }


@pytest.fixture
def sample_attack_data() -> dict:
    """Sample ATT&CK data for testing"""
    return {
        "objects": [
            {
                "type": "attack-pattern",
                "id": "attack-pattern--1",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1059"}
                ],
                "name": "Command and Scripting Interpreter"
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--2",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1055"}
                ],
                "name": "Process Injection"
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--3",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1082"}
                ],
                "name": "System Information Discovery"
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--4",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1053"}
                ],
                "name": "Scheduled Task/Job"
            },
            {
                "type": "attack-pattern",
                "id": "attack-pattern--5",
                "external_references": [
                    {"source_name": "mitre-attack", "external_id": "T1105"}
                ],
                "name": "Ingress Tool Transfer"
            }
        ]
    }


@pytest.fixture
def temp_dataset_file(sample_dataset, tmp_path) -> Path:
    """Create temporary dataset file"""
    dataset_file = tmp_path / "test_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(sample_dataset, f)
    return dataset_file


@pytest.fixture
def temp_attack_file(sample_attack_data, tmp_path) -> Path:
    """Create temporary ATT&CK file"""
    attack_file = tmp_path / "test_attack.json"
    with open(attack_file, 'w') as f:
        json.dump(sample_attack_data, f)
    return attack_file


@pytest_asyncio.fixture
async def mock_db_manager() -> AsyncGenerator[DatabaseManager, None]:
    """Mock database manager for testing"""
    db_manager = AsyncMock(spec=DatabaseManager)
    db_manager.initialize = AsyncMock()
    db_manager.cleanup = AsyncMock()
    db_manager.health_check = AsyncMock(return_value=True)

    # Mock model operations
    db_manager.save_model = AsyncMock(return_value="test-model-id")
    db_manager.get_model = AsyncMock(return_value={
        "id": "test-model-id",
        "name": "test-model",
        "model_type": "wals",
        "status": "trained",
        "hyperparameters": {"epochs": 25},
        "metrics": {"ndcg_at_20": 0.75},
        "artifacts_path": "/test/path"
    })
    db_manager.list_models = AsyncMock(return_value=[])
    db_manager.delete_model = AsyncMock()

    yield db_manager


@pytest.fixture
def mock_metrics_collector() -> MetricsCollector:
    """Mock metrics collector for testing"""
    collector = MagicMock(spec=MetricsCollector)
    collector.record_prediction = AsyncMock()
    collector.record_training = AsyncMock()
    collector.record_error = AsyncMock()
    collector.get_system_metrics = AsyncMock(return_value={
        "cpu_usage_percent": 50.0,
        "memory_usage_percent": 60.0,
        "total_predictions": 100
    })
    return collector


@pytest_asyncio.fixture
async def mock_model_manager(
    mock_db_manager, mock_metrics_collector
) -> AsyncGenerator[ModelManager, None]:
    """Mock model manager for testing"""
    model_manager = AsyncMock(spec=ModelManager)
    model_manager.initialize = AsyncMock()
    model_manager.cleanup = AsyncMock()
    model_manager.save_model = AsyncMock(return_value="test-model-id")
    model_manager.load_model = AsyncMock()
    model_manager.list_models = AsyncMock(return_value=[])
    model_manager.get_model_info = AsyncMock()
    model_manager.delete_model = AsyncMock()
    model_manager.create_dataset = AsyncMock()

    yield model_manager


@pytest_asyncio.fixture
async def mock_engine_manager(
    mock_model_manager, mock_metrics_collector
) -> AsyncGenerator[TIEEngineManager, None]:
    """Mock engine manager for testing"""
    engine_manager = AsyncMock(spec=TIEEngineManager)
    engine_manager.initialize = AsyncMock()
    engine_manager.cleanup = AsyncMock()
    engine_manager.predict_techniques = AsyncMock(return_value={
        "predicted_techniques": [
            {
                "technique_id": "T1055",
                "technique_name": "Process Injection",
                "score": 0.85,
                "in_training_data": True
            }
        ],
        "input_techniques": ["T1059"],
        "model_id": "test-model",
        "execution_time_seconds": 0.1
    })
    engine_manager.train_model = AsyncMock()
    engine_manager.evaluate_model = AsyncMock()
    engine_manager.get_attack_techniques = AsyncMock(return_value=[])

    yield engine_manager


@pytest_asyncio.fixture
async def tie_mcp_server(
    mock_engine_manager,
    mock_model_manager,
    mock_db_manager,
    mock_metrics_collector,
) -> AsyncGenerator[TIEMCPServer, None]:
    """TIE MCP Server instance for testing"""
    server = TIEMCPServer()

    # Replace components with mocks
    server.db_manager = mock_db_manager
    server.metrics_collector = mock_metrics_collector
    server.model_manager = mock_model_manager
    server.engine_manager = mock_engine_manager

    yield server


@pytest.fixture
def prediction_request():
    """Sample prediction request"""
    return {
        "techniques": ["T1059", "T1082"],
        "top_k": 5,
        "prediction_method": "dot"
    }


@pytest.fixture
def training_request():
    """Sample training request"""
    return {
        "dataset_path": "/test/dataset.json",
        "model_type": "wals",
        "embedding_dimension": 4,
        "auto_hyperparameter_tuning": True
    }


@pytest.fixture
def mock_trained_model():
    """Mock trained TIE model"""
    model = MagicMock()
    model.predict_for_new_report = MagicMock(return_value=MagicMock())
    model.precision = MagicMock(return_value=0.75)
    model.recall = MagicMock(return_value=0.65)
    model.normalized_discounted_cumulative_gain = MagicMock(return_value=0.80)
    return model


# Performance testing fixtures
@pytest.fixture
def large_dataset(tmp_path) -> Path:
    """Generate a larger dataset for performance testing"""
    reports = []
    techniques = [f"T{1000 + i}" for i in range(100)]  # 100 techniques

    for i in range(1000):  # 1000 reports
        # Randomly select 3-10 techniques per report
        num_techniques = random.randint(3, 10)
        selected_techniques = random.sample(techniques, num_techniques)

        report = {
            "id": f"perf_report_{i}",
            "mitre_techniques": dict.fromkeys(selected_techniques, 1),
            "metadata": {"source": "performance_test"}
        }
        reports.append(report)

    dataset = {"reports": reports}
    dataset_file = tmp_path / "large_dataset.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f)

    return dataset_file


@pytest.fixture
def stress_test_techniques():
    """Generate techniques for stress testing"""
    return [f"T{1000 + i}" for i in range(50)]


# Database fixtures for integration tests
@pytest_asyncio.fixture
async def real_db_manager(test_settings) -> AsyncGenerator[DatabaseManager, None]:
    """Real database manager for integration tests"""
    db_manager = DatabaseManager()

    # Override settings for test database
    original_url = test_settings.database.url
    test_settings.database.url = "sqlite+aiosqlite:///:memory:"

    await db_manager.initialize()
    yield db_manager
    await db_manager.cleanup()

    # Restore original settings
    test_settings.database.url = original_url


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_temp_files():
    """Automatically cleanup temporary files after tests"""
    yield
    # Cleanup code can be added here if needed


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Custom assertions
def assert_model_metrics_valid(metrics: dict):
    """Assert that model metrics are valid"""
    required_metrics = ["precision_at_10", "recall_at_10", "ndcg_at_10"]
    for metric in required_metrics:
        assert metric in metrics
        assert 0.0 <= metrics[metric] <= 1.0


def assert_prediction_response_valid(response: dict):
    """Assert that prediction response is valid"""
    assert "predicted_techniques" in response
    assert "input_techniques" in response
    assert "execution_time_seconds" in response

    for prediction in response["predicted_techniques"]:
        assert "technique_id" in prediction
        assert "score" in prediction
        assert 0.0 <= prediction["score"] <= 1.0
