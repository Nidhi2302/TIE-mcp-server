"""
Performance tests for TIE MCP Server - Simplified version
"""

import importlib.util
import time

import pytest

from tie_mcp.core.tie.constants import PredictionMethod
from tie_mcp.core.tie.engine import TechniqueInferenceEngine
from tie_mcp.core.tie.matrix_builder import ReportTechniqueMatrixBuilder
from tie_mcp.core.tie.recommender import WalsRecommender

# Skip performance tests gracefully when TensorFlow is not installed
_tf_missing = importlib.util.find_spec("tensorflow") is None
pytestmark = pytest.mark.skipif(_tf_missing, reason="tensorflow not installed")


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Simplified performance test suite"""

    @pytest.mark.asyncio
    async def test_basic_prediction_performance(self, large_dataset, temp_attack_file, stress_test_techniques):
        """Test basic prediction performance"""
        # Build model for testing
        data_builder = ReportTechniqueMatrixBuilder(
            combined_dataset_filepath=str(large_dataset),
            enterprise_attack_filepath=str(temp_attack_file),
        )

        training_data, test_data, validation_data = data_builder.build_train_test_validation(0.2, 0.1)

        # Create and train a simple model
        model = WalsRecommender(m=training_data.m, n=training_data.n, k=4)
        tie = TechniqueInferenceEngine(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=str(temp_attack_file),
        )

        # Quick training for performance testing
        tie.fit(epochs=5, c=0.01, regularization_coefficient=0.001)

        # Test single prediction latency
        start_time = time.time()
        predictions = tie.predict_for_new_report(frozenset(stress_test_techniques[:5]))
        single_prediction_time = time.time() - start_time

        assert single_prediction_time < 5.0, f"Single prediction took {single_prediction_time:.2f}s"
        assert len(predictions) > 0

        print(f"\nSingle prediction time: {single_prediction_time:.3f}s")

    def test_stress_test_techniques_generation(self, stress_test_techniques):
        """Test that stress test techniques are properly generated"""
        assert len(stress_test_techniques) == 50
        assert all(tech.startswith("T") for tech in stress_test_techniques)
        assert len(set(stress_test_techniques)) == 50  # All unique
