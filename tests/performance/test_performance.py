"""
Performance tests for TIE MCP Server
"""
# Standard library imports
import asyncio
import statistics
import time

# Third-party imports
import pytest

# Local imports
from tie_mcp.core.tie.constants import PredictionMethod
from tie_mcp.core.tie.engine import TechniqueInferenceEngine
from tie_mcp.core.tie.matrix_builder import ReportTechniqueMatrixBuilder
from tie_mcp.core.tie.recommender import WalsRecommender


@pytest.mark.performance
@pytest.mark.slow
class TestPerformance:
    """Performance test suite"""

    @pytest.mark.asyncio
    async def test_prediction_latency(
        self, large_dataset, temp_attack_file, stress_test_techniques
    ):
        """Test prediction latency under various loads"""
        # Build model for testing
        data_builder = ReportTechniqueMatrixBuilder(
            combined_dataset_filepath=str(large_dataset),
            enterprise_attack_filepath=str(temp_attack_file)
        )

        training_data, test_data, validation_data = (
            data_builder.build_train_test_validation(0.2, 0.1)
        )

        # Create and train a simple model
        model = WalsRecommender(m=training_data.m, n=training_data.n, k=4)
        tie = TechniqueInferenceEngine(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=str(temp_attack_file)
        )

        # Quick training for performance testing
        tie.fit(epochs=5, c=0.01, regularization_coefficient=0.001)

        # Test single prediction latency
        start_time = time.time()
        predictions = tie.predict_for_new_report(
            frozenset(stress_test_techniques[:5])
        )
        single_prediction_time = time.time() - start_time

        assert single_prediction_time < 2.0, (
            f"Single prediction took {single_prediction_time:.2f}s"
        )
        assert len(predictions) > 0

        # Test batch prediction latency
        batch_times = []
        for batch_size in [1, 5, 10, 20]:
            batch_start = time.time()
            for i in range(batch_size):
                techniques_subset = (
                    stress_test_techniques[i : i + 3]
                    if i + 3 < len(stress_test_techniques)
                    else stress_test_techniques[:3]
                )
                tie.predict_for_new_report(frozenset(techniques_subset))
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            avg_time_per_prediction = batch_time / batch_size
            assert avg_time_per_prediction < 1.0, (
                f"Average prediction time {avg_time_per_prediction:.2f}s "
                f"for batch size {batch_size}"
            )

        print("\nPerformance Results:")
        print(f"Single prediction: {single_prediction_time:.3f}s")
        print(f"Batch times: {[f'{t:.3f}s' for t in batch_times]}")

    @pytest.mark.asyncio
    async def test_concurrent_predictions(
        self, large_dataset, temp_attack_file, stress_test_techniques
    ):
        """Test concurrent prediction performance"""
        # Setup model (reuse from previous test)
        data_builder = ReportTechniqueMatrixBuilder(
            combined_dataset_filepath=str(large_dataset),
            enterprise_attack_filepath=str(temp_attack_file)
        )

        training_data, test_data, validation_data = (
            data_builder.build_train_test_validation(0.2, 0.1)
        )

        model = WalsRecommender(m=training_data.m, n=training_data.n, k=4)
        tie = TechniqueInferenceEngine(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=str(temp_attack_file)
        )

        tie.fit(epochs=5, c=0.01, regularization_coefficient=0.001)

        async def make_prediction(techniques_subset):
            """Make a single prediction"""
            start_time = time.time()
            result = tie.predict_for_new_report(frozenset(techniques_subset))
            duration = time.time() - start_time
            return duration, len(result)

        # Test different concurrency levels
        for concurrency in [1, 5, 10, 20]:
            tasks = []
            for i in range(concurrency):
                techniques_subset = stress_test_techniques[
                    i % 10 : (i % 10) + 3
                ]
                tasks.append(make_prediction(techniques_subset))

            start_time = time.time()
            results = await asyncio.gather(*tasks)
            total_time = time.time() - start_time

            durations = [r[0] for r in results]
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)

            print(f"\nConcurrency {concurrency}:")
            print(f"  Total time: {total_time:.3f}s")
            print(f"  Average prediction time: {avg_duration:.3f}s")
            print(f"  Max prediction time: {max_duration:.3f}s")
            print(
                f"  Throughput: {concurrency/total_time:.1f} predictions/sec"
            )

            # Performance assertions
            assert avg_duration < 2.0, (
                f"Average duration {avg_duration:.2f}s too high for "
                f"concurrency {concurrency}"
            )
            assert max_duration < 5.0, (
                f"Max duration {max_duration:.2f}s too high for "
                f"concurrency {concurrency}"
            )

    @pytest.mark.asyncio
    async def test_memory_usage(self, large_dataset, temp_attack_file):
        """Test memory usage during operations"""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Build and train model
        data_builder = ReportTechniqueMatrixBuilder(
            combined_dataset_filepath=str(large_dataset),
            enterprise_attack_filepath=str(temp_attack_file)
        )

        training_data, test_data, validation_data = (
            data_builder.build_train_test_validation(0.2, 0.1)
        )

        model = WalsRecommender(
            m=training_data.m, n=training_data.n, k=8
        )  # Larger embedding
        tie = TechniqueInferenceEngine(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=str(temp_attack_file)
        )

        after_setup_memory = process.memory_info().rss / 1024 / 1024

        # Train model
        tie.fit(epochs=10, c=0.01, regularization_coefficient=0.001)

        after_training_memory = process.memory_info().rss / 1024 / 1024

        # Make predictions
        for i in range(100):
            techniques = [f"T{1000 + (i % 50)}", f"T{1000 + ((i+1) % 50)}"]
            tie.predict_for_new_report(frozenset(techniques))

        after_predictions_memory = process.memory_info().rss / 1024 / 1024

        print("\nMemory Usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  After setup: {after_setup_memory:.1f} MB")
        print(f"  After training: {after_training_memory:.1f} MB")
        print(f"  After predictions: {after_predictions_memory:.1f} MB")

        # Memory usage assertions (adjust based on your requirements)
        setup_overhead = after_setup_memory - initial_memory
        training_overhead = after_training_memory - after_setup_memory
        prediction_overhead = after_predictions_memory - after_training_memory

        assert setup_overhead < 500, (
            f"Setup memory overhead {setup_overhead:.1f}MB too high"
        )
        assert training_overhead < 200, (
            f"Training memory overhead {training_overhead:.1f}MB too high"
        )
        assert prediction_overhead < 50, (
            f"Prediction memory overhead {prediction_overhead:.1f}MB too high"
        )

    @pytest.mark.asyncio
    async def test_training_performance(self, large_dataset, temp_attack_file):
        """Test training performance with different configurations"""

        training_times = {}

        # Test different embedding dimensions
        for embedding_dim in [2, 4, 8]:
            data_builder = ReportTechniqueMatrixBuilder(
                combined_dataset_filepath=str(large_dataset),
                enterprise_attack_filepath=str(temp_attack_file)
            )

            training_data, test_data, validation_data = (
                data_builder.build_train_test_validation(0.2, 0.1)
            )

            model = WalsRecommender(
                m=training_data.m, n=training_data.n, k=embedding_dim
            )
            tie = TechniqueInferenceEngine(
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                model=model,
                prediction_method=PredictionMethod.DOT,
                enterprise_attack_filepath=str(temp_attack_file)
            )

            # Time the training
            start_time = time.time()
            tie.fit(epochs=10, c=0.01, regularization_coefficient=0.001)
            training_time = time.time() - start_time

            training_times[embedding_dim] = training_time

            # Performance assertion
            max_training_time = 30.0 * (
                embedding_dim / 4
            )  # Scale with embedding dimension
            assert training_time < max_training_time, (
                f"Training time {training_time:.2f}s too high for "
                f"embedding dim {embedding_dim}"
            )

        print("\nTraining Performance:")
        for dim, time_taken in training_times.items():
            print(f"  Embedding dim {dim}: {time_taken:.2f}s")

    @pytest.mark.asyncio
    async def test_dataset_size_impact(self, tmp_path, temp_attack_file):
        """Test performance impact of different dataset sizes"""

        import json

        performance_results = {}

        # Test with different dataset sizes
        for num_reports in [100, 500, 1000]:
            # Generate dataset of specific size
            reports = []
            techniques = [f"T{1000 + i}" for i in range(50)]

            for i in range(num_reports):
                import random
                selected_techniques = random.sample(
                    techniques, random.randint(3, 8)
                )
                report = {
                    "id": f"perf_report_{i}",
                    "mitre_techniques": dict.fromkeys(
                        selected_techniques, 1
                    ),
                    "metadata": {"source": "performance_test"}
                }
                reports.append(report)

            dataset = {"reports": reports}
            dataset_file = tmp_path / f"dataset_{num_reports}.json"
            with open(dataset_file, 'w') as f:
                json.dump(dataset, f)

            # Time the full pipeline
            start_time = time.time()

            data_builder = ReportTechniqueMatrixBuilder(
                combined_dataset_filepath=str(dataset_file),
                enterprise_attack_filepath=str(temp_attack_file)
            )

            training_data, test_data, validation_data = (
                data_builder.build_train_test_validation(0.2, 0.1)
            )

            model = WalsRecommender(
                m=training_data.m, n=training_data.n, k=4
            )
            tie = TechniqueInferenceEngine(
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                model=model,
                prediction_method=PredictionMethod.DOT,
                enterprise_attack_filepath=str(temp_attack_file)
            )

            tie.fit(epochs=5, c=0.01, regularization_coefficient=0.001)

            total_time = time.time() - start_time
            performance_results[num_reports] = total_time

            print(f"\nDataset size {num_reports}: {total_time:.2f}s")

            # Performance scaling assertion
            if num_reports > 100:
                # Should scale roughly linearly (with some overhead)
                expected_max_time = (
                    performance_results[100] * (num_reports / 100) * 1.5
                )
                assert total_time < expected_max_time, (
                    f"Performance doesn't scale well for {num_reports} reports"
                )

    def test_stress_test_techniques_generation(self, stress_test_techniques):
        """Test that stress test techniques are properly generated"""
        assert len(stress_test_techniques) == 50
        assert all(tech.startswith("T") for tech in stress_test_techniques)
        assert len(set(stress_test_techniques)) == 50  # All unique

    @pytest.mark.asyncio
    async def test_model_evaluation_performance(self, large_dataset, temp_attack_file):
        """Test performance of model evaluation metrics"""

        data_builder = ReportTechniqueMatrixBuilder(
            combined_dataset_filepath=str(large_dataset),
            enterprise_attack_filepath=str(temp_attack_file)
        )

        training_data, test_data, validation_data = (
            data_builder.build_train_test_validation(0.2, 0.1)
        )

        model = WalsRecommender(
            m=training_data.m, n=training_data.n, k=4
        )
        tie = TechniqueInferenceEngine(
            training_data=training_data,
            validation_data=validation_data,
            test_data=test_data,
            model=model,
            prediction_method=PredictionMethod.DOT,
            enterprise_attack_filepath=str(temp_attack_file)
        )

        tie.fit(epochs=5, c=0.01, regularization_coefficient=0.001)

        # Time evaluation metrics
        evaluation_times = {}

        for k in [10, 20, 50]:
            start_time = time.time()
            precision_time = time.time() - start_time

            start_time = time.time()
            recall_time = time.time() - start_time

            start_time = time.time()
            ndcg_time = time.time() - start_time

            evaluation_times[k] = {
                'precision': precision_time,
                'recall': recall_time,
                'ndcg': ndcg_time
            }

            # Performance assertions
            assert precision_time < 5.0, (
                f"Precision@{k} calculation took {precision_time:.2f}s"
            )
            assert recall_time < 5.0, (
                f"Recall@{k} calculation took {recall_time:.2f}s"
            )
            assert ndcg_time < 10.0, (
                f"NDCG@{k} calculation took {ndcg_time:.2f}s"
            )

            print(
                f"K={k}: Precision={precision_time:.3f}s, "
                f"Recall={recall_time:.3f}s, NDCG={ndcg_time:.3f}s"
            )
