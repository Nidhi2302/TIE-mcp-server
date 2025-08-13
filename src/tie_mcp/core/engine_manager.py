"""
TIE Engine Manager - High-level interface for TIE functionality
"""

import asyncio
import logging
from typing import Any

from ..api.schemas import (
    AttackTechniqueInfo,
    ModelEvaluationResponse,
    PredictionResponse,
    TrainingResponse,
)
from ..config.settings import settings
from ..models.model_manager import ModelManager
from ..monitoring.metrics import MetricsCollector
from ..utils.async_utils import run_in_thread
from .tie.constants import PredictionMethod
from .tie.engine import TechniqueInferenceEngine
from .tie.matrix_builder import ReportTechniqueMatrixBuilder
from .tie.recommender import (
    BPRRecommender,
    FactorizationRecommender,
    ImplicitBPRRecommender,
    ImplicitWalsRecommender,
    TopItemsRecommender,
    WalsRecommender,
)
from .tie.utils import get_mitre_technique_ids_to_names

logger = logging.getLogger(__name__)


class TIEEngineManager:
    """High-level manager for TIE engine operations"""

    def __init__(self, model_manager: ModelManager, metrics_collector: MetricsCollector):
        self.model_manager = model_manager
        self.metrics_collector = metrics_collector
        self.current_engine: TechniqueInferenceEngine | None = None
        self.attack_techniques_cache: dict[str, str] | None = None

    async def initialize(self):
        """Initialize the engine manager"""
        logger.info("Initializing TIE Engine Manager...")

        try:
            # Load ATT&CK techniques cache
            await self._load_attack_techniques()

            # Load default model if available
            default_model = await self.model_manager.get_default_model()
            if default_model:
                await self._load_engine(default_model.id)

            logger.info("TIE Engine Manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize TIE Engine Manager: {e}")
            raise

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up TIE Engine Manager...")
        self.current_engine = None
        self.attack_techniques_cache = None

    async def predict_techniques(
        self,
        techniques: list[str],
        model_id: str | None = None,
        top_k: int = 20,
        prediction_method: str = "dot"
    ) -> PredictionResponse:
        """
        Predict MITRE ATT&CK techniques based on observed techniques

        Args:
            techniques: List of observed technique IDs
            model_id: Optional specific model to use
            top_k: Number of top predictions to return
            prediction_method: Prediction method ('dot' or 'cosine')

        Returns:
            PredictionResponse with predicted techniques and scores
        """
        start_time = asyncio.get_event_loop().time()

        try:
            # Load appropriate model
            engine = await self._get_engine(model_id)

            # Convert prediction method

            # Run prediction in thread to avoid blocking
            predictions_df = await run_in_thread(
                engine.predict_for_new_report,
                frozenset(techniques)
            )

            # Sort by predictions and get top k
            top_predictions = predictions_df.sort_values(
                by="predictions", ascending=False
            ).head(top_k)

            # Format response
            predicted_techniques = []
            for _, row in top_predictions.iterrows():
                technique_id = row.name
                score = float(row["predictions"])
                technique_name = row.get("technique_name", "Unknown")

                predicted_techniques.append({
                    "technique_id": technique_id,
                    "technique_name": technique_name,
                    "score": score,
                    "in_training_data": bool(row["training_data"])
                })

            # Record metrics
            duration = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.record_prediction(
                duration=duration,
                input_techniques_count=len(techniques),
                output_techniques_count=len(predicted_techniques),
                model_id=model_id or "default"
            )

            return PredictionResponse(
                predicted_techniques=predicted_techniques,
                input_techniques=techniques,
                model_id=model_id or "default",
                prediction_method=prediction_method,
                execution_time_seconds=duration
            )

        except Exception as e:
            logger.error(f"Error in predict_techniques: {e}")
            await self.metrics_collector.record_error("prediction", str(e))
            raise

    async def train_model(
        self,
        dataset_path: str,
        model_type: str = "wals",
        hyperparameters: dict[str, Any] | None = None,
        validation_ratio: float = 0.1,
        test_ratio: float = 0.2,
        embedding_dimension: int = 4,
        auto_hyperparameter_tuning: bool = True
    ) -> TrainingResponse:
        """
        Train a new TIE model

        Args:
            dataset_path: Path to training dataset
            model_type: Type of model to train
            hyperparameters: Model-specific hyperparameters
            validation_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
            embedding_dimension: Embedding dimension
            auto_hyperparameter_tuning: Whether to auto-tune hyperparameters

        Returns:
            TrainingResponse with training results
        """
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(f"Starting model training: {model_type}")

            # Build data matrices
            data_builder = ReportTechniqueMatrixBuilder(
                combined_dataset_filepath=dataset_path,
                enterprise_attack_filepath=settings.enterprise_attack_filepath
            )

            training_data, test_data, validation_data = await run_in_thread(
                data_builder.build_train_test_validation,
                test_ratio,
                validation_ratio
            )

            # Create model
            model = self._create_model(
                model_type=model_type,
                m=training_data.m,
                n=training_data.n,
                k=embedding_dimension
            )

            # Create TIE engine
            pred_method = PredictionMethod.DOT if model_type in ["wals", "factorization"] else PredictionMethod.COSINE

            engine = TechniqueInferenceEngine(
                training_data=training_data,
                validation_data=validation_data,
                test_data=test_data,
                model=model,
                prediction_method=pred_method,
                enterprise_attack_filepath=settings.enterprise_attack_filepath
            )

            # Train model
            if auto_hyperparameter_tuning:
                best_hyperparameters = await run_in_thread(
                    engine.fit_with_validation,
                    **self._get_default_hyperparameters(model_type)
                )
            else:
                hyperparameters = hyperparameters or {}
                await run_in_thread(engine.fit, **hyperparameters)
                best_hyperparameters = hyperparameters

            # Evaluate model
            evaluation_metrics = await self._evaluate_trained_model(engine)

            # Save model
            model_id = await self.model_manager.save_model(
                engine=engine,
                model_type=model_type,
                hyperparameters=best_hyperparameters,
                metrics=evaluation_metrics,
                dataset_path=dataset_path
            )

            duration = asyncio.get_event_loop().time() - start_time

            # Record training metrics
            await self.metrics_collector.record_training(
                duration=duration,
                model_type=model_type,
                dataset_size=training_data.m,
                success=True
            )

            logger.info(f"Model training completed: {model_id}")

            return TrainingResponse(
                model_id=model_id,
                model_type=model_type,
                hyperparameters=best_hyperparameters,
                metrics=evaluation_metrics,
                training_time_seconds=duration,
                dataset_info={
                    "training_samples": training_data.m,
                    "validation_samples": validation_data.m,
                    "test_samples": test_data.m,
                    "num_techniques": training_data.n
                }
            )

        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            await self.metrics_collector.record_training(
                duration=duration,
                model_type=model_type,
                dataset_size=0,
                success=False
            )
            logger.error(f"Error in train_model: {e}")
            raise

    async def evaluate_model(
        self,
        model_id: str,
        k_values: list[int] = None
    ) -> ModelEvaluationResponse:
        """Evaluate a trained model"""
        if k_values is None:
            k_values = [10, 20, 50]
        try:
            engine = await self._get_engine(model_id)

            metrics = {}
            for k in k_values:
                precision = await run_in_thread(engine.precision, k)
                recall = await run_in_thread(engine.recall, k)
                ndcg = await run_in_thread(engine.normalized_discounted_cumulative_gain, k)

                metrics[f"precision_at_{k}"] = precision
                metrics[f"recall_at_{k}"] = recall
                metrics[f"ndcg_at_{k}"] = ndcg

            return ModelEvaluationResponse(
                model_id=model_id,
                metrics=metrics,
                k_values=k_values
            )

        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            raise

    async def get_attack_techniques(
        self,
        technique_ids: list[str] | None = None,
        search_term: str | None = None,
        tactic: str | None = None
    ) -> list[AttackTechniqueInfo]:
        """Get information about ATT&CK techniques"""
        try:
            if not self.attack_techniques_cache:
                await self._load_attack_techniques()

            techniques = []

            for technique_id, technique_name in self.attack_techniques_cache.items():
                # Filter by technique IDs if specified
                if technique_ids and technique_id not in technique_ids:
                    continue

                # Filter by search term if specified
                if search_term and search_term.lower() not in technique_name.lower():
                    continue

                # TODO: Add tactic filtering (requires parsing ATT&CK data)

                techniques.append(AttackTechniqueInfo(
                    technique_id=technique_id,
                    technique_name=technique_name,
                    tactic=None  # TODO: Extract from ATT&CK data
                ))

            return techniques

        except Exception as e:
            logger.error(f"Error getting ATT&CK techniques: {e}")
            raise

    async def _get_engine(self, model_id: str | None = None) -> TechniqueInferenceEngine:
        """Get or load TIE engine"""
        if model_id:
            return await self.model_manager.load_model(model_id)
        elif self.current_engine:
            return self.current_engine
        else:
            # Load default model
            default_model = await self.model_manager.get_default_model()
            if default_model:
                return await self.model_manager.load_model(default_model.id)
            else:
                raise ValueError("No model available. Please train a model first.")

    async def _load_engine(self, model_id: str):
        """Load engine from model ID"""
        self.current_engine = await self.model_manager.load_model(model_id)

    async def _load_attack_techniques(self):
        """Load ATT&CK techniques from STIX file"""
        try:
            self.attack_techniques_cache = await run_in_thread(
                get_mitre_technique_ids_to_names,
                settings.enterprise_attack_filepath
            )
            logger.info(f"Loaded {len(self.attack_techniques_cache)} ATT&CK techniques")
        except Exception as e:
            logger.error(f"Failed to load ATT&CK techniques: {e}")
            self.attack_techniques_cache = {}

    def _create_model(self, model_type: str, m: int, n: int, k: int):
        """Create a model instance based on type"""
        model_classes = {
            "wals": WalsRecommender,
            "bpr": BPRRecommender,
            "implicit_wals": ImplicitWalsRecommender,
            "implicit_bpr": ImplicitBPRRecommender,
            "factorization": FactorizationRecommender,
            "top_items": TopItemsRecommender,
        }

        if model_type not in model_classes:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model_classes[model_type](m=m, n=n, k=k)

    def _get_default_hyperparameters(self, model_type: str) -> dict[str, list]:
        """Get default hyperparameter search space for model type"""
        if model_type == "wals":
            return {
                "epochs": [settings.model.wals_epochs],
                "c": settings.model.wals_c_values,
                "regularization_coefficient": settings.model.wals_regularization_values
            }
        elif model_type == "bpr":
            return {
                "epochs": [settings.model.bpr_epochs],
                "learning_rate": settings.model.bpr_learning_rates,
                "regularization": settings.model.bpr_regularization_values
            }
        else:
            return {}

    async def _evaluate_trained_model(self, engine: TechniqueInferenceEngine) -> dict[str, float]:
        """Evaluate a trained model and return metrics"""
        metrics = {}

        for k in [10, 20, 50]:
            precision = await run_in_thread(engine.precision, k)
            recall = await run_in_thread(engine.recall, k)
            ndcg = await run_in_thread(engine.normalized_discounted_cumulative_gain, k)

            metrics[f"precision_at_{k}"] = precision
            metrics[f"recall_at_{k}"] = recall
            metrics[f"ndcg_at_{k}"] = ndcg

        return metrics
