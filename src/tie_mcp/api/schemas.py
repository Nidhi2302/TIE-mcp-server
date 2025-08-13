"""
API schemas for TIE MCP Server
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class ModelType(str, Enum):
    """Supported model types"""
    WALS = "wals"
    BPR = "bpr"
    IMPLICIT_WALS = "implicit_wals"
    IMPLICIT_BPR = "implicit_bpr"
    FACTORIZATION = "factorization"
    TOP_ITEMS = "top_items"


class PredictionMethod(str, Enum):
    """Prediction methods"""
    DOT = "dot"
    COSINE = "cosine"


class ModelStatus(str, Enum):
    """Model status"""
    TRAINING = "training"
    TRAINED = "trained"
    FAILED = "failed"
    ARCHIVED = "archived"


# Request schemas
class PredictionRequest(BaseModel):
    """Request for technique prediction"""
    techniques: list[str] = Field(..., description="List of observed technique IDs")
    model_id: str | None = Field(None, description="Optional model ID to use")
    top_k: int = Field(20, ge=1, le=100, description="Number of top predictions")
    prediction_method: PredictionMethod = Field(PredictionMethod.DOT)

    @validator("techniques")
    def validate_techniques(cls, v):
        if not v:
            raise ValueError("At least one technique must be provided")
        return v


class TrainingRequest(BaseModel):
    """Request for model training"""
    dataset_path: str = Field(..., description="Path to training dataset")
    model_type: ModelType = Field(ModelType.WALS)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    validation_ratio: float = Field(0.1, ge=0.0, le=0.5)
    test_ratio: float = Field(0.2, ge=0.0, le=0.5)
    embedding_dimension: int = Field(4, ge=1, le=128)
    auto_hyperparameter_tuning: bool = Field(True)
    model_name: str | None = Field(None, description="Optional model name")
    description: str | None = Field(None, description="Model description")

    @validator("validation_ratio", "test_ratio")
    def validate_ratios(cls, v, values):
        if "validation_ratio" in values and "test_ratio" in values:
            if values["validation_ratio"] + values["test_ratio"] > 1.0:
                raise ValueError("validation_ratio + test_ratio must be <= 1.0")
        return v


class DatasetCreationRequest(BaseModel):
    """Request for dataset creation"""
    reports: list[dict[str, Any]] = Field(..., description="List of CTI reports")
    dataset_name: str = Field(..., description="Name for the dataset")
    description: str = Field("", description="Dataset description")

    @validator("reports")
    def validate_reports(cls, v):
        if not v:
            raise ValueError("At least one report must be provided")

        for i, report in enumerate(v):
            if "id" not in report:
                raise ValueError(f"Report {i} missing 'id' field")
            if "techniques" not in report:
                raise ValueError(f"Report {i} missing 'techniques' field")
            if not isinstance(report["techniques"], list):
                raise ValueError(f"Report {i} 'techniques' must be a list")

        return v


# Response schemas
class PredictedTechnique(BaseModel):
    """Single predicted technique"""
    technique_id: str
    technique_name: str
    score: float
    in_training_data: bool


class PredictionResponse(BaseModel):
    """Response for technique prediction"""
    predicted_techniques: list[PredictedTechnique]
    input_techniques: list[str]
    model_id: str
    prediction_method: str
    execution_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelMetrics(BaseModel):
    """Model performance metrics"""
    precision_at_10: float | None = None
    precision_at_20: float | None = None
    precision_at_50: float | None = None
    recall_at_10: float | None = None
    recall_at_20: float | None = None
    recall_at_50: float | None = None
    ndcg_at_10: float | None = None
    ndcg_at_20: float | None = None
    ndcg_at_50: float | None = None


class DatasetInfo(BaseModel):
    """Dataset information"""
    training_samples: int
    validation_samples: int
    test_samples: int
    num_techniques: int


class TrainingResponse(BaseModel):
    """Response for model training"""
    model_id: str
    model_type: ModelType
    hyperparameters: dict[str, Any]
    metrics: dict[str, float]
    training_time_seconds: float
    dataset_info: DatasetInfo
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str | None
    model_type: ModelType
    status: ModelStatus
    hyperparameters: dict[str, Any]
    metrics: ModelMetrics | None
    dataset_path: str | None
    dataset_info: DatasetInfo | None
    created_at: datetime
    updated_at: datetime
    is_default: bool = False
    description: str | None = None
    version: str = "1.0"


class ModelEvaluationResponse(BaseModel):
    """Response for model evaluation"""
    model_id: str
    metrics: dict[str, float]
    k_values: list[int]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AttackTechniqueInfo(BaseModel):
    """ATT&CK technique information"""
    technique_id: str
    technique_name: str
    tactic: str | None = None
    description: str | None = None
    platforms: list[str] | None = None
    data_sources: list[str] | None = None


class DatasetCreationResponse(BaseModel):
    """Response for dataset creation"""
    dataset_id: str
    dataset_name: str
    description: str
    num_reports: int
    num_techniques: int
    file_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ModelListResponse(BaseModel):
    """Response for listing models"""
    models: list[ModelInfo]
    total_count: int
    default_model_id: str | None = None


class DatasetListResponse(BaseModel):
    """Response for listing datasets"""
    datasets: list[DatasetCreationResponse]
    total_count: int


class SystemMetrics(BaseModel):
    """System performance metrics"""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    active_models: int
    total_predictions: int
    total_trainings: int
    average_prediction_time: float
    average_training_time: float
    error_rate_percent: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    components: dict[str, str] = Field(default_factory=dict)
    metrics: SystemMetrics | None = None


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str | None = None
    error_code: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Task-related schemas for async operations
class TaskStatus(str, Enum):
    """Task status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    REVOKED = "revoked"


class TaskInfo(BaseModel):
    """Task information"""
    task_id: str
    status: TaskStatus
    progress: float = Field(0.0, ge=0.0, le=100.0)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    estimated_completion: datetime | None = None


class TaskResponse(BaseModel):
    """Response for async task submission"""
    task_id: str
    status: TaskStatus
    message: str
    estimated_completion_minutes: float | None = None


# Batch operation schemas
class BatchPredictionRequest(BaseModel):
    """Request for batch prediction"""
    prediction_requests: list[PredictionRequest]
    model_id: str | None = None

    @validator("prediction_requests")
    def validate_batch_size(cls, v):
        if len(v) > 1000:  # Reasonable batch limit
            raise ValueError("Batch size cannot exceed 1000 requests")
        return v


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction"""
    results: list[PredictionResponse]
    total_requests: int
    successful_requests: int
    failed_requests: int
    execution_time_seconds: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Configuration schemas
class ModelConfig(BaseModel):
    """Model configuration"""
    model_type: ModelType
    hyperparameters: dict[str, Any]
    embedding_dimension: int = 4
    prediction_method: PredictionMethod = PredictionMethod.DOT


class TrainingConfig(BaseModel):
    """Training configuration"""
    validation_ratio: float = 0.1
    test_ratio: float = 0.2
    auto_hyperparameter_tuning: bool = True
    max_training_time_minutes: int = 60
    early_stopping: bool = True
    save_intermediate_models: bool = False
