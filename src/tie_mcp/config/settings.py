"""
Configuration settings for TIE MCP Server
"""

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    """Environment types"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


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


class DatabaseSettings(BaseSettings):
    """Database configuration"""

    url: str = Field(default="postgresql://tie:tie@localhost:5432/tie_mcp")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)
    pool_timeout: int = Field(default=30)
    pool_recycle: int = Field(default=3600)

    class Config:
        env_prefix = "DATABASE_"


class RedisSettings(BaseSettings):
    """Redis configuration"""

    url: str = Field(default="redis://localhost:6379")
    db: int = Field(default=0)
    max_connections: int = Field(default=50)
    retry_on_timeout: bool = Field(default=True)
    socket_keepalive: bool = Field(default=True)
    socket_keepalive_options: dict[str, Any] = Field(default_factory=dict)

    class Config:
        env_prefix = "REDIS_"


class CelerySettings(BaseSettings):
    """Celery configuration"""

    broker_url: str = Field(default="redis://localhost:6379/1")
    result_backend: str = Field(default="redis://localhost:6379/2")
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: list[str] = Field(default=["json"])
    timezone: str = Field(default="UTC")
    enable_utc: bool = Field(default=True)

    # Task routing
    task_routes: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "tie_mcp.tasks.training.*": {"queue": "training"},
            "tie_mcp.tasks.inference.*": {"queue": "inference"},
            "tie_mcp.tasks.data_processing.*": {"queue": "data_processing"},
        }
    )

    class Config:
        env_prefix = "CELERY_"


class MLFlowSettings(BaseSettings):
    """MLFlow configuration"""

    tracking_uri: str = Field(default="http://localhost:5000")
    experiment_name: str = Field(default="TIE_MCP_Experiments")
    registry_uri: str | None = Field(default=None)
    s3_endpoint_url: str | None = Field(default=None)
    artifact_root: str | None = Field(default=None)

    class Config:
        env_prefix = "MLFLOW_"


class MonitoringSettings(BaseSettings):
    """Monitoring configuration"""

    prometheus_port: int = Field(default=8001)
    health_check_interval: int = Field(default=30)
    metrics_retention_days: int = Field(default=30)
    alert_thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "inference_latency_p95": 5.0,  # seconds
            "error_rate": 5.0,  # percentage
        }
    )

    class Config:
        env_prefix = "MONITORING_"


class SecuritySettings(BaseSettings):
    """Security configuration"""

    secret_key: str = Field(default="your-secret-key-change-in-production")
    api_key_header: str = Field(default="X-API-Key")
    allowed_hosts: list[str] = Field(default=["*"])
    cors_origins: list[str] = Field(default=["*"])
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_hours: int = Field(default=24)
    rate_limit_requests: int = Field(default=1000)
    rate_limit_window: int = Field(default=3600)  # seconds

    class Config:
        env_prefix = "SECURITY_"


class ModelSettings(BaseSettings):
    """Model configuration"""

    default_model_type: ModelType = Field(default=ModelType.WALS)
    default_prediction_method: PredictionMethod = Field(default=PredictionMethod.DOT)
    default_embedding_dimension: int = Field(default=4)

    # Model paths
    models_directory: Path = Field(default=Path("data/models"))
    datasets_directory: Path = Field(default=Path("data/datasets"))
    configs_directory: Path = Field(default=Path("data/configs"))

    # Training parameters
    validation_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.2, ge=0.0, le=1.0)

    # WALS default hyperparameters
    wals_epochs: int = Field(default=25)
    wals_c_values: list[float] = Field(
        default=[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7]
    )
    wals_regularization_values: list[float] = Field(default=[0.0, 0.0001, 0.001, 0.01])

    # BPR default hyperparameters
    bpr_epochs: int = Field(default=20)
    bpr_learning_rates: list[float] = Field(default=[0.00001, 0.00005, 0.0001, 0.001])
    bpr_regularization_values: list[float] = Field(default=[0.0, 0.0001, 0.001, 0.01])

    # Model versioning
    max_model_versions: int = Field(default=10)
    auto_cleanup_models: bool = Field(default=True)

    @validator("validation_ratio", "test_ratio")
    def validate_ratios(cls, v, values):
        if "validation_ratio" in values:
            if v + values["validation_ratio"] > 1.0:
                raise ValueError("validation_ratio + test_ratio must be <= 1.0")
        return v

    class Config:
        env_prefix = "MODEL_"


class Settings(BaseSettings):
    """Main application settings"""

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)
    log_level: LogLevel = Field(default=LogLevel.INFO)

    # Server
    host: str = Field(default="127.0.0.1")  # Default to localhost for security
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    reload: bool = Field(default=True)

    # Application
    app_name: str = Field(default="TIE MCP Server")
    app_version: str = Field(default="1.0.0")
    api_v1_prefix: str = Field(default="/api/v1")

    # Enterprise ATT&CK data
    enterprise_attack_filepath: str = Field(
        default="data/datasets/stix/enterprise-attack.json"
    )
    default_dataset_filepath: str = Field(
        default="data/datasets/combined_dataset_full_frequency.json"
    )

    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    mlflow: MLFlowSettings = Field(default_factory=MLFlowSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    model: ModelSettings = Field(default_factory=ModelSettings)

    @validator("environment", pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    @validator("log_level", pre=True)
    def validate_log_level(cls, v):
        if isinstance(v, str):
            return LogLevel(v.upper())
        return v

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == Environment.TESTING

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
