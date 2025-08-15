"""
Database manager for TIE MCP Server
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.sql import func

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models"""

    pass


class Model(Base):
    """Database model for storing TIE models"""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="training")
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    metrics: Mapped[dict[str, float]] = mapped_column(JSON, nullable=True)
    dataset_path: Mapped[str] = mapped_column(String(500), nullable=True)
    dataset_info: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    artifacts_path: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(20), nullable=False, default="1.0")
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_models_status", "status"),
        Index("idx_models_model_type", "model_type"),
        Index("idx_models_created_at", "created_at"),
        Index("idx_models_is_default", "is_default"),
    )


class Dataset(Base):
    """Database model for storing dataset information.

    NOTE: Attribute name 'metadata' is reserved by SQLAlchemy's Declarative API.
    We use the attribute 'extra_metadata' mapped to column name 'metadata' to avoid
    a naming collision while preserving the DB column name and external dictionary key.
    """

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=True)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    num_reports: Mapped[int] = mapped_column(Integer, nullable=False)
    num_techniques: Mapped[int] = mapped_column(Integer, nullable=False)
    extra_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("idx_datasets_name", "name"),
        Index("idx_datasets_created_at", "created_at"),
    )


class PredictionLog(Base):
    """Database model for logging predictions"""

    __tablename__ = "prediction_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(String(36), nullable=False)
    input_techniques: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    predicted_techniques: Mapped[list[dict[str, Any]]] = mapped_column(
        JSON, nullable=False
    )
    prediction_method: Mapped[str] = mapped_column(String(20), nullable=False)
    execution_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    user_id: Mapped[str] = mapped_column(String(100), nullable=True)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_prediction_logs_model_id", "model_id"),
        Index("idx_prediction_logs_created_at", "created_at"),
        Index("idx_prediction_logs_user_id", "user_id"),
    )


class TrainingLog(Base):
    """Database model for logging training jobs"""

    __tablename__ = "training_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_id: Mapped[str] = mapped_column(
        String(36), nullable=True
    )  # Null if training failed
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    dataset_path: Mapped[str] = mapped_column(String(500), nullable=False)
    hyperparameters: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    final_metrics: Mapped[dict[str, float]] = mapped_column(JSON, nullable=True)
    training_time_seconds: Mapped[float] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # success, failure, running
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    __table_args__ = (
        Index("idx_training_logs_model_id", "model_id"),
        Index("idx_training_logs_status", "status"),
        Index("idx_training_logs_started_at", "started_at"),
        Index("idx_training_logs_user_id", "user_id"),
    )


class AuditLog(Base):
    """Database model for audit logging"""

    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    component: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=True)
    resource_id: Mapped[str] = mapped_column(String(36), nullable=True)
    user_id: Mapped[str] = mapped_column(String(100), nullable=True)
    ip_address: Mapped[str] = mapped_column(String(45), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("idx_audit_logs_event_type", "event_type"),
        Index("idx_audit_logs_component", "component"),
        Index("idx_audit_logs_resource_type", "resource_type"),
        Index("idx_audit_logs_user_id", "user_id"),
        Index("idx_audit_logs_created_at", "created_at"),
    )


class DatabaseManager:
    """Manages database connections and operations"""

    def __init__(self):
        self.engine = None
        self.async_session_maker = None
        self._initialized = False

    async def initialize(self):
        """Initialize database connection and create tables"""
        if self._initialized:
            return

        try:
            logger.info("Initializing database connection")

            # Create async engine
            self.engine = create_async_engine(
                settings.database.url,
                pool_size=settings.database.pool_size,
                max_overflow=settings.database.max_overflow,
                pool_timeout=settings.database.pool_timeout,
                pool_recycle=settings.database.pool_recycle,
                echo=settings.is_development(),  # Log SQL in development
            )

            # Create session maker
            self.async_session_maker = async_sessionmaker(
                bind=self.engine, class_=AsyncSession, expire_on_commit=False
            )

            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._initialized = True
            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error("Failed to initialize database", error=str(e))
            raise

    async def cleanup(self):
        """Cleanup database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")

    @asynccontextmanager
    async def get_session(self):
        """Get database session context manager"""
        if not self._initialized:
            await self.initialize()

        async with self.async_session_maker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    # Model operations
    async def save_model(self, model_data: dict[str, Any]) -> str:
        """Save model metadata to database"""
        try:
            async with self.get_session() as session:
                model = Model(**model_data)
                session.add(model)
                await session.commit()
                await session.refresh(model)

                logger.info("Model saved to database", model_id=model.id)
                return model.id

        except Exception as e:
            logger.error("Error saving model to database", error=str(e))
            raise

    async def get_model(self, model_id: str) -> dict[str, Any] | None:
        """Get model by ID"""
        try:
            async with self.get_session() as session:
                model = await session.get(Model, model_id)
                if model:
                    return self._model_to_dict(model)
                return None

        except Exception as e:
            logger.error(
                "Error getting model from database", model_id=model_id, error=str(e)
            )
            raise

    async def list_models(self, status: str | None = None) -> list[dict[str, Any]]:
        """List all models, optionally filtered by status"""
        try:
            async with self.get_session() as session:
                query = session.query(Model)
                if status:
                    query = query.filter(Model.status == status)

                result = await query.all()
                return [self._model_to_dict(model) for model in result]

        except Exception as e:
            logger.error("Error listing models from database", error=str(e))
            raise

    async def update_model(self, model_id: str, updates: dict[str, Any]):
        """Update model metadata"""
        try:
            async with self.get_session() as session:
                model = await session.get(Model, model_id)
                if model:
                    for key, value in updates.items():
                        setattr(model, key, value)
                    model.updated_at = datetime.utcnow()
                    await session.commit()
                    logger.info("Model updated in database", model_id=model_id)
                else:
                    raise ValueError(f"Model {model_id} not found")

        except Exception as e:
            logger.error(
                "Error updating model in database", model_id=model_id, error=str(e)
            )
            raise

    async def delete_model(self, model_id: str):
        """Delete model from database"""
        try:
            async with self.get_session() as session:
                model = await session.get(Model, model_id)
                if model:
                    await session.delete(model)
                    await session.commit()
                    logger.info("Model deleted from database", model_id=model_id)
                else:
                    raise ValueError(f"Model {model_id} not found")

        except Exception as e:
            logger.error(
                "Error deleting model from database", model_id=model_id, error=str(e)
            )
            raise

    async def set_default_model(self, model_id: str):
        """Set a model as the default"""
        try:
            async with self.get_session() as session:
                # Unset current default
                current_default = (
                    await session.query(Model).filter(Model.is_default).first()
                )
                if current_default:
                    current_default.is_default = False

                # Set new default
                model = await session.get(Model, model_id)
                if model:
                    model.is_default = True
                    await session.commit()
                    logger.info("Default model set", model_id=model_id)
                else:
                    raise ValueError(f"Model {model_id} not found")

        except Exception as e:
            logger.error("Error setting default model", model_id=model_id, error=str(e))
            raise

    async def get_default_model(self) -> dict[str, Any] | None:
        """Get the default model"""
        try:
            async with self.get_session() as session:
                model = await session.query(Model).filter(Model.is_default).first()
                if model:
                    return self._model_to_dict(model)
                return None

        except Exception as e:
            logger.error("Error getting default model", error=str(e))
            raise

    # Dataset operations
    async def save_dataset(self, dataset_data: dict[str, Any]) -> str:
        """Save dataset metadata to database"""
        try:
            async with self.get_session() as session:
                dataset = Dataset(**dataset_data)
                session.add(dataset)
                await session.commit()
                await session.refresh(dataset)

                logger.info("Dataset saved to database", dataset_id=dataset.id)
                return dataset.id

        except Exception as e:
            logger.error("Error saving dataset to database", error=str(e))
            raise

    async def list_datasets(self) -> list[dict[str, Any]]:
        """List all datasets"""
        try:
            async with self.get_session() as session:
                result = await session.query(Dataset).all()
                return [self._dataset_to_dict(dataset) for dataset in result]

        except Exception as e:
            logger.error("Error listing datasets from database", error=str(e))
            raise

    # Logging operations
    async def log_prediction(self, prediction_data: dict[str, Any]):
        """Log a prediction request"""
        try:
            async with self.get_session() as session:
                log = PredictionLog(**prediction_data)
                session.add(log)
                await session.commit()

        except Exception as e:
            logger.error("Error logging prediction", error=str(e))

    async def log_training(self, training_data: dict[str, Any]):
        """Log a training job"""
        try:
            async with self.get_session() as session:
                log = TrainingLog(**training_data)
                session.add(log)
                await session.commit()

        except Exception as e:
            logger.error("Error logging training", error=str(e))

    async def log_audit_event(self, audit_data: dict[str, Any]):
        """Log an audit event"""
        try:
            async with self.get_session() as session:
                log = AuditLog(**audit_data)
                session.add(log)
                await session.commit()

        except Exception as e:
            logger.error("Error logging audit event", error=str(e))

    # Utility methods
    def _model_to_dict(self, model: Model) -> dict[str, Any]:
        """Convert model ORM object to dictionary"""
        return {
            "id": model.id,
            "name": model.name,
            "model_type": model.model_type,
            "status": model.status,
            "hyperparameters": model.hyperparameters,
            "metrics": model.metrics,
            "dataset_path": model.dataset_path,
            "dataset_info": model.dataset_info,
            "artifacts_path": model.artifacts_path,
            "description": model.description,
            "version": model.version,
            "is_default": model.is_default,
            "created_at": model.created_at,
            "updated_at": model.updated_at,
        }

    def _dataset_to_dict(self, dataset: Dataset) -> dict[str, Any]:
        """Convert dataset ORM object to dictionary"""
        return {
            "id": dataset.id,
            "name": dataset.name,
            "description": dataset.description,
            "file_path": dataset.file_path,
            "num_reports": dataset.num_reports,
            "num_techniques": dataset.num_techniques,
            "metadata": dataset.extra_metadata,
            "created_at": dataset.created_at,
            "updated_at": dataset.updated_at,
        }

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.get_session() as session:
                await session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False
