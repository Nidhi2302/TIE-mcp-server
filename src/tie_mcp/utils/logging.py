"""
Logging utilities for TIE MCP Server
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from pythonjsonlogger import jsonlogger

from ..config.settings import settings


def setup_logging():
    """Setup structured logging for the application"""

    # Clear any existing handlers
    logging.getLogger().handlers.clear()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.value))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.log_level.value))

    if settings.is_development():
        # Human-readable format for development
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    else:
        # JSON format for production
        formatter = jsonlogger.JsonFormatter(
            "%(asctime)s %(name)s %(levelname)s %(message)s"
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler for production
    if settings.is_production():
        log_file = Path("logs") / "tie_mcp.log"
        log_file.parent.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set third-party library log levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)
    logging.getLogger("celery").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured successfully",
        log_level=settings.log_level.value,
        environment=settings.environment.value,
    )


class ContextualLogger:
    """Logger with contextual information"""

    def __init__(self, name: str, context: dict[str, Any] | None = None):
        self.logger = structlog.get_logger(name)
        self.context = context or {}

    def bind(self, **kwargs) -> "ContextualLogger":
        """Bind additional context to the logger"""
        new_context = {**self.context, **kwargs}
        return ContextualLogger(self.logger.name, new_context)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **{**self.context, **kwargs})

    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **{**self.context, **kwargs})

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **{**self.context, **kwargs})

    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **{**self.context, **kwargs})

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **{**self.context, **kwargs})

    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, **{**self.context, **kwargs})


class RequestLogger:
    """Logger for HTTP requests with correlation IDs"""

    def __init__(self, correlation_id: str):
        self.correlation_id = correlation_id
        self.logger = ContextualLogger("request", {"correlation_id": correlation_id})
        self.start_time = datetime.utcnow()

    def log_request(self, method: str, path: str, **kwargs):
        """Log incoming request"""
        self.logger.info("Request started", method=method, path=path, **kwargs)

    def log_response(self, status_code: int, **kwargs):
        """Log outgoing response"""
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        self.logger.info(
            "Request completed",
            status_code=status_code,
            duration_seconds=duration,
            **kwargs,
        )

    def log_error(self, error: Exception, **kwargs):
        """Log request error"""
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        self.logger.error(
            "Request failed",
            error=str(error),
            error_type=type(error).__name__,
            duration_seconds=duration,
            **kwargs,
        )


class AuditLogger:
    """Logger for audit events"""

    def __init__(self):
        self.logger = ContextualLogger("audit")

    def log_model_training_started(
        self, model_id: str, model_type: str, user_id: str | None = None
    ):
        """Log model training start"""
        self.logger.info(
            "Model training started",
            event_type="model_training_started",
            model_id=model_id,
            model_type=model_type,
            user_id=user_id,
        )

    def log_model_training_completed(
        self,
        model_id: str,
        model_type: str,
        metrics: dict[str, float],
        user_id: str | None = None,
    ):
        """Log model training completion"""
        self.logger.info(
            "Model training completed",
            event_type="model_training_completed",
            model_id=model_id,
            model_type=model_type,
            metrics=metrics,
            user_id=user_id,
        )

    def log_model_training_failed(
        self, model_type: str, error: str, user_id: str | None = None
    ):
        """Log model training failure"""
        self.logger.error(
            "Model training failed",
            event_type="model_training_failed",
            model_type=model_type,
            error=error,
            user_id=user_id,
        )

    def log_prediction_request(
        self,
        model_id: str,
        input_techniques: list,
        prediction_count: int,
        user_id: str | None = None,
    ):
        """Log prediction request"""
        self.logger.info(
            "Prediction request",
            event_type="prediction_request",
            model_id=model_id,
            input_techniques_count=len(input_techniques),
            prediction_count=prediction_count,
            user_id=user_id,
        )

    def log_model_deleted(
        self, model_id: str, model_name: str, user_id: str | None = None
    ):
        """Log model deletion"""
        self.logger.info(
            "Model deleted",
            event_type="model_deleted",
            model_id=model_id,
            model_name=model_name,
            user_id=user_id,
        )

    def log_dataset_created(
        self,
        dataset_id: str,
        dataset_name: str,
        num_reports: int,
        user_id: str | None = None,
    ):
        """Log dataset creation"""
        self.logger.info(
            "Dataset created",
            event_type="dataset_created",
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            num_reports=num_reports,
            user_id=user_id,
        )


class PerformanceLogger:
    """Logger for performance metrics"""

    def __init__(self):
        self.logger = ContextualLogger("performance")

    def log_operation_duration(self, operation: str, duration_seconds: float, **kwargs):
        """Log operation duration"""
        self.logger.info(
            f"{operation} duration",
            operation=operation,
            duration_seconds=duration_seconds,
            **kwargs,
        )

    def log_memory_usage(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage"""
        self.logger.info(
            f"{operation} memory usage",
            operation=operation,
            memory_mb=memory_mb,
            **kwargs,
        )

    def log_model_size(self, model_id: str, size_mb: float):
        """Log model file size"""
        self.logger.info("Model size", model_id=model_id, size_mb=size_mb)

    def log_dataset_size(self, dataset_id: str, size_mb: float, num_reports: int):
        """Log dataset size"""
        self.logger.info(
            "Dataset size",
            dataset_id=dataset_id,
            size_mb=size_mb,
            num_reports=num_reports,
        )


class SecurityLogger:
    """Logger for security events"""

    def __init__(self):
        self.logger = ContextualLogger("security")

    def log_authentication_attempt(
        self, user_id: str, success: bool, ip_address: str | None = None
    ):
        """Log authentication attempt"""
        self.logger.info(
            "Authentication attempt",
            event_type="authentication_attempt",
            user_id=user_id,
            success=success,
            ip_address=ip_address,
        )

    def log_authorization_failure(
        self, user_id: str, resource: str, action: str, ip_address: str | None = None
    ):
        """Log authorization failure"""
        self.logger.warning(
            "Authorization failure",
            event_type="authorization_failure",
            user_id=user_id,
            resource=resource,
            action=action,
            ip_address=ip_address,
        )

    def log_rate_limit_exceeded(
        self, user_id: str, endpoint: str, ip_address: str | None = None
    ):
        """Log rate limit exceeded"""
        self.logger.warning(
            "Rate limit exceeded",
            event_type="rate_limit_exceeded",
            user_id=user_id,
            endpoint=endpoint,
            ip_address=ip_address,
        )

    def log_suspicious_activity(
        self,
        user_id: str,
        activity: str,
        details: dict[str, Any],
        ip_address: str | None = None,
    ):
        """Log suspicious activity"""
        self.logger.warning(
            "Suspicious activity detected",
            event_type="suspicious_activity",
            user_id=user_id,
            activity=activity,
            details=details,
            ip_address=ip_address,
        )


# Global logger instances
audit_logger = AuditLogger()
performance_logger = PerformanceLogger()
security_logger = SecurityLogger()


def get_logger(name: str, context: dict[str, Any] | None = None) -> ContextualLogger:
    """Get a contextual logger instance"""
    return ContextualLogger(name, context)


def get_request_logger(correlation_id: str) -> RequestLogger:
    """Get a request logger instance"""
    return RequestLogger(correlation_id)
