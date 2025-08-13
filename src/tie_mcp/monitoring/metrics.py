"""
Metrics collection and monitoring for TIE MCP Server
"""

import asyncio
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from ..config.settings import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MetricData:
    """Container for metric data"""
    value: float
    timestamp: datetime
    labels: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and manages application metrics"""

    def __init__(self, registry: CollectorRegistry | None = None):
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()

        # Initialize Prometheus metrics
        self._init_prometheus_metrics()

        # Internal metrics storage
        self._prediction_metrics = deque(maxlen=10000)
        self._training_metrics = deque(maxlen=1000)
        self._system_metrics = deque(maxlen=1000)
        self._error_metrics = deque(maxlen=5000)

        # Counters for aggregation
        self._prediction_count = 0
        self._training_count = 0
        self._error_count = 0

        # Start background tasks
        self._start_background_tasks()

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        # Prediction metrics
        self.prediction_counter = Counter(
            'tie_predictions_total',
            'Total number of predictions made',
            ['model_id', 'status'],
            registry=self.registry
        )

        self.prediction_duration = Histogram(
            'tie_prediction_duration_seconds',
            'Time spent on predictions',
            ['model_id'],
            registry=self.registry,
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        # Training metrics
        self.training_counter = Counter(
            'tie_training_total',
            'Total number of training jobs',
            ['model_type', 'status'],
            registry=self.registry
        )

        self.training_duration = Histogram(
            'tie_training_duration_seconds',
            'Time spent on training',
            ['model_type'],
            registry=self.registry,
            buckets=[60, 300, 600, 1800, 3600, 7200, 14400]
        )

        # Model metrics
        self.model_count = Gauge(
            'tie_models_total',
            'Total number of models',
            ['status'],
            registry=self.registry
        )

        self.model_size_bytes = Gauge(
            'tie_model_size_bytes',
            'Size of model files in bytes',
            ['model_id'],
            registry=self.registry
        )

        # System metrics
        self.cpu_usage = Gauge(
            'tie_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )

        self.memory_usage = Gauge(
            'tie_memory_usage_percent',
            'Memory usage percentage',
            registry=self.registry
        )

        self.disk_usage = Gauge(
            'tie_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )

        # Error metrics
        self.error_counter = Counter(
            'tie_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )

        # HTTP metrics (if using HTTP API)
        self.http_requests = Counter(
            'tie_http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )

        self.http_duration = Histogram(
            'tie_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )

        # Application info
        self.app_info = Info(
            'tie_app_info',
            'Application information',
            registry=self.registry
        )
        self.app_info.info({
            'version': settings.app_version,
            'environment': settings.environment.value
        })

    def _start_background_tasks(self):
        """Start background tasks for metrics collection"""
        def run_system_metrics_collection():
            asyncio.run(self._collect_system_metrics_loop())

        # Start system metrics collection in background thread
        metrics_thread = threading.Thread(
            target=run_system_metrics_collection,
            daemon=True,
            name="metrics_collector"
        )
        metrics_thread.start()

    async def _collect_system_metrics_loop(self):
        """Background loop to collect system metrics"""
        while True:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(settings.monitoring.health_check_interval)
            except Exception as e:
                logger.error("Error collecting system metrics", error=str(e))
                await asyncio.sleep(5)  # Short sleep on error

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_usage.set(memory_percent)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage.set(disk_percent)

            # Store internal metrics
            timestamp = datetime.utcnow()
            with self._lock:
                self._system_metrics.append(MetricData(
                    value=cpu_percent,
                    timestamp=timestamp,
                    labels={'metric': 'cpu_usage'}
                ))
                self._system_metrics.append(MetricData(
                    value=memory_percent,
                    timestamp=timestamp,
                    labels={'metric': 'memory_usage'}
                ))
                self._system_metrics.append(MetricData(
                    value=disk_percent,
                    timestamp=timestamp,
                    labels={'metric': 'disk_usage'}
                ))

            logger.debug("System metrics collected",
                        cpu_percent=cpu_percent,
                        memory_percent=memory_percent,
                        disk_percent=disk_percent)

        except Exception as e:
            logger.error("Error collecting system metrics", error=str(e))

    async def record_prediction(self, duration: float, input_techniques_count: int,
                              output_techniques_count: int, model_id: str,
                              status: str = "success"):
        """Record prediction metrics"""
        try:
            # Update Prometheus metrics
            self.prediction_counter.labels(model_id=model_id, status=status).inc()
            self.prediction_duration.labels(model_id=model_id).observe(duration)

            # Store internal metrics
            timestamp = datetime.utcnow()
            with self._lock:
                self._prediction_metrics.append(MetricData(
                    value=duration,
                    timestamp=timestamp,
                    labels={
                        'model_id': model_id,
                        'status': status,
                        'input_count': str(input_techniques_count),
                        'output_count': str(output_techniques_count)
                    }
                ))
                self._prediction_count += 1

            logger.debug("Prediction metrics recorded",
                        duration=duration,
                        model_id=model_id,
                        status=status)

        except Exception as e:
            logger.error("Error recording prediction metrics", error=str(e))

    async def record_training(self, duration: float, model_type: str,
                            dataset_size: int, success: bool):
        """Record training metrics"""
        try:
            status = "success" if success else "failure"

            # Update Prometheus metrics
            self.training_counter.labels(model_type=model_type, status=status).inc()
            if success:
                self.training_duration.labels(model_type=model_type).observe(duration)

            # Store internal metrics
            timestamp = datetime.utcnow()
            with self._lock:
                self._training_metrics.append(MetricData(
                    value=duration,
                    timestamp=timestamp,
                    labels={
                        'model_type': model_type,
                        'status': status,
                        'dataset_size': str(dataset_size)
                    }
                ))
                self._training_count += 1

            logger.debug("Training metrics recorded",
                        duration=duration,
                        model_type=model_type,
                        success=success)

        except Exception as e:
            logger.error("Error recording training metrics", error=str(e))

    async def record_error(
        self, component: str, error_type: str, error_message: str = ""
    ):
        """Record error metrics"""
        try:
            # Update Prometheus metrics
            self.error_counter.labels(error_type=error_type, component=component).inc()

            # Store internal metrics
            timestamp = datetime.utcnow()
            with self._lock:
                self._error_metrics.append(MetricData(
                    value=1.0,
                    timestamp=timestamp,
                    labels={
                        'component': component,
                        'error_type': error_type,
                        'error_message': error_message[:100]  # Truncate long messages
                    }
                ))
                self._error_count += 1

            logger.debug("Error metrics recorded",
                        component=component,
                        error_type=error_type)

        except Exception as e:
            logger.error("Error recording error metrics", error=str(e))

    async def record_model_saved(self, model_id: str, model_type: str):
        """Record model save event"""
        try:
            self.model_count.labels(status="active").inc()
            logger.debug(
                "Model save recorded", model_id=model_id, model_type=model_type
            )
        except Exception as e:
            logger.error("Error recording model save", error=str(e))

    async def record_model_loaded(self, model_id: str):
        """Record model load event"""
        try:
            # This could be extended to track load frequency
            logger.debug("Model load recorded", model_id=model_id)
        except Exception as e:
            logger.error("Error recording model load", error=str(e))

    async def record_model_deleted(self, model_id: str):
        """Record model deletion event"""
        try:
            self.model_count.labels(status="active").dec()
            logger.debug("Model deletion recorded", model_id=model_id)
        except Exception as e:
            logger.error("Error recording model deletion", error=str(e))

    async def record_http_request(self, method: str, endpoint: str,
                                status_code: int, duration: float):
        """Record HTTP request metrics"""
        try:
            self.http_requests.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code)
            ).inc()

            self.http_duration.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            logger.debug("HTTP request recorded",
                        method=method,
                        endpoint=endpoint,
                        status_code=status_code,
                        duration=duration)

        except Exception as e:
            logger.error("Error recording HTTP request", error=str(e))

    async def get_system_metrics(self) -> dict[str, Any]:
        """Get current system metrics"""
        try:
            # Get latest system metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            with self._lock:
                prediction_count = self._prediction_count
                training_count = self._training_count

            # Calculate average prediction time (last 100 predictions)
            recent_predictions = list(self._prediction_metrics)[-100:]
            avg_prediction_time = (
                sum(m.value for m in recent_predictions) / len(recent_predictions)
                if recent_predictions else 0.0
            )

            # Calculate average training time (last 10 trainings)
            recent_trainings = list(self._training_metrics)[-10:]
            avg_training_time = (
                sum(m.value for m in recent_trainings) / len(recent_trainings)
                if recent_trainings else 0.0
            )

            # Calculate error rate (last hour)
            one_hour_ago = datetime.utcnow() - timedelta(hours=1)
            recent_errors = [
                m for m in self._error_metrics
                if m.timestamp > one_hour_ago
            ]
            recent_predictions_hour = [
                m for m in self._prediction_metrics
                if m.timestamp > one_hour_ago
            ]

            error_rate = (
                len(recent_errors) / len(recent_predictions_hour) * 100
                if recent_predictions_hour else 0.0
            )

            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": (disk.used / disk.total) * 100,
                "active_models": 0,  # Would need to query model manager
                "total_predictions": prediction_count,
                "total_trainings": training_count,
                "average_prediction_time": avg_prediction_time,
                "average_training_time": avg_training_time,
                "error_rate_percent": error_rate,
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Error getting system metrics", error=str(e))
            return {}

    async def get_prediction_metrics(
        self, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get prediction metrics since a specific time"""
        try:
            since = since or (datetime.utcnow() - timedelta(hours=24))

            with self._lock:
                metrics = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels
                    }
                    for m in self._prediction_metrics
                    if m.timestamp > since
                ]

            return metrics

        except Exception as e:
            logger.error("Error getting prediction metrics", error=str(e))
            return []

    async def get_training_metrics(
        self, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get training metrics since a specific time"""
        try:
            since = since or (datetime.utcnow() - timedelta(days=7))

            with self._lock:
                metrics = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels
                    }
                    for m in self._training_metrics
                    if m.timestamp > since
                ]

            return metrics

        except Exception as e:
            logger.error("Error getting training metrics", error=str(e))
            return []

    async def get_error_metrics(
        self, since: datetime | None = None
    ) -> list[dict[str, Any]]:
        """Get error metrics since a specific time"""
        try:
            since = since or (datetime.utcnow() - timedelta(hours=24))

            with self._lock:
                metrics = [
                    {
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "labels": m.labels
                    }
                    for m in self._error_metrics
                    if m.timestamp > since
                ]

            return metrics

        except Exception as e:
            logger.error("Error getting error metrics", error=str(e))
            return []

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        try:
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error("Error generating Prometheus metrics", error=str(e))
            return ""

    async def check_alert_thresholds(self) -> list[dict[str, Any]]:
        """Check if any metrics exceed alert thresholds"""
        alerts = []

        try:
            system_metrics = await self.get_system_metrics()
            thresholds = settings.monitoring.alert_thresholds

            # Check CPU usage
            if system_metrics.get("cpu_usage_percent", 0) > thresholds["cpu_usage"]:
                alerts.append({
                    "metric": "cpu_usage",
                    "current_value": system_metrics["cpu_usage_percent"],
                    "threshold": thresholds["cpu_usage"],
                    "severity": "warning"
                })

            # Check memory usage
            if (
                system_metrics.get("memory_usage_percent", 0)
                > thresholds["memory_usage"]
            ):
                alerts.append({
                    "metric": "memory_usage",
                    "current_value": system_metrics["memory_usage_percent"],
                    "threshold": thresholds["memory_usage"],
                    "severity": "warning"
                })

            # Check error rate
            if system_metrics.get("error_rate_percent", 0) > thresholds["error_rate"]:
                alerts.append({
                    "metric": "error_rate",
                    "current_value": system_metrics["error_rate_percent"],
                    "threshold": thresholds["error_rate"],
                    "severity": "critical"
                })

            # Check inference latency
            if (
                system_metrics.get("average_prediction_time", 0)
                > thresholds["inference_latency_p95"]
            ):
                alerts.append({
                    "metric": "inference_latency",
                    "current_value": system_metrics["average_prediction_time"],
                    "threshold": thresholds["inference_latency_p95"],
                    "severity": "warning"
                })

        except Exception as e:
            logger.error("Error checking alert thresholds", error=str(e))

        return alerts


# Global metrics collector instance
metrics_collector = MetricsCollector()
