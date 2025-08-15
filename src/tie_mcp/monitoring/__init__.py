"""
Monitoring package initialization.

Exports:
- MetricsCollector: Primary metrics collection/aggregation class.
- metrics_collector: Global default collector instance.
"""

from .metrics import MetricsCollector, metrics_collector

__all__ = ["MetricsCollector", "metrics_collector"]
