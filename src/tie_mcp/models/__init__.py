"""
Model management package initialization.

Exports:
- ModelManager: In-memory model registry used by engine and tests.
"""

from .model_manager import ModelManager

__all__ = ["ModelManager"]