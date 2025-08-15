"""
Core package initialization for TIE MCP.

Exposes key manager classes and subpackages used by tests and runtime.
"""

from .engine_manager import TIEEngineManager

__all__ = ["TIEEngineManager"]
