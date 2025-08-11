"""
TIE MCP Server - Model Context Protocol server for Technique Inference Engine

This package provides a comprehensive MCP server implementation for the 
Technique Inference Engine (TIE) project, enabling scalable inference
of MITRE ATT&CK techniques with model retraining capabilities.
"""

__version__ = "1.0.0"
__author__ = "Nidhi Trivedi"
__email__ = "nj20383@gmail.com"

from .config.settings import Settings

__all__ = ["Settings"]