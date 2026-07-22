"""
Inference API for CodKing.

Provides:
- FastAPI server for REST API
- Python client for easy integration
- Threat detection endpoints
- Batch processing support
"""

from .inference_server import app, ModelManager, run_server
from .client import (
    CodKingClient,
    ThreatDetectionResult,
    BatchThreatDetectionResult,
    ClassificationResult
)

__all__ = [
    'app',
    'ModelManager',
    'run_server',
    'CodKingClient',
    'ThreatDetectionResult',
    'BatchThreatDetectionResult',
    'ClassificationResult'
]
