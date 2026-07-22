"""
Integration layer for HNet + HRM.

Provides:
- Chunk routing strategies (variance, temporal, learned)
- Complete pipeline (CodKingPipeline)
- Cybersecurity-specific pipeline (CybersecurityPipeline)
"""

from .interface import (
    VarianceRouter,
    TemporalRouter,
    LearnedRouter,
    HNetHRMInterface
)

from .pipeline import (
    CodKingPipeline,
    CybersecurityPipeline
)

__all__ = [
    'VarianceRouter',
    'TemporalRouter',
    'LearnedRouter',
    'HNetHRMInterface',
    'CodKingPipeline',
    'CybersecurityPipeline'
]
