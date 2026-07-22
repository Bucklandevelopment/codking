"""
CodKing RLM Integration Module.

Provides integration between CodKing's efficient inference pipeline
and the RLM (Recursive Language Models) framework for infinite context processing.

Components:
- InfiniteContextManager: RAM + Disk hybrid storage for massive contexts
- CodKingClient: BaseLM implementation for RLM compatibility
- CodKingREPL: BaseEnv implementation with CodKing processing
- CodKingRLM: Unified interface combining all components
"""

from .context_manager import InfiniteContextManager
from .codking_client import CodKingClient
from .codking_repl import CodKingREPL
from .codking_rlm import CodKingRLM

__all__ = [
    'InfiniteContextManager',
    'CodKingClient',
    'CodKingREPL',
    'CodKingRLM',
]

__version__ = '0.1.0'
