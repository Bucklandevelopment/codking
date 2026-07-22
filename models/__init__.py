"""
CodKing Models Package.

This package contains all neural network architectures for CodKing:
- H-Net: Hierarchical network with compression and processing
- HRM: Hierarchical Reasoning Model with adaptive computation
- mHC: Manifold-Constrained Hyper-Connections for improved training stability
- Integration: Combined pipeline for H-Net + HRM
"""

# H-Net exports
from .hnet.hnet import HNet
from .hnet.encoder import HNetEncoder
from .hnet.decoder import HNetDecoder
from .hnet.main_network import HNetMainNetwork
from .hnet.dynamic_chunking import DynamicChunkingModule

# HRM exports
from .hrm.hrm import HRM
from .hrm.modules import HModule, LModule

# mHC exports
from .mhc.mhc_layers import (
    SinkhornKnopp,
    mHCResidual,
    mHCTransformerBlock,
    mHCMultiHeadAttention,
    mHCFeedForward,
)
from .mhc.main_network_mhc import (
    HNetMainNetworkMHC,
    HNetMainNetworkFactory,
)

# Integration exports
from .integration.pipeline import CodKingPipeline
from .integration.interface import HNetHRMInterface

__all__ = [
    # H-Net
    "HNet",
    "HNetEncoder",
    "HNetDecoder",
    "HNetMainNetwork",
    "DynamicChunkingModule",
    # HRM
    "HRM",
    "HModule",
    "LModule",
    # mHC
    "SinkhornKnopp",
    "mHCResidual",
    "mHCTransformerBlock",
    "mHCMultiHeadAttention",
    "mHCFeedForward",
    "HNetMainNetworkMHC",
    "HNetMainNetworkFactory",
    # Integration
    "CodKingPipeline",
    "HNetHRMInterface",
]
