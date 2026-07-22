"""
mHC: Manifold-Constrained Hyper-Connections for CodKing.

This module implements mHC from DeepSeek's paper (arXiv:2512.24880).
mHC projects residual mixing matrices onto the Birkhoff polytope
(doubly stochastic matrices) using differentiable Sinkhorn-Knopp iteration.

Key Components:
- SinkhornKnopp: Differentiable projection to Birkhoff polytope
- mHCResidual: Residual block with manifold-constrained hyper-connections
- mHCTransformerBlock: Full Transformer block with mHC
- mHCFeedForward: Feed-forward with mHC residual (optional)

Benefits:
- Training stability (no loss spikes)
- Smooth gradient flow
- +5-7% downstream performance
- Only 6.7% training overhead

Reference: https://arxiv.org/abs/2512.24880
"""

from .mhc_layers import (
    SinkhornKnopp,
    mHCResidual,
    mHCTransformerBlock,
    mHCMultiHeadAttention,
    mHCFeedForward,
)

__all__ = [
    "SinkhornKnopp",
    "mHCResidual",
    "mHCTransformerBlock",
    "mHCMultiHeadAttention",
    "mHCFeedForward",
]
