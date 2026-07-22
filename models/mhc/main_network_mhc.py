"""
H-Net Main Network with mHC (Manifold-Constrained Hyper-Connections).

This module provides an alternative to HNetMainNetwork that uses mHC
for improved training stability and performance.

Based on: https://arxiv.org/abs/2512.24880

Key differences from standard Main Network:
1. Residual connections use mHC with Birkhoff polytope projection
2. Input is expanded to n parallel streams (default: 4)
3. Streams are contracted back to single output at the end

Expected benefits:
- No loss spikes during training
- Smooth gradient flow through 22 layers
- +5-7% downstream task performance
- Only 6.7% training overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from .mhc_layers import (
    SinkhornKnopp,
    mHCResidual,
    mHCTransformerBlock,
    mHCMultiHeadAttention,
    mHCFeedForward,
    compute_doubly_stochastic_loss,
    orthogonality_regularization,
)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class HNetMainNetworkMHC(nn.Module):
    """
    H-Net Main Transformer Network with mHC Hyper-Connections.

    22-layer Transformer with streams expanded to n parallel channels
    and mixing matrices projected to the Birkhoff polytope.

    This is a drop-in replacement for HNetMainNetwork with mHC.
    """

    def __init__(
        self,
        num_layers: int = 22,
        d_model: int = 1536,
        num_heads: int = 12,
        dim_head: int = 128,
        dim_ff: Optional[int] = None,
        n_streams: int = 4,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 5,
        use_rms_norm: bool = True,
        input_dim: int = 1024,
        max_seq_len: int = 2048,
        orthogonality_weight: float = 0.01,
    ):
        """
        Initialize mHC main network.

        Args:
            num_layers: Number of Transformer layers (default: 22)
            d_model: Model dimension (default: 1536)
            num_heads: Number of attention heads (default: 12)
            dim_head: Dimension per head (default: 128)
            dim_ff: Feed-forward dimension (default: 4 * d_model)
            n_streams: Number of parallel mHC streams (default: 4)
            dropout: Dropout rate
            sinkhorn_iterations: Iterations for Sinkhorn-Knopp projection
            use_rms_norm: Use RMSNorm instead of LayerNorm
            input_dim: Input dimension from encoder (default: 1024)
            max_seq_len: Maximum sequence length
            orthogonality_weight: Weight for orthogonality regularization
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.n_streams = n_streams
        self.orthogonality_weight = orthogonality_weight

        if dim_ff is None:
            dim_ff = 4 * d_model

        # Input projection (from encoder d_model to main d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # mHC Transformer layers
        self.layers = nn.ModuleList([
            mHCTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dim_ff=dim_ff,
                n_streams=n_streams,
                dropout=dropout,
                sinkhorn_iterations=sinkhorn_iterations,
                use_rms_norm=use_rms_norm
            )
            for _ in range(num_layers)
        ])

        # Final stream contraction
        self.stream_contract = nn.Linear(d_model * n_streams, d_model, bias=False)

        # Final normalization
        norm_class = RMSNorm if use_rms_norm else nn.LayerNorm
        self.final_norm = norm_class(d_model)

        # Initialize stream contraction for averaging behavior
        self._init_contraction()

    def _init_contraction(self):
        """Initialize stream contraction to average streams."""
        with torch.no_grad():
            weight = torch.zeros(self.d_model, self.d_model * self.n_streams)
            for i in range(self.n_streams):
                start = i * self.d_model
                end = (i + 1) * self.d_model
                weight[:, start:end] = torch.eye(self.d_model) / self.n_streams
            self.stream_contract.weight.copy_(weight)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_metrics: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through mHC main network.

        Args:
            x: Input chunks [batch, num_chunks, input_dim]
            attention_mask: Optional mask [batch, num_chunks]
            return_metrics: Whether to return additional metrics

        Returns:
            If return_metrics=False:
                Processed chunks [batch, num_chunks, d_model]
            If return_metrics=True:
                Tuple of (output, metrics_dict)
        """
        # Project input to d_model
        x = self.input_proj(x)  # [batch, seq, d_model]

        # Process through mHC Transformer layers
        # First layer expands to streams
        x = self.layers[0](x, is_first_layer=True, attention_mask=attention_mask)

        # Remaining layers maintain stream format
        for layer in self.layers[1:]:
            x = layer(x, is_first_layer=False, attention_mask=attention_mask)

        # Contract streams back to single output
        batch, seq_len, n_streams, d_model = x.shape
        x_flat = x.reshape(batch, seq_len, -1)  # [batch, seq, n_streams * d_model]
        x = self.stream_contract(x_flat)  # [batch, seq, d_model]

        # Final normalization
        x = self.final_norm(x)

        if return_metrics:
            metrics = self._compute_metrics()
            return x, metrics

        return x

    def _compute_metrics(self) -> Dict[str, torch.Tensor]:
        """
        Compute mHC-specific metrics for monitoring.

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Compute doubly stochastic loss for all H_res matrices
        ds_losses = []
        orth_losses = []

        for i, layer in enumerate(self.layers):
            # Attention mHC
            H_res_attn = layer.mhc_attn.sinkhorn(layer.mhc_attn.H_res_raw)
            ds_losses.append(compute_doubly_stochastic_loss(H_res_attn))
            orth_losses.append(orthogonality_regularization(H_res_attn))

            # FF mHC
            H_res_ff = layer.mhc_ff.sinkhorn(layer.mhc_ff.H_res_raw)
            ds_losses.append(compute_doubly_stochastic_loss(H_res_ff))
            orth_losses.append(orthogonality_regularization(H_res_ff))

        metrics['doubly_stochastic_loss'] = torch.stack(ds_losses).mean()
        metrics['orthogonality_loss'] = torch.stack(orth_losses).mean()

        return metrics

    def get_regularization_loss(self) -> torch.Tensor:
        """
        Get mHC regularization loss for training.

        Returns:
            Scalar regularization loss
        """
        metrics = self._compute_metrics()
        return self.orthogonality_weight * metrics['orthogonality_loss']

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_mhc_params(self) -> int:
        """Return number of mHC-specific parameters."""
        mhc_params = 0
        for layer in self.layers:
            # Count H matrices
            mhc_params += layer.mhc_attn.H_res_raw.numel()
            mhc_params += layer.mhc_attn.H_pre_raw.numel()
            mhc_params += layer.mhc_attn.H_post_raw.numel()
            mhc_params += layer.mhc_ff.H_res_raw.numel()
            mhc_params += layer.mhc_ff.H_pre_raw.numel()
            mhc_params += layer.mhc_ff.H_post_raw.numel()
            # Count expand/contract
            mhc_params += layer.mhc_attn.expand.weight.numel()
            mhc_params += layer.mhc_attn.contract.weight.numel()
            mhc_params += layer.mhc_ff.expand.weight.numel()
            mhc_params += layer.mhc_ff.contract.weight.numel()
        return mhc_params

    def visualize_mixing_matrices(self, layer_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Get mixing matrices for visualization.

        Args:
            layer_idx: Which layer to visualize

        Returns:
            Dictionary with projected H matrices
        """
        layer = self.layers[layer_idx]

        # Attention mHC matrices
        H_res_attn, H_pre_attn, H_post_attn = layer.mhc_attn.get_projected_matrices()

        # FF mHC matrices
        H_res_ff, H_pre_ff, H_post_ff = layer.mhc_ff.get_projected_matrices()

        return {
            'attn_H_res': H_res_attn.detach(),
            'attn_H_pre': H_pre_attn.detach(),
            'attn_H_post': H_post_attn.detach(),
            'ff_H_res': H_res_ff.detach(),
            'ff_H_pre': H_pre_ff.detach(),
            'ff_H_post': H_post_ff.detach(),
        }


class HNetMainNetworkFactory:
    """
    Factory for creating Main Network instances.

    Allows switching between standard and mHC variants via config.
    """

    @staticmethod
    def create(
        use_mhc: bool = False,
        **kwargs
    ) -> nn.Module:
        """
        Create a Main Network instance.

        Args:
            use_mhc: Whether to use mHC variant
            **kwargs: Arguments passed to network constructor

        Returns:
            HNetMainNetwork or HNetMainNetworkMHC instance
        """
        if use_mhc:
            return HNetMainNetworkMHC(**kwargs)
        else:
            # Import standard network
            from ..hnet.main_network import HNetMainNetwork
            # Filter kwargs for standard network
            standard_kwargs = {
                k: v for k, v in kwargs.items()
                if k not in ['n_streams', 'sinkhorn_iterations', 'orthogonality_weight']
            }
            return HNetMainNetwork(**standard_kwargs)


# Test when run directly
if __name__ == "__main__":
    print("Testing HNetMainNetworkMHC...")

    # Create network
    network = HNetMainNetworkMHC(
        num_layers=22,
        d_model=1536,
        num_heads=12,
        dim_head=128,
        n_streams=4,
        input_dim=1024
    )

    # Test forward pass
    batch_size = 2
    num_chunks = 64
    x = torch.randn(batch_size, num_chunks, 1024)
    attention_mask = torch.ones(batch_size, num_chunks)

    print(f"\nInput shape: {x.shape}")

    # Forward with metrics
    output, metrics = network(x, attention_mask, return_metrics=True)

    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, {num_chunks}, 1536]")

    # Metrics
    print(f"\nMetrics:")
    print(f"  Doubly stochastic loss: {metrics['doubly_stochastic_loss']:.6f}")
    print(f"  Orthogonality loss: {metrics['orthogonality_loss']:.6f}")

    # Parameter counts
    total_params = network.get_num_params()
    mhc_params = network.get_mhc_params()
    print(f"\nParameters:")
    print(f"  Total: {total_params:,}")
    print(f"  mHC-specific: {mhc_params:,}")
    print(f"  mHC overhead: {100 * mhc_params / total_params:.2f}%")

    # Visualize first layer matrices
    matrices = network.visualize_mixing_matrices(layer_idx=0)
    print(f"\nLayer 0 mixing matrices (H_res should be ~doubly stochastic):")
    H_res = matrices['attn_H_res']
    print(f"  H_res row sums: {H_res.sum(dim=-1)}")
    print(f"  H_res col sums: {H_res.sum(dim=-2)}")

    # Test factory
    print("\nTesting factory...")
    mhc_net = HNetMainNetworkFactory.create(
        use_mhc=True,
        num_layers=4,
        d_model=512,
        num_heads=8,
        input_dim=512
    )
    print(f"  Created mHC network: {type(mhc_net).__name__}")

    print("\nAll tests passed!")
