"""
mHC: Manifold-Constrained Hyper-Connections for CodKing.

Based on: https://arxiv.org/abs/2512.24880
Authors: DeepSeek-AI (Zhenda Xie et al.)

This module implements mHC which projects residual mixing matrices onto the
Birkhoff polytope (doubly stochastic matrices) using Sinkhorn-Knopp iteration.

Key insight: Standard Hyper-Connections (HC) break identity mapping property
that stabilizes ResNets. mHC restores this by constraining mixing matrices.

Update equation:
    x_{l+1} = H_l^{res} @ x_l + H_l^{post}.T @ F(H_l^{pre} @ x_l)

Where H^{res} is doubly stochastic (projected via Sinkhorn-Knopp).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class SinkhornKnopp(nn.Module):
    """
    Differentiable projection to the Birkhoff polytope.

    The Birkhoff polytope is the set of doubly stochastic matrices:
    - All entries >= 0
    - Each row sums to 1
    - Each column sums to 1

    Uses alternating row/column normalization (Sinkhorn-Knopp algorithm).
    """

    def __init__(
        self,
        iterations: int = 5,
        eps: float = 1e-8,
        tau: float = 1.0
    ):
        """
        Initialize Sinkhorn-Knopp projector.

        Args:
            iterations: Number of Sinkhorn iterations (default: 5, paper default)
            eps: Small constant for numerical stability
            tau: Temperature parameter for softmax (lower = sharper)
        """
        super().__init__()
        self.iterations = iterations
        self.eps = eps
        self.tau = tau

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Project matrix A to the Birkhoff polytope.

        Args:
            A: Input matrix [n_streams, n_streams] or [batch, n, n]

        Returns:
            Doubly stochastic matrix with same shape
        """
        # Apply temperature scaling and exponentiate for non-negativity
        A = torch.exp(A / self.tau)

        # Sinkhorn-Knopp iterations
        for _ in range(self.iterations):
            # Row normalization: each row sums to 1
            A = A / (A.sum(dim=-1, keepdim=True) + self.eps)
            # Column normalization: each column sums to 1
            A = A / (A.sum(dim=-2, keepdim=True) + self.eps)

        return A

    def extra_repr(self) -> str:
        return f"iterations={self.iterations}, eps={self.eps}, tau={self.tau}"


class mHCResidual(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Residual Block.

    Expands the residual stream from 1 to n parallel streams with
    mixing matrices projected to the Birkhoff polytope.

    This ensures:
    1. Identity mapping can be learned (stability)
    2. Diverse information flow through parallel streams (capacity)
    """

    def __init__(
        self,
        d_model: int,
        n_streams: int = 4,
        sinkhorn_iterations: int = 5,
        init_scale: float = 0.1
    ):
        """
        Initialize mHC residual block.

        Args:
            d_model: Model dimension
            n_streams: Number of parallel streams (default: 4 as in DeepSeek)
            sinkhorn_iterations: Sinkhorn-Knopp iterations
            init_scale: Scale for initialization offset from identity
        """
        super().__init__()

        self.d_model = d_model
        self.n_streams = n_streams
        self.init_scale = init_scale

        # Learnable mixing matrices (before projection)
        # H_res: residual mixing (the key matrix that must be doubly stochastic)
        # H_pre: pre-transformation mixing
        # H_post: post-transformation mixing
        self.H_res_raw = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.H_pre_raw = nn.Parameter(torch.zeros(n_streams, n_streams))
        self.H_post_raw = nn.Parameter(torch.zeros(n_streams, n_streams))

        # Initialize close to identity for training stability
        self._init_weights()

        # Sinkhorn-Knopp projector (only for H_res as per paper)
        self.sinkhorn = SinkhornKnopp(iterations=sinkhorn_iterations)

        # Projection layers for stream expansion/contraction
        self.expand = nn.Linear(d_model, d_model * n_streams, bias=False)
        self.contract = nn.Linear(d_model * n_streams, d_model, bias=False)

        # Initialize expand/contract for identity-like behavior
        self._init_projection_weights()

    def _init_weights(self):
        """Initialize mixing matrices close to identity."""
        nn.init.eye_(self.H_res_raw)
        nn.init.eye_(self.H_pre_raw)
        nn.init.eye_(self.H_post_raw)

        # Add small noise for symmetry breaking
        with torch.no_grad():
            self.H_res_raw.add_(torch.randn_like(self.H_res_raw) * self.init_scale)
            self.H_pre_raw.add_(torch.randn_like(self.H_pre_raw) * self.init_scale)
            self.H_post_raw.add_(torch.randn_like(self.H_post_raw) * self.init_scale)

    def _init_projection_weights(self):
        """Initialize expansion/contraction for near-identity mapping."""
        # Initialize expand to replicate input to each stream
        with torch.no_grad():
            weight = torch.zeros(self.d_model * self.n_streams, self.d_model)
            for i in range(self.n_streams):
                start = i * self.d_model
                end = (i + 1) * self.d_model
                weight[start:end, :] = torch.eye(self.d_model) / math.sqrt(self.n_streams)
            self.expand.weight.copy_(weight)

            # Initialize contract to average streams
            weight = torch.zeros(self.d_model, self.d_model * self.n_streams)
            for i in range(self.n_streams):
                start = i * self.d_model
                end = (i + 1) * self.d_model
                weight[:, start:end] = torch.eye(self.d_model) / self.n_streams
            self.contract.weight.copy_(weight)

    def expand_to_streams(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input to multiple parallel streams.

        Args:
            x: Input [batch, seq_len, d_model]

        Returns:
            Expanded streams [batch, seq_len, n_streams, d_model]
        """
        batch, seq_len, _ = x.shape
        expanded = self.expand(x)  # [batch, seq_len, d_model * n_streams]
        return expanded.view(batch, seq_len, self.n_streams, self.d_model)

    def contract_from_streams(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        Contract multiple streams to single output.

        Args:
            x_streams: Streams [batch, seq_len, n_streams, d_model]

        Returns:
            Contracted output [batch, seq_len, d_model]
        """
        batch, seq_len, n_streams, d_model = x_streams.shape
        flattened = x_streams.reshape(batch, seq_len, -1)  # [batch, seq, n*d]
        return self.contract(flattened)  # [batch, seq, d_model]

    def get_projected_matrices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the projected mixing matrices.

        Returns:
            Tuple of (H_res, H_pre, H_post) where H_res is doubly stochastic
        """
        H_res = self.sinkhorn(self.H_res_raw)  # Doubly stochastic
        # H_pre and H_post use softmax for non-negativity but not full projection
        H_pre = F.softmax(self.H_pre_raw, dim=-1)
        H_post = F.softmax(self.H_post_raw, dim=-1)
        return H_res, H_pre, H_post

    def forward(
        self,
        x: torch.Tensor,
        layer_output: torch.Tensor,
        is_first_layer: bool = False
    ) -> torch.Tensor:
        """
        Apply mHC residual connection.

        Args:
            x: Input/previous residual
               - [batch, seq_len, d_model] if first layer
               - [batch, seq_len, n_streams, d_model] otherwise
            layer_output: Layer output (attention or FF) [batch, seq_len, d_model]
            is_first_layer: Whether this is the first layer (expand from 1 stream)

        Returns:
            Updated streams [batch, seq_len, n_streams, d_model]
        """
        # Get projected matrices
        H_res, _, H_post = self.get_projected_matrices()

        # Expand to streams if first layer
        if is_first_layer:
            x_streams = self.expand_to_streams(x)  # [batch, seq, n_streams, d_model]
        else:
            x_streams = x  # Already in stream format

        batch, seq_len, n_streams, d_model = x_streams.shape

        # Apply mHC update: x_{l+1} = H_res @ x_l + H_post.T @ F(x_l)

        # Residual mixing: H_res @ x_streams
        # einsum: [n, n] @ [batch, seq, n, d] -> [batch, seq, n, d]
        x_res = torch.einsum('ij,bsjd->bsid', H_res, x_streams)

        # Layer output mixing: H_post.T @ layer_output (expanded to streams)
        # Expand layer_output to all streams
        layer_expanded = layer_output.unsqueeze(2).expand(-1, -1, n_streams, -1)
        # Apply transposed H_post (ji instead of ij)
        x_layer = torch.einsum('ji,bsjd->bsid', H_post, layer_expanded)

        # Combine: residual + transformed layer output
        x_new = x_res + x_layer

        return x_new

    def get_single_output(self, x_streams: torch.Tensor) -> torch.Tensor:
        """
        Get single output from streams (for final layer).

        Args:
            x_streams: Stream tensor [batch, seq_len, n_streams, d_model]

        Returns:
            Single output [batch, seq_len, d_model]
        """
        return self.contract_from_streams(x_streams)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_streams={self.n_streams}"


class mHCMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention compatible with mHC.

    Processes averaged stream representation and returns output
    suitable for mHC residual connection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int = 64,
        dropout: float = 0.1,
        bias: bool = False
    ):
        """
        Initialize attention module.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
            bias: Whether to use bias in projections
        """
        super().__init__()

        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.scale = dim_head ** -0.5

        # Q, K, V projections
        self.to_q = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(d_model, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(d_model, self.inner_dim, bias=bias)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Attention output [batch, seq_len, d_model]
        """
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x).view(batch, seq_len, self.num_heads, self.dim_head)
        k = self.to_k(x).view(batch, seq_len, self.num_heads, self.dim_head)
        v = self.to_v(x).view(batch, seq_len, self.num_heads, self.dim_head)

        # Transpose for attention: [batch, heads, seq, dim_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq]
            scores = scores.masked_fill(~mask.bool(), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        out = self.to_out(out)

        return out


class mHCFeedForward(nn.Module):
    """Feed-Forward Network compatible with mHC."""

    def __init__(
        self,
        d_model: int,
        dim_ff: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize feed-forward network.

        Args:
            d_model: Model dimension
            dim_ff: Hidden dimension (default: 4 * d_model)
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
        """
        super().__init__()

        dim_ff = dim_ff or d_model * 4

        if activation == 'gelu':
            act = nn.GELU()
        elif activation == 'swiglu':
            # SwiGLU variant
            act = nn.SiLU()
        else:
            act = nn.ReLU()

        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class mHCTransformerBlock(nn.Module):
    """
    Transformer block with mHC residual connections.

    Replaces standard residual connections (x + F(x)) with
    manifold-constrained hyper-connections that project
    mixing matrices to the Birkhoff polytope.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int = 64,
        dim_ff: Optional[int] = None,
        n_streams: int = 4,
        dropout: float = 0.1,
        sinkhorn_iterations: int = 5,
        use_rms_norm: bool = True
    ):
        """
        Initialize mHC Transformer block.

        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dim_head: Dimension per head
            dim_ff: Feed-forward hidden dimension
            n_streams: Number of parallel streams for mHC
            dropout: Dropout rate
            sinkhorn_iterations: Iterations for Sinkhorn projection
            use_rms_norm: Use RMSNorm (True) or LayerNorm (False)
        """
        super().__init__()

        self.d_model = d_model
        self.n_streams = n_streams
        dim_ff = dim_ff or d_model * 4

        # Normalization layers
        norm_class = nn.RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm1 = norm_class(d_model)
        self.norm2 = norm_class(d_model)

        # Attention
        self.attention = mHCMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout
        )

        # Feed-forward
        self.ff = mHCFeedForward(
            d_model=d_model,
            dim_ff=dim_ff,
            dropout=dropout
        )

        # mHC residual blocks
        self.mhc_attn = mHCResidual(
            d_model=d_model,
            n_streams=n_streams,
            sinkhorn_iterations=sinkhorn_iterations
        )
        self.mhc_ff = mHCResidual(
            d_model=d_model,
            n_streams=n_streams,
            sinkhorn_iterations=sinkhorn_iterations
        )

    def forward(
        self,
        x: torch.Tensor,
        is_first_layer: bool = False,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with mHC.

        Args:
            x: Input tensor
               - [batch, seq, d_model] if first layer
               - [batch, seq, n_streams, d_model] otherwise
            is_first_layer: Whether this is the first layer
            attention_mask: Optional attention mask

        Returns:
            Updated streams [batch, seq, n_streams, d_model]
        """
        # Get single representation for attention
        if is_first_layer:
            x_single = x  # Already [batch, seq, d_model]
        else:
            # Average across streams
            x_single = x.mean(dim=2)  # [batch, seq, d_model]

        # Attention with mHC residual
        attn_input = self.norm1(x_single)
        attn_out = self.attention(attn_input, attention_mask)
        x = self.mhc_attn(x, attn_out, is_first_layer)

        # Feed-forward with mHC residual
        x_single = x.mean(dim=2)  # Average streams
        ff_input = self.norm2(x_single)
        ff_out = self.ff(ff_input)
        x = self.mhc_ff(x, ff_out, is_first_layer=False)

        return x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_streams={self.n_streams}"


# Utility functions for analysis and debugging
def compute_doubly_stochastic_loss(H: torch.Tensor) -> torch.Tensor:
    """
    Compute how far a matrix is from being doubly stochastic.
    Useful for monitoring training.

    Args:
        H: Matrix to check

    Returns:
        Scalar loss (0 = perfectly doubly stochastic)
    """
    row_sums = H.sum(dim=-1)
    col_sums = H.sum(dim=-2)

    row_loss = (row_sums - 1.0).pow(2).mean()
    col_loss = (col_sums - 1.0).pow(2).mean()

    return row_loss + col_loss


def orthogonality_regularization(H: torch.Tensor) -> torch.Tensor:
    """
    Regularization to encourage diverse stream mixing.
    Pushes H towards orthogonal rows.

    Args:
        H: Mixing matrix

    Returns:
        Scalar regularization loss
    """
    # Encourage orthogonal rows
    HHT = torch.matmul(H, H.transpose(-2, -1))
    identity = torch.eye(H.shape[-1], device=H.device, dtype=H.dtype)
    return (HHT - identity).pow(2).mean()


# Test when run directly
if __name__ == "__main__":
    print("Testing mHC modules...")

    # Test SinkhornKnopp
    print("\n1. Testing SinkhornKnopp...")
    sk = SinkhornKnopp(iterations=10)
    A = torch.randn(4, 4)
    A_ds = sk(A)
    print(f"   Input row sums: {A.sum(dim=-1)}")
    print(f"   Output row sums: {A_ds.sum(dim=-1)}")
    print(f"   Output col sums: {A_ds.sum(dim=-2)}")
    print(f"   All entries >= 0: {(A_ds >= 0).all()}")

    # Test mHCResidual
    print("\n2. Testing mHCResidual...")
    mhc_res = mHCResidual(d_model=512, n_streams=4)
    x = torch.randn(2, 16, 512)  # [batch, seq, d_model]
    layer_out = torch.randn(2, 16, 512)
    y = mhc_res(x, layer_out, is_first_layer=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")
    print(f"   Expected: [2, 16, 4, 512]")

    # Test mHCTransformerBlock
    print("\n3. Testing mHCTransformerBlock...")
    block = mHCTransformerBlock(
        d_model=512,
        num_heads=8,
        dim_head=64,
        n_streams=4
    )
    x = torch.randn(2, 16, 512)
    y = block(x, is_first_layer=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {y.shape}")

    # Contract back to single stream
    final = mhc_res.get_single_output(y)
    print(f"   Final contracted shape: {final.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in block.parameters())
    print(f"\n   Total parameters: {total_params:,}")

    print("\nAll tests passed!")
