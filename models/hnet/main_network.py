"""
H-Net Main Transformer Network.
22-layer Transformer with FlashAttention for processing compressed chunks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# Try to import FlashAttention
try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not available. Using standard attention.")


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization."""
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with optional FlashAttention.
    Falls back to standard attention if FlashAttention unavailable.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.1,
        use_flash_attn: bool = True
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head
        self.use_flash_attn = use_flash_attn and FLASH_ATTN_AVAILABLE

        self.scale = dim_head ** -0.5

        # Q, K, V projections
        self.to_q = nn.Linear(d_model, self.inner_dim, bias=False)
        self.to_k = nn.Linear(d_model, self.inner_dim, bias=False)
        self.to_v = nn.Linear(d_model, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, d_model),
            nn.Dropout(dropout)
        )

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.dim_head)
        k = k.view(batch_size, seq_len, self.num_heads, self.dim_head)
        v = v.view(batch_size, seq_len, self.num_heads, self.dim_head)

        if self.use_flash_attn:
            # FlashAttention expects [batch, seq_len, num_heads, dim_head]
            # Already in correct format
            out = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=False
            )
        else:
            # Standard scaled dot-product attention
            # Transpose to [batch, num_heads, seq_len, dim_head]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # Apply attention mask if provided
            if attention_mask is not None:
                # Expand mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
                mask = attention_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(~mask.bool(), float('-inf'))

            # Softmax
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)

            # Apply attention to values
            out = torch.matmul(attn, v)

            # Transpose back: [batch, num_heads, seq_len, dim_head] -> [batch, seq_len, num_heads, dim_head]
            out = out.transpose(1, 2)

        # Reshape and project output
        out = out.contiguous().view(batch_size, seq_len, self.inner_dim)
        out = self.to_out(out)

        return out


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""

    def __init__(
        self,
        d_model: int,
        dim_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerLayer(nn.Module):
    """Single Transformer layer with Pre-LN architecture."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_head: int,
        dim_ff: int,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
        use_rms_norm: bool = False
    ):
        super().__init__()

        # Layer norms
        norm_layer = RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)

        # Attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            use_flash_attn=use_flash_attn
        )

        # Feed-forward
        self.ff = FeedForward(d_model, dim_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with Pre-LN and residual connections.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, attention_mask)
        x = residual + x

        # Feed-forward with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x

        return x


class HNetMainNetwork(nn.Module):
    """
    H-Net Main Transformer Network.

    Processes compressed chunks from dynamic chunking module.
    22 layers, D=1536, 12 heads, FlashAttention enabled.
    """

    def __init__(
        self,
        num_layers: int = 22,
        d_model: int = 1536,
        num_heads: int = 12,
        dim_head: int = 128,
        dim_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_flash_attn: bool = True,
        use_rms_norm: bool = False,
        max_seq_len: int = 2048,
        input_dim: Optional[int] = None
    ):
        """
        Initialize main network.

        Args:
            num_layers: Number of Transformer layers (default: 22)
            d_model: Model dimension (default: 1536)
            num_heads: Number of attention heads (default: 12)
            dim_head: Dimension per head (default: 128)
            dim_ff: Feed-forward dimension (default: 4 * d_model)
            dropout: Dropout rate
            use_flash_attn: Use FlashAttention if available
            use_rms_norm: Use RMSNorm instead of LayerNorm
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        if dim_ff is None:
            dim_ff = 4 * d_model

        # Input projection (from encoder dim to main d_model)
        _input_dim = input_dim if input_dim is not None else 1024
        self.input_proj = nn.Linear(_input_dim, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_head=dim_head,
                dim_ff=dim_ff,
                dropout=dropout,
                use_flash_attn=use_flash_attn,
                use_rms_norm=use_rms_norm
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        norm_layer = RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm = norm_layer(d_model)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through main network.

        Args:
            x: Input chunks [batch, num_chunks, 1024]
            attention_mask: Optional mask [batch, num_chunks]

        Returns:
            Processed chunks [batch, num_chunks, 1536]
        """
        # Project input to d_model
        x = self.input_proj(x)

        # Process through Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)

        # Final normalization
        x = self.norm(x)

        return x

    def get_num_params(self) -> int:
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Example usage and testing
if __name__ == "__main__":
    # Test main network
    batch_size = 4
    num_chunks = 256  # After 6:1 compression from 1536 bytes
    d_encoder = 1024  # From encoder output

    main_network = HNetMainNetwork(
        num_layers=22,
        d_model=1536,
        num_heads=12,
        dim_head=128,
        use_flash_attn=True
    )

    # Random chunk input from encoder
    chunks = torch.randn(batch_size, num_chunks, d_encoder)
    attention_mask = torch.ones(batch_size, num_chunks)

    # Forward pass
    output = main_network(chunks, attention_mask)

    print(f"Input shape: {chunks.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of layers: {main_network.num_layers}")
    print(f"Number of parameters: {main_network.get_num_params():,}")
    print(f"Using FlashAttention: {FLASH_ATTN_AVAILABLE}")
