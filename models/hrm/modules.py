"""
HRM Modules: H-module (slow, abstract) and L-module (fast, detailed).

Both are 4-layer Transformers with RMSNorm (Post-Norm architecture).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in HRM paper)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class TransformerLayer(nn.Module):
    """Single Transformer layer with Post-Norm (RMSNorm after attention/ff)."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dim_head: int,
        dropout: float = 0.1,
        use_rms_norm: bool = True
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.inner_dim = num_heads * dim_head

        # Attention components
        self.to_qkv = nn.Linear(hidden_dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Norms (Post-Norm: after residual)
        norm_layer = RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm1 = norm_layer(hidden_dim)
        self.norm2 = norm_layer(hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = dim_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional context for cross-attention.

        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            context: Optional context (for L-module to attend to H-state)

        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Self-attention
        residual = x
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2),
            qkv
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.inner_dim)
        out = self.to_out(out)

        # Post-Norm: Add residual then normalize
        x = self.norm1(residual + self.dropout(out))

        # Feed-forward
        residual = x
        out = self.ff(x)
        x = self.norm2(residual + out)

        return x


class HModule(nn.Module):
    """
    H-Module: Slow, abstract planning module.

    Recurrent with 4-layer Transformer.
    Updates once per cycle after L-module converges.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        use_rms_norm: bool = True
    ):
        """
        Initialize H-module.

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of Transformer layers (default: 4)
            num_heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
            use_rms_norm: Use RMSNorm (Post-Norm)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                use_rms_norm=use_rms_norm
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        z_H: torch.Tensor,
        z_L_final: torch.Tensor,
        x_input: torch.Tensor
    ) -> torch.Tensor:
        """
        H-module forward pass (one update per cycle).

        Args:
            z_H: Previous H-state [batch, hidden_dim]
            z_L_final: Final L-state from convergence [batch, hidden_dim]
            x_input: Input representation [batch, hidden_dim]

        Returns:
            Updated H-state [batch, hidden_dim]

        Formula:
            z_H^(m) = H(z_H^(m-1), z_L^final, x̃)
        """
        # Combine inputs: [z_H_prev, z_L_final, x_input]
        # Add sequence dimension for Transformer
        combined = torch.stack([z_H, z_L_final, x_input], dim=1)
        # combined: [batch, 3, hidden_dim]

        # Process through Transformer layers
        for layer in self.layers:
            combined = layer(combined)

        # Extract updated H-state (take first token)
        z_H_new = combined[:, 0, :]  # [batch, hidden_dim]

        return z_H_new


class LModule(nn.Module):
    """
    L-Module: Fast, detailed computation module.

    Recurrent with 4-layer Transformer.
    Iterates T times per H-cycle conditioned on frozen H-state.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1,
        use_rms_norm: bool = True
    ):
        """
        Initialize L-module.

        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of Transformer layers (default: 4)
            num_heads: Number of attention heads
            dim_head: Dimension per head
            dropout: Dropout rate
            use_rms_norm: Use RMSNorm (Post-Norm)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dim_head=dim_head,
                dropout=dropout,
                use_rms_norm=use_rms_norm
            )
            for _ in range(num_layers)
        ])

    def forward(
        self,
        z_L: torch.Tensor,
        z_H_frozen: torch.Tensor,
        x_input: torch.Tensor
    ) -> torch.Tensor:
        """
        L-module forward pass (T iterations per H-cycle).

        Args:
            z_L: Previous L-state [batch, hidden_dim]
            z_H_frozen: Frozen H-state (no gradient) [batch, hidden_dim]
            x_input: Input representation [batch, hidden_dim]

        Returns:
            Updated L-state [batch, hidden_dim]

        Formula:
            z_L^(t) = L(z_L^(t-1), z_H^frozen, x̃)
        """
        # Combine inputs: [z_L_prev, z_H_frozen, x_input]
        combined = torch.stack([z_L, z_H_frozen, x_input], dim=1)
        # combined: [batch, 3, hidden_dim]

        # Process through Transformer layers
        for layer in self.layers:
            combined = layer(combined)

        # Extract updated L-state (take first token)
        z_L_new = combined[:, 0, :]  # [batch, hidden_dim]

        return z_L_new


class InputNetwork(nn.Module):
    """Input network φ_in: Projects input to working representation."""

    def __init__(
        self,
        input_dim: int = 1536,  # From H-Net main output
        hidden_dim: int = 512,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to working representation.

        Args:
            x: Input from H-Net [batch, seq_len, input_dim]

        Returns:
            Working representation [batch, seq_len, hidden_dim]
        """
        return self.net(x)


class OutputNetwork(nn.Module):
    """Output network φ_out: Extracts prediction from final H-state."""

    def __init__(
        self,
        hidden_dim: int = 512,
        output_dim: int = 512,  # Task-specific
        activation: str = 'gelu'
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, z_H_final: torch.Tensor) -> torch.Tensor:
        """
        Extract prediction from final H-state.

        Args:
            z_H_final: Final H-state [batch, hidden_dim]

        Returns:
            Prediction [batch, output_dim]
        """
        return self.net(z_H_final)


# Example usage
if __name__ == "__main__":
    batch_size = 4
    hidden_dim = 512

    # Initialize modules
    h_module = HModule(hidden_dim=hidden_dim)
    l_module = LModule(hidden_dim=hidden_dim)
    input_net = InputNetwork(input_dim=1536, hidden_dim=hidden_dim)
    output_net = OutputNetwork(hidden_dim=hidden_dim, output_dim=512)

    # Initialize states (using TruncatedNormal in practice)
    z_H = torch.randn(batch_size, hidden_dim)
    z_L = torch.randn(batch_size, hidden_dim)

    # Input from H-Net
    x_from_hnet = torch.randn(batch_size, 10, 1536)  # [batch, seq_len, 1536]
    x_input = input_net(x_from_hnet).mean(dim=1)  # Pool to [batch, 512]

    # Simulate one H-cycle
    print("Simulating HRM cycle:")

    # L-module iterations (T=8)
    z_H_frozen = z_H.detach()  # Freeze H-state
    for t in range(8):
        z_L = l_module(z_L, z_H_frozen, x_input)
        print(f"  L-iteration {t+1}: z_L shape {z_L.shape}")

    # H-module update (once per cycle)
    z_H = h_module(z_H, z_L, x_input)
    print(f"  H-update: z_H shape {z_H.shape}")

    # Output
    output = output_net(z_H)
    print(f"  Output shape: {output.shape}")

    # Count parameters
    h_params = sum(p.numel() for p in h_module.parameters())
    l_params = sum(p.numel() for p in l_module.parameters())
    print(f"\nH-module parameters: {h_params:,}")
    print(f"L-module parameters: {l_params:,}")
