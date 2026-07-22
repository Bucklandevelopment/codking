"""
H-Net Encoder with Mamba-2 SSM layers.
Processes raw byte sequences into encoded representations.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

# Try to import Mamba, fall back to LSTM if not available
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not available. Using LSTM fallback for encoder.")


class MambaLayer(nn.Module):
    """Single Mamba-2 SSM layer wrapper."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4, expand_factor: int = 2):
        super().__init__()
        self.d_model = d_model

        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand_factor
            )
        else:
            # Fallback: Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Output tensor [batch, seq_len, d_model]
        """
        if MAMBA_AVAILABLE:
            return self.mamba(x)
        else:
            output, _ = self.lstm(x)
            return output


class HNetEncoder(nn.Module):
    """
    H-Net Encoder: 4-layer Mamba-2 SSM for byte-level processing.

    Processes raw byte sequences (D=1024) before dynamic chunking.
    """

    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 1024,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        input_vocab_size: int = 256  # Byte vocabulary (0-255)
    ):
        """
        Initialize H-Net encoder.

        Args:
            num_layers: Number of Mamba layers (default: 4)
            d_model: Model dimension (default: 1024)
            d_state: State dimension for SSM
            d_conv: Convolution dimension
            expand_factor: Expansion factor for Mamba
            dropout: Dropout rate
            input_vocab_size: Size of input vocabulary (256 for bytes)
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Byte embedding
        self.embedding = nn.Embedding(input_vocab_size, d_model)

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand_factor=expand_factor
            )
            for _ in range(num_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(num_layers)
        ])

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output projection (optional, identity by default)
        self.output_proj = nn.Identity()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through encoder.

        Args:
            input_ids: Byte sequence [batch, seq_len] with values 0-255
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Tuple of:
                - encoded: Encoded sequence [batch, seq_len, d_model]
                - attention_mask: Pass-through mask
        """
        # Embed bytes
        x = self.embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        # Process through Mamba layers
        for layer, norm in zip(self.layers, self.layer_norms):
            # Pre-norm architecture
            residual = x
            x = norm(x)
            x = layer(x)
            x = residual + self.dropout(x)  # Residual connection

        # Output projection
        x = self.output_proj(x)

        # Apply attention mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        return x, attention_mask

    def get_num_params(self) -> int:
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (optional, if needed)."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pe[:, :x.size(1)]


# Example usage and testing
if __name__ == "__main__":
    # Test encoder
    batch_size = 4
    seq_len = 512
    d_model = 1024

    encoder = HNetEncoder(
        num_layers=4,
        d_model=d_model,
        d_state=64,
        d_conv=4
    )

    # Random byte sequence
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    output, mask = encoder(input_ids, attention_mask)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {encoder.get_num_params():,}")
    print(f"Using Mamba: {MAMBA_AVAILABLE}")
