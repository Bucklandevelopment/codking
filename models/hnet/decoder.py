"""
H-Net Decoder with Mamba-2 SSM layers.
Reconstructs full resolution from processed chunks.
"""

import torch
import torch.nn as nn
from typing import Optional

# Try to import Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


class MambaLayer(nn.Module):
    """Single Mamba-2 SSM layer wrapper (same as encoder)."""

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
            # Fallback: LSTM
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if MAMBA_AVAILABLE:
            return self.mamba(x)
        else:
            output, _ = self.lstm(x)
            return output


class HNetDecoder(nn.Module):
    """
    H-Net Decoder: 4-layer Mamba-2 SSM for reconstruction.

    Reconstructs full resolution from main network output (D=1536 → D=1024).
    """

    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 1024,
        d_state: int = 64,
        d_conv: int = 4,
        expand_factor: int = 2,
        dropout: float = 0.1,
        output_vocab_size: int = 256,  # Byte vocabulary
        input_dim: Optional[int] = None
    ):
        """
        Initialize H-Net decoder.

        Args:
            num_layers: Number of Mamba layers (default: 4)
            d_model: Model dimension (default: 1024)
            d_state: State dimension for SSM
            d_conv: Convolution dimension
            expand_factor: Expansion factor for Mamba
            dropout: Dropout rate
            output_vocab_size: Size of output vocabulary (256 for bytes)
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # Input projection (from main network dim to decoder d_model)
        _input_dim = input_dim if input_dim is not None else 1536
        self.input_proj = nn.Linear(_input_dim, d_model)

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

        # Output head for byte prediction
        self.output_head = nn.Linear(d_model, output_vocab_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through decoder.

        Args:
            x: Input from main network [batch, seq_len, 1536]
            attention_mask: Optional mask [batch, seq_len]

        Returns:
            Logits for byte prediction [batch, seq_len, vocab_size]
        """
        # Project to decoder dimension
        x = self.input_proj(x)  # [batch, seq_len, 1024]
        x = self.dropout(x)

        # Process through Mamba layers
        for layer, norm in zip(self.layers, self.layer_norms):
            # Pre-norm architecture
            residual = x
            x = norm(x)
            x = layer(x)
            x = residual + self.dropout(x)  # Residual connection

        # Apply attention mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # Output head for byte prediction
        logits = self.output_head(x)  # [batch, seq_len, vocab_size]

        return logits

    def get_num_params(self) -> int:
        """Return number of parameters."""
        return sum(p.numel() for p in self.parameters())


# Example usage
if __name__ == "__main__":
    # Test decoder
    batch_size = 4
    seq_len = 512
    d_main = 1536  # From main network

    decoder = HNetDecoder(
        num_layers=4,
        d_model=1024,
        d_state=64,
        output_vocab_size=256
    )

    # Random input from main network
    x = torch.randn(batch_size, seq_len, d_main)
    attention_mask = torch.ones(batch_size, seq_len)

    # Forward pass
    logits = decoder(x, attention_mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {decoder.get_num_params():,}")
    print(f"Using Mamba: {MAMBA_AVAILABLE}")
