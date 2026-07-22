"""
Complete H-Net Model with Dynamic Chunking.

Integrates:
- Encoder (4-layer Mamba-2, D=1024)
- Dynamic Chunking (routing, smoothing, STE)
- Main Network (22-layer Transformer, D=1536)
- Decoder (4-layer Mamba-2, D=1024)
- Losses (ratio loss + AR loss)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
import yaml

from .encoder import HNetEncoder
from .dynamic_chunking import DynamicChunkingModule
from .main_network import HNetMainNetwork
from .decoder import HNetDecoder
from .losses import HNetLoss, BitsPerByteLoss, CompressionMetrics


class HNet(nn.Module):
    """
    Complete H-Net model for hierarchical sequence modeling with dynamic chunking.

    Architecture:
        RAW BYTES → Encoder (Mamba-2) → Dynamic Chunking (6:1) →
        Main Network (Transformer) → Decoder (Mamba-2) → BYTE PREDICTION
    """

    def __init__(
        self,
        # Encoder config
        encoder_layers: int = 4,
        encoder_dim: int = 1024,
        encoder_d_state: int = 64,

        # Dynamic chunking config
        target_ratio: float = 6.0,
        boundary_threshold: float = 0.5,
        smoothing_momentum: float = 0.9,
        use_smoothing: bool = True,  # CRITICAL

        # Main network config
        main_layers: int = 22,
        main_dim: int = 1536,
        num_heads: int = 12,
        dim_head: int = 128,
        use_flash_attn: bool = True,

        # Decoder config
        decoder_layers: int = 4,
        decoder_dim: int = 1024,
        decoder_d_state: int = 64,

        # Loss config
        ratio_loss_weight: float = 0.03,

        # General
        dropout: float = 0.1,
        vocab_size: int = 256  # Byte vocabulary
    ):
        """
        Initialize H-Net.

        Args:
            encoder_layers: Number of encoder layers (default: 4)
            encoder_dim: Encoder dimension (default: 1024)
            encoder_d_state: Encoder SSM state dim
            target_ratio: Target compression ratio (default: 6.0)
            boundary_threshold: Threshold for boundary detection (default: 0.5)
            smoothing_momentum: Momentum for smoothing module (default: 0.9)
            use_smoothing: Enable smoothing (CRITICAL, default: True)
            main_layers: Number of main network layers (default: 22)
            main_dim: Main network dimension (default: 1536)
            num_heads: Number of attention heads
            dim_head: Dimension per head
            use_flash_attn: Use FlashAttention if available
            decoder_layers: Number of decoder layers (default: 4)
            decoder_dim: Decoder dimension (default: 1024)
            decoder_d_state: Decoder SSM state dim
            ratio_loss_weight: Weight for ratio loss (default: 0.03)
            dropout: Dropout rate
            vocab_size: Vocabulary size (256 for bytes)
        """
        super().__init__()

        # Store config
        self.target_ratio = target_ratio
        self.vocab_size = vocab_size

        # 1. Encoder: Bytes → Encoded sequence
        self.encoder = HNetEncoder(
            num_layers=encoder_layers,
            d_model=encoder_dim,
            d_state=encoder_d_state,
            dropout=dropout,
            input_vocab_size=vocab_size
        )

        # 2. Dynamic Chunking: Compressed representation
        self.chunking = DynamicChunkingModule(
            d_model=encoder_dim,
            boundary_threshold=boundary_threshold,
            smoothing_momentum=smoothing_momentum,
            use_smoothing=use_smoothing
        )

        # 3. Main Network: Process chunks
        self.main_network = HNetMainNetwork(
            num_layers=main_layers,
            d_model=main_dim,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            input_dim=encoder_dim
        )

        # 4. Decoder: Reconstruct full resolution + predict bytes
        self.decoder = HNetDecoder(
            num_layers=decoder_layers,
            d_model=decoder_dim,
            d_state=decoder_d_state,
            dropout=dropout,
            output_vocab_size=vocab_size,
            input_dim=main_dim
        )

        # 5. Loss functions
        self.loss_fn = HNetLoss(
            target_ratio=target_ratio,
            ratio_loss_weight=ratio_loss_weight
        )

        self.bpb_fn = BitsPerByteLoss()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete H-Net.

        Args:
            input_ids: Input bytes [batch, seq_len]
            attention_mask: Optional mask [batch, seq_len]
            targets: Optional target bytes [batch, seq_len] for loss
            return_loss: Whether to compute and return loss

        Returns:
            Dictionary containing:
                - logits: Byte predictions [batch, seq_len, vocab_size]
                - chunks: Compressed chunks [batch, num_chunks, main_dim]
                - boundary_probs: Boundary probabilities [batch, seq_len]
                - loss: (if return_loss=True) Combined loss
                - loss_dict: (if return_loss=True) Individual losses
                - compression_ratio: Achieved compression ratio
        """
        batch_size, seq_len = input_ids.shape

        # 1. Encode bytes
        encoded, _ = self.encoder(input_ids, attention_mask)
        # encoded: [batch, seq_len, 1024]

        # 2. Dynamic chunking
        chunks, chunk_indices, boundary_probs = self.chunking(
            encoded,
            return_boundaries=True
        )
        # chunks: [batch, num_chunks, 1024]
        # boundary_probs: [batch, seq_len]

        # 3. Process chunks through main network
        # Create attention mask for chunks
        chunk_mask = None
        if attention_mask is not None:
            chunk_mask = torch.ones(
                batch_size, chunks.size(1),
                device=chunks.device
            )

        processed_chunks = self.main_network(chunks, chunk_mask)
        # processed_chunks: [batch, num_chunks, 1536]

        # 4. Reconstruct and decode
        # First, upsample chunks back to full sequence
        boundaries = (boundary_probs >= 0.5).float()
        reconstructed = self.chunking.reconstruct(
            processed_chunks,
            chunk_indices,
            seq_len,
            boundary_probs
        )
        # reconstructed: [batch, seq_len, 1536]

        # Decode to logits
        logits = self.decoder(reconstructed, attention_mask)
        # logits: [batch, seq_len, vocab_size]

        # Compute compression ratio
        num_chunks = chunks.size(1)
        compression_ratio = seq_len / num_chunks

        # Prepare output
        output = {
            'logits': logits,
            'chunks': processed_chunks,
            'boundary_probs': boundary_probs,
            'compression_ratio': compression_ratio,
            'num_chunks': num_chunks
        }

        # Compute loss if targets provided
        if return_loss and targets is not None:
            total_loss, loss_dict = self.loss_fn(
                logits, targets, boundary_probs, boundaries
            )

            # Add BPB to loss dict
            bpb = self.bpb_fn(logits, targets)
            loss_dict['bpb'] = bpb.item()

            output['loss'] = total_loss
            output['loss_dict'] = loss_dict

        return output

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Return total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def print_summary(self):
        """Print model summary with component-wise parameter counts."""
        print(f"\n{'='*70}")
        print(f"{'H-Net Model Summary':^70}")
        print(f"{'='*70}")

        # Component parameter counts
        encoder_params = self.encoder.get_num_params()
        main_params = self.main_network.get_num_params()
        decoder_params = self.decoder.get_num_params()
        chunking_params = sum(p.numel() for p in self.chunking.parameters())

        total_params = self.get_num_params()
        trainable_params = self.get_num_params(trainable_only=True)

        print(f"\nComponent Parameters:")
        print(f"  Encoder (Mamba-2, 4L):      {encoder_params:>12,}")
        print(f"  Dynamic Chunking:           {chunking_params:>12,}")
        print(f"  Main Network (Trans, 22L):  {main_params:>12,}")
        print(f"  Decoder (Mamba-2, 4L):      {decoder_params:>12,}")
        print(f"  {'-'*40}")
        print(f"  Total Parameters:           {total_params:>12,}")
        print(f"  Trainable Parameters:       {trainable_params:>12,}")

        print(f"\nConfiguration:")
        print(f"  Target Compression Ratio:   {self.target_ratio}:1")
        print(f"  Vocabulary Size:            {self.vocab_size}")

        print(f"\n{'='*70}\n")

    @classmethod
    def from_config(cls, config_path: str) -> 'HNet':
        """
        Load H-Net from YAML config file.

        Args:
            config_path: Path to config file

        Returns:
            Initialized H-Net model
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        hnet_config = config['hnet']

        return cls(
            encoder_layers=hnet_config['encoder']['num_layers'],
            encoder_dim=hnet_config['encoder']['d_model'],
            encoder_d_state=hnet_config['encoder']['d_state'],
            target_ratio=hnet_config['chunking']['target_ratio'],
            boundary_threshold=hnet_config['chunking']['routing']['boundary_threshold'],
            smoothing_momentum=hnet_config['chunking']['smoothing']['ema_momentum'],
            use_smoothing=hnet_config['chunking']['smoothing']['enabled'],
            main_layers=hnet_config['main']['num_layers'],
            main_dim=hnet_config['main']['d_model'],
            num_heads=hnet_config['main']['num_heads'],
            dim_head=hnet_config['main']['dim_head'],
            use_flash_attn=hnet_config['main']['use_flash_attention'],
            decoder_layers=hnet_config['decoder']['num_layers'],
            decoder_dim=hnet_config['decoder']['d_model'],
            decoder_d_state=hnet_config['decoder']['d_state'],
            ratio_loss_weight=hnet_config['loss']['ratio_loss_weight']
        )


# Example usage
if __name__ == "__main__":
    # Create H-Net model
    model = HNet(
        encoder_layers=4,
        encoder_dim=1024,
        target_ratio=6.0,
        main_layers=22,
        main_dim=1536,
        decoder_layers=4,
        use_flash_attn=True
    )

    # Print summary
    model.print_summary()

    # Test forward pass
    batch_size = 2
    seq_len = 512

    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    targets = torch.randint(0, 256, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        output = model(
            input_ids,
            attention_mask=attention_mask,
            targets=targets,
            return_loss=True
        )

    print(f"\nForward Pass Results:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {output['logits'].shape}")
    print(f"  Chunks shape: {output['chunks'].shape}")
    print(f"  Compression ratio: {output['compression_ratio']:.2f}:1")
    print(f"  Target ratio: 6.0:1")

    print(f"\nLoss Values:")
    for key, value in output['loss_dict'].items():
        print(f"  {key}: {value:.4f}")
