"""
Loss functions for H-Net training.
Implements ratio loss for compression guidance and autoregressive loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RatioLoss(nn.Module):
    """
    Ratio Loss for guiding dynamic chunking compression.

    Guides the model to achieve target compression ratio (e.g., 6:1).

    Formula:
        L_ratio = (N/(N-1)) * ((N-1)*F*G + (1-F)*(1-G))

    Where:
        N = target_ratio (e.g., 6 for 6:1 compression)
        F = actual fraction of positions selected as boundaries
        G = average boundary probability
    """

    def __init__(self, target_ratio: float = 6.0):
        """
        Initialize ratio loss.

        Args:
            target_ratio: Target compression ratio (default: 6.0 for 6:1)
        """
        super().__init__()
        self.target_ratio = target_ratio

    def forward(
        self,
        boundary_probs: torch.Tensor,
        boundaries: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ratio loss.

        Args:
            boundary_probs: Boundary probabilities [batch, seq_len]
            boundaries: Binary boundaries [batch, seq_len]

        Returns:
            Ratio loss scalar
        """
        N = self.target_ratio

        # F: Actual fraction of positions selected
        F = boundaries.mean()

        # G: Average boundary probability
        G = boundary_probs.mean()

        # Ratio loss formula
        loss = (N / (N - 1)) * ((N - 1) * F * G + (1 - F) * (1 - G))

        return loss


class AutoregressiveLoss(nn.Module):
    """
    Autoregressive loss for language modeling.
    Standard cross-entropy loss for next-token prediction.
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute autoregressive loss.

        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            targets: Target tokens [batch, seq_len]

        Returns:
            Cross-entropy loss
        """
        # Flatten for cross-entropy
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

        loss = F.cross_entropy(
            logits,
            targets,
            ignore_index=self.ignore_index
        )

        return loss


class HNetLoss(nn.Module):
    """
    Combined loss for H-Net training.

    L_total = L_AR + α * L_ratio

    Where:
        L_AR = Autoregressive (task) loss
        L_ratio = Ratio loss for compression
        α = ratio_loss_weight (default: 0.03)
    """

    def __init__(
        self,
        target_ratio: float = 6.0,
        ratio_loss_weight: float = 0.03,
        ignore_index: int = -100
    ):
        """
        Initialize H-Net loss.

        Args:
            target_ratio: Target compression ratio
            ratio_loss_weight: Weight for ratio loss (α)
            ignore_index: Index to ignore in targets
        """
        super().__init__()

        self.ratio_loss_weight = ratio_loss_weight

        self.ar_loss = AutoregressiveLoss(ignore_index)
        self.ratio_loss = RatioLoss(target_ratio)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        boundary_probs: torch.Tensor,
        boundaries: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            targets: Target tokens [batch, seq_len]
            boundary_probs: Boundary probabilities [batch, seq_len]
            boundaries: Binary boundaries [batch, seq_len]

        Returns:
            Tuple of:
                - total_loss: Combined loss
                - loss_dict: Dictionary of individual losses
        """
        # Compute individual losses
        ar_loss = self.ar_loss(logits, targets)
        ratio_loss = self.ratio_loss(boundary_probs, boundaries)

        # Combined loss
        total_loss = ar_loss + self.ratio_loss_weight * ratio_loss

        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'ar_loss': ar_loss.item(),
            'ratio_loss': ratio_loss.item(),
            'compression_ratio': (1.0 / boundaries.mean()).item()
        }

        return total_loss, loss_dict


class BitsPerByteLoss(nn.Module):
    """
    Bits Per Byte (BPB) loss for evaluating compression efficiency.

    BPB = cross_entropy_loss / log(2)

    Lower BPB indicates better compression.
    Target: ~0.743 BPB (H-Net 2-stage achieves this)
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.log2 = torch.log(torch.tensor(2.0))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bits per byte.

        Args:
            logits: Model predictions [batch, seq_len, vocab_size]
            targets: Target bytes [batch, seq_len]

        Returns:
            BPB score
        """
        # Compute cross-entropy in nats
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=self.ignore_index
        )

        # Convert to bits per byte
        bpb = ce_loss / self.log2

        return bpb


class CompressionMetrics:
    """Utility class for computing compression-related metrics."""

    @staticmethod
    def compute_compression_ratio(
        original_length: int,
        compressed_length: int
    ) -> float:
        """
        Compute compression ratio.

        Args:
            original_length: Original sequence length
            compressed_length: Compressed (chunk) length

        Returns:
            Compression ratio (e.g., 6.0 for 6:1)
        """
        return original_length / compressed_length

    @staticmethod
    def compute_bytes_per_chunk(
        total_bytes: int,
        num_chunks: int
    ) -> float:
        """
        Compute average bytes per chunk.

        Args:
            total_bytes: Total number of bytes processed
            num_chunks: Number of chunks created

        Returns:
            Bytes per chunk (target: ~4.8)
        """
        return total_bytes / num_chunks

    @staticmethod
    def compute_boundary_density(boundaries: torch.Tensor) -> float:
        """
        Compute boundary density (fraction of positions that are boundaries).

        Args:
            boundaries: Binary boundary tensor [batch, seq_len]

        Returns:
            Boundary density (0-1)
        """
        return boundaries.mean().item()


# Example usage
if __name__ == "__main__":
    # Test ratio loss
    batch_size = 4
    seq_len = 512

    # Simulate boundary detection
    boundary_probs = torch.rand(batch_size, seq_len)
    boundaries = (boundary_probs > 0.5).float()

    ratio_loss_fn = RatioLoss(target_ratio=6.0)
    ratio_loss = ratio_loss_fn(boundary_probs, boundaries)

    print(f"Boundary density: {boundaries.mean():.4f}")
    print(f"Target compression: 6:1")
    print(f"Actual compression: {1.0 / boundaries.mean():.2f}:1")
    print(f"Ratio loss: {ratio_loss.item():.4f}")

    # Test combined loss
    vocab_size = 256
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    hnet_loss_fn = HNetLoss(
        target_ratio=6.0,
        ratio_loss_weight=0.03
    )

    total_loss, loss_dict = hnet_loss_fn(
        logits, targets, boundary_probs, boundaries
    )

    print(f"\nCombined Loss:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")

    # Test BPB
    bpb_fn = BitsPerByteLoss()
    bpb = bpb_fn(logits, targets)
    print(f"\nBits Per Byte: {bpb.item():.4f}")
    print(f"Target BPB: 0.743")
