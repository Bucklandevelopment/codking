"""
Deep Supervision for HRM.

Multiple segments with detached gradients between them.
Provides more frequent feedback and better regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class DeepSupervision(nn.Module):
    """
    Deep Supervision module for HRM training.

    Key concepts:
    - Divide reasoning into multiple segments
    - Each segment produces an output and contributes to loss
    - Gradients DO NOT propagate between segments (detached)
    - More frequent feedback improves training
    """

    def __init__(
        self,
        num_segments: int = 4,
        segment_loss_weight: str = 'uniform',  # 'uniform' or 'decay'
        output_network: Optional[nn.Module] = None
    ):
        """
        Initialize deep supervision.

        Args:
            num_segments: Number of segments to supervise
            segment_loss_weight: How to weight segment losses
                - 'uniform': All segments weighted equally
                - 'decay': Later segments weighted more (exponential)
            output_network: Output network to use for predictions
        """
        super().__init__()

        self.num_segments = num_segments
        self.segment_loss_weight_type = segment_loss_weight
        self.output_network = output_network

        # Compute segment loss weights
        if segment_loss_weight == 'uniform':
            self.segment_weights = [1.0 / num_segments] * num_segments
        elif segment_loss_weight == 'decay':
            # Exponential decay: later segments more important
            weights = [2 ** i for i in range(num_segments)]
            total = sum(weights)
            self.segment_weights = [w / total for w in weights]
        else:
            raise ValueError(f"Unknown segment_loss_weight: {segment_loss_weight}")

    def compute_segment_boundaries(
        self,
        total_cycles: int
    ) -> List[int]:
        """
        Compute cycle indices for segment boundaries.

        Args:
            total_cycles: Total number of H-cycles

        Returns:
            List of segment boundary indices
        """
        cycles_per_segment = max(1, total_cycles // self.num_segments)
        boundaries = [
            min((i + 1) * cycles_per_segment, total_cycles)
            for i in range(self.num_segments)
        ]
        return boundaries

    def apply_supervision(
        self,
        z_H_history: List[torch.Tensor],
        targets: torch.Tensor,
        task_loss_fn: nn.Module
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Apply deep supervision to H-state history.

        Args:
            z_H_history: List of H-states at each cycle [num_cycles, batch, hidden_dim]
            targets: Target labels [batch, ...]
            task_loss_fn: Task-specific loss function

        Returns:
            Tuple of:
                - total_loss: Weighted average of segment losses
                - loss_dict: Dictionary with segment-wise losses
        """
        num_cycles = len(z_H_history)

        if num_cycles == 0:
            return torch.tensor(0.0), {}

        # Determine segment boundaries
        boundaries = self.compute_segment_boundaries(num_cycles)

        # Compute loss for each segment
        segment_losses = []
        loss_dict = {}

        for seg_idx, boundary in enumerate(boundaries):
            # Get H-state at segment boundary
            z_H_at_boundary = z_H_history[boundary - 1]  # -1 for 0-indexing

            # CRITICAL: Detach gradients between segments
            z_H_detached = z_H_at_boundary.detach()

            # Generate prediction
            if self.output_network is not None:
                prediction = self.output_network(z_H_detached)
            else:
                prediction = z_H_detached

            # Compute loss for this segment
            seg_loss = task_loss_fn(prediction, targets)

            # Weight and accumulate
            weighted_loss = self.segment_weights[seg_idx] * seg_loss
            segment_losses.append(weighted_loss)

            loss_dict[f'segment_{seg_idx}_loss'] = seg_loss.item()
            loss_dict[f'segment_{seg_idx}_weight'] = self.segment_weights[seg_idx]

        # Total loss: sum of weighted segment losses
        total_loss = sum(segment_losses)

        loss_dict['deep_supervision_loss'] = total_loss.item()
        loss_dict['num_segments'] = len(segment_losses)

        return total_loss, loss_dict


class SegmentedTraining:
    """
    Utility class for managing segmented training with deep supervision.

    Handles:
    - Segment boundary computation
    - Gradient detachment between segments
    - Loss accumulation
    """

    def __init__(
        self,
        num_segments: int = 4,
        cycles_per_segment: Optional[int] = None
    ):
        self.num_segments = num_segments
        self.cycles_per_segment = cycles_per_segment

    def split_into_segments(
        self,
        total_cycles: int
    ) -> List[Tuple[int, int]]:
        """
        Split cycles into segments with (start, end) indices.

        Args:
            total_cycles: Total number of cycles

        Returns:
            List of (start_idx, end_idx) tuples for each segment
        """
        if self.cycles_per_segment is not None:
            # Fixed cycles per segment
            segments = []
            start = 0
            while start < total_cycles:
                end = min(start + self.cycles_per_segment, total_cycles)
                segments.append((start, end))
                start = end
        else:
            # Equal division
            cycles_per_seg = max(1, total_cycles // self.num_segments)
            segments = [
                (i * cycles_per_seg, min((i + 1) * cycles_per_seg, total_cycles))
                for i in range(self.num_segments)
            ]

        return segments

    @staticmethod
    def detach_state(state: torch.Tensor) -> torch.Tensor:
        """
        Detach state to prevent gradient flow.

        Args:
            state: State tensor

        Returns:
            Detached state
        """
        return state.detach()


# Example usage
if __name__ == "__main__":
    from ..hrm.modules import OutputNetwork

    batch_size = 4
    hidden_dim = 512
    output_dim = 10  # Classification task

    # Create output network
    output_net = OutputNetwork(hidden_dim=hidden_dim, output_dim=output_dim)

    # Create deep supervision module
    deep_sup = DeepSupervision(
        num_segments=4,
        segment_loss_weight='uniform',
        output_network=output_net
    )

    # Simulate H-state history (16 cycles)
    num_cycles = 16
    z_H_history = [
        torch.randn(batch_size, hidden_dim)
        for _ in range(num_cycles)
    ]

    # Targets for classification
    targets = torch.randint(0, output_dim, (batch_size,))

    # Task loss function
    task_loss_fn = nn.CrossEntropyLoss()

    # Apply deep supervision
    total_loss, loss_dict = deep_sup.apply_supervision(
        z_H_history,
        targets,
        task_loss_fn
    )

    print(f"Deep Supervision Results:")
    print(f"  Total cycles: {num_cycles}")
    print(f"  Num segments: {deep_sup.num_segments}")
    print(f"  Segment weights: {deep_sup.segment_weights}")
    print(f"  Total loss: {total_loss.item():.4f}")

    print(f"\nSegment-wise losses:")
    for key, value in loss_dict.items():
        if 'segment' in key:
            print(f"  {key}: {value:.4f}")

    # Test segmented training utility
    print(f"\nSegmented Training:")
    segmented = SegmentedTraining(num_segments=4)
    segments = segmented.split_into_segments(num_cycles)
    for i, (start, end) in enumerate(segments):
        print(f"  Segment {i}: cycles {start}-{end}")
