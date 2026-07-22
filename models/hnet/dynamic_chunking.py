"""
Dynamic Chunking Module for H-Net.
Implements routing (boundary detection), smoothing (EMA), and upsampling (STE).

CRITICAL: Smoothing module provides differentiability (10% performance drop without it).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-Through Estimator for binary operations.
    Forward: discrete, Backward: pass-through gradient.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """Forward pass: binarize input."""
        return (input > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Backward pass: straight-through (identity)."""
        return grad_output


class RoutingModule(nn.Module):
    """
    Routing module for boundary detection via cosine similarity.

    Key insight: Low cosine similarity = semantic discontinuity = chunk boundary
    """

    def __init__(self, d_model: int, boundary_threshold: float = 0.5):
        super().__init__()

        self.d_model = d_model
        self.boundary_threshold = boundary_threshold

        # Query and Key projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect chunk boundaries via cosine similarity.

        Args:
            x: Input sequence [batch, seq_len, d_model]

        Returns:
            Tuple of:
                - boundary_probs: Probability of boundary [batch, seq_len]
                - boundaries: Binary boundaries (with STE) [batch, seq_len]
        """
        batch_size, seq_len, d_model = x.shape

        # Project to query and key
        q = self.W_q(x)  # [batch, seq_len, d_model]
        k = self.W_k(x)  # [batch, seq_len, d_model]

        # Compute cosine similarity with previous position
        # q_t vs k_{t-1}
        q_current = q[:, 1:, :]  # [batch, seq_len-1, d_model]
        k_prev = k[:, :-1, :]     # [batch, seq_len-1, d_model]

        # Cosine similarity
        cos_sim = F.cosine_similarity(q_current, k_prev, dim=-1)  # [batch, seq_len-1]

        # Boundary probability: p_t = 0.5 * (1 - cosine_similarity)
        boundary_probs = 0.5 * (1.0 - cos_sim)  # [batch, seq_len-1]

        # Add boundary at position 0 (always start of chunk)
        boundary_probs = torch.cat([
            torch.ones(batch_size, 1, device=x.device),  # First position always boundary
            boundary_probs
        ], dim=1)  # [batch, seq_len]

        # Binary boundaries: b_t = 1 if p_t >= threshold else 0
        boundaries = (boundary_probs >= self.boundary_threshold).float()

        return boundary_probs, boundaries


class SmoothingModule(nn.Module):
    """
    CRITICAL: Smoothing module for differentiability.
    Transforms discrete chunking operation into continuous via EMA.

    10% performance drop without this module!
    """

    def __init__(self, momentum: float = 0.9):
        super().__init__()
        self.momentum = momentum

    def forward(
        self,
        chunk_embeddings: torch.Tensor,
        boundary_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply EMA smoothing to chunk embeddings.

        Args:
            chunk_embeddings: Raw chunk embeddings [batch, seq_len, d_model]
            boundary_probs: Boundary probabilities [batch, seq_len]

        Returns:
            Smoothed embeddings [batch, seq_len, d_model]

        Formula:
            z̄_t = P_t · ẑ_t + (1 - P_t) · z̄_{t-1}
        """
        batch_size, seq_len, d_model = chunk_embeddings.shape

        smoothed = []
        z_prev = torch.zeros(batch_size, d_model, device=chunk_embeddings.device)

        for t in range(seq_len):
            p_t = boundary_probs[:, t].unsqueeze(-1)  # [batch, 1]
            z_t = chunk_embeddings[:, t, :]  # [batch, d_model]

            # EMA update
            z_smoothed = p_t * z_t + (1 - p_t) * z_prev
            smoothed.append(z_smoothed)
            z_prev = z_smoothed

        smoothed = torch.stack(smoothed, dim=1)  # [batch, seq_len, d_model]
        return smoothed


class Downsampler(nn.Module):
    """
    Downsampler: Retains only boundary positions.
    Compresses sequence from input_len to num_chunks.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        boundaries: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Downsample sequence by selecting boundary positions.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            boundaries: Binary boundaries [batch, seq_len]

        Returns:
            Tuple of:
                - chunks: Downsampled chunks [batch, num_chunks, d_model]
                - chunk_indices: Indices of selected positions [batch, num_chunks]
        """
        batch_size, seq_len, d_model = x.shape

        # Find boundary indices
        # For each batch element, gather positions where boundary=1
        chunks_list = []
        indices_list = []

        for b in range(batch_size):
            boundary_mask = boundaries[b].bool()  # [seq_len]
            chunk_positions = torch.where(boundary_mask)[0]  # Indices where boundary=1

            selected_chunks = x[b, chunk_positions, :]  # [num_chunks, d_model]
            chunks_list.append(selected_chunks)
            indices_list.append(chunk_positions)

        # Pad to max num_chunks in batch
        max_chunks = max(c.size(0) for c in chunks_list)

        padded_chunks = torch.zeros(batch_size, max_chunks, d_model, device=x.device)
        padded_indices = torch.zeros(batch_size, max_chunks, dtype=torch.long, device=x.device)

        for b, (chunks, indices) in enumerate(zip(chunks_list, indices_list)):
            num_chunks = chunks.size(0)
            padded_chunks[b, :num_chunks] = chunks
            padded_indices[b, :num_chunks] = indices

        return padded_chunks, padded_indices


class Upsampler(nn.Module):
    """
    Upsampler: Reconstructs full sequence from chunks using STE.
    """

    def __init__(self):
        super().__init__()
        self.ste = StraightThroughEstimator.apply

    def forward(
        self,
        chunks: torch.Tensor,
        chunk_indices: torch.Tensor,
        target_len: int,
        boundary_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Upsample chunks back to full sequence.

        Args:
            chunks: Chunk embeddings [batch, num_chunks, d_model]
            chunk_indices: Original positions [batch, num_chunks]
            target_len: Target sequence length
            boundary_probs: Boundary probabilities [batch, target_len]

        Returns:
            Upsampled sequence [batch, target_len, d_model]
        """
        batch_size, num_chunks, d_model = chunks.shape

        output = torch.zeros(batch_size, target_len, d_model, device=chunks.device)

        for b in range(batch_size):
            for i, idx in enumerate(chunk_indices[b]):
                if idx >= target_len:
                    break

                # Compute contribution weight with STE
                p_t = boundary_probs[b, idx]
                boundary_binary = self.ste(p_t.unsqueeze(0))

                # c_t = p_t if boundary else (1 - p_t)
                weight = boundary_binary * p_t + (1 - boundary_binary) * (1 - p_t)

                output[b, idx] = weight * chunks[b, i]

        return output


class DynamicChunkingModule(nn.Module):
    """
    Complete Dynamic Chunking Module for H-Net.

    Integrates:
    1. Routing (boundary detection)
    2. Downsampling (select boundaries)
    3. Smoothing (EMA for differentiability) - CRITICAL
    4. Upsampling (reconstruct with STE)
    """

    def __init__(
        self,
        d_model: int,
        boundary_threshold: float = 0.5,
        smoothing_momentum: float = 0.9,
        use_smoothing: bool = True  # CRITICAL: Should always be True
    ):
        super().__init__()

        self.use_smoothing = use_smoothing

        self.routing = RoutingModule(d_model, boundary_threshold)
        self.downsampler = Downsampler()
        self.smoothing = SmoothingModule(smoothing_momentum) if use_smoothing else None
        self.upsampler = Upsampler()

    def forward(
        self,
        x: torch.Tensor,
        return_boundaries: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply dynamic chunking.

        Args:
            x: Input sequence [batch, seq_len, d_model]
            return_boundaries: If True, return boundary probabilities

        Returns:
            Tuple of:
                - chunks: Compressed chunks [batch, num_chunks, d_model]
                - chunk_indices: Original positions
                - boundary_probs: (optional) Boundary probabilities
        """
        # 1. Routing: Detect boundaries
        boundary_probs, boundaries = self.routing(x)

        # 2. Downsampling: Select boundary positions
        chunks, chunk_indices = self.downsampler(x, boundaries)

        # 3. Smoothing (if enabled)
        if self.use_smoothing and self.smoothing is not None:
            chunks = self.smoothing(chunks, boundary_probs[:, :chunks.size(1)])

        if return_boundaries:
            return chunks, chunk_indices, boundary_probs
        else:
            return chunks, chunk_indices, None

    def reconstruct(
        self,
        chunks: torch.Tensor,
        chunk_indices: torch.Tensor,
        target_len: int,
        boundary_probs: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct full sequence from chunks."""
        return self.upsampler(chunks, chunk_indices, target_len, boundary_probs)
