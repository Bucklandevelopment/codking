"""
HNet-HRM Interface for chunk routing.

Routes chunks from H-Net to appropriate HRM modules based on abstraction level.
Implements three strategies: variance-based, temporal, and learned routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Literal


class VarianceRouter(nn.Module):
    """
    Variance-based routing strategy (RECOMMENDED).

    High variance chunks → H-module (abstract, high-level)
    Low variance chunks → L-module (detailed, low-level)
    """

    def __init__(self, use_median_threshold: bool = True):
        """
        Initialize variance router.

        Args:
            use_median_threshold: Use median variance as threshold (dynamic)
                                 If False, uses fixed threshold of 0.0
        """
        super().__init__()
        self.use_median_threshold = use_median_threshold

    def forward(
        self,
        chunks: torch.Tensor,
        boundary_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Route chunks based on variance.

        Args:
            chunks: Chunk embeddings [batch, num_chunks, d_model]
            boundary_probs: Optional boundary probabilities [batch, num_chunks]

        Returns:
            Dictionary containing:
                - h_chunks: High-variance chunks for H-module
                - l_chunks: Low-variance chunks for L-module
                - h_indices: Original indices of H-chunks
                - l_indices: Original indices of L-chunks
                - variance: Variance values for all chunks
        """
        batch_size, num_chunks, d_model = chunks.shape

        # Compute variance across embedding dimension
        variance = chunks.var(dim=-1)  # [batch, num_chunks]

        # Determine threshold
        if self.use_median_threshold:
            threshold = variance.median()
        else:
            threshold = 0.0

        # Route based on variance
        h_mask = variance > threshold  # High variance → H-module
        l_mask = ~h_mask  # Low variance → L-module

        # Split chunks
        h_chunks_list = []
        l_chunks_list = []
        h_indices_list = []
        l_indices_list = []

        for b in range(batch_size):
            h_idx = torch.where(h_mask[b])[0]
            l_idx = torch.where(l_mask[b])[0]

            h_chunks_list.append(chunks[b, h_idx])
            l_chunks_list.append(chunks[b, l_idx])
            h_indices_list.append(h_idx)
            l_indices_list.append(l_idx)

        # Pad to max length in batch
        max_h = max(h.size(0) for h in h_chunks_list) if h_chunks_list[0].size(0) > 0 else 1
        max_l = max(l.size(0) for l in l_chunks_list) if l_chunks_list[0].size(0) > 0 else 1

        h_chunks_padded = torch.zeros(batch_size, max_h, d_model, device=chunks.device)
        l_chunks_padded = torch.zeros(batch_size, max_l, d_model, device=chunks.device)
        h_indices_padded = torch.zeros(batch_size, max_h, dtype=torch.long, device=chunks.device)
        l_indices_padded = torch.zeros(batch_size, max_l, dtype=torch.long, device=chunks.device)

        for b, (h_c, l_c, h_i, l_i) in enumerate(zip(h_chunks_list, l_chunks_list, h_indices_list, l_indices_list)):
            if h_c.size(0) > 0:
                h_chunks_padded[b, :h_c.size(0)] = h_c
                h_indices_padded[b, :h_i.size(0)] = h_i
            if l_c.size(0) > 0:
                l_chunks_padded[b, :l_c.size(0)] = l_c
                l_indices_padded[b, :l_i.size(0)] = l_i

        return {
            'h_chunks': h_chunks_padded,
            'l_chunks': l_chunks_padded,
            'h_indices': h_indices_padded,
            'l_indices': l_indices_padded,
            'variance': variance,
            'threshold': threshold
        }


class TemporalRouter(nn.Module):
    """
    Temporal routing strategy.

    Start/end chunks → H-module (global context)
    Middle chunks → L-module (local processing)
    """

    def __init__(self, boundary_ratio: float = 0.2):
        """
        Initialize temporal router.

        Args:
            boundary_ratio: Fraction of sequence considered as boundaries
                           (e.g., 0.2 = first/last 20%)
        """
        super().__init__()
        self.boundary_ratio = boundary_ratio

    def forward(
        self,
        chunks: torch.Tensor,
        boundary_probs: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Route chunks based on temporal position."""
        batch_size, num_chunks, d_model = chunks.shape

        boundary_size = max(1, int(num_chunks * self.boundary_ratio))

        # H-module: first and last boundary_size chunks
        h_indices = torch.cat([
            torch.arange(boundary_size, device=chunks.device),
            torch.arange(num_chunks - boundary_size, num_chunks, device=chunks.device)
        ])

        # L-module: middle chunks
        l_indices = torch.arange(boundary_size, num_chunks - boundary_size, device=chunks.device)

        # Extract chunks
        h_chunks = chunks[:, h_indices]  # [batch, 2*boundary_size, d_model]
        l_chunks = chunks[:, l_indices]  # [batch, num_chunks - 2*boundary_size, d_model]

        # Expand indices for batch
        h_indices_batch = h_indices.unsqueeze(0).expand(batch_size, -1)
        l_indices_batch = l_indices.unsqueeze(0).expand(batch_size, -1)

        return {
            'h_chunks': h_chunks,
            'l_chunks': l_chunks,
            'h_indices': h_indices_batch,
            'l_indices': l_indices_batch
        }


class LearnedRouter(nn.Module):
    """
    Learned routing strategy with Gumbel-Softmax.

    Neural network learns optimal routing.
    """

    def __init__(self, d_model: int, hidden_dim: int = 512, temperature: float = 1.0):
        """
        Initialize learned router.

        Args:
            d_model: Chunk embedding dimension
            hidden_dim: Hidden dimension for router network
            temperature: Gumbel-Softmax temperature
        """
        super().__init__()

        self.temperature = temperature

        # Router network: chunk → [P(H-module), P(L-module)]
        self.router = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 2)  # 2 classes: H or L
        )

    def forward(
        self,
        chunks: torch.Tensor,
        boundary_probs: Optional[torch.Tensor] = None,
        hard: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Route chunks using learned router.

        Args:
            chunks: Chunk embeddings [batch, num_chunks, d_model]
            boundary_probs: Optional boundary probabilities
            hard: Use hard (argmax) routing during inference

        Returns:
            Dictionary with routed chunks and routing probabilities
        """
        batch_size, num_chunks, d_model = chunks.shape

        # Compute routing logits
        logits = self.router(chunks)  # [batch, num_chunks, 2]

        # Gumbel-Softmax sampling
        if self.training:
            routing_probs = F.gumbel_softmax(logits, tau=self.temperature, hard=hard, dim=-1)
        else:
            # Hard routing during inference
            routing_probs = F.softmax(logits, dim=-1)
            if hard:
                routing_probs = F.one_hot(routing_probs.argmax(dim=-1), num_classes=2).float()

        # routing_probs: [batch, num_chunks, 2]
        # routing_probs[:, :, 0] = P(H-module)
        # routing_probs[:, :, 1] = P(L-module)

        # Soft routing: weight chunks by probabilities
        h_weights = routing_probs[:, :, 0].unsqueeze(-1)  # [batch, num_chunks, 1]
        l_weights = routing_probs[:, :, 1].unsqueeze(-1)

        h_chunks = chunks * h_weights  # [batch, num_chunks, d_model]
        l_chunks = chunks * l_weights

        # For hard routing, also provide indices
        h_mask = routing_probs[:, :, 0] > 0.5
        l_mask = routing_probs[:, :, 1] > 0.5

        return {
            'h_chunks': h_chunks,
            'l_chunks': l_chunks,
            'routing_probs': routing_probs,
            'h_mask': h_mask,
            'l_mask': l_mask,
            'routing_logits': logits
        }


class HNetHRMInterface:
    """
    Complete interface for routing H-Net chunks to HRM modules.

    Supports three routing strategies:
    1. Variance-based (recommended)
    2. Temporal
    3. Learned
    """

    def __init__(
        self,
        strategy: Literal['variance', 'temporal', 'learned'] = 'variance',
        d_model: int = 1536,
        **kwargs
    ):
        """
        Initialize HNet-HRM interface.

        Args:
            strategy: Routing strategy to use
            d_model: Chunk embedding dimension
            **kwargs: Strategy-specific arguments
        """
        self.strategy = strategy

        if strategy == 'variance':
            self.router = VarianceRouter(
                use_median_threshold=kwargs.get('use_median_threshold', True)
            )
        elif strategy == 'temporal':
            self.router = TemporalRouter(
                boundary_ratio=kwargs.get('boundary_ratio', 0.2)
            )
        elif strategy == 'learned':
            self.router = LearnedRouter(
                d_model=d_model,
                hidden_dim=kwargs.get('hidden_dim', 512),
                temperature=kwargs.get('temperature', 1.0)
            )
        else:
            raise ValueError(f"Unknown routing strategy: {strategy}")

    def route_chunks(
        self,
        chunks: torch.Tensor,
        boundary_probs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Route chunks to H/L modules.

        Args:
            chunks: Chunk embeddings from H-Net main network [batch, num_chunks, d_model]
            boundary_probs: Optional boundary probabilities from dynamic chunking
            **kwargs: Router-specific arguments

        Returns:
            Dictionary with routed chunks for H-module and L-module
        """
        return self.router(chunks, boundary_probs, **kwargs)

    def __call__(self, *args, **kwargs):
        """Allow interface to be called directly."""
        return self.route_chunks(*args, **kwargs)


# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_chunks = 256
    d_model = 1536

    # Test variance-based routing (RECOMMENDED)
    print("Testing Variance-based Routing:")
    chunks = torch.randn(batch_size, num_chunks, d_model)

    interface = HNetHRMInterface(strategy='variance')
    routed = interface(chunks)

    print(f"  Input chunks: {chunks.shape}")
    print(f"  H-chunks (abstract): {routed['h_chunks'].shape}")
    print(f"  L-chunks (detailed): {routed['l_chunks'].shape}")
    print(f"  Variance threshold: {routed['threshold']:.4f}")
    print(f"  Mean variance: {routed['variance'].mean():.4f}")

    # Test temporal routing
    print("\nTesting Temporal Routing:")
    interface_temporal = HNetHRMInterface(strategy='temporal', boundary_ratio=0.2)
    routed_temporal = interface_temporal(chunks)

    print(f"  H-chunks (boundaries): {routed_temporal['h_chunks'].shape}")
    print(f"  L-chunks (middle): {routed_temporal['l_chunks'].shape}")

    # Test learned routing
    print("\nTesting Learned Routing:")
    interface_learned = HNetHRMInterface(strategy='learned', d_model=d_model)
    routed_learned = interface_learned(chunks)

    print(f"  H-chunks (learned): {routed_learned['h_chunks'].shape}")
    print(f"  L-chunks (learned): {routed_learned['l_chunks'].shape}")
    print(f"  Routing probs shape: {routed_learned['routing_probs'].shape}")
    print(f"  Mean H-prob: {routed_learned['routing_probs'][:, :, 0].mean():.4f}")
    print(f"  Mean L-prob: {routed_learned['routing_probs'][:, :, 1].mean():.4f}")
