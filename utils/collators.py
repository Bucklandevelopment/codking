"""
Dynamic batch collation utilities for variable-length chunks.
Handles padding and attention masking for H-Net chunked outputs.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any


class ChunkBatchCollator:
    """
    Collator for batching variable-length chunks from H-Net.

    Handles:
    - Dynamic padding to max sequence length in batch
    - Attention mask generation for padded positions
    - Proper tensor stacking
    """

    def __init__(self, pad_value: float = 0.0, padding_side: str = 'right'):
        """
        Initialize collator.

        Args:
            pad_value: Value to use for padding
            padding_side: 'right' or 'left' padding
        """
        self.pad_value = pad_value
        self.padding_side = padding_side

    def __call__(self, batch: List[Tuple[torch.Tensor, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of (chunks, label) tuples.

        Args:
            batch: List of (chunks, label) tuples
                   chunks: Tensor of shape [num_chunks, hidden_dim]
                   label: Any label type

        Returns:
            Dictionary containing:
                - padded_chunks: [batch_size, max_chunks, hidden_dim]
                - attention_mask: [batch_size, max_chunks]
                - labels: [batch_size] or appropriate shape
                - chunk_lengths: [batch_size] original lengths
        """
        chunks_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Get dimensions
        batch_size = len(chunks_list)
        max_chunks = max(chunks.size(0) for chunks in chunks_list)
        hidden_dim = chunks_list[0].size(-1)

        # Initialize padded tensors
        padded_chunks = torch.full(
            (batch_size, max_chunks, hidden_dim),
            self.pad_value,
            dtype=chunks_list[0].dtype
        )
        attention_mask = torch.zeros(batch_size, max_chunks, dtype=torch.bool)
        chunk_lengths = torch.zeros(batch_size, dtype=torch.long)

        # Fill padded tensors
        for i, chunks in enumerate(chunks_list):
            num_chunks = chunks.size(0)
            chunk_lengths[i] = num_chunks

            if self.padding_side == 'right':
                padded_chunks[i, :num_chunks] = chunks
                attention_mask[i, :num_chunks] = True
            else:  # left padding
                padded_chunks[i, -num_chunks:] = chunks
                attention_mask[i, -num_chunks:] = True

        # Handle labels
        if isinstance(labels[0], torch.Tensor):
            labels_tensor = torch.stack(labels)
        else:
            labels_tensor = torch.tensor(labels)

        return {
            'padded_chunks': padded_chunks,
            'attention_mask': attention_mask,
            'labels': labels_tensor,
            'chunk_lengths': chunk_lengths
        }


class ByteSequenceCollator:
    """
    Collator for raw byte sequences (input to H-Net).
    """

    def __init__(self, max_length: int = 8192, pad_value: int = 0):
        """
        Initialize collator.

        Args:
            max_length: Maximum sequence length
            pad_value: Value for padding
        """
        self.max_length = max_length
        self.pad_value = pad_value

    def __call__(self, batch: List[Tuple[torch.Tensor, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of byte sequences.

        Args:
            batch: List of (bytes_tensor, label) tuples

        Returns:
            Dictionary with padded sequences and masks
        """
        sequences = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        batch_size = len(sequences)

        # Truncate or pad to max_length
        padded_seqs = torch.full(
            (batch_size, self.max_length),
            self.pad_value,
            dtype=sequences[0].dtype
        )
        attention_mask = torch.zeros(batch_size, self.max_length, dtype=torch.bool)

        for i, seq in enumerate(sequences):
            length = min(seq.size(0), self.max_length)
            padded_seqs[i, :length] = seq[:length]
            attention_mask[i, :length] = True

        # Handle labels
        if isinstance(labels[0], torch.Tensor):
            labels_tensor = torch.stack(labels)
        else:
            labels_tensor = torch.tensor(labels)

        return {
            'input_ids': padded_seqs,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }


class AdaptiveBatchSampler:
    """
    Adaptive batch sampler that groups sequences of similar length.
    Improves efficiency by minimizing padding.
    """

    def __init__(self, lengths: List[int], batch_size: int, drop_last: bool = False):
        """
        Initialize sampler.

        Args:
            lengths: List of sequence lengths
            batch_size: Desired batch size
            drop_last: Drop last incomplete batch
        """
        self.lengths = lengths
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Sort indices by length
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def __iter__(self):
        """Generate batches of similar-length sequences."""
        batches = []
        current_batch = []

        for idx in self.sorted_indices:
            current_batch.append(idx)

            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []

        # Handle last batch
        if current_batch and not self.drop_last:
            batches.append(current_batch)

        # Shuffle batches to avoid training bias
        import random
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        """Return number of batches."""
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    Create causal attention mask for autoregressive modeling.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Causal mask of shape [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return ~mask  # True for allowed positions


def create_padding_mask(lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
    """
    Create padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths [batch_size]
        max_len: Maximum sequence length (if None, use max of lengths)

    Returns:
        Padding mask [batch_size, max_len] (True for valid positions)
    """
    if max_len is None:
        max_len = lengths.max().item()

    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len)
    mask = mask < lengths.unsqueeze(1)

    return mask


def combine_masks(padding_mask: torch.Tensor, causal_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Combine padding and causal masks.

    Args:
        padding_mask: [batch_size, seq_len] padding mask
        causal_mask: [seq_len, seq_len] causal mask (optional)

    Returns:
        Combined mask [batch_size, seq_len, seq_len]
    """
    batch_size, seq_len = padding_mask.shape

    # Expand padding mask to [batch_size, seq_len, seq_len]
    # Each query position can attend to all non-padded positions
    mask = padding_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)

    # Apply causal mask if provided
    if causal_mask is not None:
        mask = mask & causal_mask.unsqueeze(0)

    return mask
