"""
Unit tests for utility functions.

Tests:
- initialization.py: TruncatedNormal, parameter counting
- metrics.py: APM, APTE, efficiency metrics
- collators.py: Dynamic batching
"""

import pytest
import torch
import torch.nn as nn

from codking.utils.initialization import (
    init_hrm_states,
    truncated_normal_,
    count_parameters
)
from codking.utils.metrics import (
    compute_apm,
    compute_apte,
    measure_inference_latency,
    measure_memory_footprint,
    compute_throughput
)
from codking.utils.collators import ChunkBatchCollator


class TestInitialization:
    """Test initialization utilities."""

    def test_truncated_normal(self, device):
        """Test truncated normal initialization."""
        tensor = torch.empty(100, 100, device=device)
        truncated_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0)

        # Check range
        assert tensor.min() >= -2.0
        assert tensor.max() <= 2.0

        # Check approximately normal
        assert abs(tensor.mean().item()) < 0.1
        assert abs(tensor.std().item() - 1.0) < 0.2

    def test_init_hrm_states(self, device, sample_batch_size):
        """Test HRM state initialization."""
        hidden_dim = 64
        z_H, z_L = init_hrm_states(
            batch_size=sample_batch_size,
            hidden_dim=hidden_dim,
            device=device,
            sigma=1.0,
            truncation=2.0
        )

        # Check shapes
        assert z_H.shape == (sample_batch_size, hidden_dim)
        assert z_L.shape == (sample_batch_size, hidden_dim)

        # Check values are in range
        assert z_H.min() >= -2.0
        assert z_H.max() <= 2.0
        assert z_L.min() >= -2.0
        assert z_L.max() <= 2.0

    def test_count_parameters(self):
        """Test parameter counting."""
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 20),  # 10*20 + 20 = 220
            nn.Linear(20, 5)    # 20*5 + 5 = 105
        )

        # Total: 325 parameters
        total_params = count_parameters(model)
        assert total_params == 325

        # Trainable only (all trainable by default)
        trainable_params = count_parameters(model, trainable_only=True)
        assert trainable_params == 325

        # Freeze one layer
        for param in model[0].parameters():
            param.requires_grad = False

        trainable_params = count_parameters(model, trainable_only=True)
        assert trainable_params == 105  # Only second layer


class TestMetrics:
    """Test metric computations."""

    def test_compute_apm(self):
        """Test Accuracy Per Million Parameters."""
        accuracy = 0.85
        num_params = 27_000_000

        apm = compute_apm(accuracy, num_params)

        # APM = accuracy / (params / 1M) = 0.85 / 27 ≈ 0.0315
        assert abs(apm - 0.0315) < 0.001

    def test_compute_apte(self):
        """Test Accuracy Per Training Example."""
        accuracy = 0.85
        num_examples = 1000

        apte = compute_apte(accuracy, num_examples)

        # APTE = accuracy / examples = 0.85 / 1000 = 0.00085
        assert abs(apte - 0.00085) < 0.00001

    def test_measure_inference_latency(self, device):
        """Test latency measurement."""
        # Simple model
        model = nn.Linear(100, 100).to(device)
        input_tensor = torch.randn(4, 100, device=device)

        latency_ms = measure_inference_latency(model, input_tensor, num_runs=10)

        # Should be positive and reasonable
        assert latency_ms > 0
        assert latency_ms < 1000  # Less than 1 second for simple model

    def test_measure_memory_footprint(self, device):
        """Test memory footprint computation."""
        model = nn.Linear(1000, 1000).to(device)

        memory_mb = measure_memory_footprint(model)

        # Linear layer: 1000*1000 params = 1M params
        # FP32: 4 bytes per param = 4MB
        # Should be around 4MB
        assert memory_mb > 3.0
        assert memory_mb < 10.0

    def test_compute_throughput(self):
        """Test throughput computation."""
        # compute_throughput takes latency_ms and batch_size
        latency_ms = 10.0
        batch_size = 8

        throughput = compute_throughput(latency_ms, batch_size=batch_size)

        # Should be positive
        assert throughput > 0
        # 8 samples / 0.01 seconds = 800 samples/sec
        assert abs(throughput - 800.0) < 1.0


class TestCollators:
    """Test batch collators."""

    def test_chunk_batch_collator_basic(self):
        """Test basic chunk batch collation."""
        collator = ChunkBatchCollator()

        # Create batch with variable-length chunks
        batch = [
            (torch.randn(10, 64), torch.tensor(0)),  # 10 chunks
            (torch.randn(15, 64), torch.tensor(1)),  # 15 chunks
            (torch.randn(8, 64), torch.tensor(0)),   # 8 chunks
        ]

        result = collator(batch)

        # Check shapes
        assert result['padded_chunks'].shape == (3, 15, 64)  # Max length is 15
        assert result['labels'].shape == (3,)
        assert result['attention_mask'].shape == (3, 15)

        # Check padding
        assert result['attention_mask'][0, :10].all()  # First 10 are valid
        assert not result['attention_mask'][0, 10:].any()  # Rest are padding

        assert result['attention_mask'][1].all()  # All 15 are valid

        assert result['attention_mask'][2, :8].all()  # First 8 are valid
        assert not result['attention_mask'][2, 8:].any()  # Rest are padding

    def test_chunk_batch_collator_padding_value(self):
        """Test custom padding value."""
        collator = ChunkBatchCollator(pad_value=-1.0)

        batch = [
            (torch.randn(5, 32), torch.tensor(0)),
            (torch.randn(3, 32), torch.tensor(1)),
        ]

        result = collator(batch)

        # Check padded region uses pad_value
        assert result['padded_chunks'].shape == (2, 5, 32)
        # Second sample, positions 3-4 should be padded with -1.0
        assert (result['padded_chunks'][1, 3:] == -1.0).all()

    def test_chunk_batch_collator_empty(self):
        """Test empty batch handling."""
        collator = ChunkBatchCollator()

        # Empty batch should raise error or handle gracefully
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            collator([])


class TestMetricsIntegration:
    """Integration tests for metrics."""

    def test_efficiency_comparison(self):
        """Test efficiency comparison against SOTA."""
        # CodKing
        codking_accuracy = 0.80
        codking_params = 27_000_000

        # SOTA (DeepSeek R1-Distill-1.5B)
        sota_accuracy = 0.85
        sota_params = 1_500_000_000

        # Compute efficiency
        codking_apm = compute_apm(codking_accuracy, codking_params)
        sota_apm = compute_apm(sota_accuracy, sota_params)

        # CodKing should be more parameter efficient
        efficiency_ratio = codking_apm / sota_apm
        assert efficiency_ratio > 50  # At least 50× more efficient

        print(f"Efficiency ratio: {efficiency_ratio:.1f}×")

    def test_data_efficiency_comparison(self):
        """Test data efficiency comparison."""
        # CodKing with few-shot
        codking_accuracy = 0.75
        codking_examples = 1000

        # SOTA
        sota_accuracy = 0.85
        sota_examples = 800_000

        # Compute data efficiency
        codking_apte = compute_apte(codking_accuracy, codking_examples)
        sota_apte = compute_apte(sota_accuracy, sota_examples)

        # CodKing should be more data efficient
        efficiency_ratio = codking_apte / sota_apte
        assert efficiency_ratio > 700  # At least 700× more efficient

        print(f"Data efficiency ratio: {efficiency_ratio:.1f}×")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
