"""
Tests for mHC (Manifold-Constrained Hyper-Connections) module.

Tests cover:
1. SinkhornKnopp projection correctness
2. mHCResidual forward/backward pass
3. mHCTransformerBlock functionality
4. HNetMainNetworkMHC integration
5. Numerical stability
6. Training dynamics
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

# Import mHC modules
from codking.models.mhc.mhc_layers import (
    SinkhornKnopp,
    mHCResidual,
    mHCTransformerBlock,
    mHCMultiHeadAttention,
    mHCFeedForward,
    compute_doubly_stochastic_loss,
    orthogonality_regularization,
)
from codking.models.mhc.main_network_mhc import (
    HNetMainNetworkMHC,
    HNetMainNetworkFactory,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mhc_config_small():
    """Small mHC configuration for testing."""
    return {
        'd_model': 64,
        'n_streams': 4,
        'sinkhorn_iterations': 5,
        'num_heads': 4,
        'dim_head': 16,
        'dim_ff': 256,
        'dropout': 0.1,
    }


@pytest.fixture
def mhc_network_config_small():
    """Small network configuration for testing."""
    return {
        'num_layers': 4,
        'd_model': 128,
        'num_heads': 4,
        'dim_head': 32,
        'dim_ff': 512,
        'n_streams': 4,
        'dropout': 0.1,
        'sinkhorn_iterations': 5,
        'input_dim': 64,
    }


@pytest.fixture
def sample_mhc_input():
    """Sample input for mHC tests."""
    batch_size = 2
    seq_len = 16
    d_model = 64
    return torch.randn(batch_size, seq_len, d_model)


@pytest.fixture
def sample_stream_input():
    """Sample stream input for mHC tests."""
    batch_size = 2
    seq_len = 16
    n_streams = 4
    d_model = 64
    return torch.randn(batch_size, seq_len, n_streams, d_model)


# =============================================================================
# SinkhornKnopp Tests
# =============================================================================

class TestSinkhornKnopp:
    """Tests for Sinkhorn-Knopp projection."""

    def test_output_is_doubly_stochastic(self):
        """Verify output is doubly stochastic (rows and cols sum to 1)."""
        sk = SinkhornKnopp(iterations=10)
        A = torch.randn(4, 4)
        A_ds = sk(A)

        # Check row sums
        row_sums = A_ds.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(4), atol=1e-5), \
            f"Row sums should be 1, got {row_sums}"

        # Check column sums
        col_sums = A_ds.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones(4), atol=1e-5), \
            f"Column sums should be 1, got {col_sums}"

    def test_output_is_non_negative(self):
        """Verify all entries are non-negative."""
        sk = SinkhornKnopp(iterations=10)
        A = torch.randn(4, 4) * 10  # Large random values
        A_ds = sk(A)

        assert (A_ds >= 0).all(), "All entries should be non-negative"

    def test_batched_input(self):
        """Test with batched input."""
        sk = SinkhornKnopp(iterations=5)
        A = torch.randn(8, 4, 4)  # Batch of 8
        A_ds = sk(A)

        assert A_ds.shape == A.shape, "Output shape should match input"

        # Check each batch element
        for i in range(8):
            row_sums = A_ds[i].sum(dim=-1)
            col_sums = A_ds[i].sum(dim=-2)
            assert torch.allclose(row_sums, torch.ones(4), atol=1e-4)
            assert torch.allclose(col_sums, torch.ones(4), atol=1e-4)

    def test_gradient_flow(self):
        """Verify gradients flow through projection."""
        sk = SinkhornKnopp(iterations=5)
        A = torch.randn(4, 4, requires_grad=True)
        A_ds = sk(A)

        # Compute loss and backprop
        loss = A_ds.sum()
        loss.backward()

        assert A.grad is not None, "Gradients should flow through Sinkhorn-Knopp"
        assert not torch.isnan(A.grad).any(), "Gradients should not contain NaN"

    def test_identity_preservation(self):
        """Near-identity input should produce near-identity output."""
        sk = SinkhornKnopp(iterations=10)
        # Input that should project to identity
        A = torch.zeros(4, 4)
        A_ds = sk(A)

        # For zero input, all entries should be equal (uniform doubly stochastic)
        expected = torch.ones(4, 4) / 4
        assert torch.allclose(A_ds, expected, atol=1e-5), \
            f"Zero input should give uniform matrix, got {A_ds}"

    def test_numerical_stability(self):
        """Test with extreme values."""
        sk = SinkhornKnopp(iterations=10)

        # Very large values
        A_large = torch.randn(4, 4) * 100
        A_ds = sk(A_large)
        assert not torch.isnan(A_ds).any(), "Should handle large values"
        assert not torch.isinf(A_ds).any(), "Should not produce Inf"

        # Very small values
        A_small = torch.randn(4, 4) * 0.001
        A_ds = sk(A_small)
        assert not torch.isnan(A_ds).any(), "Should handle small values"


# =============================================================================
# mHCResidual Tests
# =============================================================================

class TestMHCResidual:
    """Tests for mHCResidual module."""

    def test_first_layer_expansion(self, sample_mhc_input, mhc_config_small):
        """Test expansion from single stream to multiple streams."""
        mhc = mHCResidual(
            d_model=mhc_config_small['d_model'],
            n_streams=mhc_config_small['n_streams']
        )

        layer_output = torch.randn_like(sample_mhc_input)
        output = mhc(sample_mhc_input, layer_output, is_first_layer=True)

        batch, seq, d = sample_mhc_input.shape
        expected_shape = (batch, seq, mhc_config_small['n_streams'], d)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"

    def test_subsequent_layer_processing(self, sample_stream_input, mhc_config_small):
        """Test processing when already in stream format."""
        mhc = mHCResidual(
            d_model=mhc_config_small['d_model'],
            n_streams=mhc_config_small['n_streams']
        )

        layer_output = torch.randn(
            sample_stream_input.shape[0],
            sample_stream_input.shape[1],
            mhc_config_small['d_model']
        )
        output = mhc(sample_stream_input, layer_output, is_first_layer=False)

        assert output.shape == sample_stream_input.shape, \
            f"Output shape should match input stream shape"

    def test_stream_contraction(self, sample_stream_input, mhc_config_small):
        """Test contraction from streams to single output."""
        mhc = mHCResidual(
            d_model=mhc_config_small['d_model'],
            n_streams=mhc_config_small['n_streams']
        )

        output = mhc.get_single_output(sample_stream_input)

        batch, seq, n, d = sample_stream_input.shape
        expected_shape = (batch, seq, d)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"

    def test_projected_matrices_properties(self, mhc_config_small):
        """Test that projected matrices have correct properties."""
        mhc = mHCResidual(
            d_model=mhc_config_small['d_model'],
            n_streams=mhc_config_small['n_streams'],
            sinkhorn_iterations=10
        )

        H_res, H_pre, H_post = mhc.get_projected_matrices()

        # H_res should be doubly stochastic
        row_sums = H_res.sum(dim=-1)
        col_sums = H_res.sum(dim=-2)
        assert torch.allclose(row_sums, torch.ones(mhc_config_small['n_streams']), atol=1e-4)
        assert torch.allclose(col_sums, torch.ones(mhc_config_small['n_streams']), atol=1e-4)

        # All should be non-negative
        assert (H_res >= 0).all()
        assert (H_pre >= 0).all()
        assert (H_post >= 0).all()

    def test_gradient_flow(self, sample_mhc_input, mhc_config_small):
        """Verify gradients flow through mHC residual."""
        mhc = mHCResidual(
            d_model=mhc_config_small['d_model'],
            n_streams=mhc_config_small['n_streams']
        )

        x = sample_mhc_input.requires_grad_(True)
        layer_output = torch.randn_like(x)

        output = mhc(x, layer_output, is_first_layer=True)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert mhc.H_res_raw.grad is not None, "Gradients should flow to H_res"


# =============================================================================
# mHCTransformerBlock Tests
# =============================================================================

class TestMHCTransformerBlock:
    """Tests for mHCTransformerBlock."""

    def test_first_layer_forward(self, sample_mhc_input, mhc_config_small):
        """Test forward pass for first layer."""
        block = mHCTransformerBlock(
            d_model=mhc_config_small['d_model'],
            num_heads=mhc_config_small['num_heads'],
            dim_head=mhc_config_small['dim_head'],
            n_streams=mhc_config_small['n_streams'],
            dropout=0.0  # Disable dropout for deterministic test
        )

        output = block(sample_mhc_input, is_first_layer=True)

        batch, seq, d = sample_mhc_input.shape
        expected_shape = (batch, seq, mhc_config_small['n_streams'], d)
        assert output.shape == expected_shape

    def test_subsequent_layer_forward(self, sample_stream_input, mhc_config_small):
        """Test forward pass for subsequent layers."""
        block = mHCTransformerBlock(
            d_model=mhc_config_small['d_model'],
            num_heads=mhc_config_small['num_heads'],
            dim_head=mhc_config_small['dim_head'],
            n_streams=mhc_config_small['n_streams'],
            dropout=0.0
        )

        output = block(sample_stream_input, is_first_layer=False)
        assert output.shape == sample_stream_input.shape

    def test_attention_mask(self, sample_mhc_input, mhc_config_small):
        """Test that attention mask is applied correctly."""
        block = mHCTransformerBlock(
            d_model=mhc_config_small['d_model'],
            num_heads=mhc_config_small['num_heads'],
            dim_head=mhc_config_small['dim_head'],
            n_streams=mhc_config_small['n_streams'],
            dropout=0.0
        )

        batch, seq, _ = sample_mhc_input.shape
        mask = torch.ones(batch, seq)
        mask[:, seq//2:] = 0  # Mask second half

        output_masked = block(sample_mhc_input, is_first_layer=True, attention_mask=mask)
        output_unmasked = block(sample_mhc_input, is_first_layer=True)

        # Outputs should differ when mask is applied
        assert not torch.allclose(output_masked, output_unmasked, atol=1e-3)

    def test_gradient_flow(self, sample_mhc_input, mhc_config_small):
        """Test gradient flow through entire block."""
        block = mHCTransformerBlock(
            d_model=mhc_config_small['d_model'],
            num_heads=mhc_config_small['num_heads'],
            dim_head=mhc_config_small['dim_head'],
            n_streams=mhc_config_small['n_streams']
        )

        x = sample_mhc_input.clone().requires_grad_(True)
        output = block(x, is_first_layer=True)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        # Check mHC parameters receive gradients
        assert block.mhc_attn.H_res_raw.grad is not None
        assert block.mhc_ff.H_res_raw.grad is not None


# =============================================================================
# HNetMainNetworkMHC Tests
# =============================================================================

class TestHNetMainNetworkMHC:
    """Tests for full HNet Main Network with mHC."""

    def test_forward_pass(self, mhc_network_config_small):
        """Test basic forward pass."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, mhc_network_config_small['input_dim'])

        output = network(x)

        expected_shape = (batch_size, seq_len, mhc_network_config_small['d_model'])
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"

    def test_forward_with_mask(self, mhc_network_config_small):
        """Test forward pass with attention mask."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, mhc_network_config_small['input_dim'])
        mask = torch.ones(batch_size, seq_len)

        output = network(x, attention_mask=mask)
        assert output.shape == (batch_size, seq_len, mhc_network_config_small['d_model'])

    def test_metrics_computation(self, mhc_network_config_small):
        """Test that metrics are computed correctly."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, mhc_network_config_small['input_dim'])

        output, metrics = network(x, return_metrics=True)

        assert 'doubly_stochastic_loss' in metrics
        assert 'orthogonality_loss' in metrics
        assert metrics['doubly_stochastic_loss'] >= 0
        assert metrics['orthogonality_loss'] >= 0

    def test_regularization_loss(self, mhc_network_config_small):
        """Test regularization loss computation."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        # Run forward to populate gradients
        x = torch.randn(2, 16, mhc_network_config_small['input_dim'])
        _ = network(x)

        reg_loss = network.get_regularization_loss()
        assert reg_loss >= 0
        assert not torch.isnan(reg_loss)

    def test_parameter_counts(self, mhc_network_config_small):
        """Test parameter counting methods."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        total = network.get_num_params()
        mhc_specific = network.get_mhc_params()

        assert total > 0
        assert mhc_specific > 0
        assert mhc_specific < total

    def test_matrix_visualization(self, mhc_network_config_small):
        """Test mixing matrix visualization."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        matrices = network.visualize_mixing_matrices(layer_idx=0)

        assert 'attn_H_res' in matrices
        assert 'ff_H_res' in matrices
        assert matrices['attn_H_res'].shape == (
            mhc_network_config_small['n_streams'],
            mhc_network_config_small['n_streams']
        )

    def test_gradient_flow_full_network(self, mhc_network_config_small):
        """Test gradient flow through entire network."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        x = torch.randn(2, 16, mhc_network_config_small['input_dim'], requires_grad=True)
        output = network(x)

        loss = output.sum()
        loss.backward()

        assert x.grad is not None

        # Check that mHC parameters received gradients
        for layer in network.layers:
            assert layer.mhc_attn.H_res_raw.grad is not None


# =============================================================================
# Factory Tests
# =============================================================================

class TestHNetMainNetworkFactory:
    """Tests for network factory."""

    def test_create_mhc_network(self):
        """Test creating mHC network via factory."""
        network = HNetMainNetworkFactory.create(
            use_mhc=True,
            num_layers=2,
            d_model=64,
            num_heads=4,
            input_dim=32
        )

        assert isinstance(network, HNetMainNetworkMHC)

    def test_create_standard_network(self):
        """Test creating standard network via factory."""
        # This test may fail if standard network not in path
        try:
            network = HNetMainNetworkFactory.create(
                use_mhc=False,
                num_layers=2,
                d_model=64,
                num_heads=4,
                input_dim=32
            )
            assert not isinstance(network, HNetMainNetworkMHC)
        except ImportError:
            pytest.skip("Standard HNetMainNetwork not available")


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_doubly_stochastic_loss(self):
        """Test doubly stochastic loss computation."""
        # Perfect doubly stochastic matrix
        H_ds = torch.ones(4, 4) / 4
        loss = compute_doubly_stochastic_loss(H_ds)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

        # Non-doubly stochastic matrix
        H_bad = torch.randn(4, 4)
        loss = compute_doubly_stochastic_loss(H_bad)
        assert loss > 0

    def test_orthogonality_regularization(self):
        """Test orthogonality regularization."""
        # Orthogonal matrix should have low loss
        H_orth = torch.eye(4)
        loss = orthogonality_regularization(H_orth)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

        # Non-orthogonal matrix should have higher loss
        H_non_orth = torch.ones(4, 4)
        loss = orthogonality_regularization(H_non_orth)
        assert loss > 0


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestMHCIntegration:
    """Integration tests for mHC with training."""

    def test_training_step(self, mhc_network_config_small):
        """Test a single training step."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)

        x = torch.randn(2, 16, mhc_network_config_small['input_dim'])
        target = torch.randn(2, 16, mhc_network_config_small['d_model'])

        # Forward
        output, metrics = network(x, return_metrics=True)

        # Loss
        task_loss = nn.functional.mse_loss(output, target)
        reg_loss = network.get_regularization_loss()
        total_loss = task_loss + reg_loss

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        assert not torch.isnan(total_loss)

    def test_multiple_training_steps(self, mhc_network_config_small):
        """Test multiple training steps for stability."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)

        losses = []
        for step in range(10):
            x = torch.randn(2, 16, mhc_network_config_small['input_dim'])
            target = torch.randn(2, 16, mhc_network_config_small['d_model'])

            output = network(x)
            loss = nn.functional.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            assert not np.isnan(loss.item()), f"NaN loss at step {step}"

        # Loss should generally decrease or stay stable
        assert losses[-1] < losses[0] * 2, "Loss should not explode"

    def test_doubly_stochastic_property_during_training(self, mhc_network_config_small):
        """Verify doubly stochastic property is maintained during training."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)
        optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)

        for _ in range(5):
            x = torch.randn(2, 16, mhc_network_config_small['input_dim'])
            target = torch.randn(2, 16, mhc_network_config_small['d_model'])

            output, metrics = network(x, return_metrics=True)
            loss = nn.functional.mse_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check doubly stochastic property
            ds_loss = metrics['doubly_stochastic_loss']
            assert ds_loss < 0.1, f"DS loss too high: {ds_loss}"


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestMHCPerformance:
    """Performance tests for mHC."""

    def test_inference_speed(self, mhc_network_config_small, measure_time):
        """Measure inference speed."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)
        network.eval()

        x = torch.randn(4, 64, mhc_network_config_small['input_dim'])

        # Warmup
        with torch.no_grad():
            _ = network(x)

        # Measure
        with measure_time() as timer:
            with torch.no_grad():
                for _ in range(10):
                    _ = network(x)

        avg_time = timer.elapsed / 10 * 1000  # ms
        print(f"\nAverage inference time: {avg_time:.2f} ms")
        # Should be reasonably fast
        assert avg_time < 1000, f"Inference too slow: {avg_time} ms"

    def test_memory_usage(self, mhc_network_config_small):
        """Estimate memory usage."""
        network = HNetMainNetworkMHC(**mhc_network_config_small)

        # Count parameters
        total_params = sum(p.numel() for p in network.parameters())
        param_memory = total_params * 4 / 1024 / 1024  # MB (float32)

        print(f"\nParameter count: {total_params:,}")
        print(f"Parameter memory: {param_memory:.2f} MB")

        # mHC overhead
        mhc_params = network.get_mhc_params()
        overhead = mhc_params / total_params * 100
        print(f"mHC overhead: {overhead:.2f}%")

        # Overhead should be reasonable (paper claims ~6.7%)
        assert overhead < 30, f"mHC overhead too high: {overhead}%"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
