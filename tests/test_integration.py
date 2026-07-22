"""
Integration tests for HNet + HRM pipeline.

Tests:
- interface.py: Chunk routing (variance, temporal, learned)
- pipeline.py: Complete CodKingPipeline, CybersecurityPipeline
"""

import pytest
import torch
import torch.nn as nn

from codking.models.integration.interface import (
    VarianceRouter,
    TemporalRouter,
    LearnedRouter,
    HNetHRMInterface
)
from codking.models.integration.pipeline import (
    CodKingPipeline,
    CybersecurityPipeline
)


class TestRoutingStrategies:
    """Test chunk routing strategies."""

    def test_variance_router(self, device):
        """Test variance-based routing."""
        batch_size, num_chunks, d_model = 4, 32, 128
        router = VarianceRouter(use_median_threshold=True).to(device)

        chunks = torch.randn(batch_size, num_chunks, d_model, device=device)

        # Make some chunks high variance, some low variance
        chunks[:, :10, :] *= 2.0  # High variance
        chunks[:, 10:, :] *= 0.5  # Low variance

        routed = router(chunks)

        # Check outputs
        assert 'h_chunks' in routed
        assert 'l_chunks' in routed
        assert 'h_indices' in routed
        assert 'l_indices' in routed
        assert 'variance' in routed
        assert 'threshold' in routed

        # Check shapes
        assert routed['h_chunks'].dim() == 3
        assert routed['l_chunks'].dim() == 3

        # Variance should be computed for all chunks
        assert routed['variance'].shape == (batch_size, num_chunks)

    def test_temporal_router(self, device):
        """Test temporal routing."""
        batch_size, num_chunks, d_model = 4, 32, 128
        router = TemporalRouter(boundary_ratio=0.2).to(device)

        chunks = torch.randn(batch_size, num_chunks, d_model, device=device)

        routed = router(chunks)

        # Check outputs
        assert 'h_chunks' in routed
        assert 'l_chunks' in routed
        assert 'h_indices' in routed
        assert 'l_indices' in routed

        # H-chunks should be first and last 20% (boundary_ratio=0.2)
        h_size = routed['h_chunks'].shape[1]
        expected_h_size = 2 * int(num_chunks * 0.2)  # First + last
        assert h_size == expected_h_size

    def test_learned_router(self, device):
        """Test learned routing."""
        batch_size, num_chunks, d_model = 4, 32, 128
        router = LearnedRouter(d_model=d_model, hidden_dim=64).to(device)

        chunks = torch.randn(batch_size, num_chunks, d_model, device=device)

        routed = router(chunks, hard=False)

        # Check outputs
        assert 'h_chunks' in routed
        assert 'l_chunks' in routed
        assert 'routing_probs' in routed
        assert 'routing_logits' in routed

        # Routing probs should sum to 1
        probs = routed['routing_probs']
        assert probs.shape == (batch_size, num_chunks, 2)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, num_chunks, device=device))

    def test_hnet_hrm_interface(self, device):
        """Test complete HNetHRMInterface."""
        batch_size, num_chunks, d_model = 4, 32, 1536

        # Test variance strategy
        interface = HNetHRMInterface(strategy='variance', d_model=d_model)
        chunks = torch.randn(batch_size, num_chunks, d_model, device=device)

        routed = interface(chunks)

        assert 'h_chunks' in routed
        assert 'l_chunks' in routed

        # Test temporal strategy
        interface = HNetHRMInterface(strategy='temporal', d_model=d_model, boundary_ratio=0.2)
        routed = interface(chunks)

        assert 'h_chunks' in routed
        assert 'l_chunks' in routed

        # Test learned strategy
        interface = HNetHRMInterface(strategy='learned', d_model=d_model, hidden_dim=512)
        interface.router.to(device)
        routed = interface(chunks)

        assert 'h_chunks' in routed
        assert 'l_chunks' in routed


@pytest.mark.integration
class TestCodKingPipeline:
    """Test complete CodKing pipeline."""

    def test_pipeline_initialization(self, hnet_config_small, hrm_config_small):
        """Test pipeline initialization."""
        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            routing_strategy='variance',
            freeze_hnet=False
        )

        # Check components
        assert hasattr(pipeline, 'hnet')
        assert hasattr(pipeline, 'router')
        assert hasattr(pipeline, 'hrm')
        assert hasattr(pipeline, 'task_loss_fn')

    def test_pipeline_forward(self, hnet_config_small, hrm_config_small, device):
        """Test pipeline forward pass."""
        batch_size, seq_len = 4, 512
        num_classes = 10

        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=num_classes,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            routing_strategy='variance',
            freeze_hnet=False
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        output = pipeline(input_ids, targets=None, return_loss=False)

        # Check outputs
        assert 'output' in output
        assert output['output'].shape == (batch_size, num_classes)
        assert 'num_cycles' in output
        assert 'compression_ratio' in output
        assert 'num_chunks' in output

    def test_pipeline_with_loss(self, hnet_config_small, hrm_config_small, device):
        """Test pipeline with loss computation."""
        batch_size, seq_len = 4, 512
        num_classes = 10

        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=num_classes,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            routing_strategy='variance',
            freeze_hnet=False,
            loss_weight_hnet=0.3,
            loss_weight_hrm=0.7
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        output = pipeline(input_ids, targets=targets, return_loss=True)

        # Check loss outputs
        assert 'loss' in output
        assert 'loss_dict' in output
        assert output['loss'].item() >= 0

        # Check loss dict
        loss_dict = output['loss_dict']
        assert 'total_loss' in loss_dict
        assert 'loss_weight_hnet' in loss_dict
        assert 'loss_weight_hrm' in loss_dict

    def test_pipeline_frozen_hnet(self, hnet_config_small, hrm_config_small, device):
        """Test pipeline with frozen HNet."""
        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            freeze_hnet=True
        ).to(device)

        # Check HNet parameters are frozen
        for param in pipeline.hnet.parameters():
            assert not param.requires_grad

        # Check HRM parameters are trainable
        for param in pipeline.hrm.parameters():
            assert param.requires_grad

    def test_pipeline_parameter_count(self, hnet_config_small, hrm_config_small):
        """Test pipeline parameter counting."""
        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        params = pipeline.get_num_params()

        assert 'hnet' in params
        assert 'hrm' in params
        assert 'total' in params

        # Total should equal sum of components
        assert params['total'] == params['hnet'] + params['hrm'] + params['router']

    @pytest.mark.slow
    def test_pipeline_compression(self, hnet_config_small, hrm_config_small, device):
        """Test that pipeline achieves compression."""
        batch_size, seq_len = 4, 512

        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        output = pipeline(input_ids, return_loss=False)

        # Should achieve some compression
        compression_ratio = output['compression_ratio']
        assert compression_ratio > 1.0  # At least some compression

        # Ideally close to target (6:1), but for small test model may vary
        print(f"Compression ratio: {compression_ratio:.2f}:1")


@pytest.mark.integration
class TestCybersecurityPipeline:
    """Test cybersecurity-specific pipeline."""

    def test_cyber_pipeline_initialization(self, hnet_config_small, hrm_config_small):
        """Test cybersecurity pipeline initialization."""
        pipeline = CybersecurityPipeline(
            task_type='threat_detection',
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        # Should default to binary classification
        assert pipeline.num_classes == 2
        assert hasattr(pipeline, 'cybersecurity_task')

    def test_cyber_pipeline_threat_detection(self, hnet_config_small, hrm_config_small, device):
        """Test threat detection."""
        batch_size, seq_len = 4, 512

        pipeline = CybersecurityPipeline(
            task_type='threat_detection',
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        # Simulate log data (as bytes)
        log_data = torch.randint(0, 256, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 2, (batch_size,), device=device)  # Binary: 0=normal, 1=threat

        output = pipeline(log_data, targets=targets, return_loss=False)

        # Check threat-specific outputs
        assert 'output' in output
        assert output['output'].shape == (batch_size, 2)  # Binary classification

        if 'threat_score' in output:
            # Threat scores should be probabilities
            assert (output['threat_score'] >= 0).all()
            assert (output['threat_score'] <= 1).all()


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    def test_bytes_to_prediction(self, hnet_config_small, hrm_config_small, device):
        """Test complete pipeline from bytes to prediction."""
        batch_size = 2
        seq_len = 512
        num_classes = 10

        # Create pipeline
        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=num_classes,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            routing_strategy='variance'
        ).to(device)

        # Simulate byte input (from text encoding)
        text = "This is a test log entry for classification."
        byte_seq = text.encode('utf-8')
        byte_array = torch.frombuffer(byte_seq, dtype=torch.uint8)

        # Pad to seq_len
        if len(byte_array) < seq_len:
            padding = torch.zeros(seq_len - len(byte_array), dtype=torch.uint8)
            byte_array = torch.cat([byte_array, padding])
        else:
            byte_array = byte_array[:seq_len]

        # Create batch
        input_ids = byte_array.unsqueeze(0).repeat(batch_size, 1).long().to(device)

        # Forward pass
        with torch.no_grad():
            output = pipeline(input_ids, return_loss=False)

        # Check prediction
        predictions = output['output'].argmax(dim=-1)
        assert predictions.shape == (batch_size,)
        assert (predictions >= 0).all() and (predictions < num_classes).all()

    @pytest.mark.slow
    def test_training_step(self, hnet_config_small, hrm_config_small, device):
        """Test a training step."""
        batch_size = 2
        seq_len = 512
        num_classes = 10

        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=num_classes,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        optimizer = torch.optim.Adam(pipeline.parameters(), lr=1e-4)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
        targets = torch.randint(0, num_classes, (batch_size,), device=device)

        # Training step
        optimizer.zero_grad()
        output = pipeline(input_ids, targets=targets, return_loss=True)
        loss = output['loss']
        loss.backward()
        optimizer.step()

        # Loss should be computed
        assert loss.item() >= 0

    def test_inference_efficiency(self, hnet_config_small, hrm_config_small, device, measure_time):
        """Test inference efficiency."""
        batch_size = 4
        seq_len = 512
        num_classes = 10

        pipeline = CodKingPipeline(
            task_type='classification',
            num_classes=num_classes,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        pipeline.eval()

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        # Measure latency
        with torch.no_grad():
            with measure_time() as timer:
                output = pipeline(input_ids, return_loss=False)

        latency_ms = timer.elapsed * 1000

        print(f"Inference latency: {latency_ms:.2f}ms")
        print(f"Throughput: {batch_size / timer.elapsed:.2f} samples/sec")
        print(f"Compression: {output['compression_ratio']:.2f}:1")
        print(f"HRM cycles: {output['num_cycles']}")

        # For small test model on CPU, should still be reasonably fast
        assert latency_ms < 10000  # Less than 10 seconds


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not slow"])
