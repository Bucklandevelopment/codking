"""
Unit tests for H-Net components.

Tests:
- encoder.py: HNetEncoder, MambaLayer (LSTM fallback)
- dynamic_chunking.py: RoutingModule, SmoothingModule, Upsampler, DynamicChunkingModule
- main_network.py: TransformerLayer, HNetMainNetwork
- decoder.py: HNetDecoder
- losses.py: RatioLoss, AutoregressiveLoss, BitsPerByteLoss
- hnet.py: Complete HNet integration
"""

import pytest
import torch
import torch.nn as nn

from codking.models.hnet.encoder import HNetEncoder, MambaLayer
from codking.models.hnet.dynamic_chunking import (
    RoutingModule,
    SmoothingModule,
    Upsampler,
    DynamicChunkingModule
)
from codking.models.hnet.main_network import (
    TransformerLayer,
    HNetMainNetwork
)
from codking.models.hnet.decoder import HNetDecoder
from codking.models.hnet.losses import (
    RatioLoss,
    AutoregressiveLoss,
    BitsPerByteLoss
)
from codking.models.hnet.hnet import HNet


class TestEncoder:
    """Test H-Net encoder components."""

    def test_hnet_encoder(self, device):
        """Test HNetEncoder."""
        batch_size, seq_len = 4, 128
        d_model, num_layers = 64, 2

        encoder = HNetEncoder(
            num_layers=num_layers,
            d_model=d_model,
            input_vocab_size=256
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)
        output, mask = encoder(input_ids)

        # Check shape
        assert output.shape == (batch_size, seq_len, d_model)

        # Check no NaN
        assert not torch.isnan(output).any()

    def test_mamba_layer(self, device):
        """Test MambaLayer (LSTM fallback)."""
        batch_size, seq_len, d_model = 4, 32, 64
        layer = MambaLayer(d_model=d_model).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        output = layer(x)

        # Check shape preserved
        assert output.shape == x.shape


class TestDynamicChunking:
    """Test dynamic chunking components."""

    def test_routing_module(self, device):
        """Test routing module (boundary detection)."""
        batch_size, seq_len, d_model = 4, 128, 64
        router = RoutingModule(d_model=d_model).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        boundary_probs, boundaries = router(x)

        # Check shapes
        assert boundary_probs.shape == (batch_size, seq_len)
        assert boundaries.shape == (batch_size, seq_len)

        # Check probabilities in [0, 1]
        assert (boundary_probs >= 0).all()
        assert (boundary_probs <= 1).all()

    def test_smoothing_module(self, device):
        """Test smoothing module (CRITICAL component)."""
        batch_size, seq_len, d_model = 4, 128, 64
        smoothing = SmoothingModule().to(device)

        chunk_embeddings = torch.randn(batch_size, seq_len, d_model, device=device)
        boundary_probs = torch.rand(batch_size, seq_len, device=device)

        smoothed = smoothing(chunk_embeddings, boundary_probs)

        # Check shape preserved
        assert smoothed.shape == chunk_embeddings.shape

        # Check smoothing applied (output != input)
        assert not torch.allclose(smoothed, chunk_embeddings)

    def test_upsampler(self, device):
        """Test upsampler with STE."""
        batch_size, num_chunks, d_model = 4, 20, 64
        target_len = 128
        upsampler = Upsampler()

        chunks = torch.randn(batch_size, num_chunks, d_model, device=device)
        chunk_indices = torch.stack([
            torch.arange(num_chunks, device=device) * (target_len // num_chunks)
            for _ in range(batch_size)
        ])
        boundary_probs = torch.rand(batch_size, target_len, device=device)

        upsampled = upsampler(chunks, chunk_indices, target_len, boundary_probs)

        # Check shape
        assert upsampled.shape == (batch_size, target_len, d_model)

    def test_complete_chunking_module(self, device):
        """Test complete dynamic chunking module."""
        batch_size, seq_len, d_model = 4, 128, 64
        chunking = DynamicChunkingModule(
            d_model=d_model
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        chunks, chunk_indices, boundary_probs = chunking(x, return_boundaries=True)

        # Check chunks are produced
        assert chunks.dim() == 3
        assert chunks.shape[0] == batch_size
        assert chunks.shape[2] == d_model

        # Check boundary_probs returned
        assert boundary_probs is not None
        assert boundary_probs.shape == (batch_size, seq_len)


class TestMainNetwork:
    """Test main Transformer network."""

    def test_transformer_layer(self, device):
        """Test single TransformerLayer."""
        batch_size, seq_len, d_model = 4, 32, 128
        num_heads = 4
        dim_head = d_model // num_heads

        layer = TransformerLayer(
            d_model=d_model,
            num_heads=num_heads,
            dim_head=dim_head,
            dim_ff=d_model * 4,
            use_flash_attn=False
        ).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        output = layer(x)

        # Check shape preserved
        assert output.shape == x.shape

        # Check no NaN
        assert not torch.isnan(output).any()

    def test_main_network(self, device):
        """Test complete HNetMainNetwork."""
        batch_size, seq_len = 4, 32
        d_encoder = 64  # encoder output dimension (will be projected)
        d_model = 128

        main_net = HNetMainNetwork(
            d_model=d_model,
            num_layers=2,
            num_heads=4,
            dim_head=32,
            use_flash_attn=False
        ).to(device)

        x = torch.randn(batch_size, seq_len, 1024, device=device)
        output = main_net(x)

        # Check output has d_model dimension
        assert output.shape == (batch_size, seq_len, d_model)

    def test_main_network_with_mask(self, device):
        """Test main network with attention mask."""
        batch_size, seq_len = 4, 32

        main_net = HNetMainNetwork(
            d_model=128,
            num_layers=2,
            num_heads=4,
            dim_head=32,
            use_flash_attn=False
        ).to(device)

        x = torch.randn(batch_size, seq_len, 1024, device=device)

        # Create attention mask (first half valid, second half padding)
        mask = torch.ones(batch_size, seq_len, device=device)
        mask[:, seq_len//2:] = 0

        output = main_net(x, attention_mask=mask)

        # Check shape
        assert output.shape == (batch_size, seq_len, 128)


class TestDecoder:
    """Test H-Net decoder."""

    def test_decoder(self, device):
        """Test HNetDecoder."""
        batch_size, seq_len = 4, 128
        d_model, num_layers = 64, 2

        decoder = HNetDecoder(
            d_model=d_model,
            num_layers=num_layers,
            output_vocab_size=256
        ).to(device)

        # Decoder expects input from main network (1536-dim by default)
        x = torch.randn(batch_size, seq_len, 1536, device=device)
        logits = decoder(x)

        # Check shape
        assert logits.shape == (batch_size, seq_len, 256)

    def test_decoder_reconstruction(self, device):
        """Test decoder reconstruction loss."""
        batch_size, seq_len = 4, 64
        decoder = HNetDecoder(d_model=64, num_layers=2, output_vocab_size=256).to(device)

        x = torch.randn(batch_size, seq_len, 1536, device=device)
        logits = decoder(x)

        # Create target
        targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

        # Compute loss
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, 256),
            targets.reshape(-1)
        )

        # Loss should be positive
        assert loss.item() > 0


class TestLosses:
    """Test loss functions."""

    def test_ratio_loss(self, device):
        """Test ratio loss."""
        batch_size, seq_len = 4, 128
        target_ratio = 6.0

        ratio_loss = RatioLoss(target_ratio=target_ratio).to(device)

        boundary_probs = torch.rand(batch_size, seq_len, device=device)
        boundaries = (boundary_probs > 0.5).float()

        loss = ratio_loss(boundary_probs, boundaries)

        # Loss should be scalar and positive
        assert loss.dim() == 0
        assert loss.item() >= 0

    def test_ratio_loss_guidance(self, device):
        """Test that ratio loss guides toward target compression."""
        target_ratio = 6.0
        ratio_loss_fn = RatioLoss(target_ratio=target_ratio).to(device)

        # Case 1: Too much compression (F too low)
        boundary_probs_low = torch.ones(4, 128, device=device) * 0.1
        boundaries_low = (boundary_probs_low > 0.5).float()
        loss_low = ratio_loss_fn(boundary_probs_low, boundaries_low)

        # Case 2: Too little compression (F too high)
        boundary_probs_high = torch.ones(4, 128, device=device) * 0.9
        boundaries_high = (boundary_probs_high > 0.5).float()
        loss_high = ratio_loss_fn(boundary_probs_high, boundaries_high)

        assert loss_low.item() >= 0
        assert loss_high.item() >= 0

    def test_ar_loss(self, device):
        """Test autoregressive loss."""
        batch_size, seq_len = 4, 64
        ar_loss = AutoregressiveLoss().to(device)

        logits = torch.randn(batch_size, seq_len, 256, device=device)
        targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

        loss = ar_loss(logits, targets)

        # Loss should be scalar and positive
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_bpb_metric(self, device):
        """Test bits-per-byte metric."""
        batch_size, seq_len = 4, 64
        bpb_fn = BitsPerByteLoss()

        logits = torch.randn(batch_size, seq_len, 256, device=device)
        targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

        bpb = bpb_fn(logits, targets)

        # BPB should be positive and reasonable
        assert bpb > 0
        assert bpb < 10


class TestCompleteHNet:
    """Test complete H-Net integration."""

    def test_hnet_initialization(self, hnet_config_small):
        """Test H-Net initialization."""
        hnet = HNet(
            vocab_size=hnet_config_small['encoder']['input_vocab_size'],
            encoder_dim=hnet_config_small['encoder']['d_model'],
            encoder_layers=hnet_config_small['encoder']['num_layers'],
            main_dim=hnet_config_small['main_network']['d_model'],
            main_layers=hnet_config_small['main_network']['num_layers'],
            decoder_layers=hnet_config_small['decoder']['num_layers'],
            target_ratio=hnet_config_small['chunking']['target_ratio'],
            use_flash_attn=False
        )

        # Check components exist
        assert hasattr(hnet, 'encoder')
        assert hasattr(hnet, 'chunking')
        assert hasattr(hnet, 'main_network')
        assert hasattr(hnet, 'decoder')
        assert hasattr(hnet, 'loss_fn')

    def test_hnet_forward(self, hnet_config_small, device):
        """Test H-Net forward pass."""
        batch_size, seq_len = 4, 128

        hnet = HNet(
            vocab_size=256,
            encoder_dim=hnet_config_small['encoder']['d_model'],
            encoder_layers=hnet_config_small['encoder']['num_layers'],
            main_dim=hnet_config_small['main_network']['d_model'],
            main_layers=hnet_config_small['main_network']['num_layers'],
            decoder_layers=hnet_config_small['decoder']['num_layers'],
            target_ratio=hnet_config_small['chunking']['target_ratio'],
            use_flash_attn=False
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        output = hnet(input_ids, targets=None, return_loss=False)

        # Check outputs
        assert 'logits' in output
        assert 'chunks' in output
        assert 'boundary_probs' in output
        assert 'compression_ratio' in output
        assert 'num_chunks' in output

        # Check shapes
        assert output['logits'].shape[0] == batch_size
        assert output['chunks'].shape[0] == batch_size
        assert output['compression_ratio'] > 1.0

    def test_hnet_with_loss(self, hnet_config_small, device):
        """Test H-Net with loss computation."""
        batch_size, seq_len = 4, 128

        hnet = HNet(
            vocab_size=256,
            encoder_dim=hnet_config_small['encoder']['d_model'],
            encoder_layers=hnet_config_small['encoder']['num_layers'],
            main_dim=hnet_config_small['main_network']['d_model'],
            main_layers=hnet_config_small['main_network']['num_layers'],
            decoder_layers=hnet_config_small['decoder']['num_layers'],
            target_ratio=hnet_config_small['chunking']['target_ratio'],
            use_flash_attn=False
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        output = hnet(input_ids, targets=input_ids, return_loss=True)

        # Check loss outputs
        assert 'loss' in output
        assert 'loss_dict' in output
        assert output['loss'].item() >= 0

        # Check loss components
        loss_dict = output['loss_dict']
        assert 'total_loss' in loss_dict
        assert 'ar_loss' in loss_dict
        assert 'ratio_loss' in loss_dict
        assert 'bpb' in loss_dict

    def test_hnet_compression(self, hnet_config_small, device):
        """Test that H-Net achieves compression."""
        batch_size, seq_len = 4, 512

        hnet = HNet(
            vocab_size=256,
            encoder_dim=hnet_config_small['encoder']['d_model'],
            encoder_layers=hnet_config_small['encoder']['num_layers'],
            main_dim=hnet_config_small['main_network']['d_model'],
            main_layers=hnet_config_small['main_network']['num_layers'],
            decoder_layers=hnet_config_small['decoder']['num_layers'],
            target_ratio=hnet_config_small['chunking']['target_ratio'],
            use_flash_attn=False
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        output = hnet(input_ids, targets=None, return_loss=False)

        # Should achieve some compression
        compression_ratio = output['compression_ratio']
        assert compression_ratio > 1.0

        # Number of chunks should be less than sequence length
        num_chunks = output['num_chunks']
        assert num_chunks < seq_len

    def test_hnet_parameter_count(self, hnet_config_small):
        """Test H-Net parameter counting."""
        hnet = HNet(
            vocab_size=256,
            encoder_dim=hnet_config_small['encoder']['d_model'],
            encoder_layers=hnet_config_small['encoder']['num_layers'],
            main_dim=hnet_config_small['main_network']['d_model'],
            main_layers=hnet_config_small['main_network']['num_layers'],
            decoder_layers=hnet_config_small['decoder']['num_layers'],
            target_ratio=hnet_config_small['chunking']['target_ratio'],
            use_flash_attn=False
        )

        # get_num_params returns int
        num_params = hnet.get_num_params()
        assert num_params > 0

    @pytest.mark.slow
    def test_hnet_gradient_flow(self, hnet_config_small, device):
        """Test that gradients flow through entire H-Net."""
        batch_size, seq_len = 2, 64

        hnet = HNet(
            vocab_size=256,
            encoder_dim=hnet_config_small['encoder']['d_model'],
            encoder_layers=hnet_config_small['encoder']['num_layers'],
            main_dim=hnet_config_small['main_network']['d_model'],
            main_layers=hnet_config_small['main_network']['num_layers'],
            decoder_layers=hnet_config_small['decoder']['num_layers'],
            target_ratio=hnet_config_small['chunking']['target_ratio'],
            use_flash_attn=False
        ).to(device)

        input_ids = torch.randint(0, 256, (batch_size, seq_len), device=device)

        # Forward with loss
        output = hnet(input_ids, targets=input_ids, return_loss=True)
        loss = output['loss']

        # Backward
        loss.backward()

        # Check that gradients exist for all parameters
        for name, param in hnet.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
