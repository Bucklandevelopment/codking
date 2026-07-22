"""
Unit tests for HRM (Hierarchical Reasoning Model) components.

Tests:
- modules.py: H-module, L-module, InputNetwork, OutputNetwork
- act.py: ACT module, Q-learning
- deep_supervision.py: Deep supervision
- hrm.py: Complete HRM integration
"""

import pytest
import torch
import torch.nn as nn

from codking.models.hrm.modules import (
    HModule,
    LModule,
    InputNetwork,
    OutputNetwork,
    RMSNorm
)
from codking.models.hrm.act import (
    QHead,
    ACTModule
)
from codking.models.hrm.deep_supervision import (
    DeepSupervision,
    SegmentedTraining
)
from codking.models.hrm.hrm import HRM


class TestHRMModules:
    """Test HRM modules."""

    def test_rms_norm(self, device):
        """Test RMSNorm."""
        batch_size, seq_len, d_model = 4, 10, 64
        norm = RMSNorm(d_model).to(device)

        x = torch.randn(batch_size, seq_len, d_model, device=device)
        output = norm(x)

        # Check shape preserved
        assert output.shape == x.shape

        # Check normalization (RMS should be approximately 1)
        rms = (output.pow(2).mean(dim=-1, keepdim=True) + 1e-6).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    def test_input_network(self, device):
        """Test InputNetwork."""
        batch_size, seq_len = 4, 10
        input_dim, hidden_dim = 128, 64

        input_net = InputNetwork(input_dim, hidden_dim).to(device)

        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        output = input_net(x)

        # Check shape
        assert output.shape == (batch_size, seq_len, hidden_dim)

    def test_output_network(self, device):
        """Test OutputNetwork."""
        batch_size, hidden_dim, output_dim = 4, 64, 10

        output_net = OutputNetwork(hidden_dim, output_dim).to(device)

        x = torch.randn(batch_size, hidden_dim, device=device)
        output = output_net(x)

        # Check shape
        assert output.shape == (batch_size, output_dim)

    def test_h_module(self, device, hrm_config_small):
        """Test H-module."""
        batch_size = 4
        hidden_dim = hrm_config_small['input_network']['hidden_dim']

        h_module = HModule(
            hidden_dim=hidden_dim,
            num_layers=hrm_config_small['h_module']['num_layers'],
            num_heads=hrm_config_small['h_module']['num_heads'],
            dim_head=hrm_config_small['h_module']['dim_head']
        ).to(device)

        # Inputs
        z_H = torch.randn(batch_size, hidden_dim, device=device)
        z_L = torch.randn(batch_size, hidden_dim, device=device)
        x_input = torch.randn(batch_size, hidden_dim, device=device)

        # Forward
        output = h_module(z_H, z_L, x_input)

        # Check shape
        assert output.shape == (batch_size, hidden_dim)

    def test_l_module(self, device, hrm_config_small):
        """Test L-module."""
        batch_size = 4
        hidden_dim = hrm_config_small['input_network']['hidden_dim']

        l_module = LModule(
            hidden_dim=hidden_dim,
            num_layers=hrm_config_small['l_module']['num_layers'],
            num_heads=hrm_config_small['l_module']['num_heads'],
            dim_head=hrm_config_small['l_module']['dim_head']
        ).to(device)

        # Inputs
        z_L = torch.randn(batch_size, hidden_dim, device=device)
        z_H_frozen = torch.randn(batch_size, hidden_dim, device=device)
        x_input = torch.randn(batch_size, hidden_dim, device=device)

        # Forward
        output = l_module(z_L, z_H_frozen, x_input)

        # Check shape
        assert output.shape == (batch_size, hidden_dim)


class TestACT:
    """Test Adaptive Computation Time."""

    def test_q_head(self, device):
        """Test Q-head."""
        batch_size, hidden_dim = 4, 64
        q_head = QHead(hidden_dim).to(device)

        z_H = torch.randn(batch_size, hidden_dim, device=device)
        q_values = q_head(z_H)

        # Check shape: [batch, 2] for [Q_halt, Q_continue]
        assert q_values.shape == (batch_size, 2)

    def test_act_module_should_halt(self, device):
        """Test ACT halting decision."""
        batch_size, hidden_dim = 4, 64
        h_cycles_min, h_cycles_max = 2, 8

        act = ACTModule(
            hidden_dim=hidden_dim,
            h_cycles_min=h_cycles_min,
            h_cycles_max=h_cycles_max
        ).to(device)

        z_H = torch.randn(batch_size, hidden_dim, device=device)

        # Test at different cycles
        # Before min: should not halt
        should_halt, q_values = act.should_halt(z_H, current_cycle=0, training=False)
        assert not should_halt.any()

        # At max: should force halt
        should_halt, q_values = act.should_halt(z_H, current_cycle=h_cycles_max-1, training=False)
        assert should_halt.all()

        # Between min and max: depends on Q-values
        should_halt, q_values = act.should_halt(z_H, current_cycle=4, training=False)
        assert q_values.shape == (batch_size, 2)

    def test_act_compute_q_loss(self, device):
        """Test Q-loss computation."""
        batch_size, hidden_dim = 4, 64
        act = ACTModule(hidden_dim=hidden_dim).to(device)

        # Simulate Q-values history
        num_cycles = 5
        q_values_history = [
            torch.randn(batch_size, 2, device=device)
            for _ in range(num_cycles)
        ]

        rewards = torch.rand(batch_size, device=device)
        halt_steps = torch.randint(2, num_cycles, (batch_size,), device=device)

        q_loss = act.compute_q_loss(q_values_history, rewards, halt_steps)

        # Check loss is scalar and positive
        assert q_loss.dim() == 0
        assert q_loss.item() >= 0

    def test_act_efficiency_reward(self, device):
        """Test efficiency reward computation."""
        batch_size, hidden_dim = 4, 64
        h_cycles_max = 8
        act = ACTModule(
            hidden_dim=hidden_dim,
            h_cycles_max=h_cycles_max
        ).to(device)

        task_accuracy = torch.tensor([0.9, 0.8, 0.95, 0.7], device=device)
        num_cycles = torch.tensor([4, 6, 3, 7], dtype=torch.float, device=device)

        rewards = act.compute_efficiency_reward(task_accuracy, num_cycles)

        # Check shape
        assert rewards.shape == (batch_size,)

        # Higher accuracy and lower cycles should give higher reward
        assert rewards[2] > rewards[3]  # 0.95 acc, 3 cycles vs 0.7 acc, 7 cycles


class TestDeepSupervision:
    """Test Deep Supervision."""

    def test_segment_boundaries(self):
        """Test segment boundary computation."""
        deep_sup = DeepSupervision(num_segments=4)

        total_cycles = 16
        boundaries = deep_sup.compute_segment_boundaries(total_cycles)

        # Should have 4 boundaries
        assert len(boundaries) == 4

        # Should be evenly spaced
        assert boundaries == [4, 8, 12, 16]

    def test_apply_supervision(self, device):
        """Test deep supervision application."""
        batch_size, hidden_dim, output_dim = 4, 64, 10

        output_net = OutputNetwork(hidden_dim, output_dim).to(device)
        deep_sup = DeepSupervision(
            num_segments=4,
            segment_loss_weight='uniform',
            output_network=output_net
        ).to(device)

        # Simulate H-state history
        num_cycles = 16
        z_H_history = [
            torch.randn(batch_size, hidden_dim, device=device)
            for _ in range(num_cycles)
        ]

        targets = torch.randint(0, output_dim, (batch_size,), device=device)
        task_loss_fn = nn.CrossEntropyLoss()

        total_loss, loss_dict = deep_sup.apply_supervision(
            z_H_history, targets, task_loss_fn
        )

        # Check loss
        assert total_loss.dim() == 0
        assert total_loss.item() >= 0

        # Check loss dict
        assert 'deep_supervision_loss' in loss_dict
        assert 'num_segments' in loss_dict
        assert loss_dict['num_segments'] == 4

        # Should have segment losses
        assert 'segment_0_loss' in loss_dict

    def test_segmented_training(self):
        """Test segmented training utility."""
        segmented = SegmentedTraining(num_segments=4)

        total_cycles = 16
        segments = segmented.split_into_segments(total_cycles)

        # Should have 4 segments
        assert len(segments) == 4

        # Each segment should have (start, end)
        for start, end in segments:
            assert start < end

    def test_detach_state(self, device):
        """Test state detachment."""
        state = torch.randn(4, 64, device=device, requires_grad=True)

        detached = SegmentedTraining.detach_state(state)

        # Should be detached
        assert not detached.requires_grad
        assert detached.shape == state.shape


class TestCompleteHRM:
    """Test complete HRM integration."""

    def test_hrm_initialization(self, hrm_config_small, device):
        """Test HRM initialization."""
        hrm = HRM(
            input_dim=hrm_config_small['input_network']['input_dim'],
            hidden_dim=hrm_config_small['input_network']['hidden_dim'],
            output_dim=hrm_config_small['output_network']['output_dim'],
            h_layers=hrm_config_small['h_module']['num_layers'],
            l_layers=hrm_config_small['l_module']['num_layers'],
            l_cycles=hrm_config_small['recurrence']['l_cycles'],
            h_cycles_max=hrm_config_small['recurrence']['h_cycles_max'],
            use_act=True,
            use_deep_supervision=True
        ).to(device)

        # Check modules exist
        assert hasattr(hrm, 'input_network')
        assert hasattr(hrm, 'h_module')
        assert hasattr(hrm, 'l_module')
        assert hasattr(hrm, 'output_network')
        assert hasattr(hrm, 'act_module')
        assert hasattr(hrm, 'deep_supervision')

    def test_hrm_forward(self, hrm_config_small, device):
        """Test HRM forward pass."""
        batch_size, seq_len = 4, 10
        input_dim = hrm_config_small['input_network']['input_dim']
        output_dim = hrm_config_small['output_network']['output_dim']

        hrm = HRM(
            input_dim=input_dim,
            hidden_dim=hrm_config_small['input_network']['hidden_dim'],
            output_dim=output_dim,
            h_layers=hrm_config_small['h_module']['num_layers'],
            l_layers=hrm_config_small['l_module']['num_layers'],
            l_cycles=hrm_config_small['recurrence']['l_cycles'],
            h_cycles_max=hrm_config_small['recurrence']['h_cycles_max'],
            use_act=True,
            use_deep_supervision=False  # Disable for simpler test
        ).to(device)

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        output = hrm(x, targets=None, return_loss=False, use_act_inference=True)

        # Check outputs
        assert 'output' in output
        assert output['output'].shape == (batch_size, output_dim)
        assert 'num_cycles' in output
        assert 'z_H_final' in output

    def test_hrm_with_loss(self, hrm_config_small, device):
        """Test HRM with loss computation."""
        batch_size, seq_len = 4, 10
        input_dim = hrm_config_small['input_network']['input_dim']
        output_dim = hrm_config_small['output_network']['output_dim']

        hrm = HRM(
            input_dim=input_dim,
            hidden_dim=hrm_config_small['input_network']['hidden_dim'],
            output_dim=output_dim,
            h_layers=hrm_config_small['h_module']['num_layers'],
            l_layers=hrm_config_small['l_module']['num_layers'],
            l_cycles=hrm_config_small['recurrence']['l_cycles'],
            h_cycles_max=hrm_config_small['recurrence']['h_cycles_max'],
            use_act=True,
            use_deep_supervision=True
        ).to(device)

        x = torch.randn(batch_size, seq_len, input_dim, device=device)
        targets = torch.randint(0, output_dim, (batch_size,), device=device)
        task_loss_fn = nn.CrossEntropyLoss()

        output = hrm(
            x,
            targets=targets,
            task_loss_fn=task_loss_fn,
            return_loss=True,
            use_act_inference=True
        )

        # Check outputs
        assert 'loss' in output
        assert 'loss_dict' in output
        assert output['loss'].item() >= 0

        # Check loss dict
        loss_dict = output['loss_dict']
        assert 'total_loss' in loss_dict
        assert 'task_loss' in loss_dict

    @pytest.mark.slow
    def test_hrm_convergence(self, hrm_config_small, device):
        """Test HRM state convergence."""
        batch_size, seq_len = 4, 10
        input_dim = hrm_config_small['input_network']['input_dim']
        output_dim = hrm_config_small['output_network']['output_dim']

        hrm = HRM(
            input_dim=input_dim,
            hidden_dim=hrm_config_small['input_network']['hidden_dim'],
            output_dim=output_dim,
            h_layers=hrm_config_small['h_module']['num_layers'],
            l_layers=hrm_config_small['l_module']['num_layers'],
            l_cycles=hrm_config_small['recurrence']['l_cycles'],
            h_cycles_max=hrm_config_small['recurrence']['h_cycles_max'],
            use_act=False,  # Disable ACT to run all cycles
            use_deep_supervision=False
        ).to(device)

        x = torch.randn(batch_size, seq_len, input_dim, device=device)

        output = hrm(x, targets=None, return_loss=False, use_act_inference=False)

        # Should run all cycles
        assert output['num_cycles'] == hrm_config_small['recurrence']['h_cycles_max']

    def test_hrm_parameter_count(self, hrm_config_small):
        """Test HRM parameter count."""
        hrm = HRM(
            input_dim=hrm_config_small['input_network']['input_dim'],
            hidden_dim=hrm_config_small['input_network']['hidden_dim'],
            output_dim=hrm_config_small['output_network']['output_dim'],
            h_layers=hrm_config_small['h_module']['num_layers'],
            l_layers=hrm_config_small['l_module']['num_layers'],
            l_cycles=hrm_config_small['recurrence']['l_cycles'],
            h_cycles_max=hrm_config_small['recurrence']['h_cycles_max']
        )

        num_params = hrm.get_num_params()

        # Should be positive
        assert num_params > 0

        # For small config, should be reasonable
        assert num_params < 1_000_000  # Less than 1M for test config


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
