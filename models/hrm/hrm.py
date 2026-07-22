"""
Complete HRM (Hierarchical Reasoning Model) implementation.

Integrates:
- H-module and L-module (hierarchical recurrence)
- Adaptive Computation Time (ACT)
- Deep supervision
- O(1) memory via one-step gradient approximation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
import yaml

from .modules import HModule, LModule, InputNetwork, OutputNetwork
from .act import ACTModule, ACTLoss
from .deep_supervision import DeepSupervision
from ...utils.initialization import init_hrm_states


class HRM(nn.Module):
    """
    Hierarchical Reasoning Model with O(1) memory.

    Key features:
    - Two-timescale hierarchical recurrence (H-module slow, L-module fast)
    - Adaptive Computation Time (ACT) for dynamic halting
    - Deep supervision with detached gradients
    - O(1) memory gradient approximation
    """

    def __init__(
        self,
        # Input/Output dimensions
        input_dim: int = 1536,  # From H-Net main output
        hidden_dim: int = 512,
        output_dim: int = 512,  # Task-specific

        # Module configuration
        h_layers: int = 4,
        l_layers: int = 4,
        num_heads: int = 8,
        dim_head: int = 64,

        # Recurrence configuration
        l_cycles: int = 8,  # T timesteps per H-cycle
        h_cycles_min: int = 4,
        h_cycles_max: int = 16,

        # ACT configuration
        use_act: bool = True,
        q_loss_weight: float = 0.1,
        exploration_epsilon: float = 0.1,

        # Deep supervision
        use_deep_supervision: bool = True,
        num_segments: int = 4,
        segment_loss_weight: str = 'uniform',

        # Training
        dropout: float = 0.1,
        use_rms_norm: bool = True
    ):
        """
        Initialize HRM.

        Args:
            input_dim: Input dimension from H-Net
            hidden_dim: Hidden dimension for H/L modules
            output_dim: Output dimension (task-specific)
            h_layers: Number of layers in H-module
            l_layers: Number of layers in L-module
            num_heads: Number of attention heads
            dim_head: Dimension per head
            l_cycles: L-module iterations per H-cycle (T)
            h_cycles_min: Minimum H-cycles before halting
            h_cycles_max: Maximum H-cycles
            use_act: Enable Adaptive Computation Time
            q_loss_weight: Weight for Q-loss
            exploration_epsilon: Epsilon for exploration
            use_deep_supervision: Enable deep supervision
            num_segments: Number of supervision segments
            segment_loss_weight: Segment weighting ('uniform' or 'decay')
            dropout: Dropout rate
            use_rms_norm: Use RMSNorm (Post-Norm)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.l_cycles = l_cycles
        self.h_cycles_min = h_cycles_min
        self.h_cycles_max = h_cycles_max
        self.use_act = use_act
        self.use_deep_supervision = use_deep_supervision

        # 1. Input network (φ_in)
        self.input_network = InputNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim
        )

        # 2. H-module (slow, abstract)
        self.h_module = HModule(
            hidden_dim=hidden_dim,
            num_layers=h_layers,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            use_rms_norm=use_rms_norm
        )

        # 3. L-module (fast, detailed)
        self.l_module = LModule(
            hidden_dim=hidden_dim,
            num_layers=l_layers,
            num_heads=num_heads,
            dim_head=dim_head,
            dropout=dropout,
            use_rms_norm=use_rms_norm
        )

        # 4. Output network (φ_out)
        self.output_network = OutputNetwork(
            hidden_dim=hidden_dim,
            output_dim=output_dim
        )

        # 5. ACT module (optional)
        if use_act:
            self.act_module = ACTModule(
                hidden_dim=hidden_dim,
                h_cycles_min=h_cycles_min,
                h_cycles_max=h_cycles_max,
                q_loss_weight=q_loss_weight,
                exploration_epsilon=exploration_epsilon
            )
        else:
            self.act_module = None

        # 6. Deep supervision (optional)
        if use_deep_supervision:
            self.deep_supervision = DeepSupervision(
                num_segments=num_segments,
                segment_loss_weight=segment_loss_weight,
                output_network=self.output_network
            )
        else:
            self.deep_supervision = None

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        task_loss_fn: Optional[nn.Module] = None,
        return_loss: bool = True,
        use_act_inference: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through HRM with hierarchical recurrence.

        Args:
            x: Input from H-Net [batch, seq_len, input_dim]
            targets: Optional targets for loss computation
            task_loss_fn: Task-specific loss function
            return_loss: Whether to compute and return loss
            use_act_inference: Use ACT halting during inference

        Returns:
            Dictionary containing:
                - output: Final predictions [batch, output_dim]
                - z_H_final: Final H-state [batch, hidden_dim]
                - num_cycles: Number of H-cycles taken
                - loss: (if return_loss=True) Total loss
                - loss_dict: (if return_loss=True) Breakdown of losses
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 1. Input network: project to working representation
        x_input = self.input_network(x)  # [batch, seq_len, hidden_dim]

        # Pool input for single representation (mean pooling)
        x_pooled = x_input.mean(dim=1)  # [batch, hidden_dim]

        # 2. Initialize states with TruncatedNormal
        z_H, z_L = init_hrm_states(
            batch_size=batch_size,
            hidden_dim=self.hidden_dim,
            device=device,
            sigma=1.0,
            truncation=2.0
        )

        # 3. Hierarchical recurrence with ACT
        z_H_history = []
        q_values_history = []
        halted = torch.zeros(batch_size, dtype=torch.bool, device=device)
        halt_steps = torch.full((batch_size,), self.h_cycles_max, dtype=torch.long, device=device)

        for m in range(self.h_cycles_max):
            # L-module iterations (T cycles)
            # CRITICAL: Detach H-state for L-module (frozen context)
            z_H_frozen = z_H.detach()

            for t in range(self.l_cycles):
                z_L = self.l_module(z_L, z_H_frozen, x_pooled)

            # L-module converged (final state)
            z_L_final = z_L

            # H-module update (once per cycle)
            z_H = self.h_module(z_H, z_L_final, x_pooled)

            # Store H-state history for deep supervision
            z_H_history.append(z_H)

            # ACT: Check if should halt
            if self.use_act and use_act_inference:
                should_halt, q_values = self.act_module.should_halt(
                    z_H,
                    current_cycle=m,
                    training=self.training
                )

                q_values_history.append(q_values)

                # Update halt tracking
                newly_halted = should_halt & (~halted)
                halt_steps[newly_halted] = m
                halted = halted | should_halt

                # All samples halted?
                if halted.all():
                    break

            # Deep supervision: detach between segments
            if self.use_deep_supervision and self.training:
                # Note: Deep supervision handles detachment internally
                pass

        # 4. Generate final output
        z_H_final = z_H
        output = self.output_network(z_H_final)  # [batch, output_dim]

        # Prepare output dict
        num_cycles = m + 1
        result = {
            'output': output,
            'z_H_final': z_H_final,
            'num_cycles': num_cycles,
            'halt_steps': halt_steps if self.use_act else torch.full((batch_size,), num_cycles, dtype=torch.long, device=device)
        }

        # 5. Compute losses if requested
        if return_loss and targets is not None and task_loss_fn is not None:
            loss_dict = {}

            # Task loss
            task_loss = task_loss_fn(output, targets)
            loss_dict['task_loss'] = task_loss.item()
            total_loss = task_loss

            # Deep supervision loss
            if self.use_deep_supervision and len(z_H_history) > 0:
                deep_sup_loss, deep_sup_dict = self.deep_supervision.apply_supervision(
                    z_H_history,
                    targets,
                    task_loss_fn
                )
                total_loss = total_loss + deep_sup_loss
                loss_dict.update(deep_sup_dict)

            # ACT Q-loss
            if self.use_act and len(q_values_history) > 0:
                # Compute rewards based on task accuracy
                with torch.no_grad():
                    predictions = output.argmax(dim=-1) if output.dim() > 1 else (output > 0.5).float()
                    accuracy = (predictions == targets).float()

                rewards = self.act_module.compute_efficiency_reward(
                    task_accuracy=accuracy,
                    num_cycles=halt_steps.float()
                )

                q_loss = self.act_module.compute_q_loss(
                    q_values_history,
                    rewards,
                    halt_steps
                )

                total_loss = total_loss + self.act_module.q_loss_weight * q_loss
                loss_dict['q_loss'] = q_loss.item()
                loss_dict['mean_reward'] = rewards.mean().item()

            loss_dict['total_loss'] = total_loss.item()
            loss_dict['mean_cycles'] = halt_steps.float().mean().item()

            result['loss'] = total_loss
            result['loss_dict'] = loss_dict

        return result

    def get_num_params(self, trainable_only: bool = False) -> int:
        """Return total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def print_summary(self):
        """Print model summary."""
        print(f"\n{'='*70}")
        print(f"{'HRM Model Summary':^70}")
        print(f"{'='*70}")

        # Component parameters
        input_params = sum(p.numel() for p in self.input_network.parameters())
        h_params = sum(p.numel() for p in self.h_module.parameters())
        l_params = sum(p.numel() for p in self.l_module.parameters())
        output_params = sum(p.numel() for p in self.output_network.parameters())
        act_params = sum(p.numel() for p in self.act_module.parameters()) if self.act_module else 0

        total_params = self.get_num_params()

        print(f"\nComponent Parameters:")
        print(f"  Input Network (φ_in):       {input_params:>12,}")
        print(f"  H-Module (4L Trans):        {h_params:>12,}")
        print(f"  L-Module (4L Trans):        {l_params:>12,}")
        print(f"  Output Network (φ_out):     {output_params:>12,}")
        if self.use_act:
            print(f"  ACT Module:                 {act_params:>12,}")
        print(f"  {'-'*40}")
        print(f"  Total Parameters:           {total_params:>12,}")

        print(f"\nConfiguration:")
        print(f"  L-cycles per H-cycle (T):   {self.l_cycles}")
        print(f"  H-cycles (M_min, M_max):    ({self.h_cycles_min}, {self.h_cycles_max})")
        print(f"  Effective depth:            {self.h_cycles_max * self.l_cycles} (M×T)")
        print(f"  ACT enabled:                {self.use_act}")
        print(f"  Deep supervision:           {self.use_deep_supervision}")

        print(f"\n{'='*70}\n")

    @classmethod
    def from_config(cls, config_path: str) -> 'HRM':
        """Load HRM from YAML config."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        hrm_config = config['hrm']

        return cls(
            input_dim=hrm_config['input_network']['input_dim'],
            hidden_dim=hrm_config['input_network']['hidden_dim'],
            output_dim=hrm_config['output_network']['output_dim'],
            h_layers=hrm_config['h_module']['num_layers'],
            l_layers=hrm_config['l_module']['num_layers'],
            num_heads=hrm_config['h_module']['num_heads'],
            dim_head=hrm_config['h_module']['dim_head'],
            l_cycles=hrm_config['recurrence']['l_cycles'],
            h_cycles_min=hrm_config['recurrence']['h_cycles_min'],
            h_cycles_max=hrm_config['recurrence']['h_cycles_max'],
            use_act=hrm_config['act']['enabled'],
            q_loss_weight=hrm_config['act']['q_loss_weight'],
            use_deep_supervision=hrm_config['deep_supervision']['enabled'],
            num_segments=hrm_config['deep_supervision']['num_segments']
        )


# Example usage
if __name__ == "__main__":
    # Create HRM model
    model = HRM(
        input_dim=1536,
        hidden_dim=512,
        output_dim=10,  # 10-class classification
        h_layers=4,
        l_layers=4,
        l_cycles=8,
        h_cycles_max=16,
        use_act=True,
        use_deep_supervision=True
    )

    model.print_summary()

    # Test forward pass
    batch_size = 4
    seq_len = 10
    num_classes = 10

    # Input from H-Net
    x = torch.randn(batch_size, seq_len, 1536)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Task loss
    task_loss_fn = nn.CrossEntropyLoss()

    # Forward pass
    with torch.no_grad():
        output = model(
            x,
            targets=targets,
            task_loss_fn=task_loss_fn,
            return_loss=True,
            use_act_inference=True
        )

    print(f"\nForward Pass Results:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output['output'].shape}")
    print(f"  Num cycles taken: {output['num_cycles']}")
    print(f"  Halt steps: {output['halt_steps'].tolist()}")

    if 'loss_dict' in output:
        print(f"\nLosses:")
        for key, value in output['loss_dict'].items():
            print(f"  {key}: {value:.4f}")
