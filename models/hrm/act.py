"""
Adaptive Computation Time (ACT) for HRM.
Q-learning based halting mechanism without replay buffers.

CRITICAL: ACT during inference significantly improves performance!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class QHead(nn.Module):
    """
    Q-learning head for halt decision.

    Outputs: [Q_halt, Q_continue]
    """

    def __init__(self, hidden_dim: int = 512, q_head_dim: int = 2):
        super().__init__()

        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, q_head_dim)
        )

    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for halt vs continue.

        Args:
            z_H: H-state [batch, hidden_dim]

        Returns:
            Q-values [batch, 2] = [Q_halt, Q_continue]
        """
        return self.q_network(z_H)


class ACTModule(nn.Module):
    """
    Adaptive Computation Time module for HRM.

    Decides when to halt reasoning based on Q-learning.

    Halt conditions:
        1. Segments >= M_max (forced halt)
        2. (Q_halt > Q_continue) AND (segments >= M_min)
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        q_head_dim: int = 2,
        h_cycles_min: int = 4,
        h_cycles_max: int = 16,
        q_loss_weight: float = 0.1,
        exploration_epsilon: float = 0.1
    ):
        """
        Initialize ACT module.

        Args:
            hidden_dim: Hidden dimension of H-state
            q_head_dim: Q-head output dimension (always 2)
            h_cycles_min: Minimum cycles before halting allowed
            h_cycles_max: Maximum cycles (forced halt)
            q_loss_weight: Weight for Q-loss in total loss
            exploration_epsilon: Epsilon for epsilon-greedy exploration
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.h_cycles_min = h_cycles_min
        self.h_cycles_max = h_cycles_max
        self.q_loss_weight = q_loss_weight
        self.exploration_epsilon = exploration_epsilon

        # Q-learning head
        self.q_head = QHead(hidden_dim, q_head_dim)

    def should_halt(
        self,
        z_H: torch.Tensor,
        current_cycle: int,
        training: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decide whether to halt based on Q-values.

        Args:
            z_H: Current H-state [batch, hidden_dim]
            current_cycle: Current cycle number (0-indexed)
            training: Whether in training mode (for exploration)

        Returns:
            Tuple of:
                - halt_decision: Binary halt decision [batch]
                - q_values: Q-values [batch, 2]
        """
        batch_size = z_H.size(0)

        # Compute Q-values
        q_values = self.q_head(z_H)  # [batch, 2]
        q_halt = q_values[:, 0]
        q_continue = q_values[:, 1]

        # Forced halt if max cycles reached
        if current_cycle >= self.h_cycles_max - 1:
            halt_decision = torch.ones(batch_size, dtype=torch.bool, device=z_H.device)
            return halt_decision, q_values

        # Can't halt before minimum cycles
        if current_cycle < self.h_cycles_min:
            halt_decision = torch.zeros(batch_size, dtype=torch.bool, device=z_H.device)
            return halt_decision, q_values

        # Q-learning decision: halt if Q_halt > Q_continue
        halt_decision = (q_halt > q_continue)

        # Epsilon-greedy exploration during training
        if training and self.exploration_epsilon > 0:
            random_mask = torch.rand(batch_size, device=z_H.device) < self.exploration_epsilon
            random_decision = torch.rand(batch_size, device=z_H.device) > 0.5
            halt_decision = torch.where(random_mask, random_decision, halt_decision)

        return halt_decision, q_values

    def compute_q_loss(
        self,
        q_values_history: list,
        rewards: torch.Tensor,
        halt_steps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-learning loss.

        Args:
            q_values_history: List of Q-values at each cycle [num_cycles, batch, 2]
            rewards: Final rewards [batch] (e.g., task accuracy)
            halt_steps: Number of steps taken before halting [batch]

        Returns:
            Q-loss for training
        """
        if len(q_values_history) == 0:
            return torch.tensor(0.0, device=rewards.device)

        # Stack Q-values across time
        q_values_stack = torch.stack(q_values_history, dim=0)  # [num_cycles, batch, 2]
        num_cycles, batch_size, _ = q_values_stack.shape

        # Compute target Q-values using reward
        # Simple reward assignment: assign final reward to halt action
        target_q_values = torch.zeros_like(q_values_stack)

        for b in range(batch_size):
            halt_step = min(int(halt_steps[b].item()), num_cycles - 1)
            # Assign reward to Q_halt at halting step
            target_q_values[halt_step, b, 0] = rewards[b]  # Q_halt
            # Penalize Q_continue after halting
            if halt_step < num_cycles - 1:
                target_q_values[halt_step + 1:, b, 1] = -0.1  # Small penalty

        # Compute MSE loss
        q_loss = F.mse_loss(q_values_stack, target_q_values)

        return q_loss

    def compute_efficiency_reward(
        self,
        task_accuracy: torch.Tensor,
        num_cycles: torch.Tensor,
        efficiency_weight: float = 0.1
    ) -> torch.Tensor:
        """
        Compute reward that balances accuracy and efficiency.

        Reward = accuracy - efficiency_weight * (num_cycles / max_cycles)

        Args:
            task_accuracy: Task accuracy [batch]
            num_cycles: Number of cycles used [batch]
            efficiency_weight: Weight for efficiency penalty

        Returns:
            Rewards [batch]
        """
        # Normalize cycles to [0, 1]
        normalized_cycles = num_cycles.float() / self.h_cycles_max

        # Reward: high accuracy, low cycles
        rewards = task_accuracy - efficiency_weight * normalized_cycles

        return rewards


class ACTLoss(nn.Module):
    """
    Combined loss for ACT training.

    L_total = L_task + q_loss_weight * L_Q
    """

    def __init__(self, q_loss_weight: float = 0.1):
        super().__init__()
        self.q_loss_weight = q_loss_weight

    def forward(
        self,
        task_loss: torch.Tensor,
        q_loss: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            task_loss: Task-specific loss
            q_loss: Q-learning loss

        Returns:
            Tuple of:
                - total_loss: Combined loss
                - loss_dict: Dictionary of individual losses
        """
        total_loss = task_loss + self.q_loss_weight * q_loss

        loss_dict = {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'q_loss': q_loss.item()
        }

        return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    batch_size = 4
    hidden_dim = 512

    # Initialize ACT module
    act = ACTModule(
        hidden_dim=hidden_dim,
        h_cycles_min=4,
        h_cycles_max=16,
        q_loss_weight=0.1
    )

    # Simulate adaptive halting
    print("Simulating ACT halting:")

    z_H = torch.randn(batch_size, hidden_dim)
    q_values_history = []
    halt_steps = torch.zeros(batch_size, dtype=torch.long)
    halted = torch.zeros(batch_size, dtype=torch.bool)

    for cycle in range(16):
        # Check halt decision
        should_halt, q_values = act.should_halt(z_H, cycle, training=True)

        q_values_history.append(q_values)

        # Update halt steps
        newly_halted = should_halt & (~halted)
        halt_steps[newly_halted] = cycle
        halted = halted | should_halt

        print(f"  Cycle {cycle}: Q_halt={q_values[:, 0].mean():.3f}, "
              f"Q_continue={q_values[:, 1].mean():.3f}, "
              f"Halted: {halted.sum()}/{batch_size}")

        # All halted?
        if halted.all():
            print(f"  All samples halted at cycle {cycle}")
            break

        # Update z_H (simulate)
        z_H = z_H + 0.1 * torch.randn_like(z_H)

    # Compute Q-loss
    rewards = torch.rand(batch_size)  # Simulate task rewards
    q_loss = act.compute_q_loss(q_values_history, rewards, halt_steps)

    print(f"\nHalt steps: {halt_steps.tolist()}")
    print(f"Rewards: {rewards.tolist()}")
    print(f"Q-loss: {q_loss.item():.4f}")

    # Compute efficiency reward
    task_accuracy = torch.tensor([0.9, 0.85, 0.92, 0.88])
    eff_rewards = act.compute_efficiency_reward(task_accuracy, halt_steps.float())
    print(f"\nEfficiency rewards: {eff_rewards.tolist()}")
