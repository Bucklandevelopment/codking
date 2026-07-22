"""
Initialization utilities for CodKing models.
Implements TruncatedNormal initialization as specified in HRM paper.
"""

import torch
import torch.nn as nn
try:
    from scipy.stats import truncnorm  # noqa: F401 - optional, not used directly
except ImportError:
    pass


def truncated_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                      a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Initialize tensor with truncated normal distribution.

    Args:
        tensor: Tensor to initialize
        mean: Mean of the normal distribution
        std: Standard deviation
        a: Lower truncation bound (in units of std)
        b: Upper truncation bound (in units of std)

    Returns:
        Initialized tensor

    Note:
        HRM uses TruncatedNormal(σ=1, truncation=2) for z_H and z_L states.
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < b) & (tmp > a)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def init_hrm_states(batch_size: int, hidden_dim: int, device: torch.device,
                    sigma: float = 1.0, truncation: float = 2.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Initialize HRM hidden states (z_H, z_L) with truncated normal distribution.

    Args:
        batch_size: Batch size
        hidden_dim: Hidden dimension
        device: Device to create tensors on
        sigma: Standard deviation for truncated normal
        truncation: Truncation bounds (±truncation * sigma)

    Returns:
        Tuple of (z_H, z_L) initialized tensors
    """
    z_H = torch.empty(batch_size, hidden_dim, device=device)
    z_L = torch.empty(batch_size, hidden_dim, device=device)

    truncated_normal_(z_H, mean=0.0, std=sigma, a=-truncation, b=truncation)
    truncated_normal_(z_L, mean=0.0, std=sigma, a=-truncation, b=truncation)

    return z_H, z_L


def init_weights(module: nn.Module, method: str = 'xavier_uniform'):
    """
    Initialize module weights using specified method.

    Args:
        module: Module to initialize
        method: Initialization method ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if method == 'xavier_uniform':
            nn.init.xavier_uniform_(module.weight)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

        if module.bias is not None:
            nn.init.zeros_(module.bias)

    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, name: str = "Model"):
    """
    Print model summary with parameter counts.

    Args:
        model: PyTorch model
        name: Model name for display
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)

    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"{'='*60}\n")
