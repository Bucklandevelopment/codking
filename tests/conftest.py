"""
Pytest configuration and shared fixtures for CodKing tests.

Provides fixtures for:
- Sample data
- Model configurations
- Temporary directories
- Mock models
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


# Test configuration
@pytest.fixture(scope="session")
def device():
    """Get device for testing (prefer CPU for stability)."""
    return torch.device('cpu')


@pytest.fixture(scope="session")
def use_small_models():
    """Use smaller models for faster testing."""
    return True


# Sample data fixtures
@pytest.fixture
def sample_batch_size():
    """Small batch size for tests."""
    return 4


@pytest.fixture
def sample_seq_len():
    """Small sequence length for tests."""
    return 512  # Smaller than production 8192


@pytest.fixture
def sample_input_ids(sample_batch_size, sample_seq_len):
    """Sample byte-level input."""
    return torch.randint(0, 256, (sample_batch_size, sample_seq_len))


@pytest.fixture
def sample_targets(sample_batch_size):
    """Sample classification targets."""
    return torch.randint(0, 10, (sample_batch_size,))


@pytest.fixture
def sample_binary_targets(sample_batch_size):
    """Sample binary targets for threat detection."""
    return torch.randint(0, 2, (sample_batch_size,))


@pytest.fixture
def sample_text_data(sample_batch_size):
    """Sample text data."""
    return [f"Sample text number {i} for testing." for i in range(sample_batch_size)]


@pytest.fixture
def sample_log_data(sample_batch_size):
    """Sample log data for cybersecurity."""
    logs = [
        "Normal user login from 192.168.1.10",
        "CRITICAL: Multiple failed SSH attempts from 203.0.113.42",
        "INFO: System backup completed successfully",
        "WARNING: Unusual network traffic detected on port 443"
    ]
    return logs[:sample_batch_size]


# Configuration fixtures
@pytest.fixture
def hnet_config_small():
    """Small H-Net configuration for testing."""
    return {
        'encoder': {
            'input_vocab_size': 256,
            'd_model': 64,  # Small for testing
            'num_layers': 2,  # Reduced layers
            'use_mamba': False
        },
        'chunking': {
            'target_ratio': 6.0,
            'smoothing_enabled': True
        },
        'main_network': {
            'd_model': 128,  # Small for testing
            'num_layers': 4,  # Reduced layers
            'num_heads': 4,
            'use_flash_attn': False  # Disable for compatibility
        },
        'decoder': {
            'num_layers': 2  # Reduced layers
        }
    }


@pytest.fixture
def hrm_config_small():
    """Small HRM configuration for testing."""
    return {
        'input_network': {
            'input_dim': 128,
            'hidden_dim': 64  # Small for testing
        },
        'h_module': {
            'num_layers': 2,  # Reduced layers
            'num_heads': 4,
            'dim_head': 16
        },
        'l_module': {
            'num_layers': 2,  # Reduced layers
            'num_heads': 4,
            'dim_head': 16
        },
        'output_network': {
            'output_dim': 10
        },
        'recurrence': {
            'l_cycles': 4,  # Reduced cycles
            'h_cycles_min': 2,
            'h_cycles_max': 8
        },
        'act': {
            'enabled': True,
            'q_loss_weight': 0.1
        },
        'deep_supervision': {
            'enabled': True,
            'num_segments': 2  # Reduced segments
        }
    }


@pytest.fixture
def training_config_small():
    """Small training configuration for testing."""
    return {
        'training': {
            'task_type': 'classification',
            'num_classes': 10,
            'learning_rate': 1e-4,
            'weight_decay': 0.1,
            'batch_size': 4,
            'max_epochs': 2,
            'warmup_steps': 10,
            'max_steps': 100
        },
        'data': {
            'dataset_type': 'text',
            'max_length': 512,
            'use_augmentation': False,
            'batch_size': 4
        }
    }


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def temp_checkpoint_dir(temp_dir):
    """Create temporary checkpoint directory."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@pytest.fixture
def temp_config_dir(temp_dir):
    """Create temporary config directory."""
    config_dir = temp_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# Mock data fixtures
@pytest.fixture
def mock_dataset(sample_text_data, sample_targets):
    """Mock dataset for testing."""
    return list(zip(sample_text_data, sample_targets.tolist()))


@pytest.fixture
def mock_log_dataset(sample_log_data, sample_binary_targets):
    """Mock log dataset for cybersecurity testing."""
    return list(zip(sample_log_data, sample_binary_targets.tolist()))


# Utility fixtures
@pytest.fixture
def assert_shape():
    """Helper function to assert tensor shapes."""
    def _assert_shape(tensor, expected_shape, name="tensor"):
        assert tensor.shape == torch.Size(expected_shape), \
            f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
    return _assert_shape


@pytest.fixture
def assert_no_nan():
    """Helper function to check for NaN values."""
    def _assert_no_nan(tensor, name="tensor"):
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf values"
    return _assert_no_nan


@pytest.fixture
def assert_range():
    """Helper function to check value ranges."""
    def _assert_range(tensor, min_val, max_val, name="tensor"):
        assert tensor.min() >= min_val, f"{name} min value {tensor.min()} < {min_val}"
        assert tensor.max() <= max_val, f"{name} max value {tensor.max()} > {max_val}"
    return _assert_range


# Performance testing fixtures
@pytest.fixture
def measure_time():
    """Measure execution time."""
    import time

    class Timer:
        def __enter__(self):
            self.start = time.time()
            return self

        def __exit__(self, *args):
            self.end = time.time()
            self.elapsed = self.end - self.start

    return Timer


@pytest.fixture
def measure_memory():
    """Measure memory usage."""
    def _measure_memory():
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2  # MB
        return 0.0
    return _measure_memory


# Seed for reproducibility
@pytest.fixture(autouse=True)
def set_seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# Skip markers
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "api: marks tests that test API endpoints"
    )
