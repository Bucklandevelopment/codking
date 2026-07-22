"""
Unit tests for training infrastructure.

Tests:
- lightning_module.py: CodKingLightningModule
- data_module.py: CodKingDataModule, datasets, collators, augmentation
- Training loop functionality
"""

import pytest
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pathlib import Path
import tempfile

from codking.training.lightning_module import CodKingLightningModule
from codking.training.data_module import (
    CodKingDataModule,
    ByteLevelDataset,
    TextToByteDataset,
    FewShotAugmentation,
)


class TestDatasets:
    """Test dataset implementations."""

    def test_byte_level_dataset(self):
        """Test ByteLevelDataset."""
        data = [
            b"Sample text for testing",
            b"Another line of text",
            b"Third line here",
        ]
        labels = [0, 1, 0]

        dataset = ByteLevelDataset(
            data=data,
            labels=labels,
            max_length=512,
        )

        # Should have 3 samples
        assert len(dataset) == 3

        # Test sample
        input_ids, label = dataset[0]
        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(label, torch.Tensor)

        # Check that input is bytes (0-255)
        assert input_ids.min() >= 0
        assert input_ids.max() <= 255

        # Check length
        assert input_ids.shape == (512,)

    def test_byte_level_dataset_with_labels(self):
        """Test dataset with labels."""
        data = [
            b"Normal log entry",
            b"CRITICAL: Attack detected",
        ]
        labels = [0, 1]

        dataset = ByteLevelDataset(
            data=data,
            labels=labels,
            max_length=512,
        )

        assert len(dataset) == 2

        # Check labels
        _, label_0 = dataset[0]
        _, label_1 = dataset[1]

        assert label_0.item() == 0
        assert label_1.item() == 1

    def test_text_to_byte_dataset(self):
        """Test TextToByteDataset."""
        texts = ["Hello world", "Test entry"]
        labels = [0, 1]

        dataset = TextToByteDataset(
            texts=texts,
            labels=labels,
            max_length=256,
        )

        assert len(dataset) == 2

        input_ids, label = dataset[0]
        assert input_ids.shape == (256,)
        assert input_ids.min() >= 0
        assert input_ids.max() <= 255


class TestAugmentation:
    """Test data augmentation."""

    def test_few_shot_augmentation(self):
        """Test few-shot augmentation."""
        augment = FewShotAugmentation(
            noise_level=0.5,
            dropout_prob=0.1,
            swap_prob=0.05
        )

        # Original byte sequence
        original = b"This is a test log entry for augmentation."

        # Apply augmentation
        augmented = augment(original)

        # Should be same length
        assert len(augmented) == len(original)

        # Should still be bytes
        assert isinstance(augmented, bytes)

    def test_augmentation_preserves_length(self):
        """Test that augmentation preserves sequence length."""
        augment = FewShotAugmentation()

        for length in [10, 50, 100, 500]:
            original = bytes([i % 256 for i in range(length)])
            augmented = augment(original)
            assert len(augmented) == length

    def test_augmentation_variations(self):
        """Test that augmentation creates variations."""
        augment = FewShotAugmentation(noise_level=0.5, dropout_prob=0.2, swap_prob=0.1)
        original = b"Test log entry for variation check"

        # Generate multiple augmented versions
        augmented_versions = [augment(original) for _ in range(10)]

        # At least some should be different from each other
        unique_versions = set(augmented_versions)
        assert len(unique_versions) > 1


class TestDataModule:
    """Test PyTorch Lightning data module."""

    def test_data_module_initialization(self):
        """Test data module initialization."""
        data_module = CodKingDataModule(
            train_data=[b"sample1", b"sample2"],
            train_labels=[0, 1],
            val_data=[b"val1"],
            val_labels=[0],
            batch_size=2,
            max_length=512,
            num_workers=0
        )

        # Check attributes
        assert data_module.batch_size == 2
        assert data_module.max_length == 512

    def test_data_module_setup(self):
        """Test data module setup."""
        train_data = [b"Training sample %d" % i for i in range(10)]
        train_labels = [i % 3 for i in range(10)]
        val_data = [b"Val sample %d" % i for i in range(3)]
        val_labels = [0, 1, 2]

        data_module = CodKingDataModule(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            batch_size=4,
            max_length=512,
            num_workers=0
        )

        # Setup
        data_module.setup('fit')

        # Check datasets created
        assert data_module.train_dataset is not None
        assert data_module.val_dataset is not None
        assert len(data_module.train_dataset) == 10
        assert len(data_module.val_dataset) == 3

    def test_data_module_dataloaders(self):
        """Test dataloader creation."""
        train_data = [b"Sample %d" % i for i in range(20)]
        train_labels = [i % 5 for i in range(20)]
        val_data = [b"Val sample %d" % i for i in range(5)]
        val_labels = [i % 5 for i in range(5)]

        data_module = CodKingDataModule(
            train_data=train_data,
            train_labels=train_labels,
            val_data=val_data,
            val_labels=val_labels,
            batch_size=4,
            max_length=512,
            num_workers=0
        )

        data_module.setup('fit')

        # Get dataloaders
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        # Check loaders
        assert train_loader is not None
        assert val_loader is not None

        # Check batch
        batch = next(iter(train_loader))
        input_ids, labels = batch
        assert input_ids.shape[0] == 4  # batch_size

    def test_data_module_with_augmentation(self):
        """Test data module with augmentation."""
        train_data = [b"Sample %d content" % i for i in range(5)]
        train_labels = [0, 1, 0, 1, 0]

        # With augmentation (10x factor)
        data_module = CodKingDataModule(
            train_data=train_data,
            train_labels=train_labels,
            dataset_type='byte',
            batch_size=4,
            max_length=512,
            use_augmentation=True,
            augmentation_factor=10,
            num_workers=0
        )

        data_module.setup('fit')

        # Should have 5 * 10 = 50 training samples
        assert len(data_module.train_dataset) == 50


class TestLightningModule:
    """Test PyTorch Lightning module."""

    def test_lightning_module_initialization(self, hnet_config_small, hrm_config_small):
        """Test Lightning module initialization."""
        module = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            learning_rate=1e-4,
            weight_decay=0.1
        )

        # Check model created
        assert hasattr(module, 'model')
        assert module.hparams.learning_rate == 1e-4

    def test_lightning_module_training_step(self, hnet_config_small, hrm_config_small, device):
        """Test training step."""
        module = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        # Create batch as tuple (input_ids, targets)
        batch = (
            torch.randint(0, 256, (4, 512), device=device),
            torch.randint(0, 10, (4,), device=device)
        )

        # Training step
        loss = module.training_step(batch, batch_idx=0)

        # Check loss
        assert loss is not None
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_lightning_module_validation_step(self, hnet_config_small, hrm_config_small, device):
        """Test validation step."""
        module = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        batch = (
            torch.randint(0, 256, (4, 512), device=device),
            torch.randint(0, 10, (4,), device=device)
        )

        # Validation step
        loss = module.validation_step(batch, batch_idx=0)

        # Check loss
        assert loss is not None
        assert loss.item() >= 0

    def test_lightning_module_configure_optimizers(self, hnet_config_small, hrm_config_small):
        """Test optimizer configuration."""
        module = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            learning_rate=1e-4,
            weight_decay=0.1,
            warmup_steps=100,
            max_steps=1000
        )

        # Configure optimizers
        config = module.configure_optimizers()

        # Check optimizer
        assert 'optimizer' in config
        assert isinstance(config['optimizer'], torch.optim.AdamW)

        # Check scheduler
        assert 'lr_scheduler' in config
        assert config['lr_scheduler'] is not None

    def test_lightning_module_forward(self, hnet_config_small, hrm_config_small, device):
        """Test forward pass."""
        module = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        ).to(device)

        input_ids = torch.randint(0, 256, (4, 512), device=device)

        # Forward pass
        with torch.no_grad():
            output = module(input_ids)

        # Check outputs
        assert 'output' in output
        assert output['output'].shape == (4, 10)

    @pytest.mark.slow
    def test_lightning_module_full_training_step(self, hnet_config_small, hrm_config_small, device):
        """Test complete training step with backward."""
        module = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small,
            learning_rate=1e-4
        ).to(device)

        # Configure optimizer manually for this test
        optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)

        batch = (
            torch.randint(0, 256, (2, 256), device=device),
            torch.randint(0, 10, (2,), device=device)
        )

        # Training step
        optimizer.zero_grad()
        loss = module.training_step(batch, batch_idx=0)
        loss.backward()
        optimizer.step()

        # Loss should still be valid after backward
        assert not torch.isnan(loss)


class TestTrainingIntegration:
    """Integration tests for training."""

    def test_checkpoint_saving(self, hnet_config_small, hrm_config_small, temp_dir):
        """Test checkpoint saving and loading."""
        # Create model
        model = CodKingLightningModule(
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        # Save checkpoint
        checkpoint_path = temp_dir / "checkpoint.ckpt"
        trainer = pl.Trainer(
            max_epochs=0,
            logger=False,
            enable_checkpointing=False,
            accelerator='cpu'
        )
        trainer.strategy.connect(model)
        trainer.save_checkpoint(str(checkpoint_path))

        # Load checkpoint
        loaded_model = CodKingLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            task_type='classification',
            num_classes=10,
            hnet_config=hnet_config_small,
            hrm_config=hrm_config_small
        )

        # Models should have same parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(),
            loaded_model.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
