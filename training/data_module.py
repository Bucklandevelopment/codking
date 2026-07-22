"""
PyTorch Lightning DataModule for CodKing.

Handles:
- Data loading and preprocessing
- Train/val/test splits
- Byte-level tokenization
- Dynamic batching
- Data augmentation for few-shot learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Optional, Callable, List, Tuple
import numpy as np
from pathlib import Path

from ..utils.collators import ChunkBatchCollator


class ByteLevelDataset(Dataset):
    """
    Byte-level dataset for CodKing.

    Converts text/binary data to byte sequences.
    """

    def __init__(
        self,
        data: List[bytes],
        labels: List[int],
        max_length: int = 8192,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.

        Args:
            data: List of byte sequences
            labels: List of labels
            max_length: Maximum sequence length
            transform: Optional data augmentation
        """
        self.data = data
        self.labels = labels
        self.max_length = max_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item.

        Returns:
            Tuple of (input_ids, label)
            input_ids: [seq_len] byte values (0-255)
            label: scalar label
        """
        # Get byte sequence
        byte_seq = self.data[idx]
        label = self.labels[idx]

        # Apply transform if any
        if self.transform:
            byte_seq = self.transform(byte_seq)

        # Convert to byte array
        byte_array = np.frombuffer(byte_seq, dtype=np.uint8)

        # Truncate or pad to max_length
        if len(byte_array) > self.max_length:
            byte_array = byte_array[:self.max_length]
        else:
            padding = np.zeros(self.max_length - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate([byte_array, padding])

        # Convert to tensors
        input_ids = torch.from_numpy(byte_array).long()
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label


class TextToByteDataset(Dataset):
    """
    Text dataset that converts to bytes on-the-fly.
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        max_length: int = 8192,
        encoding: str = 'utf-8'
    ):
        """
        Initialize text dataset.

        Args:
            texts: List of text strings
            labels: List of labels
            max_length: Maximum byte length
            encoding: Text encoding
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.encoding = encoding

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get text converted to bytes."""
        text = self.texts[idx]
        label = self.labels[idx]

        # Convert text to bytes
        byte_seq = text.encode(self.encoding)

        # Convert to byte array
        byte_array = np.frombuffer(byte_seq, dtype=np.uint8)

        # Truncate or pad
        if len(byte_array) > self.max_length:
            byte_array = byte_array[:self.max_length]
        else:
            padding = np.zeros(self.max_length - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate([byte_array, padding])

        input_ids = torch.from_numpy(byte_array).long()
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label


class CybersecurityLogDataset(Dataset):
    """
    Dataset for cybersecurity log data.

    Handles:
    - Log file parsing
    - Threat labeling
    - Byte-level encoding
    """

    def __init__(
        self,
        log_files: List[Path],
        labels: List[int],
        max_length: int = 8192
    ):
        """
        Initialize cybersecurity log dataset.

        Args:
            log_files: List of log file paths
            labels: List of labels (0=normal, 1=threat)
            max_length: Maximum byte length
        """
        self.log_files = log_files
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.log_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and process log file."""
        log_file = self.log_files[idx]
        label = self.labels[idx]

        # Read log file as bytes
        with open(log_file, 'rb') as f:
            byte_seq = f.read(self.max_length)

        # Convert to array
        byte_array = np.frombuffer(byte_seq, dtype=np.uint8)

        # Pad if necessary
        if len(byte_array) < self.max_length:
            padding = np.zeros(self.max_length - len(byte_array), dtype=np.uint8)
            byte_array = np.concatenate([byte_array, padding])

        input_ids = torch.from_numpy(byte_array).long()
        label = torch.tensor(label, dtype=torch.long)

        return input_ids, label


class FewShotAugmentation:
    """
    Data augmentation for few-shot learning.

    Generates synthetic variants to reach 1000 training examples.
    """

    def __init__(
        self,
        noise_level: float = 0.01,
        dropout_prob: float = 0.05,
        swap_prob: float = 0.05
    ):
        """
        Initialize augmentation.

        Args:
            noise_level: Level of byte noise to add
            dropout_prob: Probability of dropping bytes
            swap_prob: Probability of swapping adjacent bytes
        """
        self.noise_level = noise_level
        self.dropout_prob = dropout_prob
        self.swap_prob = swap_prob

    def __call__(self, byte_seq: bytes) -> bytes:
        """
        Apply augmentation to byte sequence.

        Args:
            byte_seq: Input byte sequence

        Returns:
            Augmented byte sequence
        """
        byte_array = np.frombuffer(byte_seq, dtype=np.uint8).copy()

        # Add noise
        if self.noise_level > 0:
            noise = np.random.randint(-5, 6, size=len(byte_array), dtype=np.int16)
            noise_mask = np.random.rand(len(byte_array)) < self.noise_level
            byte_array = byte_array.astype(np.int16)
            byte_array[noise_mask] += noise[noise_mask]
            byte_array = np.clip(byte_array, 0, 255).astype(np.uint8)

        # Dropout (replace with zeros)
        if self.dropout_prob > 0:
            dropout_mask = np.random.rand(len(byte_array)) < self.dropout_prob
            byte_array[dropout_mask] = 0

        # Swap adjacent bytes
        if self.swap_prob > 0:
            for i in range(len(byte_array) - 1):
                if np.random.rand() < self.swap_prob:
                    byte_array[i], byte_array[i+1] = byte_array[i+1], byte_array[i]

        return byte_array.tobytes()


class CodKingDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for CodKing.

    Handles all data loading and preprocessing.
    """

    def __init__(
        self,
        # Data configuration
        train_data: Optional[List] = None,
        val_data: Optional[List] = None,
        test_data: Optional[List] = None,
        train_labels: Optional[List] = None,
        val_labels: Optional[List] = None,
        test_labels: Optional[List] = None,

        # Dataset type
        dataset_type: str = 'byte',  # 'byte', 'text', or 'cybersecurity'

        # Preprocessing
        max_length: int = 8192,
        encoding: str = 'utf-8',

        # Few-shot augmentation
        use_augmentation: bool = False,
        augmentation_factor: int = 10,  # Generate 10× examples
        noise_level: float = 0.01,
        dropout_prob: float = 0.05,
        swap_prob: float = 0.05,

        # DataLoader configuration
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize DataModule.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            train_labels: Training labels
            val_labels: Validation labels
            test_labels: Test labels
            dataset_type: Type of dataset ('byte', 'text', 'cybersecurity')
            max_length: Maximum sequence length in bytes
            encoding: Text encoding (for text datasets)
            use_augmentation: Use few-shot augmentation
            augmentation_factor: Number of augmented copies per example
            noise_level: Augmentation noise level
            dropout_prob: Augmentation dropout probability
            swap_prob: Augmentation swap probability
            batch_size: Batch size
            num_workers: DataLoader workers
            pin_memory: Pin memory for faster GPU transfer
        """
        super().__init__()

        self.save_hyperparameters(ignore=['train_data', 'val_data', 'test_data',
                                          'train_labels', 'val_labels', 'test_labels'])

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.test_labels = test_labels

        self.dataset_type = dataset_type
        self.max_length = max_length
        self.encoding = encoding

        # Augmentation
        self.use_augmentation = use_augmentation
        self.augmentation_factor = augmentation_factor
        if use_augmentation:
            self.augmentation = FewShotAugmentation(
                noise_level=noise_level,
                dropout_prob=dropout_prob,
                swap_prob=swap_prob
            )
        else:
            self.augmentation = None

        # DataLoader config
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Datasets (initialized in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        if stage == 'fit' or stage is None:
            # Apply augmentation to training data if needed
            train_data = self.train_data
            train_labels = self.train_labels

            if self.use_augmentation and train_data is not None:
                augmented_data = []
                augmented_labels = []

                for data, label in zip(train_data, train_labels):
                    # Original example
                    augmented_data.append(data)
                    augmented_labels.append(label)

                    # Generate augmented copies
                    for _ in range(self.augmentation_factor - 1):
                        if self.dataset_type == 'byte':
                            aug_data = self.augmentation(data)
                        elif self.dataset_type == 'text':
                            # Convert text to bytes, augment, convert back
                            byte_seq = data.encode(self.encoding)
                            aug_bytes = self.augmentation(byte_seq)
                            aug_data = aug_bytes.decode(self.encoding, errors='ignore')
                        else:
                            aug_data = data  # No augmentation for files

                        augmented_data.append(aug_data)
                        augmented_labels.append(label)

                train_data = augmented_data
                train_labels = augmented_labels

            # Create training dataset
            if train_data is not None:
                if self.dataset_type == 'byte':
                    self.train_dataset = ByteLevelDataset(
                        train_data, train_labels, self.max_length
                    )
                elif self.dataset_type == 'text':
                    self.train_dataset = TextToByteDataset(
                        train_data, train_labels, self.max_length, self.encoding
                    )
                elif self.dataset_type == 'cybersecurity':
                    self.train_dataset = CybersecurityLogDataset(
                        train_data, train_labels, self.max_length
                    )

            # Create validation dataset
            if self.val_data is not None:
                if self.dataset_type == 'byte':
                    self.val_dataset = ByteLevelDataset(
                        self.val_data, self.val_labels, self.max_length
                    )
                elif self.dataset_type == 'text':
                    self.val_dataset = TextToByteDataset(
                        self.val_data, self.val_labels, self.max_length, self.encoding
                    )
                elif self.dataset_type == 'cybersecurity':
                    self.val_dataset = CybersecurityLogDataset(
                        self.val_data, self.val_labels, self.max_length
                    )

        if stage == 'test' or stage is None:
            # Create test dataset
            if self.test_data is not None:
                if self.dataset_type == 'byte':
                    self.test_dataset = ByteLevelDataset(
                        self.test_data, self.test_labels, self.max_length
                    )
                elif self.dataset_type == 'text':
                    self.test_dataset = TextToByteDataset(
                        self.test_data, self.test_labels, self.max_length, self.encoding
                    )
                elif self.dataset_type == 'cybersecurity':
                    self.test_dataset = CybersecurityLogDataset(
                        self.test_data, self.test_labels, self.max_length
                    )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0)
        )

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=(self.num_workers > 0)
        )


# Example usage
if __name__ == "__main__":
    print("Testing CodKing DataModule\n")

    # Create synthetic data
    num_train = 100
    num_val = 20
    num_test = 20
    num_classes = 10

    # Generate random text data
    print("Generating synthetic text data...")
    train_texts = [f"This is training example {i} with random content." * 10 for i in range(num_train)]
    val_texts = [f"This is validation example {i}." * 10 for i in range(num_val)]
    test_texts = [f"This is test example {i}." * 10 for i in range(num_test)]

    train_labels = np.random.randint(0, num_classes, num_train).tolist()
    val_labels = np.random.randint(0, num_classes, num_val).tolist()
    test_labels = np.random.randint(0, num_classes, num_test).tolist()

    # Create DataModule
    data_module = CodKingDataModule(
        train_data=train_texts,
        val_data=val_texts,
        test_data=test_texts,
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
        dataset_type='text',
        max_length=4096,
        use_augmentation=True,
        augmentation_factor=5,
        batch_size=8,
        num_workers=0
    )

    # Setup
    data_module.setup('fit')

    print(f"\nDataset sizes:")
    print(f"  Training: {len(data_module.train_dataset)} (with augmentation)")
    print(f"  Validation: {len(data_module.val_dataset)}")
    print(f"  Original training: {num_train}")
    print(f"  Augmentation factor: {data_module.augmentation_factor}×")

    # Test dataloaders
    train_loader = data_module.train_dataloader()
    print(f"\nTraining batches: {len(train_loader)}")

    # Get one batch
    batch = next(iter(train_loader))
    input_ids, labels = batch

    print(f"\nBatch shapes:")
    print(f"  Input IDs: {input_ids.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Byte range: [{input_ids.min().item()}, {input_ids.max().item()}]")

    # Test augmentation
    print("\n" + "="*80)
    print("Testing Few-Shot Augmentation\n")

    augmentation = FewShotAugmentation(noise_level=0.02, dropout_prob=0.1, swap_prob=0.1)

    original_text = "Hello, this is a test!"
    original_bytes = original_text.encode('utf-8')

    print(f"Original: {original_bytes[:20]}")

    for i in range(3):
        augmented_bytes = augmentation(original_bytes)
        print(f"Augmented {i+1}: {augmented_bytes[:20]}")

    print("\nAugmentation applied successfully!")
