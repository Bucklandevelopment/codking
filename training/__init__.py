"""
Training infrastructure for CodKing.

Provides:
- PyTorch Lightning modules (CodKingLightningModule, CybersecurityLightningModule)
- Data loading and preprocessing (CodKingDataModule)
- Few-shot augmentation
- W&B logging integration
"""

from .lightning_module import (
    CodKingLightningModule,
    CybersecurityLightningModule
)

from .data_module import (
    CodKingDataModule,
    ByteLevelDataset,
    TextToByteDataset,
    CybersecurityLogDataset,
    FewShotAugmentation
)

__all__ = [
    'CodKingLightningModule',
    'CybersecurityLightningModule',
    'CodKingDataModule',
    'ByteLevelDataset',
    'TextToByteDataset',
    'CybersecurityLogDataset',
    'FewShotAugmentation'
]
