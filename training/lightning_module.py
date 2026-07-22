"""
PyTorch Lightning module for CodKing training.

Handles:
- Training loop with combined HNet + HRM losses
- Validation and testing
- Optimizer configuration (AdamW with warmup)
- Learning rate scheduling
- Gradient clipping
- Checkpoint management
- W&B logging integration
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Optional, Any, Tuple
import wandb

from ..models.integration.pipeline import CodKingPipeline, CybersecurityPipeline
from ..utils.metrics import (
    compute_apm, compute_apte, measure_inference_latency,
    measure_memory_footprint, compute_throughput
)


class CodKingLightningModule(pl.LightningModule):
    """
    PyTorch Lightning wrapper for CodKing pipeline.

    Features:
    - Combined HNet + HRM training
    - Adaptive Computation Time (ACT) support
    - Deep supervision
    - Efficiency metrics tracking
    - W&B integration
    """

    def __init__(
        self,
        # Model configuration
        task_type: str = 'classification',
        num_classes: int = 10,
        hnet_config: Optional[Dict] = None,
        hrm_config: Optional[Dict] = None,
        routing_strategy: str = 'variance',
        routing_config: Optional[Dict] = None,
        freeze_hnet: bool = False,
        loss_weight_hnet: float = 0.3,
        loss_weight_hrm: float = 0.7,

        # Training configuration
        learning_rate: float = 1e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 1000,
        max_steps: int = 50000,
        gradient_clip_val: float = 1.0,

        # Efficiency tracking
        track_efficiency: bool = True,
        num_training_examples: int = 1000,

        # W&B configuration
        use_wandb: bool = True,
        log_every_n_steps: int = 50,
        **kwargs
    ):
        """
        Initialize Lightning module.

        Args:
            task_type: Type of task ('classification', 'detection', 'generation')
            num_classes: Number of output classes
            hnet_config: H-Net configuration
            hrm_config: HRM configuration
            routing_strategy: Chunk routing strategy
            routing_config: Routing-specific config
            freeze_hnet: Freeze HNet during training
            loss_weight_hnet: Weight for HNet losses
            loss_weight_hrm: Weight for HRM losses
            learning_rate: Initial learning rate
            weight_decay: AdamW weight decay
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            gradient_clip_val: Max gradient norm
            track_efficiency: Track efficiency metrics
            num_training_examples: Number of training examples (for APTE)
            use_wandb: Enable W&B logging
            log_every_n_steps: W&B logging frequency
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Initialize model
        self.model = CodKingPipeline(
            task_type=task_type,
            num_classes=num_classes,
            hnet_config=hnet_config,
            hrm_config=hrm_config,
            routing_strategy=routing_strategy,
            routing_config=routing_config,
            freeze_hnet=freeze_hnet,
            loss_weight_hnet=loss_weight_hnet,
            loss_weight_hrm=loss_weight_hrm
        )

        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_clip_val = gradient_clip_val

        # Efficiency tracking
        self.track_efficiency = track_efficiency
        self.num_training_examples = num_training_examples
        self.num_params = self.model.get_num_params()['total']

        # W&B configuration
        self.use_wandb = use_wandb
        self.log_every_n_steps = log_every_n_steps

        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model."""
        return self.model(
            input_ids=input_ids,
            targets=targets,
            return_loss=(targets is not None),
            **kwargs
        )

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: (input_ids, targets)
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        input_ids, targets = batch

        # Forward pass
        output = self(
            input_ids=input_ids,
            targets=targets,
            use_act_inference=True,
            return_intermediate=False
        )

        loss = output['loss']

        # Log losses
        if 'loss_dict' in output:
            for key, value in output['loss_dict'].items():
                self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=(key == 'total_loss'))

        # Log compression and efficiency metrics
        self.log('train/compression_ratio', output['compression_ratio'], on_step=True, on_epoch=True)
        self.log('train/num_chunks', output['num_chunks'], on_step=True, on_epoch=True)
        self.log('train/mean_cycles', output['halt_steps'].float().mean(), on_step=True, on_epoch=True)

        # Compute accuracy
        if self.hparams.task_type == 'classification':
            preds = output['output'].argmax(dim=-1)
            acc = (preds == targets).float().mean()
            self.log('train/accuracy', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Validation step.

        Args:
            batch: (input_ids, targets)
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        input_ids, targets = batch

        # Forward pass
        output = self(
            input_ids=input_ids,
            targets=targets,
            use_act_inference=True,
            return_intermediate=False
        )

        # Compute accuracy
        preds = output['output'].argmax(dim=-1) if self.hparams.task_type == 'classification' else (output['output'] > 0.5).long()
        acc = (preds == targets).float().mean()

        # Store outputs
        result = {
            'val_loss': output['loss'],
            'val_accuracy': acc,
            'compression_ratio': output['compression_ratio'],
            'num_cycles': output['halt_steps'].float().mean()
        }

        self.validation_step_outputs.append(result)

        return result

    def on_validation_epoch_end(self):
        """Aggregate validation metrics."""
        if not self.validation_step_outputs:
            return

        # Average metrics
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in self.validation_step_outputs]).mean()
        avg_compression = torch.stack([x['compression_ratio'] for x in self.validation_step_outputs]).mean()
        avg_cycles = torch.stack([x['num_cycles'] for x in self.validation_step_outputs]).mean()

        # Log metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/accuracy', avg_acc, prog_bar=True)
        self.log('val/compression_ratio', avg_compression)
        self.log('val/num_cycles', avg_cycles)

        # Compute efficiency metrics
        if self.track_efficiency:
            apm = compute_apm(avg_acc.item(), self.num_params)
            apte = compute_apte(avg_acc.item(), self.num_training_examples)

            self.log('val/apm', apm)
            self.log('val/apte', apte)

            # Log parameter efficiency comparison
            deepseek_efficiency = 1_500_000_000 / self.num_params
            self.log('val/efficiency_vs_deepseek', deepseek_efficiency)

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Test step with latency measurement.

        Args:
            batch: (input_ids, targets)
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        input_ids, targets = batch

        # Measure latency
        latency_ms = measure_inference_latency(
            self.model,
            input_ids,
            num_runs=10
        )

        # Forward pass
        output = self(
            input_ids=input_ids,
            targets=targets,
            use_act_inference=True,
            return_intermediate=False
        )

        # Compute accuracy
        preds = output['output'].argmax(dim=-1) if self.hparams.task_type == 'classification' else (output['output'] > 0.5).long()
        acc = (preds == targets).float().mean()

        result = {
            'test_loss': output['loss'],
            'test_accuracy': acc,
            'latency_ms': torch.tensor(latency_ms),
            'compression_ratio': output['compression_ratio'],
            'num_cycles': output['halt_steps'].float().mean()
        }

        self.test_step_outputs.append(result)

        return result

    def on_test_epoch_end(self):
        """Aggregate test metrics."""
        if not self.test_step_outputs:
            return

        # Average metrics
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in self.test_step_outputs]).mean()
        avg_latency = torch.stack([x['latency_ms'] for x in self.test_step_outputs]).mean()
        avg_compression = torch.stack([x['compression_ratio'] for x in self.test_step_outputs]).mean()
        avg_cycles = torch.stack([x['num_cycles'] for x in self.test_step_outputs]).mean()

        # Log metrics
        self.log('test/loss', avg_loss)
        self.log('test/accuracy', avg_acc)
        self.log('test/latency_ms', avg_latency)
        self.log('test/compression_ratio', avg_compression)
        self.log('test/num_cycles', avg_cycles)

        # Efficiency metrics
        if self.track_efficiency:
            apm = compute_apm(avg_acc.item(), self.num_params)
            apte = compute_apte(avg_acc.item(), self.num_training_examples)
            memory_mb = measure_memory_footprint(self.model)
            throughput = compute_throughput(avg_latency.item(), batch_size=4)

            self.log('test/apm', apm)
            self.log('test/apte', apte)
            self.log('test/memory_mb', memory_mb)
            self.log('test/throughput_samples_per_sec', throughput)

            # Check if targets met
            target_latency = 10.0  # ms
            target_efficiency_min = 50.0  # vs SOTA
            target_efficiency_max = 260.0

            deepseek_efficiency = 1_500_000_000 / self.num_params

            self.log('test/latency_target_met', float(avg_latency.item() < target_latency))
            self.log('test/efficiency_target_met', float(target_efficiency_min <= deepseek_efficiency <= target_efficiency_max))

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

        Uses AdamW with linear warmup and cosine decay.
        """
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Learning rate scheduler
        def lr_lambda(current_step: int):
            # Warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(max(1, self.max_steps - self.warmup_steps))
            return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359))))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def on_train_start(self):
        """Log model summary at training start."""
        if self.use_wandb and wandb.run is not None:
            # Log model architecture
            wandb.config.update({
                'total_params': self.num_params,
                'trainable_params': self.model.get_num_params(trainable_only=True)['total'],
                'hnet_params': self.num_params_dict['hnet'],
                'hrm_params': self.num_params_dict['hrm'],
                'routing_strategy': self.hparams.routing_strategy,
                'freeze_hnet': self.hparams.freeze_hnet
            })

    @property
    def num_params_dict(self) -> Dict[str, int]:
        """Get parameter counts as dictionary."""
        return self.model.get_num_params()


class CybersecurityLightningModule(CodKingLightningModule):
    """
    Specialized Lightning module for cybersecurity tasks.

    Extensions:
    - Threat detection metrics (precision, recall, F1)
    - False positive rate tracking
    - Cost-benefit analysis
    """

    def __init__(
        self,
        task_type: str = 'threat_detection',
        **kwargs
    ):
        # Map cybersecurity tasks to base types
        task_mapping = {
            'threat_detection': 'classification',
            'anomaly_detection': 'detection',
            'osint_screening': 'classification'
        }

        base_task = task_mapping.get(task_type, 'classification')

        # Default to binary classification
        if 'num_classes' not in kwargs:
            kwargs['num_classes'] = 2

        super().__init__(task_type=base_task, **kwargs)

        self.cybersecurity_task = task_type

    def validation_step(self, batch, batch_idx):
        """Add cybersecurity-specific metrics."""
        result = super().validation_step(batch, batch_idx)

        input_ids, targets = batch
        output = self(input_ids=input_ids, targets=targets)

        # Compute precision, recall, F1 for threat detection
        if self.hparams.num_classes == 2:
            preds = output['output'].argmax(dim=-1)

            # True positives, false positives, false negatives
            tp = ((preds == 1) & (targets == 1)).sum().float()
            fp = ((preds == 1) & (targets == 0)).sum().float()
            fn = ((preds == 0) & (targets == 1)).sum().float()
            tn = ((preds == 0) & (targets == 0)).sum().float()

            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            fpr = fp / (fp + tn + 1e-8)  # False positive rate

            result.update({
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'fpr': fpr
            })

        return result

    def on_validation_epoch_end(self):
        """Log cybersecurity-specific metrics."""
        super().on_validation_epoch_end()

        if self.validation_step_outputs and 'precision' in self.validation_step_outputs[0]:
            avg_precision = torch.stack([x['precision'] for x in self.validation_step_outputs]).mean()
            avg_recall = torch.stack([x['recall'] for x in self.validation_step_outputs]).mean()
            avg_f1 = torch.stack([x['f1'] for x in self.validation_step_outputs]).mean()
            avg_fpr = torch.stack([x['fpr'] for x in self.validation_step_outputs]).mean()

            self.log('val/precision', avg_precision, prog_bar=True)
            self.log('val/recall', avg_recall, prog_bar=True)
            self.log('val/f1', avg_f1, prog_bar=True)
            self.log('val/false_positive_rate', avg_fpr)


# Example usage
if __name__ == "__main__":
    print("Testing CodKing Lightning Module\n")

    # Create Lightning module
    lightning_module = CodKingLightningModule(
        task_type='classification',
        num_classes=10,
        learning_rate=1e-4,
        warmup_steps=1000,
        max_steps=50000,
        use_wandb=False
    )

    print("Model initialized successfully!")
    lightning_module.model.print_summary()

    # Test forward pass
    batch_size = 4
    seq_len = 8192
    input_ids = torch.randint(0, 256, (batch_size, seq_len))
    targets = torch.randint(0, 10, (batch_size,))

    print("\nTesting training step:")
    loss = lightning_module.training_step((input_ids, targets), batch_idx=0)
    print(f"  Loss: {loss.item():.4f}")

    # Test optimizer configuration
    print("\nTesting optimizer configuration:")
    optimizer_config = lightning_module.configure_optimizers()
    print(f"  Optimizer: {optimizer_config['optimizer'].__class__.__name__}")
    print(f"  Scheduler: {optimizer_config['lr_scheduler']['scheduler'].__class__.__name__}")

    # Test cybersecurity module
    print("\n" + "="*80)
    print("Testing Cybersecurity Lightning Module\n")

    cyber_module = CybersecurityLightningModule(
        task_type='threat_detection',
        use_wandb=False
    )

    print("Cybersecurity Configuration:")
    print(f"  Task: {cyber_module.cybersecurity_task}")
    print(f"  Classes: {cyber_module.hparams.num_classes} (binary)")

    # Test with binary data
    threat_labels = torch.randint(0, 2, (batch_size,))
    loss = cyber_module.training_step((input_ids, threat_labels), batch_idx=0)
    print(f"\n  Threat detection loss: {loss.item():.4f}")
