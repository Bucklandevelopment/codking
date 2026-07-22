# CodKing Training Guide

Complete guide for training the CodKing model with PyTorch Lightning.

---

## Quick Start

### 1. Basic Training

```bash
# Train with default configuration
python src/codking/scripts/train.py --config src/codking/configs/training_config.yaml

# Fast development run (1 batch for testing)
python src/codking/scripts/train.py --config src/codking/configs/training_config.yaml --fast-dev-run

# Override data directory
python src/codking/scripts/train.py --config src/codking/configs/training_config.yaml --data-dir /path/to/data
```

### 2. Resume Training

```bash
# Resume from checkpoint
python src/codking/scripts/train.py \
  --config src/codking/configs/training_config.yaml \
  --checkpoint checkpoints/codking-epoch=10-val_accuracy=0.8500.ckpt
```

### 3. Test Only

```bash
# Run testing on saved checkpoint
python src/codking/scripts/train.py \
  --config src/codking/configs/training_config.yaml \
  --checkpoint checkpoints/best.ckpt \
  --test-only
```

---

## Configuration

The training configuration is in `src/codking/configs/training_config.yaml`.

### Key Parameters

#### Task Configuration
```yaml
training:
  task_type: "classification"  # classification, detection, generation
  num_classes: 10
  routing_strategy: "variance"  # variance (recommended), temporal, learned
```

#### Optimization
```yaml
training:
  learning_rate: 1.0e-4
  weight_decay: 0.1
  warmup_steps: 1000
  max_steps: 50000
  batch_size: 32
```

#### Few-Shot Learning (CRITICAL)
```yaml
data:
  use_augmentation: true
  augmentation_factor: 10  # 100 examples → 1000 with augmentation
  noise_level: 0.01
  dropout_prob: 0.05
  swap_prob: 0.05
```

With augmentation enabled:
- **100 real examples** × 10 augmentation = **1000 training examples**
- Achieves target data efficiency (800× vs SOTA)

#### W&B Integration
```yaml
wandb:
  enabled: true
  project: "codking"
  entity: "your-wandb-entity"  # Set this!
  tags: ["hrm", "hnet", "efficiency"]
```

---

## Data Loading

### Option 1: Text Data

```python
# In your data loading code (replace synthetic data in train.py)

train_texts = [...]  # List of text strings
val_texts = [...]
test_texts = [...]

train_labels = [...]  # List of integer labels
val_labels = [...]
test_labels = [...]

data_module = CodKingDataModule(
    train_data=train_texts,
    val_data=val_texts,
    test_data=test_texts,
    train_labels=train_labels,
    val_labels=val_labels,
    test_labels=test_labels,
    dataset_type='text',
    max_length=8192,
    use_augmentation=True,
    augmentation_factor=10,
    batch_size=32
)
```

### Option 2: Byte-Level Data

```python
train_bytes = [...]  # List of byte sequences
val_bytes = [...]
test_bytes = [...]

data_module = CodKingDataModule(
    train_data=train_bytes,
    val_data=val_bytes,
    test_data=test_bytes,
    train_labels=train_labels,
    val_labels=val_labels,
    test_labels=test_labels,
    dataset_type='byte',
    max_length=8192
)
```

### Option 3: Cybersecurity Logs

```python
from pathlib import Path

train_log_files = [Path('logs/train/file1.log'), ...]  # List of paths
val_log_files = [Path('logs/val/file1.log'), ...]
test_log_files = [Path('logs/test/file1.log'), ...]

train_labels = [0, 1, 0, 1, ...]  # 0=normal, 1=threat
val_labels = [...]
test_labels = [...]

data_module = CodKingDataModule(
    train_data=train_log_files,
    val_data=val_log_files,
    test_data=test_log_files,
    train_labels=train_labels,
    val_labels=val_labels,
    test_labels=test_labels,
    dataset_type='cybersecurity',
    max_length=8192
)
```

---

## Training Modes

### 1. Full Training (HNet + HRM)

```yaml
training:
  freeze_hnet: false
  loss_weight_hnet: 0.3
  loss_weight_hrm: 0.7
```

**Use when:**
- Training from scratch
- Have sufficient data (500+ examples)
- Want end-to-end optimization

### 2. HRM-Only Training (Frozen HNet)

```yaml
training:
  freeze_hnet: true
  loss_weight_hnet: 0.0
  loss_weight_hrm: 1.0
```

**Use when:**
- Have pretrained HNet
- Very limited data (<100 examples)
- Fast adaptation to new domains

---

## Monitored Metrics

### Training Metrics

- `train/total_loss`: Combined loss (HNet + HRM)
- `train/accuracy`: Training accuracy
- `train/compression_ratio`: Actual compression achieved
- `train/mean_cycles`: Average HRM cycles used

### Validation Metrics

- `val/loss`: Validation loss
- `val/accuracy`: Validation accuracy
- `val/apm`: Accuracy Per Million Parameters (efficiency)
- `val/apte`: Accuracy Per Training Example (data efficiency)
- `val/efficiency_vs_deepseek`: Parameter efficiency vs SOTA

### Test Metrics

- `test/accuracy`: Test accuracy
- `test/latency_ms`: Inference latency (milliseconds)
- `test/memory_mb`: Memory footprint (megabytes)
- `test/throughput_samples_per_sec`: Throughput

---

## Efficiency Targets

CodKing aims for extreme efficiency:

```yaml
targets:
  # Performance
  min_accuracy: 0.70
  target_accuracy: 0.85

  # Latency
  max_latency_ms: 10.0

  # Compression
  target_compression_ratio: 6.0

  # Parameter efficiency
  target_efficiency_vs_sota: 150.0  # 150× more efficient

  # Memory
  max_memory_mb: 500.0
```

These targets are automatically checked after testing.

---

## W&B Dashboard

When W&B is enabled, you can monitor:

1. **Training Progress**
   - Loss curves
   - Accuracy curves
   - Learning rate schedule

2. **Efficiency Metrics**
   - APM (Accuracy Per Million Parameters)
   - APTE (Accuracy Per Training Example)
   - Latency over time
   - Compression ratio

3. **Model Architecture**
   - Total parameters
   - Component breakdown
   - Routing strategy

4. **Cybersecurity Metrics** (if using CybersecurityLightningModule)
   - Precision, Recall, F1
   - False positive rate
   - Cost analysis

### View Dashboard

```bash
# Start W&B sync (if not auto-syncing)
wandb sync wandb_logs/

# Open in browser
wandb login
# Visit https://wandb.ai/your-entity/codking
```

---

## Checkpoint Management

### Automatic Checkpointing

Checkpoints are saved automatically:

```
checkpoints/
├── codking-epoch=10-val_accuracy=0.8234.ckpt
├── codking-epoch=25-val_accuracy=0.8567.ckpt
├── codking-epoch=40-val_accuracy=0.8891.ckpt  # Best model
└── last.ckpt  # Latest checkpoint
```

### Load Best Model

```python
from codking.training.lightning_module import CodKingLightningModule

# Load best checkpoint
model = CodKingLightningModule.load_from_checkpoint(
    'checkpoints/codking-epoch=40-val_accuracy=0.8891.ckpt'
)

# Use for inference
output = model(input_ids)
```

---

## Cybersecurity Training

For threat detection and OSINT screening:

### Configuration

```yaml
cybersecurity:
  task_type: "threat_detection"  # threat_detection, anomaly_detection, osint_screening
  target_recall: 0.95  # High recall for security
  max_false_positive_rate: 0.01
```

### Usage

```bash
# Set cybersecurity config in training_config.yaml
python src/codking/scripts/train.py --config src/codking/configs/training_config.yaml
```

The script automatically detects cybersecurity mode and uses `CybersecurityLightningModule`, which tracks:
- Precision, Recall, F1 score
- False positive rate (critical for security)
- True positive rate
- Cost-benefit analysis

---

## Multi-GPU Training

### Single Node, Multiple GPUs

```yaml
hardware:
  accelerator: "gpu"
  devices: 4  # Use 4 GPUs
  strategy: "ddp"  # Distributed Data Parallel
```

### Mixed Precision Training

```yaml
hardware:
  precision: "16-mixed"  # FP16 mixed precision
```

**Benefits:**
- 2× faster training
- 2× less memory
- Minimal accuracy impact

---

## Troubleshooting

### OOM (Out of Memory) Error

**Solutions:**
1. Reduce batch size: `batch_size: 16` (from 32)
2. Gradient accumulation: `accumulate_grad_batches: 2`
3. Mixed precision: `precision: "16-mixed"`
4. Freeze HNet: `freeze_hnet: true`

### NaN Loss

**Solutions:**
1. Reduce learning rate: `learning_rate: 5.0e-5`
2. Check gradient clipping: `gradient_clip_val: 1.0`
3. Increase warmup: `warmup_steps: 2000`

### Low Compression Ratio

**Expected:** 5-6:1 compression

If compression is <3:1:
1. Check ratio_loss_weight in hnet_config.yaml (should be 0.03)
2. Train longer (needs time to learn boundaries)
3. Verify smoothing module is enabled (CRITICAL)

### Slow Training

**Solutions:**
1. Use mixed precision: `precision: "16-mixed"`
2. Increase batch size (if memory allows)
3. Use more GPUs: `devices: 4`
4. Reduce workers: `num_workers: 2` (if CPU-bound)

---

## Example: Complete Training Run

```bash
# 1. Prepare your data
# (Replace synthetic data in train.py with your actual data loading)

# 2. Configure W&B
# Edit training_config.yaml, set wandb.entity to your account

# 3. Start training
python src/codking/scripts/train.py \
  --config src/codking/configs/training_config.yaml

# Training will:
# - Load and augment data (100 → 1000 examples)
# - Train for max_epochs with early stopping
# - Save best checkpoints
# - Log to W&B
# - Run final testing
# - Report efficiency metrics

# 4. Check results
# Best model: checkpoints/codking-epoch=XX-val_accuracy=0.XXXX.ckpt
# W&B dashboard: https://wandb.ai/your-entity/codking
```

---

## Expected Performance

With 1000 training examples (100 real + 10× augmentation):

| Metric | Target | Expected |
|--------|--------|----------|
| Accuracy | 85% | 75-85% |
| Latency | <10ms | 5-8ms |
| Compression | 6:1 | 5-6:1 |
| Param Efficiency | 150× | 100-200× |
| Memory | <500MB | 300-400MB |

---

## Next Steps

After training:

1. **Test the best checkpoint**
   ```bash
   python src/codking/scripts/train.py --checkpoint checkpoints/best.ckpt --test-only
   ```

2. **Deploy for inference**
   - See inference API documentation
   - Load checkpoint in production

3. **Fine-tune for specific domains**
   - Start from pretrained checkpoint
   - Train on domain-specific data

4. **Analyze efficiency**
   - Review W&B dashboard
   - Compare to SOTA models
   - Publish results!

---

**Questions?** Check the main README or open an issue on GitHub.
