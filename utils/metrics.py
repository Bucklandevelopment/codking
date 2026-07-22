"""
Efficiency metrics for CodKing models.
Implements APM (Accuracy per Million Parameters) and APTE (Accuracy per Training Example).
"""

import time
from typing import Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class EfficiencyMetrics:
    """Container for efficiency metrics."""
    # Performance metrics
    accuracy: float

    # Parameter efficiency
    total_params: int
    apm: float  # Accuracy Per Million Parameters

    # Data efficiency
    training_examples: int
    apte: float  # Accuracy Per Training Example

    # Computational efficiency
    inference_latency_ms: float
    throughput_samples_per_sec: float
    memory_footprint_mb: float

    # Energy (optional)
    energy_joules_per_inference: float = 0.0


def compute_apm(accuracy: float, num_parameters: int) -> float:
    """
    Compute Accuracy Per Million Parameters.

    Args:
        accuracy: Model accuracy (0-100 or 0-1)
        num_parameters: Total number of parameters

    Returns:
        APM score
    """
    return accuracy / (num_parameters / 1e6)


def compute_apte(accuracy: float, num_training_examples: int) -> float:
    """
    Compute Accuracy Per Training Example.

    Args:
        accuracy: Model accuracy (0-100 or 0-1)
        num_training_examples: Number of training examples used

    Returns:
        APTE score
    """
    return accuracy / num_training_examples


def measure_inference_latency(model: nn.Module, input_tensor: torch.Tensor,
                               num_runs: int = 100, warmup_runs: int = 10) -> float:
    """
    Measure average inference latency in milliseconds.

    Args:
        model: Model to measure
        input_tensor: Sample input tensor
        num_runs: Number of inference runs to average
        warmup_runs: Number of warmup runs (not counted)

    Returns:
        Average latency in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)

    # Measure
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.perf_counter()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.perf_counter()

    avg_latency_ms = ((end_time - start_time) / num_runs) * 1000
    return avg_latency_ms


def measure_memory_footprint(model: nn.Module) -> float:
    """
    Measure model memory footprint in MB.

    Args:
        model: Model to measure

    Returns:
        Memory footprint in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.numel() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / (1024 ** 2)
    return total_size_mb


def compute_throughput(latency_ms: float, batch_size: int = 1) -> float:
    """
    Compute throughput in samples per second.

    Args:
        latency_ms: Inference latency in milliseconds
        batch_size: Batch size used

    Returns:
        Throughput in samples/second
    """
    latency_sec = latency_ms / 1000
    return batch_size / latency_sec


def compute_efficiency_metrics(
    model: nn.Module,
    accuracy: float,
    num_training_examples: int,
    sample_input: torch.Tensor,
    batch_size: int = 1
) -> EfficiencyMetrics:
    """
    Compute all efficiency metrics for a model.

    Args:
        model: Model to evaluate
        accuracy: Achieved accuracy (0-100 or 0-1)
        num_training_examples: Number of training examples used
        sample_input: Sample input for latency measurement
        batch_size: Batch size for throughput calculation

    Returns:
        EfficiencyMetrics object
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Compute parameter efficiency
    apm = compute_apm(accuracy, total_params)

    # Compute data efficiency
    apte = compute_apte(accuracy, num_training_examples)

    # Measure latency
    latency_ms = measure_inference_latency(model, sample_input)

    # Compute throughput
    throughput = compute_throughput(latency_ms, batch_size)

    # Measure memory
    memory_mb = measure_memory_footprint(model)

    return EfficiencyMetrics(
        accuracy=accuracy,
        total_params=total_params,
        apm=apm,
        training_examples=num_training_examples,
        apte=apte,
        inference_latency_ms=latency_ms,
        throughput_samples_per_sec=throughput,
        memory_footprint_mb=memory_mb
    )


def compare_with_baseline(
    metrics: EfficiencyMetrics,
    baseline_metrics: EfficiencyMetrics
) -> Dict[str, float]:
    """
    Compare metrics with baseline (e.g., DeepSeek R1-Distill-1.5B).

    Args:
        metrics: Current model metrics
        baseline_metrics: Baseline model metrics

    Returns:
        Dictionary of improvement ratios
    """
    return {
        "param_efficiency_gain": metrics.apm / baseline_metrics.apm,
        "data_efficiency_gain": metrics.apte / baseline_metrics.apte,
        "speedup": baseline_metrics.inference_latency_ms / metrics.inference_latency_ms,
        "memory_reduction": baseline_metrics.memory_footprint_mb / metrics.memory_footprint_mb,
        "throughput_gain": metrics.throughput_samples_per_sec / baseline_metrics.throughput_samples_per_sec
    }


def print_efficiency_report(metrics: EfficiencyMetrics, baseline: EfficiencyMetrics = None):
    """
    Print formatted efficiency report.

    Args:
        metrics: Model metrics
        baseline: Optional baseline metrics for comparison
    """
    print(f"\n{'='*70}")
    print(f"{'CodKing Efficiency Report':^70}")
    print(f"{'='*70}")

    print(f"\n📊 Performance:")
    print(f"  Accuracy: {metrics.accuracy:.2f}%")

    print(f"\n🔢 Parameter Efficiency:")
    print(f"  Total Parameters: {metrics.total_params:,}")
    print(f"  APM (Accuracy per Million Params): {metrics.apm:.4f}")

    print(f"\n📚 Data Efficiency:")
    print(f"  Training Examples: {metrics.training_examples:,}")
    print(f"  APTE (Accuracy per Training Example): {metrics.apte:.6f}")

    print(f"\n⚡ Computational Efficiency:")
    print(f"  Inference Latency: {metrics.inference_latency_ms:.2f} ms")
    print(f"  Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
    print(f"  Memory Footprint: {metrics.memory_footprint_mb:.2f} MB")

    if baseline:
        print(f"\n📈 Comparison with Baseline:")
        comparison = compare_with_baseline(metrics, baseline)
        print(f"  Parameter Efficiency Gain: {comparison['param_efficiency_gain']:.1f}×")
        print(f"  Data Efficiency Gain: {comparison['data_efficiency_gain']:.1f}×")
        print(f"  Speedup: {comparison['speedup']:.1f}×")
        print(f"  Memory Reduction: {comparison['memory_reduction']:.1f}×")
        print(f"  Throughput Gain: {comparison['throughput_gain']:.1f}×")

    print(f"\n{'='*70}\n")
