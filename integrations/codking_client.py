"""
CodKing Client for RLM Integration.

Implements the RLM BaseLM interface to allow CodKing to be used as a
language model backend within the RLM framework.

This enables RLM's recursive decomposition to use CodKing's efficient
27M parameter model for processing each chunk.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging

# Import RLM base types
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'rlm_framework'))

try:
    from rlm.clients.base_lm import BaseLM
    from rlm.core.types import UsageSummary, ModelUsageSummary
    RLM_AVAILABLE = True
except ImportError:
    # Create stub classes if RLM not available
    RLM_AVAILABLE = False

    class BaseLM:
        def __init__(self, model_name: str, **kwargs):
            self.model_name = model_name
            self.kwargs = kwargs

    @dataclass
    class ModelUsageSummary:
        total_calls: int = 0
        total_input_tokens: int = 0
        total_output_tokens: int = 0

        def to_dict(self):
            return {
                'total_calls': self.total_calls,
                'total_input_tokens': self.total_input_tokens,
                'total_output_tokens': self.total_output_tokens,
            }

    @dataclass
    class UsageSummary:
        model_usage_summaries: Dict[str, ModelUsageSummary] = field(default_factory=dict)

        def to_dict(self):
            return {
                'model_usage_summaries': {
                    k: v.to_dict() for k, v in self.model_usage_summaries.items()
                }
            }

logger = logging.getLogger(__name__)


@dataclass
class CodKingResult:
    """Result from CodKing inference."""
    output: torch.Tensor
    prediction: int
    confidence: float
    threat_score: Optional[float] = None
    compression_ratio: float = 1.0
    num_cycles: int = 0
    latency_ms: float = 0.0


class CodKingClient(BaseLM):
    """
    CodKing pipeline as an RLM-compatible language model client.

    This allows RLM to use CodKing for processing chunks during
    recursive decomposition, providing extreme parameter efficiency
    (27M parameters vs 1.5B-7B for typical LLMs).

    Features:
    - Loads CodKing checkpoint on initialization
    - Converts text prompts to byte sequences
    - Returns classification/detection results as text
    - Tracks usage statistics

    Example:
        >>> client = CodKingClient(
        ...     checkpoint_path="checkpoints/threat_detection.ckpt",
        ...     task_type="threat_detection"
        ... )
        >>> result = client.completion("Suspicious log entry here...")
        >>> print(result)  # "THREAT: confidence=0.95"
    """

    # Class labels for different task types
    TASK_LABELS = {
        'classification': ['class_0', 'class_1', 'class_2', 'class_3', 'class_4',
                          'class_5', 'class_6', 'class_7', 'class_8', 'class_9'],
        'threat_detection': ['NORMAL', 'THREAT'],
        'anomaly_detection': ['NORMAL', 'ANOMALY'],
        'vulnerability_scan': ['SAFE', 'VULNERABLE'],
    }

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        task_type: str = 'classification',
        num_classes: int = 2,
        device: str = 'auto',
        max_input_bytes: int = 8192,
        class_labels: Optional[List[str]] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize CodKing client.

        Args:
            checkpoint_path: Path to trained CodKing checkpoint
            task_type: Type of task ('classification', 'threat_detection', etc.)
            num_classes: Number of output classes
            device: Device to use ('cuda', 'cpu', or 'auto')
            max_input_bytes: Maximum input size (default 8192)
            class_labels: Custom labels for classes
            verbose: Enable verbose logging
            **kwargs: Additional arguments passed to BaseLM
        """
        super().__init__(model_name="codking-local", **kwargs)

        self.checkpoint_path = checkpoint_path
        self.task_type = task_type
        self.num_classes = num_classes
        self.max_input_bytes = max_input_bytes
        self.verbose = verbose

        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Class labels
        self.class_labels = class_labels or self.TASK_LABELS.get(
            task_type, [f'class_{i}' for i in range(num_classes)]
        )

        # Load pipeline
        self.pipeline = None
        if checkpoint_path:
            self._load_pipeline()
        else:
            logger.warning("No checkpoint provided. Using mock inference.")

        # Usage tracking
        self._usage = ModelUsageSummary(
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0
        )

        logger.info(
            f"CodKingClient initialized: task={task_type}, "
            f"device={self.device}, classes={num_classes}"
        )

    def _load_pipeline(self) -> None:
        """Load CodKing pipeline from checkpoint."""
        try:
            # Try to import CodKing pipeline
            from ..models.integration.pipeline import CodKingPipeline, CybersecurityPipeline

            checkpoint = Path(self.checkpoint_path)
            if not checkpoint.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

            # Determine pipeline type
            if self.task_type in ['threat_detection', 'anomaly_detection', 'osint_screening']:
                self.pipeline = CybersecurityPipeline(
                    task_type=self.task_type,
                    num_classes=self.num_classes
                )
            else:
                self.pipeline = CodKingPipeline(
                    task_type='classification',
                    num_classes=self.num_classes
                )

            # Load weights if checkpoint exists
            if checkpoint.suffix in ['.ckpt', '.pt', '.pth']:
                state_dict = torch.load(checkpoint, map_location=self.device)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.pipeline.load_state_dict(state_dict, strict=False)

            self.pipeline = self.pipeline.to(self.device)
            self.pipeline.eval()

            logger.info(f"Loaded CodKing pipeline from {checkpoint}")

        except ImportError as e:
            logger.warning(f"Could not import CodKing pipeline: {e}")
            logger.warning("Using mock inference mode")
            self.pipeline = None

        except Exception as e:
            logger.error(f"Error loading pipeline: {e}")
            self.pipeline = None

    def _text_to_bytes(self, text: str) -> torch.Tensor:
        """Convert text to byte tensor for CodKing input."""
        # Encode to bytes
        byte_data = text.encode('utf-8', errors='replace')

        # Truncate or pad to max_input_bytes
        if len(byte_data) > self.max_input_bytes:
            byte_data = byte_data[:self.max_input_bytes]
        elif len(byte_data) < self.max_input_bytes:
            # Pad with zeros
            byte_data = byte_data + b'\x00' * (self.max_input_bytes - len(byte_data))

        # Convert to tensor
        byte_list = list(byte_data)
        return torch.tensor([byte_list], dtype=torch.long, device=self.device)

    def _format_output(self, result: CodKingResult) -> str:
        """Format CodKing result as string response."""
        label = self.class_labels[result.prediction] if result.prediction < len(self.class_labels) else f"class_{result.prediction}"

        if self.task_type == 'threat_detection':
            if result.threat_score is not None:
                return f"{label}: threat_score={result.threat_score:.3f}, confidence={result.confidence:.3f}"
            return f"{label}: confidence={result.confidence:.3f}"

        elif self.task_type == 'anomaly_detection':
            return f"{label}: confidence={result.confidence:.3f}, cycles={result.num_cycles}"

        else:
            return f"{label}: confidence={result.confidence:.3f}"

    def _mock_inference(self, input_tensor: torch.Tensor) -> CodKingResult:
        """Mock inference when pipeline not available."""
        # Generate random prediction based on input
        input_sum = input_tensor.sum().item()
        prediction = int(input_sum) % self.num_classes
        confidence = 0.5 + (input_sum % 50) / 100  # 0.5-1.0

        return CodKingResult(
            output=torch.zeros(1, self.num_classes),
            prediction=prediction,
            confidence=confidence,
            threat_score=confidence if self.task_type == 'threat_detection' else None,
            compression_ratio=6.0,
            num_cycles=8,
            latency_ms=1.0
        )

    def _run_inference(self, input_tensor: torch.Tensor) -> CodKingResult:
        """Run actual CodKing inference."""
        start_time = time.perf_counter()

        with torch.no_grad():
            output = self.pipeline(
                input_ids=input_tensor,
                return_loss=False,
                use_act_inference=True
            )

        latency_ms = (time.perf_counter() - start_time) * 1000

        # Extract results
        logits = output['output']
        probs = torch.softmax(logits, dim=-1)
        prediction = probs.argmax(dim=-1).item()
        confidence = probs[0, prediction].item()

        result = CodKingResult(
            output=logits,
            prediction=prediction,
            confidence=confidence,
            compression_ratio=output.get('compression_ratio', 1.0),
            num_cycles=output.get('num_cycles', 0),
            latency_ms=latency_ms
        )

        # Add threat score for threat detection
        if self.task_type == 'threat_detection' and 'threat_score' in output:
            result.threat_score = output['threat_score'][0].item()
        elif self.task_type == 'threat_detection' and self.num_classes == 2:
            result.threat_score = probs[0, 1].item()  # P(threat)

        return result

    def completion(self, prompt: Union[str, Dict[str, Any]]) -> str:
        """
        Process input through CodKing pipeline.

        This is the main interface method required by RLM's BaseLM.

        Args:
            prompt: Text input or message dict

        Returns:
            Formatted string result
        """
        # Handle dict prompts (message format)
        if isinstance(prompt, dict):
            if 'content' in prompt:
                text = str(prompt['content'])
            else:
                text = str(prompt)
        elif isinstance(prompt, list):
            # List of messages
            text = ' '.join(
                str(m.get('content', m)) if isinstance(m, dict) else str(m)
                for m in prompt
            )
        else:
            text = str(prompt)

        # Convert to bytes tensor
        input_tensor = self._text_to_bytes(text)

        # Run inference
        if self.pipeline is not None:
            result = self._run_inference(input_tensor)
        else:
            result = self._mock_inference(input_tensor)

        # Track usage
        self._usage.total_calls += 1
        self._usage.total_input_tokens += len(text)
        self._usage.total_output_tokens += 50  # Approximate output size

        if self.verbose:
            logger.info(
                f"CodKing inference: {result.prediction} "
                f"({result.confidence:.3f}) in {result.latency_ms:.2f}ms"
            )

        return self._format_output(result)

    async def acompletion(self, prompt: Union[str, Dict[str, Any]]) -> str:
        """
        Async version of completion.

        Currently runs synchronously as CodKing inference is fast enough.
        """
        return self.completion(prompt)

    def get_usage_summary(self) -> UsageSummary:
        """Get aggregated usage statistics."""
        return UsageSummary(
            model_usage_summaries={
                self.model_name: ModelUsageSummary(
                    total_calls=self._usage.total_calls,
                    total_input_tokens=self._usage.total_input_tokens,
                    total_output_tokens=self._usage.total_output_tokens
                )
            }
        )

    def get_last_usage(self) -> UsageSummary:
        """Get last call's usage (same as total for this implementation)."""
        return self.get_usage_summary()

    def reset_usage(self) -> None:
        """Reset usage statistics."""
        self._usage = ModelUsageSummary(
            total_calls=0,
            total_input_tokens=0,
            total_output_tokens=0
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        info = {
            'model_name': self.model_name,
            'task_type': self.task_type,
            'num_classes': self.num_classes,
            'device': self.device,
            'max_input_bytes': self.max_input_bytes,
            'class_labels': self.class_labels,
            'pipeline_loaded': self.pipeline is not None,
        }

        if self.pipeline is not None:
            try:
                params = self.pipeline.get_num_params()
                info['parameters'] = params
            except Exception:
                pass

        return info


# =============================================================================
# Factory Function
# =============================================================================

def create_codking_client(
    checkpoint_path: str,
    task_type: str = 'threat_detection',
    device: str = 'auto',
    **kwargs
) -> CodKingClient:
    """
    Factory function to create a CodKing client.

    Args:
        checkpoint_path: Path to model checkpoint
        task_type: Type of task
        device: Device to use
        **kwargs: Additional arguments

    Returns:
        Configured CodKingClient instance
    """
    return CodKingClient(
        checkpoint_path=checkpoint_path,
        task_type=task_type,
        device=device,
        **kwargs
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("CodKingClient Demo")
    print("=" * 60)

    # Create client (mock mode without checkpoint)
    client = CodKingClient(
        task_type='threat_detection',
        num_classes=2,
        verbose=True
    )

    # Test completions
    test_inputs = [
        "Normal system log entry: user login successful",
        "ALERT: Multiple failed authentication attempts from IP 192.168.1.100",
        "ERROR: Segmentation fault in process 1234",
        "WARNING: Unusual network traffic detected on port 443",
    ]

    print("\nProcessing test inputs:")
    print("-" * 60)

    for text in test_inputs:
        result = client.completion(text)
        print(f"Input: {text[:50]}...")
        print(f"Result: {result}")
        print()

    # Print usage
    usage = client.get_usage_summary()
    print(f"\nUsage Summary:")
    print(f"  Total calls: {usage.model_usage_summaries['codking-local'].total_calls}")
    print(f"  Total input tokens: {usage.model_usage_summaries['codking-local'].total_input_tokens}")

    print("\n" + "=" * 60)
    print("Demo complete!")
