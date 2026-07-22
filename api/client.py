"""
Python client for CodKing Inference API.

Provides convenient methods for interacting with the inference server.

Usage:
    from codking.api.client import CodKingClient

    client = CodKingClient("http://localhost:8000")

    # Detect threat in log
    result = client.detect_threat("Suspicious SSH login attempt from 192.168.1.100")
    print(result.is_threat, result.threat_score)

    # Batch detection
    logs = ["Log 1", "Log 2", "Log 3"]
    results = client.detect_threats_batch(logs)

    # Classify text
    result = client.classify("Some text to classify")
    print(result.predicted_class, result.confidence)
"""

import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import json


class CodKingClient:
    """Client for CodKing Inference API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize client.

        Args:
            base_url: Base URL of the inference server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make POST request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.post(url, json=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    def _get(self, endpoint: str) -> Dict[str, Any]:
        """Make GET request."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    def health(self) -> Dict[str, Any]:
        """
        Get server health status.

        Returns:
            Health status dictionary
        """
        return self._get("/health")

    def load_model(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_cybersecurity: bool = True
    ) -> Dict[str, Any]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_cybersecurity: Use cybersecurity-specific model

        Returns:
            Load status
        """
        return self._post("/load_model", {
            "checkpoint_path": checkpoint_path,
            "device": device,
            "use_cybersecurity": use_cybersecurity
        })

    def detect_threat(
        self,
        log_content: str,
        source: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> 'ThreatDetectionResult':
        """
        Detect threat in log data.

        Args:
            log_content: Log content as text
            source: Log source (e.g., 'firewall', 'web_server')
            timestamp: Log timestamp

        Returns:
            ThreatDetectionResult
        """
        data = {
            "log_content": log_content,
            "source": source,
            "timestamp": timestamp
        }
        result = self._post("/detect_threat", data)
        return ThreatDetectionResult(**result)

    def detect_threats_batch(
        self,
        log_contents: List[str],
        sources: Optional[List[str]] = None,
        timestamps: Optional[List[str]] = None
    ) -> 'BatchThreatDetectionResult':
        """
        Batch threat detection.

        Args:
            log_contents: List of log contents
            sources: List of log sources (optional)
            timestamps: List of log timestamps (optional)

        Returns:
            BatchThreatDetectionResult
        """
        inputs = []
        for i, content in enumerate(log_contents):
            inputs.append({
                "log_content": content,
                "source": sources[i] if sources else None,
                "timestamp": timestamps[i] if timestamps else None
            })

        result = self._post("/detect_threats_batch", inputs)
        return BatchThreatDetectionResult(**result)

    def classify(
        self,
        text: str,
        return_scores: bool = True,
        return_metadata: bool = True
    ) -> 'ClassificationResult':
        """
        Classify text.

        Args:
            text: Input text
            return_scores: Return confidence scores
            return_metadata: Return processing metadata

        Returns:
            ClassificationResult
        """
        data = {
            "text": text,
            "return_scores": return_scores,
            "return_metadata": return_metadata
        }
        result = self._post("/classify", data)
        return ClassificationResult(**result)

    def classify_batch(
        self,
        texts: List[str],
        return_scores: bool = True,
        return_metadata: bool = True
    ) -> List['ClassificationResult']:
        """
        Batch text classification.

        Args:
            texts: List of texts
            return_scores: Return confidence scores
            return_metadata: Return processing metadata

        Returns:
            List of ClassificationResult
        """
        data = {
            "texts": texts,
            "return_scores": return_scores,
            "return_metadata": return_metadata
        }
        result = self._post("/classify_batch", data)

        return [ClassificationResult(**r) for r in result['results']]

    def analyze_log_file(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze log file.

        Args:
            file_path: Path to log file

        Returns:
            Analysis results
        """
        url = f"{self.base_url}/analyze_log_file"
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f)}
            response = self.session.post(url, files=files, timeout=self.timeout)
            response.raise_for_status()
            return response.json()

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get inference metrics.

        Returns:
            Metrics dictionary
        """
        return self._get("/metrics")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.session.close()


# Result classes
class ThreatDetectionResult:
    """Threat detection result."""

    def __init__(
        self,
        is_threat: bool,
        threat_score: float,
        confidence: float,
        prediction_class: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.is_threat = is_threat
        self.threat_score = threat_score
        self.confidence = confidence
        self.prediction_class = prediction_class
        self.metadata = metadata or {}

    def __repr__(self):
        return f"ThreatDetectionResult(is_threat={self.is_threat}, threat_score={self.threat_score:.3f}, confidence={self.confidence:.3f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'is_threat': self.is_threat,
            'threat_score': self.threat_score,
            'confidence': self.confidence,
            'prediction_class': self.prediction_class,
            'metadata': self.metadata
        }


class BatchThreatDetectionResult:
    """Batch threat detection result."""

    def __init__(
        self,
        results: List[Dict[str, Any]],
        total_processed: int,
        processing_time_ms: float,
        throughput_samples_per_sec: float
    ):
        self.results = [ThreatDetectionResult(**r) for r in results]
        self.total_processed = total_processed
        self.processing_time_ms = processing_time_ms
        self.throughput = throughput_samples_per_sec

    def __repr__(self):
        threats = sum(1 for r in self.results if r.is_threat)
        return f"BatchThreatDetectionResult(total={self.total_processed}, threats={threats}, throughput={self.throughput:.1f} samples/sec)"

    def get_threats(self) -> List[ThreatDetectionResult]:
        """Get only threat detections."""
        return [r for r in self.results if r.is_threat]

    def get_normal(self) -> List[ThreatDetectionResult]:
        """Get only normal (non-threat) detections."""
        return [r for r in self.results if not r.is_threat]


class ClassificationResult:
    """Classification result."""

    def __init__(
        self,
        predicted_class: int,
        class_label: Optional[str] = None,
        confidence: Optional[float] = None,
        scores: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.predicted_class = predicted_class
        self.class_label = class_label
        self.confidence = confidence
        self.scores = scores
        self.metadata = metadata or {}

    def __repr__(self):
        return f"ClassificationResult(class={self.predicted_class}, confidence={self.confidence:.3f if self.confidence else 0})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'predicted_class': self.predicted_class,
            'class_label': self.class_label,
            'confidence': self.confidence,
            'scores': self.scores,
            'metadata': self.metadata
        }


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test CodKing API client')
    parser.add_argument('--url', type=str, default='http://localhost:8000',
                       help='API base URL')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Load model from checkpoint')

    args = parser.parse_args()

    # Create client
    print(f"Connecting to {args.url}")
    client = CodKingClient(args.url)

    # Check health
    print("\nChecking server health...")
    health = client.health()
    print(f"  Status: {health['status']}")
    print(f"  Model loaded: {health['model_loaded']}")
    print(f"  Device: {health['device']}")

    # Load model if checkpoint provided
    if args.checkpoint:
        print(f"\nLoading model from {args.checkpoint}...")
        result = client.load_model(args.checkpoint)
        print(f"  {result['message']}")

    # Test threat detection
    print("\n" + "="*80)
    print("Testing Threat Detection")
    print("="*80)

    test_logs = [
        "Normal user login from 192.168.1.10",
        "CRITICAL: Multiple failed SSH attempts from 203.0.113.42",
        "INFO: System backup completed successfully",
        "WARNING: Unusual network traffic detected on port 443"
    ]

    for log in test_logs:
        result = client.detect_threat(log)
        threat_flag = "🚨 THREAT" if result.is_threat else "✅ NORMAL"
        print(f"\n{threat_flag}")
        print(f"  Log: {log[:60]}...")
        print(f"  Threat score: {result.threat_score:.3f}")
        print(f"  Confidence: {result.confidence:.3f}")

    # Test batch detection
    print("\n" + "="*80)
    print("Testing Batch Threat Detection")
    print("="*80)

    batch_result = client.detect_threats_batch(test_logs)
    print(f"\nProcessed: {batch_result.total_processed} logs")
    print(f"Threats found: {len(batch_result.get_threats())}")
    print(f"Processing time: {batch_result.processing_time_ms:.2f}ms")
    print(f"Throughput: {batch_result.throughput:.1f} samples/sec")

    # Test classification
    print("\n" + "="*80)
    print("Testing Text Classification")
    print("="*80)

    test_texts = [
        "This is a positive review",
        "This is a negative review",
        "Neutral statement"
    ]

    for text in test_texts:
        result = client.classify(text)
        print(f"\nText: {text}")
        print(f"  Class: {result.predicted_class}")
        print(f"  Confidence: {result.confidence:.3f}")
        if result.scores:
            print(f"  Scores: {[f'{s:.3f}' for s in result.scores]}")

    # Get metrics
    print("\n" + "="*80)
    print("Server Metrics")
    print("="*80)

    metrics = client.get_metrics()
    print(f"\nTotal requests: {metrics['total_requests']}")
    print(f"Average latency: {metrics['avg_latency_ms']:.2f}ms")
    print(f"Uptime: {metrics['uptime_seconds']:.1f}s")

    print("\n✅ All tests completed successfully!")
