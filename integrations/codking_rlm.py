"""
CodKing RLM - Unified Interface for Infinite Context Processing.

Combines CodKing's efficient inference with RLM's recursive decomposition
to enable processing of arbitrarily large contexts.

This is the main entry point for using CodKing with RLM capabilities.
"""

import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from .context_manager import InfiniteContextManager
from .codking_client import CodKingClient
from .codking_repl import CodKingREPL

logger = logging.getLogger(__name__)


@dataclass
class CodKingRLMResult:
    """Result from CodKing RLM processing."""
    response: str
    chunks_processed: int
    total_context_size: int
    execution_time: float
    chunk_results: List[str] = field(default_factory=list)
    aggregated_stats: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkResult:
    """Result from processing a single chunk."""
    index: int
    response: str
    confidence: float
    latency_ms: float
    error: Optional[str] = None


class CodKingRLM:
    """
    Unified interface for CodKing + RLM infinite context processing.

    This class provides a simple API for:
    - Loading contexts of any size (KB to TB)
    - Processing with CodKing's efficient 27M parameter model
    - Recursive decomposition for complex analysis
    - Result aggregation across chunks

    Memory Management:
    - Hot data cached in RAM (configurable limit)
    - Cold data stored on disk (memory-mapped)
    - Automatic LRU eviction

    Processing Modes:
    - Sequential: Process chunks one at a time
    - Parallel: Process multiple chunks concurrently
    - Streaming: Process chunks as they arrive

    Example:
        >>> rlm = CodKingRLM(
        ...     checkpoint_path="checkpoints/threat_detection.ckpt",
        ...     task_type="threat_detection",
        ...     ram_limit_gb=4.0
        ... )
        >>> # Process a massive log file
        >>> result = rlm.analyze_file(
        ...     "/var/log/security.log",
        ...     query="Find all security threats"
        ... )
        >>> print(f"Found {result.aggregated_stats['threats']} threats")
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        task_type: str = 'classification',
        num_classes: int = 2,
        ram_limit_gb: float = 1.0,
        cache_dir: Optional[str] = None,
        chunk_size: int = 8192,
        device: str = 'auto',
        parallel_workers: int = 4,
        verbose: bool = False,
    ):
        """
        Initialize CodKing RLM.

        Args:
            checkpoint_path: Path to CodKing checkpoint
            task_type: Task type ('classification', 'threat_detection', etc.)
            num_classes: Number of output classes
            ram_limit_gb: RAM limit for context caching
            cache_dir: Directory for disk-based storage
            chunk_size: Size of processing chunks (default 8192)
            device: Device for inference ('cuda', 'cpu', 'auto')
            parallel_workers: Number of parallel processing workers
            verbose: Enable verbose logging
        """
        self.task_type = task_type
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.parallel_workers = parallel_workers
        self.verbose = verbose

        # Initialize context manager
        ram_limit_bytes = int(ram_limit_gb * 1024 * 1024 * 1024)
        self.context_manager = InfiniteContextManager(
            ram_limit_bytes=ram_limit_bytes,
            cache_dir=cache_dir,
            chunk_size=chunk_size
        )

        # Initialize CodKing client
        self.client = CodKingClient(
            checkpoint_path=checkpoint_path,
            task_type=task_type,
            num_classes=num_classes,
            device=device,
            verbose=verbose
        )

        # State
        self._context_loaded = False

        logger.info(
            f"CodKingRLM initialized: task={task_type}, "
            f"ram_limit={ram_limit_gb}GB, workers={parallel_workers}"
        )

    # =========================================================================
    # High-Level API
    # =========================================================================

    def completion(
        self,
        context: Union[str, bytes, Path],
        query: Optional[str] = None,
        parallel: bool = True,
        aggregate: bool = True
    ) -> CodKingRLMResult:
        """
        Main entry point for processing context with optional query.

        This replaces the standard `llm.completion()` call with
        infinite context support.

        Args:
            context: Context data (string, bytes, or file path)
            query: Optional query to answer about the context
            parallel: Enable parallel chunk processing
            aggregate: Aggregate results across chunks

        Returns:
            CodKingRLMResult with response and metadata
        """
        start_time = time.perf_counter()

        # Load context
        if isinstance(context, Path) or (isinstance(context, str) and Path(context).exists()):
            self._load_file(context)
        elif isinstance(context, bytes):
            self.context_manager.load_from_bytes(context)
        else:
            self.context_manager.load_from_string(context)

        self._context_loaded = True

        # Process all chunks
        if parallel and self.parallel_workers > 1:
            chunk_results = self._process_parallel(query)
        else:
            chunk_results = self._process_sequential(query)

        # Aggregate results
        if aggregate:
            response, stats = self._aggregate_results(chunk_results, query)
        else:
            response = "\n".join(r.response for r in chunk_results)
            stats = {}

        execution_time = time.perf_counter() - start_time

        return CodKingRLMResult(
            response=response,
            chunks_processed=len(chunk_results),
            total_context_size=self.context_manager.total_size,
            execution_time=execution_time,
            chunk_results=[r.response for r in chunk_results],
            aggregated_stats=stats,
            metadata={
                'task_type': self.task_type,
                'parallel': parallel,
                'workers': self.parallel_workers if parallel else 1,
                'chunk_size': self.chunk_size,
            }
        )

    def analyze_file(
        self,
        file_path: Union[str, Path],
        query: Optional[str] = None,
        **kwargs
    ) -> CodKingRLMResult:
        """
        Analyze a file of any size.

        Args:
            file_path: Path to file
            query: Optional query about the file
            **kwargs: Additional arguments for completion()

        Returns:
            Analysis result
        """
        return self.completion(Path(file_path), query=query, **kwargs)

    def analyze_text(
        self,
        text: str,
        query: Optional[str] = None,
        **kwargs
    ) -> CodKingRLMResult:
        """
        Analyze text data.

        Args:
            text: Text to analyze
            query: Optional query about the text
            **kwargs: Additional arguments for completion()

        Returns:
            Analysis result
        """
        return self.completion(text, query=query, **kwargs)

    def analyze_stream(
        self,
        stream: Iterator[bytes],
        query: Optional[str] = None,
        **kwargs
    ) -> CodKingRLMResult:
        """
        Analyze streaming data.

        Args:
            stream: Byte stream iterator
            query: Optional query
            **kwargs: Additional arguments

        Returns:
            Analysis result
        """
        start_time = time.perf_counter()

        self.context_manager.load_from_stream(stream)
        self._context_loaded = True

        # Process
        chunk_results = self._process_parallel(query) if kwargs.get('parallel', True) else self._process_sequential(query)
        response, stats = self._aggregate_results(chunk_results, query)

        return CodKingRLMResult(
            response=response,
            chunks_processed=len(chunk_results),
            total_context_size=self.context_manager.total_size,
            execution_time=time.perf_counter() - start_time,
            chunk_results=[r.response for r in chunk_results],
            aggregated_stats=stats,
            metadata={'streaming': True}
        )

    # =========================================================================
    # Specialized Analysis Methods
    # =========================================================================

    def detect_threats(
        self,
        context: Union[str, bytes, Path],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Specialized threat detection analysis.

        Args:
            context: Context to analyze
            threshold: Threat confidence threshold

        Returns:
            Dict with threats, normal entries, and statistics
        """
        result = self.completion(context, parallel=True)

        threats = []
        normal = []

        for idx, chunk_result in enumerate(result.chunk_results):
            if "THREAT" in chunk_result.upper():
                threats.append({
                    'chunk_index': idx,
                    'result': chunk_result
                })
            else:
                normal.append(idx)

        return {
            'threats': threats,
            'threat_count': len(threats),
            'normal_count': len(normal),
            'total_chunks': result.chunks_processed,
            'threat_ratio': len(threats) / max(result.chunks_processed, 1),
            'execution_time': result.execution_time
        }

    def find_anomalies(
        self,
        context: Union[str, bytes, Path],
        sensitivity: float = 0.7
    ) -> Dict[str, Any]:
        """
        Find anomalies in context data.

        Args:
            context: Context to analyze
            sensitivity: Anomaly detection sensitivity

        Returns:
            Dict with anomalies and statistics
        """
        result = self.completion(context, parallel=True)

        anomalies = []
        for idx, chunk_result in enumerate(result.chunk_results):
            if "ANOMALY" in chunk_result.upper() or "UNUSUAL" in chunk_result.upper():
                anomalies.append({
                    'chunk_index': idx,
                    'result': chunk_result
                })

        return {
            'anomalies': anomalies,
            'anomaly_count': len(anomalies),
            'total_chunks': result.chunks_processed,
            'execution_time': result.execution_time
        }

    # =========================================================================
    # Internal Processing Methods
    # =========================================================================

    def _load_file(self, path: Union[str, Path]) -> None:
        """Load context from file."""
        self.context_manager.load_from_file(Path(path))

    def _process_sequential(self, query: Optional[str]) -> List[ChunkResult]:
        """Process chunks sequentially."""
        results = []

        for idx, chunk in self.context_manager.iter_chunks():
            result = self._process_chunk(idx, chunk, query)
            results.append(result)

            if self.verbose and idx % 100 == 0:
                logger.info(f"Processed chunk {idx}/{self.context_manager.num_chunks}")

        return results

    def _process_parallel(self, query: Optional[str]) -> List[ChunkResult]:
        """Process chunks in parallel."""
        results = []

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = {
                executor.submit(self._process_chunk, idx, chunk, query): idx
                for idx, chunk in self.context_manager.iter_chunks()
            }

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing chunk {idx}: {e}")
                    results.append(ChunkResult(
                        index=idx,
                        response=f"Error: {e}",
                        confidence=0.0,
                        latency_ms=0.0,
                        error=str(e)
                    ))

        # Sort by index
        results.sort(key=lambda r: r.index)
        return results

    def _process_chunk(
        self,
        index: int,
        chunk: bytes,
        query: Optional[str]
    ) -> ChunkResult:
        """Process a single chunk."""
        start_time = time.perf_counter()

        try:
            # Decode chunk
            text = chunk.decode('utf-8', errors='replace')

            # Build prompt
            if query:
                prompt = f"Query: {query}\n\nContext chunk {index}:\n{text}"
            else:
                prompt = text

            # Get response
            response = self.client.completion(prompt)
            latency_ms = (time.perf_counter() - start_time) * 1000

            # Extract confidence from response
            confidence = 0.5
            if "confidence=" in response:
                try:
                    conf_str = response.split("confidence=")[1].split(",")[0].split()[0]
                    confidence = float(conf_str)
                except Exception:
                    pass

            return ChunkResult(
                index=index,
                response=response,
                confidence=confidence,
                latency_ms=latency_ms
            )

        except Exception as e:
            return ChunkResult(
                index=index,
                response=f"Error: {e}",
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000,
                error=str(e)
            )

    def _aggregate_results(
        self,
        chunk_results: List[ChunkResult],
        query: Optional[str]
    ) -> tuple[str, Dict[str, Any]]:
        """Aggregate results from all chunks."""
        # Collect statistics
        stats = {
            'total_chunks': len(chunk_results),
            'errors': sum(1 for r in chunk_results if r.error),
            'avg_confidence': sum(r.confidence for r in chunk_results) / max(len(chunk_results), 1),
            'avg_latency_ms': sum(r.latency_ms for r in chunk_results) / max(len(chunk_results), 1),
            'total_latency_ms': sum(r.latency_ms for r in chunk_results),
        }

        # Task-specific aggregation
        if self.task_type == 'threat_detection':
            threats = [r for r in chunk_results if 'THREAT' in r.response.upper()]
            stats['threats'] = len(threats)
            stats['threat_chunks'] = [r.index for r in threats]

            if threats:
                response = f"THREATS DETECTED: {len(threats)} chunks flagged\n"
                response += "\n".join(f"- Chunk {r.index}: {r.response}" for r in threats[:10])
                if len(threats) > 10:
                    response += f"\n... and {len(threats) - 10} more"
            else:
                response = "No threats detected in any chunk."

        elif self.task_type == 'anomaly_detection':
            anomalies = [r for r in chunk_results if 'ANOMALY' in r.response.upper()]
            stats['anomalies'] = len(anomalies)

            if anomalies:
                response = f"ANOMALIES DETECTED: {len(anomalies)} chunks\n"
                response += "\n".join(f"- Chunk {r.index}" for r in anomalies[:10])
            else:
                response = "No anomalies detected."

        else:
            # Generic aggregation
            response = f"Processed {len(chunk_results)} chunks.\n"
            response += f"Average confidence: {stats['avg_confidence']:.3f}\n"

            # Show sample results
            sample = chunk_results[:5]
            response += "\nSample results:\n"
            response += "\n".join(f"- Chunk {r.index}: {r.response[:100]}..." for r in sample)

        return response, stats

    # =========================================================================
    # Resource Management
    # =========================================================================

    def cleanup(self) -> None:
        """Release all resources."""
        self.context_manager.cleanup()
        self._context_loaded = False

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        ctx_stats = self.context_manager.get_stats() if self._context_loaded else None
        client_info = self.client.get_model_info()

        return {
            'context': {
                'loaded': self._context_loaded,
                'total_size': ctx_stats.total_size if ctx_stats else 0,
                'num_chunks': ctx_stats.num_chunks if ctx_stats else 0,
                'ram_cached': ctx_stats.ram_cached_chunks if ctx_stats else 0,
            },
            'model': client_info,
            'config': {
                'task_type': self.task_type,
                'chunk_size': self.chunk_size,
                'parallel_workers': self.parallel_workers,
            }
        }

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_threat_detector(
    checkpoint_path: Optional[str] = None,
    ram_limit_gb: float = 2.0,
    **kwargs
) -> CodKingRLM:
    """Create a CodKingRLM configured for threat detection."""
    return CodKingRLM(
        checkpoint_path=checkpoint_path,
        task_type='threat_detection',
        num_classes=2,
        ram_limit_gb=ram_limit_gb,
        **kwargs
    )


def create_anomaly_detector(
    checkpoint_path: Optional[str] = None,
    ram_limit_gb: float = 2.0,
    **kwargs
) -> CodKingRLM:
    """Create a CodKingRLM configured for anomaly detection."""
    return CodKingRLM(
        checkpoint_path=checkpoint_path,
        task_type='anomaly_detection',
        num_classes=2,
        ram_limit_gb=ram_limit_gb,
        **kwargs
    )


def create_code_analyzer(
    checkpoint_path: Optional[str] = None,
    ram_limit_gb: float = 4.0,
    **kwargs
) -> CodKingRLM:
    """Create a CodKingRLM configured for code analysis."""
    return CodKingRLM(
        checkpoint_path=checkpoint_path,
        task_type='vulnerability_scan',
        num_classes=2,
        ram_limit_gb=ram_limit_gb,
        **kwargs
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("CodKingRLM Demo - Infinite Context Processing")
    print("=" * 60)

    # Create RLM instance
    with CodKingRLM(
        task_type='threat_detection',
        ram_limit_gb=0.1,  # 100MB for demo
        verbose=True
    ) as rlm:

        # Generate test data (simulating a large log)
        print("\n1. Generating test log data...")
        normal_entries = ["INFO: User login successful - " + f"user_{i}" for i in range(100)]
        threat_entries = [
            "ALERT: Multiple failed auth attempts from 192.168.1.100",
            "WARNING: Suspicious process spawned: /tmp/evil.sh",
            "ERROR: Privilege escalation attempt detected",
        ]

        all_entries = normal_entries + threat_entries
        test_log = "\n".join(all_entries)

        print(f"   Generated {len(all_entries)} log entries ({len(test_log)} bytes)")

        # Analyze
        print("\n2. Analyzing with CodKingRLM...")
        result = rlm.completion(
            test_log,
            query="Find all security threats",
            parallel=True
        )

        print(f"\n3. Results:")
        print(f"   Chunks processed: {result.chunks_processed}")
        print(f"   Execution time: {result.execution_time:.2f}s")
        print(f"   Response: {result.response[:500]}...")

        # Get statistics
        print("\n4. Statistics:")
        stats = rlm.get_stats()
        print(f"   Context size: {stats['context']['total_size']} bytes")
        print(f"   Chunks: {stats['context']['num_chunks']}")
        print(f"   RAM cached: {stats['context']['ram_cached']} chunks")

        # Specialized threat detection
        print("\n5. Specialized threat detection:")
        threats = rlm.detect_threats(test_log)
        print(f"   Threats found: {threats['threat_count']}")
        print(f"   Normal entries: {threats['normal_count']}")
        print(f"   Threat ratio: {threats['threat_ratio']:.2%}")

    print("\n" + "=" * 60)
    print("Demo complete!")
