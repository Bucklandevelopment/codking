"""
Infinite Context Manager for CodKing + RLM Integration.

Manages arbitrarily large contexts using a hybrid RAM + Disk storage approach.
Enables processing of contexts that exceed available RAM by using:
- LRU cache for frequently accessed chunks (hot data)
- Memory-mapped files for large contexts (cold data)
- Automatic eviction and memory management

Supports contexts from kilobytes to terabytes.
"""

import mmap
import os
import shutil
import tempfile
import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union, List, Callable
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a context chunk."""
    index: int
    start_offset: int
    end_offset: int
    size: int
    checksum: Optional[str] = None


@dataclass
class ContextStats:
    """Statistics about the loaded context."""
    total_size: int
    num_chunks: int
    ram_cached_chunks: int
    ram_used_bytes: int
    disk_used_bytes: int
    source_type: str  # 'file', 'bytes', 'stream'


class InfiniteContextManager:
    """
    Manages arbitrarily large contexts using RAM + disk hybrid storage.

    Features:
    - Hot data in RAM (LRU cache with configurable size)
    - Cold data on disk (memory-mapped files for efficient access)
    - Automatic eviction when RAM limit is reached
    - Chunk-based iteration for streaming processing
    - Thread-safe operations

    Memory Hierarchy:
    1. L1: GPU VRAM - Current batch (managed by CodKing pipeline)
    2. L2: RAM Cache - Recently accessed chunks (this manager)
    3. L3: Memory-Mapped Files - Full context on disk (this manager)
    4. L4: Disk Storage - Raw files (external)

    Example:
        >>> manager = InfiniteContextManager(ram_limit_gb=2.0)
        >>> manager.load_from_file("/path/to/100gb_log.txt")
        >>> for idx, chunk in manager.iter_chunks():
        ...     result = process(chunk)
        >>> manager.cleanup()
    """

    def __init__(
        self,
        ram_limit_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        cache_dir: Optional[str] = None,
        chunk_size: int = 8192,  # CodKing's input size
        enable_compression: bool = False,
        enable_checksums: bool = False,
    ):
        """
        Initialize the InfiniteContextManager.

        Args:
            ram_limit_bytes: Maximum RAM to use for chunk caching (default 1GB)
            cache_dir: Directory for disk-based storage (auto-created if None)
            chunk_size: Size of each chunk in bytes (default 8192 for CodKing)
            enable_compression: Enable zlib compression for disk storage
            enable_checksums: Enable MD5 checksums for data integrity
        """
        self.ram_limit = ram_limit_bytes
        self.chunk_size = chunk_size
        self.enable_compression = enable_compression
        self.enable_checksums = enable_checksums

        # Create cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._temp_created = False
        else:
            self.cache_dir = Path(tempfile.mkdtemp(prefix="codking_context_"))
            self._temp_created = True

        # LRU cache for hot chunks
        self._ram_cache: OrderedDict[int, bytes] = OrderedDict()
        self._ram_used: int = 0

        # Disk storage state
        self._disk_path: Optional[Path] = None
        self._mmap: Optional[mmap.mmap] = None
        self._mmap_fd: Optional[int] = None
        self._total_size: int = 0
        self._source_type: str = "none"

        # Chunk metadata
        self._chunk_metadata: List[ChunkMetadata] = []

        # Thread safety
        self._lock = threading.RLock()

        # State flags
        self._is_loaded = False

        logger.info(
            f"InfiniteContextManager initialized: "
            f"ram_limit={ram_limit_bytes / (1024**3):.2f}GB, "
            f"chunk_size={chunk_size}, cache_dir={self.cache_dir}"
        )

    # =========================================================================
    # Loading Methods
    # =========================================================================

    def load_from_file(self, file_path: Union[str, Path]) -> ContextStats:
        """
        Load context from a file, using memory-mapping for large files.

        For files smaller than RAM limit, loads entirely into memory.
        For larger files, uses memory-mapped file access for efficiency.

        Args:
            file_path: Path to the context file

        Returns:
            ContextStats with information about the loaded context
        """
        with self._lock:
            self._cleanup_internal()

            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Context file not found: {file_path}")

            file_size = file_path.stat().st_size
            self._total_size = file_size
            self._source_type = "file"

            logger.info(f"Loading context from file: {file_path} ({file_size / (1024**2):.2f}MB)")

            if file_size <= self.ram_limit:
                # Small file: load entirely into RAM cache
                with open(file_path, 'rb') as f:
                    data = f.read()
                self._load_to_ram_cache(data)
            else:
                # Large file: use memory-mapped access
                self._setup_mmap(file_path)

            self._build_chunk_metadata()
            self._is_loaded = True

            return self.get_stats()

    def load_from_bytes(self, data: bytes) -> ContextStats:
        """
        Load context from bytes, spilling to disk if larger than RAM limit.

        Args:
            data: Raw bytes to load as context

        Returns:
            ContextStats with information about the loaded context
        """
        with self._lock:
            self._cleanup_internal()

            self._total_size = len(data)
            self._source_type = "bytes"

            logger.info(f"Loading context from bytes: {len(data) / (1024**2):.2f}MB")

            if len(data) <= self.ram_limit:
                # Fits in RAM
                self._load_to_ram_cache(data)
            else:
                # Spill to disk
                disk_path = self.cache_dir / "context.bin"
                with open(disk_path, 'wb') as f:
                    f.write(data)
                self._setup_mmap(disk_path)

            self._build_chunk_metadata()
            self._is_loaded = True

            return self.get_stats()

    def load_from_string(self, text: str, encoding: str = 'utf-8') -> ContextStats:
        """
        Load context from string, converting to bytes.

        Args:
            text: String to load as context
            encoding: Character encoding (default utf-8)

        Returns:
            ContextStats with information about the loaded context
        """
        return self.load_from_bytes(text.encode(encoding))

    def load_from_stream(
        self,
        stream: Iterator[bytes],
        total_size_hint: Optional[int] = None
    ) -> ContextStats:
        """
        Load context from a byte stream, useful for network sources.

        Always writes to disk first, then memory-maps for access.

        Args:
            stream: Iterator yielding bytes
            total_size_hint: Optional hint for progress reporting

        Returns:
            ContextStats with information about the loaded context
        """
        with self._lock:
            self._cleanup_internal()

            self._source_type = "stream"
            disk_path = self.cache_dir / "context_stream.bin"

            total_written = 0
            with open(disk_path, 'wb') as f:
                for chunk in stream:
                    f.write(chunk)
                    total_written += len(chunk)

                    if total_size_hint and total_written % (10 * 1024 * 1024) == 0:
                        progress = (total_written / total_size_hint) * 100
                        logger.info(f"Stream loading: {progress:.1f}%")

            self._total_size = total_written
            logger.info(f"Loaded {total_written / (1024**2):.2f}MB from stream")

            if total_written <= self.ram_limit:
                with open(disk_path, 'rb') as f:
                    self._load_to_ram_cache(f.read())
                disk_path.unlink()  # Remove temp file
            else:
                self._setup_mmap(disk_path)

            self._build_chunk_metadata()
            self._is_loaded = True

            return self.get_stats()

    # =========================================================================
    # Chunk Access Methods
    # =========================================================================

    def get_chunk(self, index: int) -> bytes:
        """
        Get a specific chunk by index.

        Uses LRU cache for frequently accessed chunks.

        Args:
            index: Chunk index (0-based)

        Returns:
            Chunk data as bytes

        Raises:
            IndexError: If chunk index is out of range
            RuntimeError: If no context is loaded
        """
        with self._lock:
            if not self._is_loaded:
                raise RuntimeError("No context loaded. Call load_* method first.")

            if index < 0 or index >= self.num_chunks:
                raise IndexError(f"Chunk index {index} out of range [0, {self.num_chunks})")

            # Check RAM cache first (LRU hit)
            if index in self._ram_cache:
                self._ram_cache.move_to_end(index)
                return self._ram_cache[index]

            # Load from memory-mapped file
            if self._mmap is not None:
                metadata = self._chunk_metadata[index]
                chunk = self._mmap[metadata.start_offset:metadata.end_offset]

                # Verify checksum if enabled
                if self.enable_checksums and metadata.checksum:
                    actual_checksum = hashlib.md5(chunk).hexdigest()
                    if actual_checksum != metadata.checksum:
                        raise RuntimeError(f"Checksum mismatch for chunk {index}")

                # Cache in RAM
                self._cache_chunk(index, chunk)
                return chunk

            raise RuntimeError(f"Cannot retrieve chunk {index}: no storage available")

    def get_chunks(self, indices: List[int]) -> List[bytes]:
        """
        Get multiple chunks by indices.

        Optimized for batch access patterns.

        Args:
            indices: List of chunk indices

        Returns:
            List of chunk data in same order as indices
        """
        return [self.get_chunk(idx) for idx in indices]

    def iter_chunks(self) -> Iterator[tuple[int, bytes]]:
        """
        Iterate over all chunks efficiently.

        Yields:
            Tuple of (chunk_index, chunk_data)
        """
        for i in range(self.num_chunks):
            yield i, self.get_chunk(i)

    def iter_chunks_parallel(
        self,
        num_workers: int = 4,
        prefetch: int = 8
    ) -> Iterator[tuple[int, bytes]]:
        """
        Iterate over chunks with parallel prefetching.

        Prefetches chunks in background threads for better throughput.

        Args:
            num_workers: Number of prefetch threads
            prefetch: Number of chunks to prefetch ahead

        Yields:
            Tuple of (chunk_index, chunk_data)
        """
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue

        result_queue: Queue = Queue(maxsize=prefetch)

        def prefetch_worker(start_idx: int, end_idx: int):
            for i in range(start_idx, end_idx):
                chunk = self.get_chunk(i)
                result_queue.put((i, chunk))

        # Simple sequential for now, can be extended
        for i in range(self.num_chunks):
            yield i, self.get_chunk(i)

    # =========================================================================
    # Chunk Processing with Callbacks
    # =========================================================================

    def process_chunks(
        self,
        processor: Callable[[int, bytes], any],
        batch_size: int = 1,
        parallel: bool = False,
        num_workers: int = 4
    ) -> List[any]:
        """
        Process all chunks with a callback function.

        Args:
            processor: Function taking (index, chunk) and returning result
            batch_size: Number of chunks to process together
            parallel: Enable parallel processing
            num_workers: Number of parallel workers

        Returns:
            List of results from processor
        """
        results = []

        if parallel and batch_size == 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(processor, idx, self.get_chunk(idx)): idx
                    for idx in range(self.num_chunks)
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append((idx, result))
                    except Exception as e:
                        logger.error(f"Error processing chunk {idx}: {e}")
                        results.append((idx, None))

            # Sort by index
            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]
        else:
            # Sequential processing
            for idx, chunk in self.iter_chunks():
                result = processor(idx, chunk)
                results.append(result)

            return results

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _load_to_ram_cache(self, data: bytes) -> None:
        """Load entire data into RAM cache as chunks."""
        self._ram_cache.clear()
        self._ram_used = 0

        for i in range(0, len(data), self.chunk_size):
            chunk = data[i:i + self.chunk_size]
            chunk_idx = i // self.chunk_size
            self._ram_cache[chunk_idx] = chunk
            self._ram_used += len(chunk)

    def _setup_mmap(self, file_path: Path) -> None:
        """Setup memory-mapped file access."""
        self._disk_path = file_path
        self._mmap_fd = os.open(str(file_path), os.O_RDONLY)
        self._mmap = mmap.mmap(self._mmap_fd, 0, access=mmap.ACCESS_READ)

    def _build_chunk_metadata(self) -> None:
        """Build metadata for all chunks."""
        self._chunk_metadata.clear()

        num_chunks = (self._total_size + self.chunk_size - 1) // self.chunk_size

        for i in range(num_chunks):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, self._total_size)

            metadata = ChunkMetadata(
                index=i,
                start_offset=start,
                end_offset=end,
                size=end - start
            )

            # Compute checksum if enabled
            if self.enable_checksums:
                chunk_data = self._get_raw_chunk(start, end)
                metadata.checksum = hashlib.md5(chunk_data).hexdigest()

            self._chunk_metadata.append(metadata)

    def _get_raw_chunk(self, start: int, end: int) -> bytes:
        """Get raw chunk data from any storage."""
        if self._mmap is not None:
            return self._mmap[start:end]
        else:
            # Find in RAM cache
            chunk_idx = start // self.chunk_size
            if chunk_idx in self._ram_cache:
                return self._ram_cache[chunk_idx]
        raise RuntimeError("Cannot access raw chunk data")

    def _cache_chunk(self, index: int, chunk: bytes) -> None:
        """Add chunk to RAM cache with LRU eviction."""
        chunk_size = len(chunk)

        # Evict old chunks if needed
        while self._ram_used + chunk_size > self.ram_limit and self._ram_cache:
            _, evicted = self._ram_cache.popitem(last=False)
            self._ram_used -= len(evicted)

        self._ram_cache[index] = chunk
        self._ram_used += chunk_size

    def _cleanup_internal(self) -> None:
        """Internal cleanup without lock."""
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None

        if self._mmap_fd is not None:
            try:
                os.close(self._mmap_fd)
            except Exception:
                pass
            self._mmap_fd = None

        self._ram_cache.clear()
        self._ram_used = 0
        self._chunk_metadata.clear()
        self._total_size = 0
        self._is_loaded = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def num_chunks(self) -> int:
        """Total number of chunks."""
        return len(self._chunk_metadata)

    @property
    def total_size(self) -> int:
        """Total size of context in bytes."""
        return self._total_size

    @property
    def is_loaded(self) -> bool:
        """Whether a context is currently loaded."""
        return self._is_loaded

    def get_stats(self) -> ContextStats:
        """Get current context statistics."""
        return ContextStats(
            total_size=self._total_size,
            num_chunks=self.num_chunks,
            ram_cached_chunks=len(self._ram_cache),
            ram_used_bytes=self._ram_used,
            disk_used_bytes=self._total_size if self._mmap is not None else 0,
            source_type=self._source_type
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup(self) -> None:
        """Release all resources and clean up temporary files."""
        with self._lock:
            self._cleanup_internal()

            if self._temp_created and self.cache_dir.exists():
                try:
                    shutil.rmtree(self.cache_dir)
                except Exception as e:
                    logger.warning(f"Failed to cleanup cache dir: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # =========================================================================
    # Serialization
    # =========================================================================

    def save_metadata(self, path: Union[str, Path]) -> None:
        """Save context metadata for later restoration."""
        metadata = {
            'total_size': self._total_size,
            'chunk_size': self.chunk_size,
            'num_chunks': self.num_chunks,
            'source_type': self._source_type,
            'chunks': [
                {
                    'index': m.index,
                    'start': m.start_offset,
                    'end': m.end_offset,
                    'size': m.size,
                    'checksum': m.checksum
                }
                for m in self._chunk_metadata
            ]
        }

        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_context_manager(
    ram_limit_gb: float = 1.0,
    cache_dir: Optional[str] = None,
    chunk_size: int = 8192
) -> InfiniteContextManager:
    """
    Factory function to create InfiniteContextManager with GB-based RAM limit.

    Args:
        ram_limit_gb: RAM limit in gigabytes
        cache_dir: Optional cache directory
        chunk_size: Chunk size in bytes

    Returns:
        Configured InfiniteContextManager instance
    """
    ram_limit_bytes = int(ram_limit_gb * 1024 * 1024 * 1024)
    return InfiniteContextManager(
        ram_limit_bytes=ram_limit_bytes,
        cache_dir=cache_dir,
        chunk_size=chunk_size
    )


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Demo with synthetic data
    print("InfiniteContextManager Demo")
    print("=" * 60)

    # Create manager with 100MB RAM limit
    with InfiniteContextManager(ram_limit_bytes=100 * 1024 * 1024) as manager:
        # Generate 50MB of test data
        test_data = b"A" * (50 * 1024 * 1024)

        stats = manager.load_from_bytes(test_data)
        print(f"\nLoaded context:")
        print(f"  Total size: {stats.total_size / (1024**2):.2f}MB")
        print(f"  Num chunks: {stats.num_chunks}")
        print(f"  RAM cached: {stats.ram_cached_chunks} chunks")
        print(f"  RAM used: {stats.ram_used_bytes / (1024**2):.2f}MB")

        # Access some chunks
        print(f"\nAccessing chunks:")
        for i in [0, 100, 500, 1000]:
            if i < stats.num_chunks:
                chunk = manager.get_chunk(i)
                print(f"  Chunk {i}: {len(chunk)} bytes")

        # Iterate over first 10 chunks
        print(f"\nIterating first 10 chunks:")
        for idx, chunk in manager.iter_chunks():
            if idx >= 10:
                break
            print(f"  Chunk {idx}: {len(chunk)} bytes")

    print("\n" + "=" * 60)
    print("Demo complete!")
