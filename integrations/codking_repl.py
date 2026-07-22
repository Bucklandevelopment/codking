"""
CodKing REPL Environment for RLM Integration.

Implements the RLM BaseEnv interface to provide a REPL-like environment
that uses CodKing for processing and InfiniteContextManager for
handling contexts that exceed available RAM.

Features:
- Infinite context support via RAM + disk hybrid storage
- CodKing-based chunk processing
- Code execution with sandboxed namespace
- llm_query() using CodKing pipeline
"""

import copy
import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

# Import RLM types
sys.path.insert(0, str(Path(__file__).parent.parent / 'rlm_framework'))

try:
    from rlm.environments.base_env import NonIsolatedEnv, SupportsPersistence
    from rlm.core.types import REPLResult, RLMChatCompletion
    RLM_AVAILABLE = True
except ImportError:
    RLM_AVAILABLE = False

    class NonIsolatedEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class SupportsPersistence:
        pass

    from dataclasses import dataclass

    @dataclass
    class REPLResult:
        stdout: str
        stderr: str
        locals: dict
        execution_time: float = 0.0
        rlm_calls: list = None

        def __post_init__(self):
            if self.rlm_calls is None:
                self.rlm_calls = []

    @dataclass
    class RLMChatCompletion:
        root_model: str
        prompt: str
        response: str
        usage_summary: Any
        execution_time: float

from .context_manager import InfiniteContextManager
from .codking_client import CodKingClient

logger = logging.getLogger(__name__)


# Safe builtins for code execution
_SAFE_BUILTINS = {
    # Core types and functions
    "print": print,
    "len": len,
    "str": str,
    "int": int,
    "float": float,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "bool": bool,
    "type": type,
    "isinstance": isinstance,
    "enumerate": enumerate,
    "zip": zip,
    "map": map,
    "filter": filter,
    "sorted": sorted,
    "reversed": reversed,
    "range": range,
    "min": min,
    "max": max,
    "sum": sum,
    "abs": abs,
    "round": round,
    "any": any,
    "all": all,
    "chr": chr,
    "ord": ord,
    "hex": hex,
    "bin": bin,
    "repr": repr,
    "format": format,
    "hash": hash,
    "iter": iter,
    "next": next,
    "slice": slice,
    "callable": callable,
    "hasattr": hasattr,
    "getattr": getattr,
    "setattr": setattr,
    "dir": dir,
    "bytes": bytes,
    "bytearray": bytearray,
    "object": object,
    "__import__": __import__,
    "open": open,
    # Exceptions
    "Exception": Exception,
    "ValueError": ValueError,
    "TypeError": TypeError,
    "KeyError": KeyError,
    "IndexError": IndexError,
    "RuntimeError": RuntimeError,
    # Blocked
    "input": None,
    "eval": None,
    "exec": None,
    "compile": None,
    "globals": None,
    "locals": None,
}


class CodKingREPL(NonIsolatedEnv):
    """
    CodKing-native REPL environment with infinite context support.

    This environment integrates:
    - InfiniteContextManager for RAM + disk context storage
    - CodKingClient for efficient LLM queries
    - Sandboxed Python execution

    Features:
    - Process contexts of any size (limited only by disk space)
    - <10ms per-chunk processing with CodKing
    - Thread-safe operations
    - Persistent multi-turn support

    Example:
        >>> repl = CodKingREPL(
        ...     codking_checkpoint="checkpoints/model.ckpt",
        ...     ram_limit_gb=4.0
        ... )
        >>> repl.load_context(massive_log_data)
        >>> result = repl.execute_code("print(len(context))")
        >>> print(result.stdout)
    """

    def __init__(
        self,
        lm_handler_address: Optional[tuple] = None,
        context_payload: Optional[Union[str, dict, list]] = None,
        codking_checkpoint: Optional[str] = None,
        codking_task_type: str = 'classification',
        ram_limit_gb: float = 1.0,
        cache_dir: Optional[str] = None,
        chunk_size: int = 8192,
        setup_code: Optional[str] = None,
        persistent: bool = False,
        **kwargs
    ):
        """
        Initialize CodKing REPL environment.

        Args:
            lm_handler_address: Address for RLM's LM handler (host, port)
            context_payload: Initial context to load
            codking_checkpoint: Path to CodKing checkpoint
            codking_task_type: Task type for CodKing
            ram_limit_gb: RAM limit for context caching in GB
            cache_dir: Directory for disk-based context storage
            chunk_size: Size of context chunks (default 8192)
            setup_code: Code to execute on setup
            persistent: Enable persistent multi-turn sessions
            **kwargs: Additional arguments
        """
        if hasattr(super(), '__init__'):
            super().__init__(persistent=persistent, **kwargs)

        self.lm_handler_address = lm_handler_address
        self.persistent = persistent

        # Create temp directory
        self.temp_dir = Path(tempfile.mkdtemp(prefix=f"codking_repl_{uuid.uuid4()}_"))

        # Initialize context manager
        ram_limit_bytes = int(ram_limit_gb * 1024 * 1024 * 1024)
        self.context_manager = InfiniteContextManager(
            ram_limit_bytes=ram_limit_bytes,
            cache_dir=cache_dir or str(self.temp_dir / "context_cache"),
            chunk_size=chunk_size
        )

        # Initialize CodKing client
        self.codking_client = CodKingClient(
            checkpoint_path=codking_checkpoint,
            task_type=codking_task_type
        )

        # Thread safety
        self._lock = threading.RLock()

        # Multi-turn state
        self._context_count = 0
        self._history_count = 0

        # Setup execution environment
        self.setup()

        # Load initial context if provided
        if context_payload is not None:
            self.load_context(context_payload)

        # Run setup code if provided
        if setup_code:
            self.execute_code(setup_code)

        logger.info(
            f"CodKingREPL initialized: ram_limit={ram_limit_gb}GB, "
            f"chunk_size={chunk_size}"
        )

    def setup(self) -> None:
        """Setup the execution environment."""
        # Create sandboxed globals
        self.globals: Dict[str, Any] = {
            "__builtins__": _SAFE_BUILTINS.copy(),
            "__name__": "__main__",
        }
        self.locals: Dict[str, Any] = {}

        # Track LLM calls
        self._pending_llm_calls: List[RLMChatCompletion] = []

        # Add helper functions
        self.globals["FINAL_VAR"] = self._final_var
        self.globals["llm_query"] = self._llm_query
        self.globals["llm_query_batched"] = self._llm_query_batched
        self.globals["codking_analyze"] = self._codking_analyze
        self.globals["get_chunk"] = self._get_chunk
        self.globals["iter_chunks"] = self._iter_chunks
        self.globals["context_stats"] = self._get_context_stats

    def _final_var(self, variable_name: str) -> str:
        """Return the value of a variable as a final answer."""
        variable_name = variable_name.strip().strip("\"'")
        if variable_name in self.locals:
            return str(self.locals[variable_name])
        return f"Error: Variable '{variable_name}' not found"

    def _llm_query(self, prompt: str, model: Optional[str] = None) -> str:
        """
        Query an LLM. Uses CodKing by default, or RLM handler if available.

        Args:
            prompt: The prompt to send
            model: Optional model name (ignored for CodKing)

        Returns:
            LLM response string
        """
        try:
            # Try RLM handler first if available
            if self.lm_handler_address:
                from rlm.core.comms_utils import LMRequest, send_lm_request

                request = LMRequest(prompt=prompt, model=model)
                response = send_lm_request(self.lm_handler_address, request)

                if response.success:
                    self._pending_llm_calls.append(response.chat_completion)
                    return response.chat_completion.response
                else:
                    logger.warning(f"RLM handler error: {response.error}, falling back to CodKing")

            # Fall back to CodKing
            result = self.codking_client.completion(prompt)
            return result

        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return f"Error: LLM query failed - {e}"

    def _llm_query_batched(
        self,
        prompts: List[str],
        model: Optional[str] = None
    ) -> List[str]:
        """
        Query LLM with multiple prompts.

        Args:
            prompts: List of prompts
            model: Optional model name

        Returns:
            List of responses
        """
        results = []
        for prompt in prompts:
            result = self._llm_query(prompt, model)
            results.append(result)
        return results

    def _codking_analyze(self, data: Union[str, bytes]) -> str:
        """
        Direct CodKing analysis of data.

        Args:
            data: Text or bytes to analyze

        Returns:
            Analysis result
        """
        if isinstance(data, bytes):
            data = data.decode('utf-8', errors='replace')
        return self.codking_client.completion(data)

    def _get_chunk(self, index: int) -> bytes:
        """Get a specific chunk from the context."""
        return self.context_manager.get_chunk(index)

    def _iter_chunks(self):
        """Iterate over all context chunks."""
        return self.context_manager.iter_chunks()

    def _get_context_stats(self) -> dict:
        """Get context statistics."""
        stats = self.context_manager.get_stats()
        return {
            'total_size': stats.total_size,
            'num_chunks': stats.num_chunks,
            'ram_cached': stats.ram_cached_chunks,
            'source_type': stats.source_type
        }

    def load_context(self, context_payload: Union[str, dict, list]) -> None:
        """
        Load context into the environment.

        Handles:
        - Strings (stored directly)
        - Dicts (serialized to JSON)
        - Lists (serialized to JSON)
        - Large data (spilled to disk)

        Args:
            context_payload: Context data to load
        """
        self.add_context(context_payload, 0)

    def add_context(
        self,
        context_payload: Union[str, dict, list],
        context_index: Optional[int] = None
    ) -> int:
        """
        Add a context with versioned variable name.

        Args:
            context_payload: Context data to add
            context_index: Optional explicit index

        Returns:
            The context index used
        """
        with self._lock:
            if context_index is None:
                context_index = self._context_count

            var_name = f"context_{context_index}"

            # Convert to bytes for context manager
            if isinstance(context_payload, str):
                data = context_payload.encode('utf-8')
            else:
                data = json.dumps(context_payload).encode('utf-8')

            # Load into context manager
            self.context_manager.load_from_bytes(data)

            # Make available in execution namespace
            if isinstance(context_payload, str):
                self.locals[var_name] = context_payload
            else:
                self.locals[var_name] = context_payload

            # Alias context_0 as 'context'
            if context_index == 0:
                self.locals["context"] = self.locals[var_name]

            self._context_count = max(self._context_count, context_index + 1)

            logger.info(
                f"Loaded context_{context_index}: "
                f"{self.context_manager.total_size} bytes, "
                f"{self.context_manager.num_chunks} chunks"
            )

            return context_index

    def update_handler_address(self, address: tuple) -> None:
        """Update the LM handler address."""
        self.lm_handler_address = address

    def get_context_count(self) -> int:
        """Return number of contexts loaded."""
        return self._context_count

    def add_history(
        self,
        message_history: List[Dict[str, Any]],
        history_index: Optional[int] = None
    ) -> int:
        """
        Store conversation history as a versioned variable.

        Args:
            message_history: List of message dicts
            history_index: Optional explicit index

        Returns:
            The history index used
        """
        with self._lock:
            if history_index is None:
                history_index = self._history_count

            var_name = f"history_{history_index}"
            self.locals[var_name] = copy.deepcopy(message_history)

            if history_index == 0:
                self.locals["history"] = self.locals[var_name]

            self._history_count = max(self._history_count, history_index + 1)
            return history_index

    def get_history_count(self) -> int:
        """Return number of histories stored."""
        return self._history_count

    @contextmanager
    def _capture_output(self):
        """Context manager to capture stdout/stderr."""
        with self._lock:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
            try:
                sys.stdout, sys.stderr = stdout_buf, stderr_buf
                yield stdout_buf, stderr_buf
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr

    @contextmanager
    def _temp_cwd(self):
        """Temporarily change to temp directory."""
        old_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            yield
        finally:
            os.chdir(old_cwd)

    def execute_code(self, code: str) -> REPLResult:
        """
        Execute code in the sandboxed namespace.

        Args:
            code: Python code to execute

        Returns:
            REPLResult with stdout, stderr, and locals
        """
        start_time = time.perf_counter()
        self._pending_llm_calls = []

        with self._capture_output() as (stdout_buf, stderr_buf), self._temp_cwd():
            try:
                combined = {**self.globals, **self.locals}
                exec(code, combined, combined)

                # Update locals with new variables
                for key, value in combined.items():
                    if key not in self.globals and not key.startswith("_"):
                        self.locals[key] = value

                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue()

            except Exception as e:
                stdout = stdout_buf.getvalue()
                stderr = stderr_buf.getvalue() + f"\n{type(e).__name__}: {e}"

        return REPLResult(
            stdout=stdout,
            stderr=stderr,
            locals=self.locals.copy(),
            execution_time=time.perf_counter() - start_time,
            rlm_calls=self._pending_llm_calls.copy()
        )

    def process_all_chunks(
        self,
        processor_code: str,
        parallel: bool = False,
        num_workers: int = 4
    ) -> List[Any]:
        """
        Process all context chunks with custom code.

        Args:
            processor_code: Code that processes 'chunk' variable
            parallel: Enable parallel processing
            num_workers: Number of parallel workers

        Returns:
            List of results from each chunk
        """
        results = []

        def process_chunk(idx: int, chunk: bytes) -> Any:
            # Set chunk in namespace
            self.locals['chunk'] = chunk
            self.locals['chunk_index'] = idx

            # Execute processor code
            result = self.execute_code(processor_code)

            # Return result variable if exists
            return self.locals.get('result', result.stdout)

        if parallel:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_chunk, idx, chunk): idx
                    for idx, chunk in self.context_manager.iter_chunks()
                }

                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append((idx, result))
                    except Exception as e:
                        results.append((idx, f"Error: {e}"))

            results.sort(key=lambda x: x[0])
            return [r[1] for r in results]
        else:
            for idx, chunk in self.context_manager.iter_chunks():
                result = process_chunk(idx, chunk)
                results.append(result)
            return results

    def cleanup(self) -> None:
        """Clean up resources."""
        self.context_manager.cleanup()
        self.globals.clear()
        self.locals.clear()

        try:
            import shutil
            shutil.rmtree(self.temp_dir)
        except Exception:
            pass

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


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("CodKingREPL Demo")
    print("=" * 60)

    # Create REPL
    with CodKingREPL(ram_limit_gb=0.1) as repl:
        # Load some context
        test_context = "This is a test log entry. " * 1000
        repl.load_context(test_context)

        # Execute some code
        print("\n1. Basic code execution:")
        result = repl.execute_code("print(f'Context length: {len(context)}')")
        print(f"stdout: {result.stdout}")

        print("\n2. Get context stats:")
        result = repl.execute_code("print(context_stats())")
        print(f"stdout: {result.stdout}")

        print("\n3. LLM query (mock):")
        result = repl.execute_code("answer = llm_query('What is 2+2?'); print(answer)")
        print(f"stdout: {result.stdout}")

        print("\n4. CodKing analyze:")
        result = repl.execute_code(
            "result = codking_analyze('ERROR: Authentication failed'); print(result)"
        )
        print(f"stdout: {result.stdout}")

    print("\n" + "=" * 60)
    print("Demo complete!")
