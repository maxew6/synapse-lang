"""
runtime.py — Synapse Runtime Executor

Takes a Python source string (produced by the Transpiler) and executes it
safely inside an isolated execution context.

Two execution strategies are provided:

  1. TempFileExecutor (default)
     Writes the source to a temporary .py file and runs it with subprocess.
     This gives full process isolation and lets PyTorch/CUDA initialise
     normally (some torch operations require being in __main__).

  2. InProcessExecutor
     Executes the source string via exec() in a fresh namespace.
     Faster, but shares the Python interpreter and signal handlers.
     Useful for introspection / testing.

The Runtime class selects TempFileExecutor by default but the strategy
can be swapped out at construction time for extensibility.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int

    def print_output(self) -> None:
        if self.stdout:
            print(self.stdout, end="")
        if self.stderr:
            print(self.stderr, end="", file=sys.stderr)


# ---------------------------------------------------------------------------
# Abstract executor interface
# ---------------------------------------------------------------------------

class BaseExecutor(ABC):
    @abstractmethod
    def execute(self, source: str) -> ExecutionResult:
        """Execute a Python source string and return an ExecutionResult."""
        ...


# ---------------------------------------------------------------------------
# Temp-file subprocess executor (default)
# ---------------------------------------------------------------------------

class TempFileExecutor(BaseExecutor):
    """
    Writes `source` to a temporary .py file and runs it with the current
    Python interpreter via subprocess.  This is the safest strategy because:
      - Full process isolation (segfaults don't kill the CLI)
      - torch CUDA initialisation works correctly
      - stdout/stderr are properly captured
    """

    def __init__(self, timeout: Optional[float] = 300.0) -> None:
        self.timeout = timeout

    def execute(self, source: str) -> ExecutionResult:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            prefix="synapse_run_",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(source)
            tmp_path = f.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                returncode=result.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr=f"Execution timed out after {self.timeout}s",
                returncode=-1,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# In-process executor (for testing / introspection)
# ---------------------------------------------------------------------------

class InProcessExecutor(BaseExecutor):
    """
    Executes the source string via exec() in a fresh namespace dict.
    Shares the interpreter — useful for unit tests, not recommended for
    production use with untrusted code.
    """

    def execute(self, source: str) -> ExecutionResult:
        import io
        from contextlib import redirect_stdout, redirect_stderr

        namespace: dict = {"__name__": "__main__"}
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compile(source, "<synapse>", "exec"), namespace)  # noqa: S102
            return ExecutionResult(
                success=True,
                stdout=stdout_buf.getvalue(),
                stderr=stderr_buf.getvalue(),
                returncode=0,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                stdout=stdout_buf.getvalue(),
                stderr=f"{type(exc).__name__}: {exc}",
                returncode=1,
            )


# ---------------------------------------------------------------------------
# High-level Runtime facade
# ---------------------------------------------------------------------------

class Runtime:
    """
    Facade over the execution strategy.

    Usage:
        rt     = Runtime()                     # TempFileExecutor by default
        result = rt.run(python_source_string)
        result.print_output()
    """

    def __init__(self, executor: Optional[BaseExecutor] = None) -> None:
        self._executor = executor or TempFileExecutor()

    def run(self, source: str) -> ExecutionResult:
        """Execute the generated Python source and return an ExecutionResult."""
        return self._executor.execute(source)

    def run_file(self, path: str | Path) -> ExecutionResult:
        """
        Load an already-generated .py file from disk and execute it.
        (Useful for debugging — inspect the generated file first, then run.)
        """
        source = Path(path).read_text(encoding="utf-8")
        return self._executor.execute(source)

    def save_and_run(
        self, source: str, output_path: str | Path
    ) -> ExecutionResult:
        """
        Write the generated source to `output_path` for inspection, then run it.
        The file is NOT deleted after execution so the user can examine it.
        """
        output_path = Path(output_path)
        output_path.write_text(source, encoding="utf-8")
        return self._executor.execute(source)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def execute(source: str, strategy: str = "subprocess") -> ExecutionResult:
    """
    Execute a Python source string.

    Args:
        source:   The Python code to execute.
        strategy: 'subprocess' (default, isolated) or 'inprocess' (fast).
    """
    executor: BaseExecutor
    if strategy == "inprocess":
        executor = InProcessExecutor()
    else:
        executor = TempFileExecutor()
    return Runtime(executor).run(source)
