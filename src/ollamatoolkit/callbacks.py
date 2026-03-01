# ./ollamatoolkit/callbacks.py
"""
OllamaToolkit Callbacks
=======================
Progress callback protocols for long-running operations.

Usage:
    from ollamatoolkit.callbacks import ProgressCallback

    def my_callback(progress: float, message: str, data: dict = None):
        print(f"[{progress:.0%}] {message}")

    result = await processor.process_pdf("doc.pdf", on_progress=my_callback)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


class ProgressCallback(Protocol):
    """Protocol for progress callbacks."""

    def __call__(
        self,
        progress: float,
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Called during long-running operations.

        Args:
            progress: Progress from 0.0 to 1.0
            message: Human-readable status message
            data: Optional additional data (page_num, chunk_index, etc.)
        """
        ...


@dataclass
class ProgressEvent:
    """Structured progress event."""

    progress: float  # 0.0 to 1.0
    message: str
    current: int = 0  # Current item index
    total: int = 0  # Total items
    data: Optional[Dict[str, Any]] = None

    @property
    def percent(self) -> int:
        """Progress as percentage (0-100)."""
        return int(self.progress * 100)


def create_console_callback(prefix: str = "") -> ProgressCallback:
    """Create a simple console progress callback."""

    def callback(progress: float, message: str, data: Optional[Dict[str, Any]] = None):
        pct = int(progress * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r{prefix}[{bar}] {pct:3d}% {message}", end="", flush=True)
        if progress >= 1.0:
            print()  # New line when complete

    return callback


def create_rich_callback():
    """Create a Rich-based progress callback (requires rich)."""
    try:
        from rich.progress import Progress, SpinnerColumn, TextColumn
    except ImportError:
        return create_console_callback()

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
    )

    task_id = None

    def callback(prog: float, message: str, data: Optional[Dict[str, Any]] = None):
        nonlocal task_id
        if task_id is None:
            task_id = progress.add_task(message)
            progress.start()

        progress.update(task_id, completed=prog * 100, description=message)

        if prog >= 1.0:
            progress.stop()

    return callback


# Common no-op callback
def null_callback(progress: float, message: str, data: Optional[Dict[str, Any]] = None):
    """No-op callback for when progress reporting is disabled."""
    pass
