# ./ollamatoolkit/utils.py
"""
Ollama Toolkit - Utilities
==========================
Generic validation, file, time helpers, and retry mechanisms.

Key Utilities:
    - retry: Decorator for retrying functions on transient failures
    - retry_async: Async version of retry decorator
    - generate_safe_filename: Create safe filenames from strings
    - load_and_parse_json: Safely load JSON files
    - clean_json_response: Extract JSON from markdown/raw text
"""

import asyncio
import functools
import json
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Retry Decorators
# =============================================================================


def retry(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions on transient failures.

    Args:
        attempts: Maximum retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt) called before each retry

    Returns:
        Decorated function that retries on specified exceptions

    Example:
        @retry(attempts=3, delay=1.0, exceptions=(ConnectionError, TimeoutError))
        def fetch_data(url):
            return requests.get(url)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {attempts} attempts: {e}"
                        )

            raise last_exception or RuntimeError(
                f"{func.__name__} failed without raising a captured exception."
            )

        return wrapper

    return decorator


def retry_async(
    attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """
    Async version of retry decorator.

    Args:
        attempts: Maximum retry attempts (default: 3)
        delay: Initial delay between retries in seconds (default: 1.0)
        backoff: Multiplier for delay after each retry (default: 2.0)
        exceptions: Tuple of exception types to catch and retry
        on_retry: Optional callback(exception, attempt) called before each retry

    Returns:
        Decorated async function that retries on specified exceptions

    Example:
        @retry_async(attempts=3, exceptions=(aiohttp.ClientError,))
        async def fetch_data(url):
            async with aiohttp.get(url) as resp:
                return await resp.json()
    """

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            current_delay = delay

            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        if on_retry:
                            on_retry(e, attempt)
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {attempts} attempts: {e}"
                        )

            raise last_exception or RuntimeError(
                f"{func.__name__} failed without raising a captured exception."
            )

        return wrapper

    return decorator


# =============================================================================
# Time Utilities
# =============================================================================


def now_utc() -> datetime:
    """Returns the current UTC datetime."""
    return datetime.now(timezone.utc)


def generate_safe_filename(
    name: str, suffix: str = "", timestamp: bool = True, unique_id: bool = False
) -> str:
    """Generates a safe filename from a string."""
    if not name:
        name = "Unknown_Entity"

    safe_name = "".join(
        c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name
    ).strip()
    safe_name = safe_name.replace(" ", "_")
    safe_name = re.sub(r"[_]+", "_", safe_name)
    safe_name = re.sub(r"[-]+", "-", safe_name)
    safe_name = safe_name[:60]

    parts = [safe_name]
    if timestamp:
        parts.append(datetime.now().strftime("%Y%m%d_%H%M%S"))
    if unique_id:
        parts.append(str(uuid.uuid4())[:8])

    base_name = "_".join(parts)
    return f"{base_name}{suffix}"


def load_and_parse_json(file_path: Path) -> Optional[Dict[str, Any]]:
    """Safely loads and parses a JSON file."""
    if not file_path.exists():
        logger.error(f"JSON file not found at {file_path}")
        return None
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Failed to load or parse JSON file at {file_path}: {e}")
        return None


def save_json_output(data: Dict[str, Any], filename: str, output_dir: Path):
    """Saves a dictionary to a pretty-printed JSON file."""
    output_path: Path = output_dir / filename
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Successfully saved output to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {output_path}: {e}")


def update_json_field(
    file_path: Path, field_path: str, new_value: Any, source_url: Optional[str] = None
) -> bool:
    """
    Updates a specific field in a JSON file using dot notation.
    """
    data = load_and_parse_json(file_path)
    if data is None:
        return False

    keys = field_path.split(".")
    current_level = data

    # Traverse to the parent
    try:
        for key in keys[:-1]:
            current_level = current_level.setdefault(key, {})

        last_key = keys[-1]

        # Check if target is a value/source object
        if (
            isinstance(current_level.get(last_key), dict)
            and "value" in current_level[last_key]
        ):
            current_level[last_key]["value"] = new_value
            if source_url:
                current_level[last_key]["source_url"] = source_url
        else:
            current_level[last_key] = new_value

        save_json_output(data, file_path.name, file_path.parent)
        logger.info(f"Updated field '{field_path}' in {file_path.name}")
        return True
    except Exception as e:
        logger.error(f"Error updating JSON field: {e}")
        return False


def check_dependencies():
    """Calculates installed dependencies at startup to prevent runtime errors."""
    missing = []

    # 1. Telemetry
    try:
        import llm_telemetry_toolkit  # noqa: F401 - availability check
    except ImportError:
        missing.append("llm-telemetry-toolkit")

    # 2. Web Scraper
    try:
        import web_scraper_toolkit  # noqa: F401 - availability check
    except ImportError:
        missing.append("web-scraper-toolkit")

    if missing:
        msg = "\n".join([f"  pip install {pkg}" for pkg in missing])
        raise ImportError(
            f"Missing required dependencies:\n{msg}\n"
            "Please install them to use OllamaToolkit."
        )


def clean_json_response(text: str) -> str:
    """Attempts to extract JSON from markdown or raw text."""
    text = text.strip()
    if "```json" in text:
        parts = text.split("```json")
        if len(parts) > 1:
            return parts[1].split("```")[0].strip()
    elif "```" in text:
        parts = text.split("```")
        if len(parts) > 1:
            return parts[1].strip()
    return text
