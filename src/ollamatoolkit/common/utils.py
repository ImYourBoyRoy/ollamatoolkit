# ./ollamatoolkit/common/utils.py
"""
Ollama Toolkit - Utilities
==========================
Generic validation, file, and time helpers.
"""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


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
