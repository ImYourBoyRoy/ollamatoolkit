# ./ollamatoolkit/tools/models.py
"""
Ollama Toolkit - Model Inspector
================================
Comprehensive model inspection, listing, and export tool.

Features:
- List all installed models with detailed metadata
- Filter by capability (vision, embedding, completion, tools)
- Export to JSON (bulk or individual files)
- Sort by name, size, modified date, family
- Include/exclude specific metadata fields

Usage:
    inspector = ModelInspector(base_url="http://localhost:11434")

    # Get all models as dict
    models = inspector.get_all_models()

    # Get specific model details
    details = inspector.get_model("ministral-3:latest")

    # Filter by capability
    vision_models = inspector.filter_by_capability("vision")

    # Export to files
    inspector.export_to_path("./test_output", individual=True)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set

import ollama

logger = logging.getLogger(__name__)


@dataclass
class ExportOptions:
    """Configuration for model export."""

    output_path: Path = field(default_factory=lambda: Path("./model_output"))
    individual_files: bool = False
    bulk_filename: str = "all_models.json"
    include_fields: Optional[Set[str]] = None  # None = all fields
    exclude_fields: Optional[Set[str]] = None
    sort_by: Literal["name", "size", "modified_at", "family"] = "name"
    sort_descending: bool = False
    pretty_print: bool = True


class ModelInspector:
    """
    Comprehensive model inspection and export tool.
    Uses ollama-python for direct API access.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize ModelInspector.

        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.client = ollama.Client(host=base_url)
        self._models_cache: Optional[Dict[str, Dict]] = None
        logger.info(f"ModelInspector initialized for {base_url}")

    # =========================================================================
    # Core Listing Methods
    # =========================================================================

    def list_models(self) -> List[str]:
        """Get list of all installed model names."""
        try:
            response = self.client.list()
            return [m.model for m in response.models if m.model]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def get_model(self, model_name: str) -> Dict[str, Any]:
        """
        Get detailed information for a specific model.

        Args:
            model_name: Model name (e.g., "ministral-3:latest")

        Returns:
            Dict with all model metadata
        """
        try:
            # Use show() for detailed info
            response = self.client.show(model_name)

            # Convert to dict, handling various response types
            if hasattr(response, "model_dump"):
                data = response.model_dump()
            elif hasattr(response, "__dict__"):
                data = dict(response.__dict__)
            else:
                data = dict(response)

            # Add model name for clarity
            data["model"] = model_name

            # Parse capabilities from template/modelinfo
            data["capabilities"] = self._detect_capabilities(data)

            return data

        except Exception as e:
            logger.error(f"Failed to get model {model_name}: {e}")
            return {"model": model_name, "error": str(e)}

    def get_all_models(self, refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information for all installed models.

        Args:
            refresh: Force refresh cache

        Returns:
            Dict mapping model name to metadata
        """
        if self._models_cache and not refresh:
            return self._models_cache

        models = {}
        model_names = self.list_models()

        logger.info(f"Inspecting {len(model_names)} models...")

        for name in model_names:
            try:
                models[name] = self.get_model(name)
            except Exception as e:
                logger.warning(f"Failed to inspect {name}: {e}")
                models[name] = {"model": name, "error": str(e)}

        self._models_cache = models
        return models

    # =========================================================================
    # Capability Detection
    # =========================================================================

    def _detect_capabilities(self, model_data: Dict) -> List[str]:
        """
        Detect model capabilities from metadata.

        Returns list of capabilities: vision, embedding, completion, tools, reasoning
        """
        caps = set()

        # Always has completion
        caps.add("completion")

        # Check details
        details = model_data.get("details", {}) or {}
        family = details.get("family", "").lower() if details else ""
        families = details.get("families", []) or []

        # Check modelinfo
        modelinfo = model_data.get("modelinfo", {}) or {}

        # Check template
        template = model_data.get("template", "") or ""

        # Vision detection - expanded for VL (vision-language) and OCR models
        vision_indicators = [
            "vision" in family,
            "llava" in family,
            "vl" in family,  # qwen3vl, llava, etc.
            "ocr" in family,  # deepseek-ocr
            "mllama" in family,  # Meta Llama with vision
            "clip" in str(families).lower(),
            "vision" in str(families).lower(),
            any("vision" in f.lower() for f in families if f),
            any("vl" in f.lower() for f in families if f),
            "image" in template.lower(),
            modelinfo.get("general.architecture", "").lower()
            in ["llava", "mllama", "qwen2vl", "qwen3vl"],
        ]
        if any(vision_indicators):
            caps.add("vision")

        # Embedding detection
        embed_indicators = [
            "embed" in family,
            "nomic" in family,
            "embedding" in str(families).lower(),
            any("embed" in f.lower() for f in families if f),
            modelinfo.get("general.architecture", "").lower() in ["nomic-bert", "bert"],
        ]
        if any(embed_indicators):
            caps.add("embedding")

        # Tools/Function calling detection
        tools_indicators = [
            ".tools" in template.lower(),
            "tool_calls" in template.lower(),
            "{%- for tool in tools" in template,
        ]
        if any(tools_indicators):
            caps.add("tools")

        # Reasoning detection (thinking models)
        reasoning_indicators = [
            "thinking" in template.lower(),
            "<think>" in template.lower(),
            "reasoning" in family,
            "deepseek" in family and "r1" in family,
        ]
        if any(reasoning_indicators):
            caps.add("reasoning")

        return sorted(list(caps))

    # =========================================================================
    # Filtering Methods
    # =========================================================================

    def filter_by_capability(
        self,
        capability: Literal["vision", "embedding", "completion", "tools", "reasoning"],
        models: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Filter models by capability.

        Args:
            capability: Capability to filter by
            models: Optional dict of models (uses cache if not provided)

        Returns:
            Dict of models with the specified capability
        """
        if models is None:
            models = self.get_all_models()

        return {
            name: data
            for name, data in models.items()
            if capability in data.get("capabilities", [])
        }

    def filter_by_family(
        self, family: str, models: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Filter models by family (e.g., 'llama', 'mistral')."""
        if models is None:
            models = self.get_all_models()

        family_lower = family.lower()
        return {
            name: data
            for name, data in models.items()
            if family_lower in (data.get("details", {}) or {}).get("family", "").lower()
        }

    def search(
        self, query: str, models: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Search models by name or family."""
        if models is None:
            models = self.get_all_models()

        query_lower = query.lower()
        return {
            name: data
            for name, data in models.items()
            if query_lower in name.lower()
            or query_lower in (data.get("details", {}) or {}).get("family", "").lower()
        }

    # =========================================================================
    # Sorting Methods
    # =========================================================================

    def sort_models(
        self,
        models: Dict[str, Dict],
        sort_by: Literal["name", "size", "modified_at", "family"] = "name",
        descending: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sort models by specified field.

        Args:
            models: Dict of models to sort
            sort_by: Field to sort by
            descending: Sort in descending order

        Returns:
            Sorted dict (insertion-order preserved)
        """

        def get_sort_key(item):
            name, data = item
            if sort_by == "name":
                return name.lower()
            elif sort_by == "size":
                details = data.get("details", {}) or {}
                size_str = details.get("parameter_size", "0")
                # Parse "8B", "70B", etc.
                try:
                    if size_str and isinstance(size_str, str):
                        num = float(
                            size_str.replace("B", "").replace("M", "").replace("K", "")
                        )
                        if "B" in size_str:
                            num *= 1e9
                        elif "M" in size_str:
                            num *= 1e6
                        elif "K" in size_str:
                            num *= 1e3
                        return num
                except (ValueError, AttributeError):
                    pass
                return 0
            elif sort_by == "modified_at":
                return data.get("modified_at", "") or ""
            elif sort_by == "family":
                details = data.get("details", {}) or {}
                return (details.get("family", "") or "").lower()
            return name

        sorted_items = sorted(models.items(), key=get_sort_key, reverse=descending)
        return dict(sorted_items)

    # =========================================================================
    # Export Methods
    # =========================================================================

    def _filter_fields(
        self,
        data: Dict[str, Any],
        include: Optional[Set[str]] = None,
        exclude: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Filter dict fields based on include/exclude sets."""
        if include:
            return {k: v for k, v in data.items() if k in include}
        if exclude:
            return {k: v for k, v in data.items() if k not in exclude}
        return data

    def export_to_path(
        self,
        output_path: str,
        models: Optional[Dict[str, Dict]] = None,
        individual_files: bool = False,
        bulk_filename: str = "all_models.json",
        include_fields: Optional[Set[str]] = None,
        exclude_fields: Optional[Set[str]] = None,
        sort_by: Literal["name", "size", "modified_at", "family"] = "name",
        sort_descending: bool = False,
        pretty_print: bool = True,
    ) -> Dict[str, str]:
        """
        Export models to JSON file(s).

        Args:
            output_path: Directory to export to
            models: Optional models dict (uses cache if not provided)
            individual_files: Create separate file for each model
            bulk_filename: Filename for bulk export
            include_fields: Only include these fields
            exclude_fields: Exclude these fields
            sort_by: Sort field
            sort_descending: Sort direction
            pretty_print: Format JSON with indentation

        Returns:
            Dict mapping filenames to paths
        """
        if models is None:
            models = self.get_all_models()

        # Sort
        models = self.sort_models(models, sort_by, sort_descending)

        # Create output directory
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        indent = 2 if pretty_print else None
        exported = {}

        if individual_files:
            # Export each model to its own file
            for name, data in models.items():
                # Sanitize filename
                safe_name = name.replace(":", "_").replace("/", "_")
                filename = f"{safe_name}.json"
                filepath = out_dir / filename

                filtered_data = self._filter_fields(
                    data, include_fields, exclude_fields
                )

                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(filtered_data, f, indent=indent, default=str)

                exported[name] = str(filepath)
                logger.debug(f"Exported {name} to {filepath}")

        # Always create bulk file
        bulk_path = out_dir / bulk_filename
        bulk_data = {
            name: self._filter_fields(data, include_fields, exclude_fields)
            for name, data in models.items()
        }

        with open(bulk_path, "w", encoding="utf-8") as f:
            json.dump(bulk_data, f, indent=indent, default=str)

        exported["_bulk"] = str(bulk_path)
        logger.info(f"Exported {len(models)} models to {out_dir}")

        return exported

    def export_options(self, options: ExportOptions) -> Dict[str, str]:
        """Export using ExportOptions dataclass."""
        return self.export_to_path(
            output_path=str(options.output_path),
            individual_files=options.individual_files,
            bulk_filename=options.bulk_filename,
            include_fields=options.include_fields,
            exclude_fields=options.exclude_fields,
            sort_by=options.sort_by,
            sort_descending=options.sort_descending,
            pretty_print=options.pretty_print,
        )

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    def get_vision_models(self) -> Dict[str, Dict]:
        """Get all models with vision capability."""
        return self.filter_by_capability("vision")

    def get_embedding_models(self) -> Dict[str, Dict]:
        """Get all models with embedding capability."""
        return self.filter_by_capability("embedding")

    def get_tool_models(self) -> Dict[str, Dict]:
        """Get all models with tool/function calling capability."""
        return self.filter_by_capability("tools")

    def get_reasoning_models(self) -> Dict[str, Dict]:
        """Get all models with reasoning/thinking capability."""
        return self.filter_by_capability("reasoning")

    def summary(self) -> Dict[str, Any]:
        """
        Get summary of all installed models.

        Returns:
            Dict with counts and lists by capability
        """
        models = self.get_all_models()

        summary = {
            "total_models": len(models),
            "model_names": sorted(models.keys()),
            "by_capability": {
                "vision": list(self.filter_by_capability("vision", models).keys()),
                "embedding": list(
                    self.filter_by_capability("embedding", models).keys()
                ),
                "tools": list(self.filter_by_capability("tools", models).keys()),
                "reasoning": list(
                    self.filter_by_capability("reasoning", models).keys()
                ),
                "completion": list(
                    self.filter_by_capability("completion", models).keys()
                ),
            },
            "by_family": {},
        }

        # Group by family
        families: Dict[str, List[str]] = {}
        for name, data in models.items():
            family = (data.get("details", {}) or {}).get("family", "unknown")
            if family not in families:
                families[family] = []
            families[family].append(name)

        summary["by_family"] = families

        return summary

    def __repr__(self) -> str:
        return f"ModelInspector(base_url='{self.base_url}')"
