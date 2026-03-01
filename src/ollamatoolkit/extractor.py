# ./src/ollamatoolkit/extractor.py
"""Schema-first field extraction helper built on top of ``SimpleAgent``.

Run via code import: ``from ollamatoolkit.extractor import SimpleExtractor``.
Inputs: unstructured content text, target field names, optional field definitions,
and a configured ``SimpleAgent`` instance.
Outputs: dictionary keyed by requested field, each with value/confidence/excerpt.
Side effects: performs LLM requests through the provided agent; no filesystem writes.
Operational note: extraction truncates very large inputs for bounded prompt size.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Type, cast

from pydantic import BaseModel, Field, create_model

from .agents.simple import SimpleAgent

logger = logging.getLogger(__name__)


class ContentSource(BaseModel):
    """Container for source text and source label metadata."""

    content: str
    source_label: str = "Input Text"


class ExtractionItem(BaseModel):
    """One extracted field value with evidence and confidence."""

    value: Any = Field(None, description="Extracted value or null when unavailable")
    confidence: float = Field(0.0, description="Confidence score between 0.0 and 1.0")
    excerpt: str = Field("", description="Exact supporting quote from the source")


class SimpleExtractor:
    """Reusable extractor that asks a ``SimpleAgent`` for structured field outputs."""

    def __init__(self, agent: SimpleAgent) -> None:
        self.agent = agent

    @staticmethod
    def _build_dynamic_model(fields: List[str]) -> Type[BaseModel]:
        field_definitions: Dict[str, Any] = {
            field_name: (
                ExtractionItem,
                Field(..., description=f"Extraction result for '{field_name}'"),
            )
            for field_name in fields
        }
        dynamic_model = create_model("DynamicExtraction", **field_definitions)
        return cast(Type[BaseModel], dynamic_model)

    @staticmethod
    def _build_prompt(
        content: str,
        fields: List[str],
        definitions: str,
        source_label: str,
    ) -> str:
        return (
            "EXTRACT DATA TASK\n"
            "-----------------\n"
            "Goal: Extract the target fields from the source text.\n"
            f"Target Fields: {fields}\n\n"
            "Definitions:\n"
            f"{definitions}\n\n"
            f"Source ({source_label}):\n"
            "-----------------------\n"
            f"{content[:30000]}\n"
            "...(content truncated if too long)\n"
            "-----------------------\n\n"
            "Return JSON only."
        )

    def extract_fields(
        self,
        content: str,
        fields: List[str],
        definitions: str = "",
        source_label: str = "Text",
    ) -> Dict[str, Any]:
        """Synchronously extract requested fields from content."""
        dynamic_model = self._build_dynamic_model(fields)
        prompt = self._build_prompt(content, fields, definitions, source_label)

        try:
            result = self.agent.run_structured(prompt, dynamic_model)
            return result.model_dump()
        except Exception as exc:
            logger.error("Extraction failed: %s", exc)
            return {}

    async def extract_fields_async(
        self,
        content: str,
        fields: List[str],
        definitions: str = "",
        source_label: str = "Text",
    ) -> Dict[str, Any]:
        """Asynchronously extract requested fields from content."""
        dynamic_model = self._build_dynamic_model(fields)
        prompt = self._build_prompt(content, fields, definitions, source_label)

        try:
            result = await self.agent.run_structured_async(prompt, dynamic_model)
            return result.model_dump()
        except Exception as exc:
            logger.error("Async extraction failed: %s", exc)
            return {}
