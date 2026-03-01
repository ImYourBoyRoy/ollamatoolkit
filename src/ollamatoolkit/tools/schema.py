# ./src/ollamatoolkit/tools/schema.py
"""
OllamaToolkit - JSON Schema Tools
=================================
Tools for JSON schema validation, generation, and data structure enforcement.

Enables Ollama models to:
- Validate JSON data against schemas
- Generate sample data from schemas
- Convert Pydantic models to schemas
- Enforce output structure

Usage:
    from ollamatoolkit.tools.schema import SchemaTools

    tools = SchemaTools()

    # Validate data
    result = tools.validate(data, schema)

    # Generate sample
    sample = tools.generate_sample(schema)
"""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of schema validation."""

    valid: bool
    errors: List[str]
    data: Optional[Any] = None  # Cleaned/coerced data if valid


class SchemaTools:
    """
    JSON Schema validation and generation tools.

    Provides utilities for structured data handling without external dependencies.
    """

    # JSON Schema type mapping
    TYPE_MAP: Dict[str, Union[type, Tuple[type, ...]]] = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    # Default values by type
    DEFAULTS = {
        "string": "",
        "number": 0.0,
        "integer": 0,
        "boolean": False,
        "array": [],
        "object": {},
        "null": None,
    }

    # Sample values for generation
    SAMPLES = {
        "string": "example",
        "number": 3.14,
        "integer": 42,
        "boolean": True,
        "array": [],
        "object": {},
        "null": None,
    }

    def validate(
        self, data: Any, schema: Dict[str, Any], strict: bool = False
    ) -> ValidationResult:
        """
        Validate data against a JSON schema.

        Args:
            data: Data to validate
            schema: JSON Schema object
            strict: If True, fail on extra properties

        Returns:
            ValidationResult with valid status and errors
        """
        errors: List[str] = []
        self._validate_value(data, schema, errors, path="$", strict=strict)

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            data=data if len(errors) == 0 else None,
        )

    def _validate_value(
        self,
        value: Any,
        schema: Dict[str, Any],
        errors: List[str],
        path: str,
        strict: bool,
    ):
        """Recursively validate a value against schema."""
        # Handle null
        if value is None:
            if schema.get("type") != "null" and not schema.get("nullable", False):
                errors.append(f"{path}: Expected non-null value")
            return

        # Type validation
        expected_type = schema.get("type")
        if expected_type:
            python_types = self.TYPE_MAP.get(expected_type)
            if python_types and not isinstance(value, python_types):
                errors.append(
                    f"{path}: Expected {expected_type}, got {type(value).__name__}"
                )
                return

        # String constraints
        if expected_type == "string" and isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(f"{path}: String too short (min: {schema['minLength']})")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(f"{path}: String too long (max: {schema['maxLength']})")
            if "pattern" in schema:
                import re

                if not re.match(schema["pattern"], value):
                    errors.append(f"{path}: String doesn't match pattern")
            if "enum" in schema and value not in schema["enum"]:
                errors.append(f"{path}: Value not in enum: {schema['enum']}")

        # Number constraints
        if expected_type in ("number", "integer") and isinstance(value, (int, float)):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}: Value too small (min: {schema['minimum']})")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}: Value too large (max: {schema['maximum']})")

        # Array validation
        if expected_type == "array" and isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                errors.append(f"{path}: Array too short (min: {schema['minItems']})")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                errors.append(f"{path}: Array too long (max: {schema['maxItems']})")
            if "items" in schema:
                for i, item in enumerate(value):
                    self._validate_value(
                        item, schema["items"], errors, f"{path}[{i}]", strict
                    )

        # Object validation
        if expected_type == "object" and isinstance(value, dict):
            properties = schema.get("properties", {})
            required = schema.get("required", [])

            # Check required properties
            for prop in required:
                if prop not in value:
                    errors.append(f"{path}: Missing required property '{prop}'")

            # Validate each property
            for prop, prop_schema in properties.items():
                if prop in value:
                    self._validate_value(
                        value[prop], prop_schema, errors, f"{path}.{prop}", strict
                    )

            # Check for extra properties
            if strict:
                extra = set(value.keys()) - set(properties.keys())
                for prop in extra:
                    errors.append(f"{path}: Unexpected property '{prop}'")

    def generate_sample(self, schema: Dict[str, Any], use_examples: bool = True) -> Any:
        """
        Generate sample data from a JSON schema.

        Args:
            schema: JSON Schema object
            use_examples: If True, use 'example' fields from schema

        Returns:
            Sample data matching the schema
        """
        return self._generate_value(schema, use_examples)

    def _generate_value(self, schema: Dict[str, Any], use_examples: bool) -> Any:
        """Recursively generate a value from schema."""
        # Use example if available
        if use_examples and "example" in schema:
            return schema["example"]

        # Use default if available
        if "default" in schema:
            return schema["default"]

        # Use enum first value
        if "enum" in schema and schema["enum"]:
            return schema["enum"][0]

        schema_type = schema.get("type", "object")

        if schema_type == "object":
            result = {}
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                result[prop] = self._generate_value(prop_schema, use_examples)
            return result

        elif schema_type == "array":
            item_schema = schema.get("items", {"type": "string"})
            min_items = schema.get("minItems", 1)
            return [
                self._generate_value(item_schema, use_examples)
                for _ in range(min_items)
            ]

        else:
            return self.SAMPLES.get(schema_type, None)

    def from_pydantic(self, model: Type) -> Dict[str, Any]:
        """
        Generate JSON schema from a Pydantic model.

        Args:
            model: Pydantic model class

        Returns:
            JSON Schema dict
        """
        try:
            if hasattr(model, "model_json_schema"):
                return model.model_json_schema()
            elif hasattr(model, "schema"):
                return model.schema()
            else:
                raise ValueError("Not a Pydantic model")
        except Exception as e:
            logger.error(f"Failed to generate schema from model: {e}")
            raise

    def from_example(self, example: Any) -> Dict[str, Any]:
        """
        Infer JSON schema from an example value.

        Args:
            example: Example data to infer schema from

        Returns:
            Inferred JSON Schema
        """
        return self._infer_schema(example)

    def _infer_schema(self, value: Any) -> Dict[str, Any]:
        """Recursively infer schema from value."""
        if value is None:
            return {"type": "null"}

        elif isinstance(value, bool):
            return {"type": "boolean", "example": value}

        elif isinstance(value, int):
            return {"type": "integer", "example": value}

        elif isinstance(value, float):
            return {"type": "number", "example": value}

        elif isinstance(value, str):
            return {"type": "string", "example": value}

        elif isinstance(value, list):
            if not value:
                return {"type": "array", "items": {"type": "string"}}
            # Infer from first item
            return {"type": "array", "items": self._infer_schema(value[0])}

        elif isinstance(value, dict):
            properties = {}
            for k, v in value.items():
                properties[k] = self._infer_schema(v)
            return {
                "type": "object",
                "properties": properties,
                "required": list(value.keys()),
            }

        return {"type": "string"}

    def to_prompt(self, schema: Dict[str, Any]) -> str:
        """
        Convert schema to a prompt-friendly description for LLMs.

        Args:
            schema: JSON Schema

        Returns:
            Human-readable schema description
        """
        lines = ["Expected JSON structure:"]
        sample = self.generate_sample(schema)
        lines.append("```json")
        lines.append(json.dumps(sample, indent=2))
        lines.append("```")

        # Add constraints
        if "properties" in schema:
            lines.append("\nField constraints:")
            for prop, prop_schema in schema["properties"].items():
                constraints = []
                if "type" in prop_schema:
                    constraints.append(f"type: {prop_schema['type']}")
                if "minLength" in prop_schema:
                    constraints.append(f"min length: {prop_schema['minLength']}")
                if "maxLength" in prop_schema:
                    constraints.append(f"max length: {prop_schema['maxLength']}")
                if "enum" in prop_schema:
                    constraints.append(f"allowed: {prop_schema['enum']}")
                if constraints:
                    lines.append(f"- {prop}: {', '.join(constraints)}")

        return "\n".join(lines)
