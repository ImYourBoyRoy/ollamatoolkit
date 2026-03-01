# ./ollamatoolkit/tools/vision/metadata.py
"""
Ollama Toolkit - Vision Metadata
================================
Scene description and structured data extraction.
"""

import json
from typing import Dict


class MetadataProcessor:
    def __init__(self, client):
        self.client = client
        self.config = client.config

    def describe_scene(self, image_path: str) -> Dict[str, str]:
        """
        Returns structured metadata about the scene.
        Uses config.meta_map to align keys if provided.
        """
        # Default Schema
        keys = ["objects", "colors", "atmosphere", "text_content"]

        # Apply Mapping
        # If meta_map has {'author': 'creator'}, we might add 'creator' to request?
        # For now, let's just ask for a robust JSON and try to map output keys.

        prompt = (
            "Analyze this image and return a valid JSON object. "
            f"Required keys: {', '.join(keys)}. "
            "Ensure 'objects' is a list, 'colors' is a list of hex codes or names."
        )

        raw = self.client.analyze_image(image_path, prompt)

        # Parse JSON
        try:
            # Cleanup markdown
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)

            # Post-Process Mapping
            if self.config.meta_map:
                remapped = {}
                for k, v in data.items():
                    # Check if 'k' should be mapped to something else?
                    # Usually meta_map is Key->MappedKey.
                    new_key = self.config.meta_map.get(k, k)
                    remapped[new_key] = v
                return remapped

            return data
        except Exception:
            # Fallback to raw text wrapped
            return {"raw_description": raw, "error": "JSON parse failed"}
