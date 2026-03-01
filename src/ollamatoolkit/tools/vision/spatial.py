# ./ollamatoolkit/tools/vision/spatial.py
"""
Ollama Toolkit - Vision Spatial
===============================
Visual Grounding and Object Detection logic.
"""

import os
import cv2
import re
from typing import Optional, List


class SpatialProcessor:
    def __init__(self, client):
        self.client = client

    def detect_objects(self, image_path: str, target: str) -> str:
        """
        Locates an object and attempts to draw a bounding box on a debug copy.
        """
        prompt = (
            f"Detect the '{target}' in this image. "
            "Return the bounding box in [ymin, xmin, ymax, xmax] format (0-1000 normalised) or [x1, y1, x2, y2]. "
            "Also provide a textual description."
        )

        response = self.client.analyze_image(image_path, prompt)

        # Attempt to visualize?
        # Parsing LLM output is hard without strict structured output.
        # But we can try regex for [0-9, ...]

        coords = self._extract_coords(response)
        if coords:
            debug_path = self._draw_box(image_path, coords, target)
            return f"{response}\n\n[Debug] Visualized at: {debug_path}"

        return response

    def _extract_coords(self, text: str) -> Optional[List[int]]:
        # Regex for [100, 200, 300, 400]
        # Supports space or comma
        match = re.search(r"\[\s*(\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)\s*\]", text)
        if match:
            return [int(g) for g in match.groups()]
        return None

    def _draw_box(self, image_path: str, coords: List[int], label: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            return ""

        h, w = img.shape[:2]
        # Assume coords are [x1, y1, x2, y2]
        # Normalization check: if all < 1000, likely normalized 0-1000 standard (Qwen/LLaVA)
        # OR pixel values?

        c = coords
        is_normalized = all(v <= 1000 for v in c)

        if is_normalized:
            # Convert 1000-scale to pixels
            x1 = int((c[0] / 1000) * w)  # Assuming [x1, y1, x2, y2] order usually
            y1 = int((c[1] / 1000) * h)
            x2 = int((c[2] / 1000) * w)
            y2 = int((c[3] / 1000) * h)

            # Note: Some models output [ymin, xmin, ymax, xmax]
            # Heuristic: if y1 > x1 consistently? Hard to know without model spec.
            # We assume [x1,y1, x2,y2] for now as common standard.
        else:
            x1, y1, x2, y2 = c

        # Draw
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        base, ext = os.path.splitext(image_path)
        out_path = f"{base}_detected{ext}"
        cv2.imwrite(out_path, img)
        return out_path
