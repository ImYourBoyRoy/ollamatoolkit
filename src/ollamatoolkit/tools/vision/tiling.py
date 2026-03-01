# ./ollamatoolkit/tools/vision/tiling.py
"""
Ollama Toolkit - Vision Tiling
==============================
Splits high-res images to preserve detail for VLMs.
"""

import os
import cv2
import tempfile
from typing import List


class ImageTiler:
    def __init__(self):
        pass

    def tile_image(self, image_path: str, grid: int = 2) -> List[str]:
        """
        Splits image into grid x grid tiles + original (resized).
        Returns list of paths: [Original, Tile1, Tile2, ...].
        """
        if not os.path.exists(image_path):
            return []

        img = cv2.imread(image_path)
        if img is None:
            return [image_path]  # Fail gracefully

        h, w = img.shape[:2]

        # Output setup
        temp_dir = os.path.join(tempfile.gettempdir(), "ollama_vision_tiles")
        os.makedirs(temp_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(image_path))[0]

        paths = []

        # 1. Add Original (maybe downscaled if massive?)
        # For now, keep original as "Master View"
        paths.append(image_path)

        # 2. Generate tiles
        # Step size
        step_h = h // grid
        step_w = w // grid

        count = 0
        for y in range(0, grid):
            for x in range(0, grid):
                y_start = y * step_h
                x_start = x * step_w
                y_end = y_start + step_h
                x_end = x_start + step_w

                crop = img[y_start:y_end, x_start:x_end]

                # Verify chunk size
                if crop.size == 0:
                    continue

                out_path = os.path.join(temp_dir, f"{base}_tile_{y}_{x}.jpg")
                cv2.imwrite(out_path, crop)
                paths.append(out_path)
                count += 1

        return paths
