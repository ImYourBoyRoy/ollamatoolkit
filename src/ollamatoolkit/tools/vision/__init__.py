# ./src/ollamatoolkit/tools/vision/__init__.py
"""
Ollama Toolkit - Vision Package
===============================
Unified interface for Multimodal capabilities.
Supports Images, PDFs, Videos, and Visual Grounding.
"""

import base64
import os
import litellm
from typing import Dict, List, Union


from ollamatoolkit.config import VisionConfig
from ollamatoolkit.tools.vision.ocr import OCRProcessor
from ollamatoolkit.tools.vision.analysis import AnalysisProcessor
from ollamatoolkit.tools.vision.metadata import MetadataProcessor
from ollamatoolkit.tools.vision.video import VideoProcessor
from ollamatoolkit.tools.vision.tiling import ImageTiler
from ollamatoolkit.tools.vision.spatial import SpatialProcessor


class VisionTools:
    """
    Facade for visual analysis, OCR, video processing, and grounding.
    """

    def __init__(self, config: VisionConfig, base_url: str):
        self.config = config
        self.base_url = base_url

        # Sub-Processors
        self.ocr = OCRProcessor(self)
        self.analysis = AnalysisProcessor(self)
        self.metadata = MetadataProcessor(self)
        self.video = VideoProcessor()
        self.tiler = ImageTiler()
        self.spatial = SpatialProcessor(self)

    def _encode_image(self, image_path: str) -> str:
        """Encodes an image file to base64 string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def analyze_image(
        self,
        image_input: Union[str, List[str]],
        instruction: str = "Describe this.",
        high_res: bool = False,
    ) -> str:
        """
        Core Internal Method: Calls the VLM.
        Args:
            image_input: Single path, or List of paths.
            high_res: If True and input is single image, applies Tiling (Zoom) to catch details.
        """
        try:
            images = [image_input] if isinstance(image_input, str) else image_input

            # Apply Tiling if requested on single image
            if high_res and isinstance(image_input, str):
                images = self.tiler.tile_image(image_input)

            content_blocks: List[Dict[str, object]] = [
                {"type": "text", "text": instruction}
            ]

            for img_path in images:
                b64_image = self._encode_image(img_path)
                # Guess mime
                mime_type = "image/jpeg"
                if img_path.lower().endswith(".png"):
                    mime_type = "image/png"
                elif img_path.lower().endswith(".webp"):
                    mime_type = "image/webp"

                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{b64_image}"},
                    }
                )

            messages = [{"role": "user", "content": content_blocks}]

            response = litellm.completion(
                model=self.config.model,
                messages=messages,
                base_url=self.base_url,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error analyzing vision content: {e}"

    # --- Video & Advanced Methods ---

    def analyze_video(
        self, video_path: str, instruction: str = "Summarize the events in this video."
    ) -> str:
        """
        Extracts key frames (smart scene detection) and analyzes.
        """
        try:
            # Smart Extraction
            frames = self.video.extract_keyframes(
                video_path, min_interval_sec=self.config.video_sample_interval
            )

            if not frames:
                return "No video content extracted."

            # Batching? Sending 50 images might break context.
            # We'll rely on the model's context window.
            result = self.analyze_image(frames, instruction)

            # Cleanup
            for f in frames:
                try:
                    os.remove(f)
                except Exception:
                    pass

            return result
        except Exception as e:
            return f"Video analysis failed: {e}"

    def detect_objects(self, image_path: str, target: str) -> str:
        """
        Visual Grounding: Asks the model to locate an object.
        Draws debug box if coordinates parseable.
        """
        return self.spatial.detect_objects(image_path, target)

    def compare_images(
        self, image1: str, image2: str, criteria: str = "What are the differences?"
    ) -> str:
        return self.analyze_image([image1, image2], criteria)

    # --- Facade Delegates ---

    def ocr_image(self, image_path: str, language_hint: str = "English") -> str:
        return self.ocr.process(image_path, language_hint)

    def image_to_code(
        self, image_path: str, target_format: str = "HTML/Tailwind"
    ) -> str:
        return self.analysis.image_to_code(image_path, target_format)

    def describe_scene(self, image_path: str) -> Dict[str, str]:
        return self.metadata.describe_scene(image_path)
