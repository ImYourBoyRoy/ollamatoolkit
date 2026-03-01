# ./src/ollamatoolkit/tools/vision/ocr.py
"""OCR helper that routes image/PDF inputs through a vision-capable model.

Run via ``VisionTools`` composition; this class is typically instantiated internally.
Inputs: image/PDF file path and optional language hint.
Outputs: extracted OCR text as a string.
Side effects: creates temporary PDF page images and deletes them after processing.
"""

from __future__ import annotations

import os
from typing import List

from ollamatoolkit.tools.pdf import PDFHandler


class OCRProcessor:
    """OCR processor that delegates image analysis to the parent vision client."""

    def __init__(self, client: object) -> None:
        self.client = client
        self.pdf_handler = PDFHandler()

    def process(self, file_path: str, language_hint: str = "English") -> str:
        """Extract text from either image files or multi-page PDFs."""
        if not os.path.exists(file_path):
            return f"Error: File not found {file_path}"

        extension = os.path.splitext(file_path)[1].lower()
        if extension == ".pdf":
            return self._process_pdf(file_path, language_hint)
        return self._process_image(file_path, language_hint)

    def _process_image(self, image_path: str, language_hint: str) -> str:
        prompt = (
            "Transcribe ALL text in this image verbatim. "
            "Do not describe the image, only output text. "
            f"Language hint: {language_hint}."
        )
        analyze_image = getattr(self.client, "analyze_image")
        return str(analyze_image(image_path, prompt))

    def _process_pdf(self, pdf_path: str, language_hint: str) -> str:
        try:
            image_paths = self.pdf_handler.pdf_to_images(pdf_path)
        except Exception as exc:
            return f"PDF conversion failed: {exc}"

        report_parts: List[str] = [f"--- OCR REPORT: {os.path.basename(pdf_path)} ---"]
        for page_index, image_path in enumerate(image_paths, start=1):
            report_parts.append(f"\n[Page {page_index}]")
            report_parts.append(self._process_image(image_path, language_hint))
            try:
                os.remove(image_path)
            except Exception:
                pass

        return "\n".join(report_parts)
