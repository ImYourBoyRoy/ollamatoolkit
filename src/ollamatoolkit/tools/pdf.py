# ./src/ollamatoolkit/tools/pdf.py
"""PDF helper utilities shared by vision/document workflows.

Run via import: ``from ollamatoolkit.tools.pdf import PDFHandler``.
Inputs: PDF file paths.
Outputs: extracted text strings or temporary image paths for each PDF page.
Side effects: reads local files and writes temporary PNG images for OCR workflows.
"""

from __future__ import annotations

import os
import tempfile
from typing import List


class PDFHandler:
    """Small utility wrapper for common PDF read/convert operations."""

    def extract_text(self, pdf_path: str) -> str:
        """Extract plain text from all pages using pypdf."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return "Error: pypdf not installed. Install with 'pip install ollamatoolkit[files]'."

        if not os.path.exists(pdf_path):
            return f"Error: File not found: {pdf_path}"

        try:
            reader = PdfReader(pdf_path)
            chunks: List[str] = []
            for index, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                chunks.append(f"\n--- Page {index} ---\n{page_text}")
            return "".join(chunks)
        except Exception as exc:
            return f"Error reading PDF: {exc}"

    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """Convert PDF pages to temporary PNG image files and return their paths."""
        try:
            from pdf2image import convert_from_path
        except ImportError as exc:
            raise ImportError("pdf2image not installed.") from exc

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        try:
            images = convert_from_path(pdf_path)
            temp_dir = os.path.join(tempfile.gettempdir(), "ollamatoolkit_pdf_ocr")
            os.makedirs(temp_dir, exist_ok=True)

            output_paths: List[str] = []
            base_name = os.path.basename(pdf_path)
            for index, image in enumerate(images, start=1):
                output_path = os.path.join(temp_dir, f"{base_name}_page_{index}.png")
                image.save(output_path, "PNG")
                output_paths.append(output_path)
            return output_paths
        except Exception as exc:
            if "poppler" in str(exc).lower():
                raise RuntimeError(
                    "Poppler not found. Install poppler-utils (Linux/macOS) or poppler binaries (Windows)."
                ) from exc
            raise
