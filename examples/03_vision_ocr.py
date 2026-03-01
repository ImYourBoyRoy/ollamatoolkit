# ./examples/03_vision_ocr.py
"""
Vision & OCR Example
====================
Demonstrates PDF text extraction and image analysis.

Run: python examples/03_vision_ocr.py
"""

import asyncio
from pathlib import Path

from ollamatoolkit.tools.document import DocumentProcessor
from ollamatoolkit import ModelSelector


async def main():
    # Check for vision models
    selector = ModelSelector()
    vision_model = selector.get_best_vision_model()

    if not vision_model:
        print("No vision model found. Run: ollama pull llava")
        return

    print(f"Using vision model: {vision_model}")

    # Create processor
    processor = DocumentProcessor()

    # Example 1: Smart PDF processing (text extraction with OCR fallback)
    pdf_path = Path("test_input_samples/sample.pdf")
    if pdf_path.exists():
        print("\n--- PDF Processing ---")
        result = await processor.smart_process_pdf(str(pdf_path))

        if result["success"]:
            print(f"Method: {result['method']}")
            print(f"Extracted {result['chars']} characters")
            print(f"Preview: {result['content'][:200]}...")
        else:
            print(f"Error: {result.get('error')}")
    else:
        print(f"\nNote: Create {pdf_path} to test PDF extraction")

    # Example 2: Image text extraction (if you have an image)
    image_path = Path("test_input_samples/document.png")
    if image_path.exists():
        print("\n--- Image OCR ---")
        page_result = await processor.process_page_image(
            image_path.read_bytes(), page_num=1, model=vision_model
        )

        if page_result.success:
            print(f"Extracted: {page_result.markdown[:200]}...")
        else:
            print(f"Error: {page_result.error}")


if __name__ == "__main__":
    asyncio.run(main())
