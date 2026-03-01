# ./ollamatoolkit/tools/document.py
"""
Ollama Toolkit - Document Processor
===================================
Vision-based document processing for PDFs and images.

Flow:
    PDF → Image (per page) → Vision Model → Markdown

Supports:
    - PDF to images (via pymupdf/fitz)
    - Multi-page handling
    - Multiple vision model comparison
    - HTML to markdown (via WebScraperToolkit or bs4)

Usage:
    processor = DocumentProcessor(base_url)

    # Process PDF with vision model
    result = await processor.process_pdf("document.pdf", model="ministral-3:latest")

    # Compare models
    results = await processor.compare_vision_models("document.pdf")

Requires:
    - pymupdf (pip install pymupdf)
    - ollama-python
"""

import base64
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import ollama
from ollamatoolkit.types import ChatResponse, GenerateResponse

logger = logging.getLogger(__name__)


@dataclass
class PageResult:
    """Result of processing a single page."""

    page_num: int
    model: str
    success: bool
    markdown: str
    latency_ms: float
    error: Optional[str] = None


@dataclass
class DocumentResult:
    """Result of processing an entire document."""

    file_path: str
    model: str
    total_pages: int
    success: bool
    combined_markdown: str
    pages: List[PageResult]
    total_latency_ms: float
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "model": self.model,
            "total_pages": self.total_pages,
            "success": self.success,
            "combined_markdown": self.combined_markdown,
            "pages": [asdict(p) for p in self.pages],
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
        }


class DocumentProcessor:
    """
    Vision-based document processor.
    Converts PDFs to images and uses vision models for extraction.

    Auto-selects available vision models if none specified.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        Initialize DocumentProcessor.

        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.async_client = ollama.AsyncClient(host=base_url)
        self.sync_client = ollama.Client(host=base_url)
        self._template_cache: Dict[str, bool] = {}  # model -> uses_raw_prompt
        self._vision_models_cache: Optional[List[str]] = None
        logger.info(f"DocumentProcessor initialized for {base_url}")

    def _get_vision_models(self) -> List[str]:
        """Get available vision models from the server."""
        if self._vision_models_cache is not None:
            return self._vision_models_cache

        try:
            from ollamatoolkit.tools.models import ModelInspector

            inspector = ModelInspector(base_url=self.base_url)
            vision_models = list(inspector.get_vision_models().keys())
            self._vision_models_cache = vision_models
            return vision_models
        except Exception as e:
            logger.warning(f"Failed to get vision models: {e}")
            return []

    def _select_vision_model(self) -> Optional[str]:
        """Auto-select the best available vision model."""
        models = self._get_vision_models()
        if not models:
            return None
        # Return first available (could add smarter selection later)
        return models[0]

    def uses_raw_prompt_template(self, model: str) -> bool:
        """
        Check if a model uses raw prompt template ({{ .Prompt }}) vs chat template.

        Models with raw prompt templates require the `generate` API.
        Models with chat templates (using .Messages) require the `chat` API.

        Args:
            model: Model name to check

        Returns:
            True if model uses raw prompt template (needs generate API)
        """
        # Use cache to avoid repeated API calls
        if model in self._template_cache:
            return self._template_cache[model]

        try:
            info = self.sync_client.show(model)
            template = info.get("template", "")

            # Raw prompt templates are simple: {{ .Prompt }} or {{ .System }} {{ .Prompt }}
            # Chat templates use .Messages, .Content (for message content), etc.
            uses_raw = (
                "{{ .Prompt }}" in template
                and ".Messages" not in template
                and ".Content" not in template
            )

            self._template_cache[model] = uses_raw
            logger.debug(f"Model {model}: uses_raw_prompt={uses_raw}")
            return uses_raw

        except Exception as e:
            logger.warning(
                f"Could not determine template for {model}: {e}, defaulting to chat API"
            )
            self._template_cache[model] = False
            return False

    def extract_pdf_text(self, pdf_path: str, clean: bool = False) -> str:
        """
        Extract text directly from PDF using pypdf.

        This is fast and reliable for text-based PDFs.
        Returns empty string for scanned/image-based PDFs.

        Args:
            pdf_path: Path to PDF file
            clean: If True, apply LLM-optimized cleaning

        Returns:
            Extracted text (empty if PDF is image-based)
        """
        try:
            import pypdf

            reader = pypdf.PdfReader(pdf_path)
            text_parts = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
            text = "\n".join(text_parts)
            logger.info(f"Extracted {len(text)} chars from PDF via pypdf")

            if clean:
                text = self.clean_for_llm(text)
                logger.info(f"Cleaned text: {len(text)} chars")

            return text
        except ImportError:
            logger.warning("pypdf not installed, falling back to vision OCR")
            return ""
        except Exception as e:
            logger.warning(
                f"PDF text extraction failed: {e}, falling back to vision OCR"
            )
            return ""

    def clean_for_llm(self, text: str) -> str:
        """
        Clean extracted text for optimal LLM consumption.

        Fixes common PDF extraction issues:
        - Normalizes whitespace and line breaks
        - Fixes split words from column layouts
        - Normalizes bullets and special characters
        - Fixes phone number spacing
        - Converts smart quotes to standard quotes

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text ready for LLM processing
        """
        import re

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Fix common PDF column-break word splits
        common_fixes = {
            "EDUCA TION": "EDUCATION",
            "CERTIFICA TION": "CERTIFICATION",
            "EXPERI ENCE": "EXPERIENCE",
            "EMPLOY MENT": "EMPLOYMENT",
            "MANAGE MENT": "MANAGEMENT",
        }
        for broken, fixed in common_fixes.items():
            text = re.sub(broken, fixed, text, flags=re.IGNORECASE)

        # Fix phone number spacing: "801-657 -9912" -> "801-657-9912"
        text = re.sub(r"(\d{3})\s*-\s*(\d{3})\s*-\s*(\d{4})", r"\1-\2-\3", text)

        # Normalize various bullet characters to standard dash
        # Note: Do NOT include 'o' as it breaks regular words
        bullet_chars = ["•", "·", "ò", "▪", "►", "■", "●", "○"]
        for bullet in bullet_chars:
            text = text.replace(bullet, "-")

        # Convert smart quotes to standard quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")
        text = text.replace("–", "-").replace("—", "-")

        # Collapse multiple spaces to single space
        text = re.sub(r"[ \t]+", " ", text)

        # Remove trailing spaces from lines
        text = re.sub(r" +\n", "\n", text)
        text = re.sub(r" +$", "", text, flags=re.MULTILINE)

        # Collapse multiple blank lines to max 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Join lines that were split mid-sentence (line doesn't end with punctuation)
        # But preserve intentional line breaks (after periods, colons, etc.)
        lines = text.split("\n")
        joined_lines = []
        buffer = ""

        for line in lines:
            line = line.strip()
            if not line:
                if buffer:
                    joined_lines.append(buffer)
                    buffer = ""
                joined_lines.append("")
                continue

            if buffer:
                # Check if previous line ended with continuation indicators
                if buffer[-1] in ".,;:!?-":
                    joined_lines.append(buffer)
                    buffer = line
                else:
                    # Join with previous line
                    buffer = buffer + " " + line
            else:
                buffer = line

        if buffer:
            joined_lines.append(buffer)

        text = "\n".join(joined_lines)

        return text.strip()

    async def smart_process_pdf(
        self,
        pdf_path: str,
        vision_model: Optional[str] = None,
        min_text_chars: int = 100,
        dpi: int = 150,
        save_images_dir: Optional[str] = None,
        clean_text: bool = True,
    ) -> Dict[str, Any]:
        """
        Hybrid PDF processing: text extraction first, vision OCR fallback.

        Strategy:
            1. Try direct text extraction via pypdf (fast, reliable)
            2. If text is too short, fall back to vision model OCR

        Args:
            pdf_path: Path to PDF file
            vision_model: Model to use for OCR (auto-selected if None)
            min_text_chars: Minimum chars to consider extraction successful
            dpi: Resolution for vision OCR (if needed)
            save_images_dir: Optional directory to save images (for inspection)
            clean_text: If True, apply LLM-optimized cleaning to extracted text

        Returns:
            Dict with 'method' (text|vision), 'content', 'chars', 'success'
        """
        path = Path(pdf_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {pdf_path}"}

        start_time = time.time()

        # Step 1: Try direct text extraction (with optional cleaning)
        logger.info(f"Smart processing PDF: {path.name}")
        text = self.extract_pdf_text(pdf_path, clean=clean_text)

        if len(text.strip()) >= min_text_chars:
            # Success! Text-based PDF
            latency = (time.time() - start_time) * 1000
            logger.info(
                f"  Text extraction successful: {len(text)} chars in {latency:.0f}ms"
            )
            return {
                "success": True,
                "method": "text",
                "content": text,
                "chars": len(text),
                "latency_ms": latency,
                "model": None,
                "message": "Direct text extraction (fast path)",
            }

        # Step 2: Fall back to vision OCR
        logger.info(
            f"  Text extraction insufficient ({len(text)} chars), using vision OCR"
        )

        # Auto-select vision model if not provided
        if vision_model is None:
            vision_model = self._select_vision_model()
            if vision_model is None:
                return {
                    "success": False,
                    "error": "No vision model available",
                    "method": "vision",
                }
            logger.info(f"  Auto-selected vision model: {vision_model}")

        try:
            result = await self.process_pdf(
                pdf_path, model=vision_model, dpi=dpi, save_images_dir=save_images_dir
            )

            latency = (time.time() - start_time) * 1000
            return {
                "success": result.success,
                "method": "vision",
                "content": result.combined_markdown,
                "chars": len(result.combined_markdown),
                "latency_ms": latency,
                "model": vision_model,
                "pages": result.total_pages,
                "message": "Vision OCR fallback (scanned/image PDF)",
            }
        except Exception as e:
            return {"success": False, "error": str(e), "method": "vision"}

    # =========================================================================
    # PDF Processing
    # =========================================================================

    def pdf_to_images(
        self, pdf_path: str, dpi: int = 150, save_dir: Optional[str] = None
    ) -> List[bytes]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for rendering (150 dpi is good balance)
            save_dir: Optional directory to save images (for inspection)

        Returns:
            List of PNG image bytes (one per page)
        """
        try:
            import fitz  # pymupdf
        except ImportError:
            raise ImportError(
                "pymupdf is required for PDF processing.\n"
                "Install with: pip install pymupdf"
            )

        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Converting PDF to images: {path.name} (dpi={dpi})")

        # Create save directory if specified
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  Saving images to: {save_path}")

        images = []
        doc = fitz.open(str(path))

        for page_num in range(len(doc)):
            page = doc[page_num]
            # Render at specified DPI
            zoom = dpi / 72  # 72 is default PDF DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

            # Convert to PNG bytes
            img_bytes = pix.tobytes("png")
            images.append(img_bytes)

            # Save if directory specified
            if save_dir:
                img_filename = f"{path.stem}_page_{page_num + 1}.png"
                img_path = Path(save_dir) / img_filename
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
                logger.info(
                    f"  Page {page_num + 1}: {pix.width}x{pix.height} px → {img_filename}"
                )
            else:
                logger.debug(f"  Page {page_num + 1}: {pix.width}x{pix.height} px")

        doc.close()
        logger.info(f"Converted {len(images)} pages to images")
        return images

    async def process_page_image(
        self, image_bytes: bytes, page_num: int, model: str, prompt: str = None
    ) -> PageResult:
        """
        Process a single page image with a vision model.

        Args:
            image_bytes: PNG image bytes
            page_num: Page number (1-indexed)
            model: Vision model to use
            prompt: Custom prompt (default: extract document contents)

        Returns:
            PageResult with extracted markdown
        """
        if prompt is None:
            # Model-specific prompts for better OCR accuracy
            model_lower = model.lower()

            if "deepseek-ocr" in model_lower:
                # DeepSeek OCR needs a simple, direct OCR prompt
                prompt = (
                    "OCR this document image. "
                    "Extract and output ALL visible text exactly as it appears. "
                    "Preserve the original formatting, layout, and structure."
                )
            elif "qwen3-vl" in model_lower or "qwen" in model_lower:
                # Qwen3-VL works better with simpler, direct prompts
                prompt = (
                    "Extract all text content from this document page. "
                    "Format the output as clean markdown with proper headings, "
                    "lists, and structure. Preserve the document's organization."
                )
            else:
                # Default for Mistral-based models (ministral-3, devstral, mistral-small3.2)
                prompt = (
                    "Extract all text content from this document page. "
                    "Format the output as clean markdown with proper headings, "
                    "lists, and structure. Preserve the document's organization. "
                    "Include all names, contact information, dates, and details exactly as written."
                )

        logger.info(f"  [Page {page_num}] Processing with {model}...")
        start_time = time.time()

        try:
            # Encode image
            image_b64 = base64.b64encode(image_bytes).decode()

            # Determine API based on model template
            # Models with raw prompt templates ({{ .Prompt }}) need generate API
            # Models with chat templates (.Messages) need chat API
            use_generate = self.uses_raw_prompt_template(model)

            if use_generate:
                # Use generate API for raw prompt models (e.g., deepseek-ocr, qwen3-vl)
                logger.debug(f"  [Page {page_num}] Using generate API for {model}")
                generate_response = cast(
                    GenerateResponse,
                    await self.async_client.generate(
                        model=model,
                        prompt=prompt,
                        images=[image_b64],
                        options={"num_predict": 4096},
                        keep_alive="0",
                    ),
                )
                markdown = generate_response.response
            else:
                # Use chat API for chat-template models (e.g., ministral-3)
                logger.debug(f"  [Page {page_num}] Using chat API for {model}")
                chat_response = cast(
                    ChatResponse,
                    await self.async_client.chat(
                        model=model,
                        messages=[
                            {"role": "user", "content": prompt, "images": [image_b64]}
                        ],
                        options={"num_predict": 4096},
                        keep_alive="0",
                    ),
                )
                markdown = chat_response.message.content or ""
            latency = (time.time() - start_time) * 1000

            logger.info(
                f"  [Page {page_num}] Done: {len(markdown)} chars, {latency:.0f}ms"
            )

            return PageResult(
                page_num=page_num,
                model=model,
                success=True,
                markdown=markdown,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error(f"  [Page {page_num}] Error: {e}")
            return PageResult(
                page_num=page_num,
                model=model,
                success=False,
                markdown="",
                latency_ms=latency,
                error=str(e),
            )

    async def process_pdf(
        self,
        pdf_path: str,
        model: Optional[str] = None,
        dpi: int = 150,
        prompt: str = None,
        save_images_dir: Optional[str] = None,
    ) -> DocumentResult:
        """
        Process an entire PDF with a vision model.

        Args:
            pdf_path: Path to PDF file
            model: Vision model to use (auto-selected if None)
            dpi: Image resolution
            prompt: Custom extraction prompt
            save_images_dir: Directory to save page images for inspection

        Returns:
            DocumentResult with combined markdown from all pages
        """
        # Auto-select model if not provided
        if model is None:
            model = self._select_vision_model()
            if model is None:
                raise ValueError(
                    "No vision model available. Please install one with: "
                    "ollama pull llava"
                )

        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"  Model: {model}")

        start_time = time.time()

        # Convert PDF to images (optionally save them)
        images = self.pdf_to_images(pdf_path, dpi=dpi, save_dir=save_images_dir)

        # Process each page
        pages = []
        for i, img_bytes in enumerate(images):
            page_result = await self.process_page_image(
                img_bytes, page_num=i + 1, model=model, prompt=prompt
            )
            pages.append(page_result)

        # Combine results
        combined_md = "\n\n---\n\n".join(
            f"## Page {p.page_num}\n\n{p.markdown}" for p in pages if p.success
        )

        total_latency = (time.time() - start_time) * 1000
        success = all(p.success for p in pages)

        logger.info(
            f"PDF processing complete: {len(pages)} pages, {total_latency:.0f}ms total"
        )

        return DocumentResult(
            file_path=pdf_path,
            model=model,
            total_pages=len(pages),
            success=success,
            combined_markdown=combined_md,
            pages=pages,
            total_latency_ms=total_latency,
            timestamp=datetime.now().isoformat(),
        )

    async def compare_vision_models(
        self, pdf_path: str, models: List[str] = None, dpi: int = 150
    ) -> Dict[str, DocumentResult]:
        """
        Process PDF with multiple vision models for comparison.

        Args:
            pdf_path: Path to PDF
            models: List of models to compare (default: all vision models)
            dpi: Image resolution

        Returns:
            Dict mapping model name to DocumentResult
        """
        if models is None:
            # Auto-detect vision models
            models = self._get_vision_models()
            if not models:
                logger.warning("No vision models installed")
                return {}

        logger.info(f"Comparing {len(models)} vision models on: {pdf_path}")

        results = {}
        for model in models:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Testing: {model}")
            logger.info(f"{'=' * 50}")

            try:
                result = await self.process_pdf(pdf_path, model=model, dpi=dpi)
                results[model] = result
            except Exception as e:
                logger.error(f"Model {model} failed: {e}")
                results[model] = DocumentResult(
                    file_path=pdf_path,
                    model=model,
                    total_pages=0,
                    success=False,
                    combined_markdown="",
                    pages=[],
                    total_latency_ms=0,
                    timestamp=datetime.now().isoformat(),
                )

        return results

    # =========================================================================
    # HTML Processing
    # =========================================================================

    def html_to_markdown(self, html_path: str) -> str:
        """
        Convert local HTML file to markdown using BeautifulSoup.

        Args:
            html_path: Path to HTML file

        Returns:
            Markdown text
        """
        from bs4 import BeautifulSoup
        import re

        path = Path(html_path)
        if not path.exists():
            raise FileNotFoundError(f"HTML not found: {html_path}")

        logger.info(f"Converting HTML to markdown: {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            html = f.read()

        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, and control elements
        for element in soup(["script", "style", "head", "nav", "footer"]):
            element.decompose()

        # Get title
        title = soup.title.string if soup.title else ""

        # Extract text with structure
        lines = []
        if title:
            lines.append(f"# {title.strip()}\n")

        # Process main content
        for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "div"]):
            text = element.get_text(strip=True)
            if not text:
                continue

            if element.name == "h1":
                lines.append(f"# {text}")
            elif element.name == "h2":
                lines.append(f"## {text}")
            elif element.name == "h3":
                lines.append(f"### {text}")
            elif element.name == "h4":
                lines.append(f"#### {text}")
            elif element.name == "li":
                lines.append(f"- {text}")
            elif element.name in ["p", "div"]:
                # Only add if substantial
                if len(text) > 20:
                    lines.append(text)

        markdown = "\n\n".join(lines)

        # Clean up excessive whitespace
        markdown = re.sub(r"\n{3,}", "\n\n", markdown)

        logger.info(f"Converted HTML: {len(markdown)} chars")
        return markdown

    # =========================================================================
    # Export
    # =========================================================================

    def export_results(
        self, results: Dict[str, DocumentResult], output_dir: str
    ) -> Dict[str, str]:
        """Export comparison results to JSON files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        exported = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for model, result in results.items():
            safe_name = model.replace(":", "_").replace("/", "_")

            # Export full result
            json_path = out_path / f"doc_{safe_name}_{timestamp}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

            # Export markdown
            md_path = out_path / f"doc_{safe_name}_{timestamp}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(f"# Document Extraction: {model}\n\n")
                f.write(f"- File: {result.file_path}\n")
                f.write(f"- Pages: {result.total_pages}\n")
                f.write(f"- Latency: {result.total_latency_ms:.0f}ms\n\n")
                f.write("---\n\n")
                f.write(result.combined_markdown)

            exported[model] = str(json_path)
            logger.info(f"Exported: {json_path.name}")

        return exported
