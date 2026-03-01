# ./ollamatoolkit/tools/web.py
"""
Ollama Toolkit - Web Tools
==========================
Advanced web capabilities strictly using WebScraperToolkit.

Usage:
    web = WebTools(config)
    result = await web.scrape_url("https://example.com")
    markdown = await web.scrape_local_html("./page.html")

Requirements:
    - web-scraper-toolkit must be installed
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from ollamatoolkit.config import WebToolConfig

logger = logging.getLogger(__name__)


class WebTools:
    """
    Web tools strictly using WebScraperToolkit for all content extraction.
    """

    def __init__(self, config: WebToolConfig):
        self.config = config
        self._verify_toolkit()

    def _verify_toolkit(self):
        """Verify WebScraperToolkit is available."""
        try:
            import web_scraper_toolkit

            logger.info(
                f"WebScraperToolkit v{web_scraper_toolkit.__version__} available"
            )
        except ImportError:
            raise ImportError(
                "web-scraper-toolkit is required but not installed.\n"
                "Please install it with: pip install web-scraper-toolkit"
            )

    # =========================================================================
    # URL Scraping
    # =========================================================================

    async def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape URL and return structured result with markdown.

        Returns:
            Dict with keys: markdown, success, url
        """
        try:
            from web_scraper_toolkit import read_website_markdown

            markdown = await read_website_markdown(url)

            return {"success": True, "markdown": markdown or "", "url": url}
        except Exception as e:
            logger.error(f"Scrape failed for {url}: {e}")
            return {"success": False, "error": str(e), "url": url}

    async def scrape_urls(self, urls: List[str]) -> List[Dict[str, Any]]:
        """Scrape multiple URLs concurrently."""
        import asyncio

        tasks = [self.scrape_url(url) for url in urls]
        return await asyncio.gather(*tasks)

    # =========================================================================
    # Local HTML Processing
    # =========================================================================

    async def scrape_local_html(self, file_path: str) -> Dict[str, Any]:
        """
        Process local HTML file using WebScraperToolkit's MarkdownConverter.
        """
        path = Path(file_path).resolve()
        if not path.exists():
            return {"success": False, "error": f"File not found: {file_path}"}

        try:
            from web_scraper_toolkit import MarkdownConverter

            with open(path, "r", encoding="utf-8") as f:
                html_content = f.read()

            converter = MarkdownConverter()
            markdown = converter.convert(html_content)

            return {
                "success": True,
                "markdown": markdown,
                "title": self._extract_title(html_content),
                "links": self._extract_links(html_content),
                "source": str(path),
            }
        except Exception as e:
            logger.error(f"Local HTML processing failed: {e}")
            return {"success": False, "error": str(e), "source": str(path)}

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        import re

        match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def _extract_links(self, html: str) -> List[str]:
        """Extract links from HTML."""
        import re

        return re.findall(r'href=["\']([^"\']+)["\']', html)

    # =========================================================================
    # Contact Extraction (via WebScraperToolkit)
    # =========================================================================

    async def extract_contacts(self, text: str) -> Dict[str, Any]:
        """
        Extract contact information from text/markdown.
        Uses WebScraperToolkit's contact extraction.
        """
        try:
            from web_scraper_toolkit import (
                extract_emails,
                extract_phones,
                extract_socials,
            )

            return {
                "success": True,
                "emails": extract_emails(text),
                "phones": extract_phones(text),
                "socials": extract_socials(text),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Sitemap Discovery (via WebScraperToolkit)
    # =========================================================================

    async def discover_urls(self, base_url: str, max_urls: int = 100) -> Dict[str, Any]:
        """
        Discover URLs from a domain using smart discovery.
        """
        try:
            from web_scraper_toolkit import smart_discover_urls

            urls = await smart_discover_urls(base_url, max_results=max_urls)

            return {
                "success": True,
                "urls": list(urls),
                "count": len(urls),
                "base_url": base_url,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
