# ./tests/test_thorough.py
"""
Comprehensive Test Suite for OllamaToolkit
===========================================
Config-driven tests for all critical components:
1. OllamaConnector (Management Tasks)
2. WebTools (Strict WebScraperToolkit)
3. VisionTools (Image Analysis)
4. VectorTools (PDF/HTML Ingestion)
5. Telemetry Integration

Environment:
- Uses config.json for host/model configuration
- Samples: test_input_samples/
"""

import json
import unittest
import logging
import os
from pathlib import Path

from ollamatoolkit.connector import OllamaConnector
from ollamatoolkit.config import WebToolConfig, VectorConfig
from ollamatoolkit.tools.vector import VectorTools
from ollamatoolkit.tools.web import WebTools

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("TestThorough")

# Load config
CONFIG_PATH = Path(__file__).parents[1] / "config.json"
if CONFIG_PATH.exists():
    with open(CONFIG_PATH, "r") as f:
        CONFIG = json.load(f)
    BASE_URL = CONFIG.get("agent", {}).get("base_url", "http://localhost:11434")
    CHAT_MODEL = CONFIG.get("agent", {}).get("model", "ollama/ministral-3:latest")
    VISION_MODEL = CONFIG.get("vision", {}).get("model", "ollama/ministral-3:latest")
    EMBED_MODEL = CONFIG.get("vector", {}).get(
        "embedding_model", "ollama/nomic-embed-text"
    )
else:
    BASE_URL = "http://localhost:11434"
    CHAT_MODEL = "ollama/ministral-3:latest"
    VISION_MODEL = "ollama/ministral-3:latest"
    EMBED_MODEL = "ollama/nomic-embed-text"

# Optional gating for live/integration-heavy suite.
BASE_URL = os.getenv("OLLAMA_TEST_BASE_URL", BASE_URL)
ENABLE_THOROUGH = os.getenv("OLLAMA_RUN_THOROUGH_TESTS", "0").strip() == "1"

logger.info(f"Config: BASE_URL={BASE_URL}, CHAT_MODEL={CHAT_MODEL}")

# Sample paths
SAMPLES_DIR = Path(__file__).parents[1] / "test_input_samples"
WEB_HTML = SAMPLES_DIR / "web.html"
DOC_PDF = SAMPLES_DIR / "document.pdf"
IMG_JPEG = SAMPLES_DIR / "image.jpeg"


@unittest.skipUnless(
    ENABLE_THOROUGH,
    "Set OLLAMA_RUN_THOROUGH_TESTS=1 to run live thorough integration tests.",
)
class TestOllamaConnector(unittest.TestCase):
    """Test OllamaConnector management tasks."""

    def setUp(self):
        self.connector = OllamaConnector(base_url=BASE_URL)

    def test_health_check(self):
        """Verify connection to remote Ollama server."""
        logger.info(f"Testing health check against {BASE_URL}...")
        health = OllamaConnector.check_health(BASE_URL)
        self.assertTrue(health["online"], f"Ollama at {BASE_URL} should be online")
        logger.info(f"Health check passed: {health}")

    def test_list_models(self):
        """Verify model listing works."""
        logger.info("Testing model listing...")
        models = self.connector.get_models()
        self.assertIsInstance(models, list)
        self.assertTrue(len(models) > 0, "Should have at least one model")
        logger.info(f"Found {len(models)} models: {models[:5]}...")

    def test_ps(self):
        """Test listing running models."""
        logger.info("Testing PS (running models)...")
        resp = self.connector.list_running()
        self.assertIsNotNone(resp)
        logger.info(f"Running models: {resp}")


@unittest.skipUnless(
    ENABLE_THOROUGH,
    "Set OLLAMA_RUN_THOROUGH_TESTS=1 to run live thorough integration tests.",
)
class TestVectorTools(unittest.TestCase):
    """Test VectorTools with PDF ingestion."""

    def setUp(self):
        self.cfg = VectorConfig(
            storage_path="./test_vectors.json", embedding_model=EMBED_MODEL
        )
        self.vt = VectorTools(self.cfg)

    def test_pdf_ingestion(self):
        """Test PDF file ingestion and search."""
        if not DOC_PDF.exists():
            self.skipTest(f"Missing {DOC_PDF}")

        logger.info(f"Testing PDF ingestion: {DOC_PDF}")
        result = self.vt.ingest_file(str(DOC_PDF))
        logger.info(f"Ingest result: {result}")
        self.assertIn("Ingested", result)

        # Search for content
        hits = self.vt.search_memory("experience", top_k=2)
        logger.info(f"Search results: {len(hits)} hits")
        self.assertTrue(len(hits) > 0, "Should find relevant chunks")

    def test_text_ingestion(self):
        """Test plain text ingestion."""
        logger.info("Testing text ingestion...")
        result = self.vt.ingest_text(
            "A software engineer has experience in AI.",
            metadata={"source": "test"},
        )
        self.assertIn("Ingested", result)

    def tearDown(self):
        # Cleanup test vectors
        test_file = Path("./test_vectors.json")
        if test_file.exists():
            test_file.unlink()


@unittest.skipUnless(
    ENABLE_THOROUGH,
    "Set OLLAMA_RUN_THOROUGH_TESTS=1 to run live thorough integration tests.",
)
class TestWebTools(unittest.IsolatedAsyncioTestCase):
    """Test WebTools with WebScraperToolkit."""

    def setUp(self):
        self.config = WebToolConfig(timeout=30)
        self.web = WebTools(self.config)

    async def test_local_html_scraping(self):
        """Test local HTML file processing."""
        if not WEB_HTML.exists():
            self.skipTest(f"Missing {WEB_HTML}")

        logger.info(f"Testing local HTML scraping: {WEB_HTML}")
        result = await self.web.scrape_local_html(str(WEB_HTML))

        self.assertTrue(result.get("success"), f"Scrape failed: {result}")
        self.assertIn("markdown", result)
        logger.info(f"Extracted {len(result.get('markdown', ''))} chars of markdown")


@unittest.skipUnless(
    ENABLE_THOROUGH,
    "Set OLLAMA_RUN_THOROUGH_TESTS=1 to run live thorough integration tests.",
)
class TestVisionIntegration(unittest.IsolatedAsyncioTestCase):
    """Test vision capabilities with image.jpeg."""

    async def test_image_analysis(self):
        """Test image analysis using vision model."""
        if not IMG_JPEG.exists():
            self.skipTest(f"Missing {IMG_JPEG}")

        logger.info(f"Testing vision with {VISION_MODEL}...")

        import ollama
        import base64

        with open(IMG_JPEG, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        # Strip ollama/ prefix if present
        model = VISION_MODEL
        if model.startswith("ollama/"):
            model = model[7:]

        try:
            client = ollama.Client(host=BASE_URL)
            response = client.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": "Describe this image briefly.",
                        "images": [b64],
                    }
                ],
            )

            content = response.message.content
            logger.info(f"Vision response: {content[:100]}...")
            self.assertTrue(len(content) > 10, "Response too short")
        except Exception as e:
            logger.error(f"Vision test failed: {e}")
            self.fail(f"Vision test failed: {e}")


class TestTelemetryIntegration(unittest.TestCase):
    """Test telemetry toolkit integration."""

    def test_logger_creation(self):
        """Test LLMLogger can be created."""
        from ollamatoolkit.telemetry import get_logger, LLMLogger

        logger_instance = get_logger("test_session")
        self.assertIsInstance(logger_instance, LLMLogger)

    def test_log_interaction(self):
        """Test logging an interaction."""
        from ollamatoolkit.telemetry import log_interaction

        # Should not raise
        log_interaction(
            prompt="Test prompt", response="Test response", model="test-model"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
