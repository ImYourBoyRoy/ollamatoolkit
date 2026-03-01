# ./ollamatoolkit/tools/vector.py
"""
Ollama Toolkit - Vector Intelligence (RAG)
==========================================
Production-grade RAG engine with:
- Async embedding support via ollama-python
- PDF, HTML, and text ingestion
- Semantic search with cosine similarity
- Persistent JSON storage with backup

Usage:
    vt = VectorTools(config)
    await vt.aingest_file("document.pdf")
    results = await vt.asearch_memory("query")

Inputs:
    - VectorConfig from config.json
    - Text, PDF, or HTML files

Outputs:
    - Ingested chunks stored in vector_store.json
    - Search results with similarity scores
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ollamatoolkit.config import VectorConfig

logger = logging.getLogger(__name__)


class VectorTools:
    """
    Production-grade RAG with async embedding via ollama-python.
    """

    def __init__(self, config: VectorConfig):
        import ollama

        self.config = config
        self.memory: List[Dict[str, Any]] = []
        self._storage_path = Path(config.storage_path)
        # Create client with configured base URL (supports remote Ollama servers)
        self._client = ollama.Client(host=config.base_url)
        self._load_index()

    # =========================================================================
    # Embedding Generation
    # =========================================================================

    def _get_embedding_sync(self, text: str) -> List[float]:
        """Synchronous embedding via ollama-python with configured base URL."""
        # Extract model name without provider prefix
        model = self.config.embedding_model
        if model.startswith("ollama/"):
            model = model[7:]

        try:
            response = self._client.embed(model=model, input=text)
            return list(response.embeddings[0])
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise RuntimeError(f"Embedding failed: {e}")

    async def _get_embedding_async(self, text: str) -> List[float]:
        """Async embedding via ollama-python with configured base URL."""
        import ollama

        model = self.config.embedding_model
        if model.startswith("ollama/"):
            model = model[7:]

        try:
            # Create async client with same base URL
            client = ollama.AsyncClient(host=self.config.base_url)
            response = await client.embed(model=model, input=text)
            return list(response.embeddings[0])
        except Exception as e:
            logger.error(f"Async embedding failed: {e}")
            raise RuntimeError(f"Async embedding failed: {e}")

    # =========================================================================
    # Text Chunking
    # =========================================================================

    def _chunk_text(self, text: str) -> List[str]:
        """
        Intelligent chunking with sentence awareness.
        Tries to break at sentence boundaries when possible.
        """
        if not text:
            return []

        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        # Simple sentence-aware chunking
        sentences = text.replace("\n", " ").split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Add period back if it was removed
            if not sentence.endswith("."):
                sentence += "."

            potential = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential) <= chunk_size:
                current_chunk = potential
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap by including end of previous chunk at start of next
        if overlap > 0 and len(chunks) > 1:
            overlapped = [chunks[0]]
            for i in range(1, len(chunks)):
                prev_end = (
                    chunks[i - 1][-overlap:]
                    if len(chunks[i - 1]) > overlap
                    else chunks[i - 1]
                )
                overlapped.append(prev_end + " " + chunks[i])
            chunks = overlapped

        return chunks

    # =========================================================================
    # File Reading
    # =========================================================================

    def _read_pdf(self, file_path: str) -> str:
        """Read PDF file using pypdf."""
        try:
            import pypdf

            reader = pypdf.PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(extracted)
            return "\n".join(text_parts)
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")
        except Exception as e:
            raise RuntimeError(f"PDF read error: {e}")

    def _read_html(self, file_path: str) -> str:
        """Read HTML file and extract text."""
        try:
            from web_scraper_toolkit.utils.markdown import html_to_markdown

            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            return html_to_markdown(html)
        except ImportError:
            # Fallback to basic HTML stripping
            import re

            with open(file_path, "r", encoding="utf-8") as f:
                html = f.read()
            # Strip tags
            text = re.sub(r"<[^>]+>", " ", html)
            # Clean whitespace
            text = re.sub(r"\s+", " ", text)
            return text.strip()
        except Exception as e:
            raise RuntimeError(f"HTML read error: {e}")

    def _read_file(self, file_path: str) -> str:
        """Read file based on extension."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self._read_pdf(file_path)
        elif suffix in [".html", ".htm"]:
            return self._read_html(file_path)
        else:
            # Plain text
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    # =========================================================================
    # Ingestion (Sync + Async)
    # =========================================================================

    def ingest_text(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Synchronous text ingestion."""
        if not text:
            return "Empty text provided."

        chunks = self._chunk_text(text)
        count = 0

        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue
            try:
                vec = self._get_embedding_sync(chunk)
                self.memory.append(
                    {"text": chunk, "vector": vec, "metadata": metadata or {}}
                )
                count += 1
            except Exception as e:
                logger.warning(f"Chunk embedding failed: {e}")

        self._save_index()
        return f"Ingested {count} chunks."

    async def aingest_text(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Async text ingestion."""
        if not text:
            return "Empty text provided."

        chunks = self._chunk_text(text)
        count = 0

        for chunk in chunks:
            if len(chunk.strip()) < 10:
                continue
            try:
                vec = await self._get_embedding_async(chunk)
                self.memory.append(
                    {"text": chunk, "vector": vec, "metadata": metadata or {}}
                )
                count += 1
            except Exception as e:
                logger.warning(f"Async chunk embedding failed: {e}")

        self._save_index()
        return f"Ingested {count} chunks."

    def ingest_file(self, file_path: str) -> str:
        """Synchronous file ingestion."""
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        try:
            content = self._read_file(file_path)
            return self.ingest_text(content, metadata={"source": str(path.resolve())})
        except Exception as e:
            return f"Error: {e}"

    async def aingest_file(self, file_path: str) -> str:
        """Async file ingestion."""
        path = Path(file_path)
        if not path.exists():
            return f"File not found: {file_path}"

        try:
            content = self._read_file(file_path)
            return await self.aingest_text(
                content, metadata={"source": str(path.resolve())}
            )
        except Exception as e:
            return f"Error: {e}"

    # =========================================================================
    # Search (Sync + Async)
    # =========================================================================

    def _cosine_similarity(self, a: Sequence[float], b: Sequence[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a_np = np.array(a)
        b_np = np.array(b)
        norm_a = np.linalg.norm(a_np)
        norm_b = np.linalg.norm(b_np)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a_np, b_np) / (norm_a * norm_b))

    def search_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        """Synchronous semantic search."""
        if not self.memory:
            return []

        query_vec = self._get_embedding_sync(query)
        results = []

        for item in self.memory:
            sim = self._cosine_similarity(query_vec, item["vector"])
            results.append(
                {"text": item["text"], "similarity": sim, "metadata": item["metadata"]}
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    async def asearch_memory(self, query: str, top_k: int = 3) -> List[Dict]:
        """Async semantic search."""
        if not self.memory:
            return []

        query_vec = await self._get_embedding_async(query)
        results = []

        for item in self.memory:
            sim = self._cosine_similarity(query_vec, item["vector"])
            results.append(
                {"text": item["text"], "similarity": sim, "metadata": item["metadata"]}
            )

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    # =========================================================================
    # Persistence
    # =========================================================================

    def _save_index(self):
        """Save memory to disk with backup."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Backup existing if present
            if self._storage_path.exists():
                backup = self._storage_path.with_suffix(".json.bak")
                backup.write_text(self._storage_path.read_text())

            with open(self._storage_path, "w", encoding="utf-8") as f:
                json.dump(self.memory, f, indent=2)

            logger.debug(f"Saved {len(self.memory)} items to {self._storage_path}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def _load_index(self):
        """Load memory from disk."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                logger.info(
                    f"Loaded {len(self.memory)} items from {self._storage_path}"
                )
            except Exception as e:
                logger.warning(f"Failed to load index: {e}")
                self.memory = []
        else:
            self.memory = []

    def clear_memory(self):
        """Clear all stored vectors."""
        self.memory = []
        self._save_index()
        return "Memory cleared."
