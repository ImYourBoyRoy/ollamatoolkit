# ./ollamatoolkit/tools/cache.py
"""
OllamaToolkit - Cache Tools
===========================
Local caching for LLM responses and embeddings to improve performance.

Enables:
- Response caching to avoid redundant LLM calls
- Embedding caching for vector operations
- TTL-based expiration
- Disk persistence

Usage:
    from ollamatoolkit.tools.cache import CacheTools

    cache = CacheTools(cache_dir="./cache")

    # Cache LLM responses
    key = cache.make_key(prompt="Hello", model="llama3")
    if cached := cache.get(key):
        response = cached
    else:
        response = llm.generate(...)
        cache.set(key, response, ttl=3600)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    expires_at: Optional[float] = None
    hits: int = 0
    size_bytes: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "hits": self.hits,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CacheEntry":
        """Deserialize from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            created_at=data["created_at"],
            expires_at=data.get("expires_at"),
            hits=data.get("hits", 0),
        )


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int
    total_size_bytes: int
    hits: int
    misses: int
    expired_evictions: int

    @property
    def hit_rate(self) -> float:
        """Calculate hit rate percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0


class CacheTools:
    """
    Local caching for LLM responses and embeddings.

    Supports both in-memory and disk-based caching with TTL expiration.
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_entries: int = 1000,
        default_ttl: Optional[int] = None,
        persist: bool = True,
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for disk persistence (None = memory only)
            max_entries: Maximum number of entries to keep
            default_ttl: Default TTL in seconds (None = no expiration)
            persist: Whether to persist to disk
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.persist = persist and cache_dir is not None

        self._cache: Dict[str, CacheEntry] = {}
        self._stats = CacheStats(
            total_entries=0,
            total_size_bytes=0,
            hits=0,
            misses=0,
            expired_evictions=0,
        )

        # Load from disk if available
        if self.persist:
            self._load_from_disk()

    def make_key(self, **kwargs) -> str:
        """
        Generate a cache key from keyword arguments.

        Args:
            **kwargs: Key-value pairs to hash (prompt, model, temperature, etc.)

        Returns:
            SHA256 hash key
        """
        # Sort keys for consistent hashing
        sorted_items = sorted(kwargs.items())
        key_str = json.dumps(sorted_items, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired:
            self._evict(key)
            self._stats.misses += 1
            self._stats.expired_evictions += 1
            return None

        entry.hits += 1
        self._stats.hits += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (None uses default)
        """
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = (time.time() + ttl) if ttl else None

        # Calculate size
        try:
            size_bytes = len(json.dumps(value, default=str).encode())
        except Exception:
            size_bytes = 0

        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            expires_at=expires_at,
            size_bytes=size_bytes,
        )

        # Evict if at capacity
        if len(self._cache) >= self.max_entries:
            self._evict_oldest()

        self._cache[key] = entry
        self._stats.total_entries = len(self._cache)
        self._stats.total_size_bytes += size_bytes

        # Persist to disk
        if self.persist:
            self._save_entry_to_disk(entry)

    def delete(self, key: str) -> bool:
        """
        Delete an entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        if key in self._cache:
            self._evict(key)
            return True
        return False

    def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        self._stats.total_entries = 0
        self._stats.total_size_bytes = 0

        if self.persist and self.cache_dir:
            for f in self.cache_dir.glob("*.cache"):
                f.unlink()

        logger.info(f"Cleared {count} cache entries")
        return count

    def cleanup(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        expired_keys = [k for k, v in self._cache.items() if v.is_expired]
        for key in expired_keys:
            self._evict(key)
            self._stats.expired_evictions += 1
        return len(expired_keys)

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._stats.total_entries = len(self._cache)
        return self._stats

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        Get from cache or compute and cache the value.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: TTL in seconds

        Returns:
            Cached or computed value
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        value = compute_fn()
        self.set(key, value, ttl=ttl)
        return value

    # -------------------------------------------------------------------------
    # Embedding-specific methods
    # -------------------------------------------------------------------------

    def cache_embedding(
        self,
        text: str,
        model: str,
        embedding: List[float],
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache an embedding vector.

        Args:
            text: Source text
            model: Embedding model used
            embedding: Embedding vector
            ttl: TTL in seconds
        """
        key = self.make_key(text=text, model=model, type="embedding")
        self.set(key, embedding, ttl=ttl)

    def get_embedding(
        self,
        text: str,
        model: str,
    ) -> Optional[List[float]]:
        """
        Get cached embedding.

        Args:
            text: Source text
            model: Embedding model

        Returns:
            Cached embedding or None
        """
        key = self.make_key(text=text, model=model, type="embedding")
        return self.get(key)

    # -------------------------------------------------------------------------
    # LLM Response caching
    # -------------------------------------------------------------------------

    def cache_response(
        self,
        prompt: str,
        model: str,
        response: str,
        temperature: float = 0.0,
        ttl: Optional[int] = None,
    ) -> None:
        """
        Cache an LLM response.

        Args:
            prompt: Input prompt
            model: Model used
            response: LLM response
            temperature: Temperature used (only cache temp=0 by default)
            ttl: TTL in seconds
        """
        # Only cache deterministic responses by default
        if temperature > 0:
            logger.debug("Not caching non-deterministic response (temp > 0)")
            return

        key = self.make_key(prompt=prompt, model=model, type="response")
        self.set(key, response, ttl=ttl)

    def get_response(
        self,
        prompt: str,
        model: str,
    ) -> Optional[str]:
        """
        Get cached LLM response.

        Args:
            prompt: Input prompt
            model: Model

        Returns:
            Cached response or None
        """
        key = self.make_key(prompt=prompt, model=model, type="response")
        return self.get(key)

    # -------------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------------

    def _evict(self, key: str) -> None:
        """Remove an entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._stats.total_size_bytes -= entry.size_bytes

            if self.persist and self.cache_dir:
                cache_file = self.cache_dir / f"{key}.cache"
                if cache_file.exists():
                    cache_file.unlink()

    def _evict_oldest(self) -> None:
        """Evict the oldest entry."""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        self._evict(oldest_key)

    def _save_entry_to_disk(self, entry: CacheEntry) -> None:
        """Save a single entry to disk."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{entry.key}.cache"

        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(entry.to_dict(), f)
        except Exception as e:
            logger.warning(f"Failed to persist cache entry: {e}")

    def _load_from_disk(self) -> None:
        """Load all entries from disk."""
        if not self.cache_dir or not self.cache_dir.exists():
            return

        loaded = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    entry = CacheEntry.from_dict(data)

                    # Skip expired entries
                    if entry.is_expired:
                        cache_file.unlink()
                        continue

                    self._cache[entry.key] = entry
                    loaded += 1
            except Exception as e:
                logger.warning(f"Failed to load cache entry {cache_file}: {e}")

        if loaded > 0:
            logger.info(f"Loaded {loaded} entries from cache")
