# ./ollamatoolkit/config/core.py
"""
Ollama Toolkit - Core Configuration
===================================
Dataclass-based configuration with JSON loading and validation.

Usage:
    config = ToolkitConfig.load("config.json")
    print(config.agent.model)
    print(config.models.vision.primary)
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Error Classes
# =============================================================================


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


class ModelNotFoundError(ConfigError):
    """Requested model not found in configuration."""

    pass


# =============================================================================
# Base Config Classes
# =============================================================================


@dataclass
class AgentConfig:
    """Core agent/LLM configuration."""

    model: str = "ollama/ministral-3:latest"
    temperature: float = 0.0
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    caching: bool = False
    fallbacks: List[str] = field(default_factory=list)
    history_limit: int = 50
    strict_mode: bool = False
    max_tokens: int = 4096


@dataclass
class ToolConfig:
    """Tool permissions and paths."""

    root_dir: str = "./"
    read_only: bool = False
    allowed_tools: List[str] = field(
        default_factory=lambda: [
            "files",
            "math",
            "server",
            "db",
            "system",
            "web",
            "vision",
            "vector",
            "system_health",
            "models",
        ]
    )
    custom_tools: List[str] = field(default_factory=list)
    mcp_servers: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class WebToolConfig:
    """Web scraping configuration."""

    user_agent: str = "OllamaToolkit/1.0 (Research)"
    timeout: int = 30
    verify_ssl: bool = True
    proxies: Optional[Dict[str, str]] = None


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    file_path: Optional[str] = "agent.log"
    verbose: bool = False


@dataclass
class DashboardConfig:
    """Dashboard display configuration."""

    enabled: bool = False
    refresh_rate_ms: int = 100
    show_system: bool = True
    show_tools: bool = True
    show_context_usage: bool = True


@dataclass
class VisionConfig:
    """Vision model configuration."""

    model: str = "ollama/ministral-3:latest"
    temperature: float = 0.2
    max_tokens: int = 1024
    meta_map: Dict[str, str] = field(default_factory=dict)
    video_sample_interval: int = 2
    enable_grounding: bool = True
    use_chat_api: bool = True


@dataclass
class VectorConfig:
    """Vector/embedding configuration."""

    embedding_model: str = "ollama/nomic-embed-text"
    base_url: str = "http://localhost:11434"
    chunk_size: int = 500
    chunk_overlap: int = 50
    storage_path: str = "./vector_store.json"


@dataclass
class MemoryConfig:
    """Conversation memory configuration."""

    max_messages: int = 100
    summarize_threshold: int = 80
    storage_path: Optional[str] = None
    auto_save: bool = False


@dataclass
class DocumentConfig:
    """Document processing configuration."""

    default_vision_model: Optional[str] = None  # Auto-select if None
    dpi: int = 150
    min_text_chars: int = 100
    clean_text: bool = True
    save_images: bool = False


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    unload_wait_time: float = 3.0
    max_unload_wait: float = 30.0
    unload_check_interval: float = 2.0


@dataclass
class TelemetryConfig:
    """Telemetry/logging configuration."""

    enabled: bool = True
    session_dir: str = "./logs/sessions"
    max_sessions: int = 100


# =============================================================================
# Model Configuration (NEW)
# =============================================================================


@dataclass
class ModelSlotConfig:
    """Configuration for a specific model slot (capability)."""

    primary: str = ""
    fallbacks: List[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 4096
    context_length: Optional[int] = None  # Auto-detect from model

    def get_model(self, fallback_index: int = -1) -> str:
        """Get model name, optionally from fallback list."""
        if fallback_index < 0:
            return self.primary
        if fallback_index < len(self.fallbacks):
            return self.fallbacks[fallback_index]
        return self.primary


@dataclass
class ModelsConfig:
    """Per-capability model configuration."""

    completion: ModelSlotConfig = field(
        default_factory=lambda: ModelSlotConfig(
            primary="ministral-3:latest",
            fallbacks=["llama3.1:latest"],
            temperature=0.0,
            max_tokens=4096,
        )
    )
    vision: ModelSlotConfig = field(
        default_factory=lambda: ModelSlotConfig(
            primary="ministral-3:latest",
            fallbacks=["devstral-small-2:latest"],
            temperature=0.2,
            max_tokens=1024,
        )
    )
    embedding: ModelSlotConfig = field(
        default_factory=lambda: ModelSlotConfig(
            primary="qwen3-embedding:4b",
            fallbacks=["mxbai-embed-large:latest", "nomic-embed-text-v2-moe:latest"],
            temperature=0.0,
            max_tokens=512,
        )
    )
    reasoning: ModelSlotConfig = field(
        default_factory=lambda: ModelSlotConfig(
            primary="qwen3:14b",
            fallbacks=["qwen3:8b", "gpt-oss:20b"],
            temperature=0.7,
            max_tokens=8192,
        )
    )
    tools: ModelSlotConfig = field(
        default_factory=lambda: ModelSlotConfig(
            primary="ministral-3:latest",
            fallbacks=["llama3.1:latest"],
            temperature=0.0,
            max_tokens=4096,
        )
    )

    def get_slot(self, capability: str) -> Optional[ModelSlotConfig]:
        """Get model slot by capability name."""
        return getattr(self, capability, None)


# =============================================================================
# Root Configuration
# =============================================================================


@dataclass
class ToolkitConfig:
    """Root configuration object."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    web: WebToolConfig = field(default_factory=WebToolConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    document: DocumentConfig = field(default_factory=DocumentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)

    @classmethod
    def load(cls, config_path: Union[str, Path, None] = None) -> "ToolkitConfig":
        """
        Load configuration from JSON file with proper error handling.

        Args:
            config_path: Path to config.json

        Returns:
            Loaded ToolkitConfig instance

        Raises:
            ConfigError: If config file exists but is invalid
        """
        cfg = cls()

        if config_path:
            path = Path(config_path)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    cfg = cls._load_from_dict(data)
                    logger.info(f"Loaded config from {path}")
                except json.JSONDecodeError as e:
                    raise ConfigError(f"Invalid JSON in {path}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to load config from {path}: {e}")

        return cfg

    @classmethod
    def _load_from_dict(cls, data: Dict[str, Any]) -> "ToolkitConfig":
        """Load config from a dictionary."""
        cfg = cls()

        # Helper to safely load dataclass
        def load_dataclass(dc_class, key):
            if key in data and isinstance(data[key], dict):
                valid_keys = {k for k in dc_class.__annotations__}
                filtered = {k: v for k, v in data[key].items() if k in valid_keys}
                return dc_class(**filtered)
            return dc_class()

        cfg.agent = load_dataclass(AgentConfig, "agent")
        cfg.tools = load_dataclass(ToolConfig, "tools")
        cfg.web = load_dataclass(WebToolConfig, "web")
        cfg.logging = load_dataclass(LoggingConfig, "logging")
        cfg.dashboard = load_dataclass(DashboardConfig, "dashboard")
        cfg.vision = load_dataclass(VisionConfig, "vision")
        cfg.vector = load_dataclass(VectorConfig, "vector")
        cfg.memory = load_dataclass(MemoryConfig, "memory")
        cfg.document = load_dataclass(DocumentConfig, "document")
        cfg.benchmark = load_dataclass(BenchmarkConfig, "benchmark")
        cfg.telemetry = load_dataclass(TelemetryConfig, "telemetry")

        # Load models config (nested)
        if "models" in data and isinstance(data["models"], dict):
            models_data = data["models"]
            cfg.models = ModelsConfig()

            for slot_name in [
                "completion",
                "vision",
                "embedding",
                "reasoning",
                "tools",
            ]:
                if slot_name in models_data:
                    slot_data = models_data[slot_name]
                    setattr(
                        cfg.models,
                        slot_name,
                        ModelSlotConfig(
                            **{
                                k: v
                                for k, v in slot_data.items()
                                if k in ModelSlotConfig.__annotations__
                            }
                        ),
                    )

        return cfg

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        logger.info(f"Saved config to {path}")

    def get_model_for_task(self, task_type: str) -> str:
        """
        Get the appropriate model for a task type.

        Args:
            task_type: One of "completion", "vision", "embedding", "reasoning", "tools"

        Returns:
            Model name string
        """
        slot = self.models.get_slot(task_type)
        if slot:
            return slot.primary
        return self.agent.model

    def get_temperature_for_task(self, task_type: str) -> float:
        """Get recommended temperature for task type."""
        slot = self.models.get_slot(task_type)
        if slot:
            return slot.temperature
        return self.agent.temperature

    @classmethod
    def generate_sample(cls) -> "ToolkitConfig":
        """Generate a sample configuration with all defaults."""
        return cls()
