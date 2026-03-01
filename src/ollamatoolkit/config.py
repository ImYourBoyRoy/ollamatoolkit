# ./ollamatoolkit/config.py
"""
Ollama Toolkit - Configuration
==============================
Centralized configuration management for the toolkit.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class AgentConfig:
    model: str = "ollama/mistral"
    temperature: float = 0.0
    base_url: str = "http://localhost:11434"
    api_key: Optional[str] = None
    caching: bool = False
    fallbacks: List[str] = field(default_factory=list)
    history_limit: int = 50
    strict_mode: bool = False  # If True, fails if critical capabilities are missing
    auto_pull: bool = True  # If True, automatically pull missing models


@dataclass
class ToolConfig:
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
            "system_health",  # Added system_health
        ]
    )
    custom_tools: List[str] = field(
        default_factory=list
    )  # e.g. ["my_lib.my_func", "webscrapertoolkit.scrape"]
    mcp_servers: Dict[str, Dict] = field(
        default_factory=dict
    )  # {"name": {"command": "...", "args": []}}


@dataclass
class WebToolConfig:
    user_agent: str = "OllamaToolkit/1.0 (Research)"
    timeout: int = 15
    verify_ssl: bool = True
    proxies: Optional[Dict[str, str]] = None


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_path: Optional[str] = "agent.log"
    verbose: bool = False


@dataclass
class DashboardConfig:
    enabled: bool = False
    refresh_rate_ms: int = 100
    show_system: bool = True
    show_tools: bool = True
    show_context_usage: bool = True


@dataclass
class VisionConfig:
    model: str = "ollama/qwen3vl"  # Updated to modern default (User refers to qwen3vl, 2.5 is current stable VL)
    temperature: float = 0.2
    max_tokens: int = 1024
    meta_map: Dict[str, str] = field(
        default_factory=dict
    )  # e.g. {"date": "creation_date"}
    video_sample_interval: int = 2  # Seconds between frames
    enable_grounding: bool = True  # Enable coordinate parsing
    use_chat_api: bool = True  # Prefer /api/chat over /api/generate for vision


@dataclass
class VectorConfig:
    embedding_model: str = "ollama/nomic-embed-text"
    chunk_size: int = 500
    chunk_overlap: int = 50
    storage_path: str = "./vector_store.json"


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

    @classmethod
    def load(cls, config_path: Union[str, Path, None] = None) -> "ToolkitConfig":
        """
        Loads configuration from a JSON file, environment variables, or defaults.
        Priority: Config File > Defaults.
        (CLI args should override this object after loading).
        """
        # Start with defaults
        cfg = cls()

        if config_path:
            path = Path(config_path)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))

                    # Update Agent Config
                    if "agent" in data:
                        cfg.agent = AgentConfig(
                            **{
                                k: v
                                for k, v in data["agent"].items()
                                if k in AgentConfig.__annotations__
                            }
                        )

                    # Update Tool Config
                    if "tools" in data:
                        cfg.tools = ToolConfig(
                            **{
                                k: v
                                for k, v in data["tools"].items()
                                if k in ToolConfig.__annotations__
                            }
                        )

                    # Update Web Config
                    if "web" in data:
                        cfg.web = WebToolConfig(
                            **{
                                k: v
                                for k, v in data["web"].items()
                                if k in WebToolConfig.__annotations__
                            }
                        )

                    # Update Logging Config
                    if "logging" in data:
                        cfg.logging = LoggingConfig(
                            **{
                                k: v
                                for k, v in data["logging"].items()
                                if k in LoggingConfig.__annotations__
                            }
                        )

                    # Update Dashboard Config
                    if "dashboard" in data:
                        cfg.dashboard = DashboardConfig(
                            **{
                                k: v
                                for k, v in data["dashboard"].items()
                                if k in DashboardConfig.__annotations__
                            }
                        )

                    # Update Vision Config
                    if "vision" in data:
                        cfg.vision = VisionConfig(
                            **{
                                k: v
                                for k, v in data["vision"].items()
                                if k in VisionConfig.__annotations__
                            }
                        )

                    # Update Vector Config
                    if "vector" in data:
                        cfg.vector = VectorConfig(
                            **{
                                k: v
                                for k, v in data["vector"].items()
                                if k in VectorConfig.__annotations__
                            }
                        )

                except Exception as e:
                    print(f"Warning: Failed to load config from {path}: {e}")

        return cfg

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def generate_sample(cls) -> "ToolkitConfig":
        """Returns a configuration object populated with all default values."""
        return cls()

    def save_sample(self, path: str = "sample_config.json"):
        """Dumps current (or default) config to a file."""
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def create_default_sample_file(cls, path: str = "sample_config.json"):
        """Creates a sample config file with all defaults."""
        cls().save_sample(path)
