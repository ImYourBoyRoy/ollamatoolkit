# ./src/ollamatoolkit/telemetry.py
"""
Ollama Toolkit - Telemetry Integration
======================================
Production-grade observability using llm-telemetry-toolkit.

Usage:
    from ollamatoolkit.telemetry import get_logger, log_interaction

    logger = get_logger("my_session")
    logger.log(LLMInteraction(prompt="...", response="..."))

Exports:
    - LLMLogger: Session-based logging
    - LLMInteraction: Structured interaction schema
    - TelemetryConfig: Configuration options
    - SessionContext: Context manager for grouped operations
    - monitor_interaction: Decorator for automatic logging
    - get_logger: Factory function for creating loggers
"""

import logging
from datetime import datetime, timezone
from typing import Optional

# Strict import - fails fast if not installed
from llm_telemetry_toolkit import (
    LLMLogger,
    LLMInteraction,
    TelemetryConfig,
    SessionContext,
    monitor_interaction,
)

logger = logging.getLogger(__name__)

# Module-level default logger instance
_default_logger: Optional[LLMLogger] = None
_default_config: Optional[TelemetryConfig] = None


def get_logger(session_id: Optional[str] = None) -> LLMLogger:
    """
    Get or create an LLMLogger instance.

    Args:
        session_id: Optional session identifier. If None, uses "ollamatoolkit".

    Returns:
        Configured LLMLogger instance.
    """
    global _default_logger, _default_config

    sid = session_id or "ollamatoolkit"

    # Create config with session_id
    config = TelemetryConfig(session_id=sid)

    if session_id:
        # Return a new logger for specific session
        return LLMLogger(config)

    # Return cached default logger
    if _default_logger is None:
        _default_config = TelemetryConfig(session_id="ollamatoolkit")
        _default_logger = LLMLogger(_default_config)

    return _default_logger


def log_interaction(
    prompt: str,
    response: str,
    model: str = "unknown",
    session_id: Optional[str] = None,
    response_time_seconds: float = 0.0,
    **metadata,
):
    """
    Convenience function to log a single interaction.

    Args:
        prompt: The input prompt
        response: The model response
        model: Model identifier
        session_id: Optional session ID
        **metadata: Additional metadata to log
    """
    sid = session_id or "ollamatoolkit"
    log = get_logger(sid)

    interaction = LLMInteraction(
        session_id=sid,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        model_name=model,
        prompt=prompt,
        response=response,
        response_time_seconds=float(response_time_seconds),
        metadata=metadata or {},
    )

    log.log(interaction)


# Re-export for convenience
__all__ = [
    "LLMLogger",
    "LLMInteraction",
    "TelemetryConfig",
    "SessionContext",
    "monitor_interaction",
    "get_logger",
    "log_interaction",
]
