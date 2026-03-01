# ./ollamatoolkit/agents/__init__.py
"""
Ollama Toolkit - Agent Components
=================================
High-level agents for conversational AI with tool usage.

Main Components:
- SimpleAgent: Core agent with tool execution and streaming
- RoleAgent: Personality-driven agents loaded from JSON profiles
- AgentTeam: Multi-agent orchestration with various strategies
- ConversationMemory: Intelligent history management with summarization

Usage:
    from ollamatoolkit.agents import SimpleAgent, AgentTeam

    # Create a simple agent
    agent = SimpleAgent(
        name="assistant",
        system_message="You are a helpful assistant.",
        model_config={"model": "ollama/qwen3:8b"}
    )

    # Run with tool support
    response = agent.run("What's 2+2?")
"""

from .simple import SimpleAgent, AgentHooks
from .role import RoleAgent
from .team import AgentTeam, AgentRole, TeamStrategy, TeamResult
from .memory import ConversationMemory, MemoryConfig

__all__ = [
    # Core Agent
    "SimpleAgent",
    "AgentHooks",
    # Role-based Agent
    "RoleAgent",
    # Multi-Agent Orchestration
    "AgentTeam",
    "AgentRole",
    "TeamStrategy",
    "TeamResult",
    # Memory Management
    "ConversationMemory",
    "MemoryConfig",
]
