# ./ollamatoolkit/agents/role.py
"""
Ollama Toolkit - Role Agent
===========================
Defines an agent based on a structured JSON role definition.
"""

from pathlib import Path
from typing import Callable, Dict, Optional, Union

from .simple import SimpleAgent
from ..common.utils import load_and_parse_json


class RoleAgent(SimpleAgent):
    """
    A SimpleAgent initialized from a "Persona" or "Role" definition.
    """

    @classmethod
    def from_role_file(
        cls,
        role_path: Union[str, Path],
        model_config: Dict,
        tool_registry: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ) -> "RoleAgent":
        """
        Loads a role definition from a JSON file and returns an agent.

        JSON Schema expected:
        {
            "role": "Research Specialist",
            "goal": "Find deep information...",
            "instructions": ["Search X", "Verify Y"],
            "tools": ["search_tool", "scrape_tool"]
        }
        """
        path = Path(role_path)
        data = load_and_parse_json(path)
        if not data:
            raise ValueError(f"Could not load role from {path}")

        role_title = data.get("role", "Assistant")
        goal = data.get("goal", "Help the user.")
        instructions = data.get("instructions", [])

        # specific_model_config = data.get("model_config", {})
        # (Could merge with passed model_config if desired)

        # Build System Message
        system_message = f"""
        ROLE: {role_title}
        GOAL: {goal}
        
        INSTRUCTIONS:
        """
        for i, instr in enumerate(instructions, 1):
            system_message += f"\n{i}. {instr}"

        # Resolve Tools

        requested_tools = data.get("tools", [])
        if requested_tools and tool_registry:
            for tool_name in requested_tools:
                if tool_name in tool_registry:
                    # func = tool_registry[tool_name]
                    # We need to generate schema for it.
                    # SimpleAgent can generate schema if we use register_tool logic,
                    # but here we want to pass `tools` list to init.
                    # We can use the internal schema generator from SimpleAgent (static/class method refactor?)
                    # OR just instantiate the agent and register them after.
                    pass
                else:
                    print(
                        f"Warning: Tool '{tool_name}' requested by role but not found in registry."
                    )

        # Initialize
        agent = cls(
            name=role_title,
            system_message=system_message.strip(),
            model_config=model_config,
            **kwargs,
        )

        # Register tools now that we have the agent instance
        if requested_tools and tool_registry:
            for tool_name in requested_tools:
                if tool_name in tool_registry:
                    agent.register_tool(tool_registry[tool_name])

        return agent
