# ./ollamatoolkit/agents/team.py
"""
Ollama Toolkit - Multi-Agent Team Orchestration
================================================
Coordinate multiple agents for complex workflows.

This module provides:
- AgentTeam: Orchestrates multiple agents with different strategies
- TeamStrategy: Defines how agents collaborate (supervisor, round-robin, parallel)
- AgentRole: Defines capabilities and priority for each team member

Usage:
    from ollamatoolkit.agents.team import AgentTeam, AgentRole, TeamStrategy
    from ollamatoolkit.agents.simple import SimpleAgent

    # Create agents
    researcher = SimpleAgent("researcher", "You research topics.", model_config)
    writer = SimpleAgent("writer", "You write content.", model_config)

    # Define roles
    research_role = AgentRole(
        name="researcher",
        agent=researcher,
        capabilities=["research", "fact-checking"],
    )

    # Create team with supervisor
    supervisor = SimpleAgent("supervisor", "You coordinate the team.", model_config)
    team = AgentTeam("content_team", strategy=TeamStrategy.SUPERVISOR, supervisor=supervisor)
    team.add_agent(research_role)

    # Run task
    result = team.run("Write an article about AI trends")

Strategies:
    - SUPERVISOR: One agent delegates tasks to specialists
    - ROUND_ROBIN: Agents take turns responding
    - PARALLEL: All agents work simultaneously, results merged
    - SEQUENTIAL: Agents process in order, passing context
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class TeamStrategy(Enum):
    """Defines how agents collaborate within a team."""

    SUPERVISOR = "supervisor"
    """One agent (supervisor) delegates tasks to specialists."""

    ROUND_ROBIN = "round_robin"
    """Agents take turns responding to the conversation."""

    PARALLEL = "parallel"
    """All agents work simultaneously, results are merged."""

    SEQUENTIAL = "sequential"
    """Agents process in order, each passing context to the next."""


@dataclass
class AgentRole:
    """
    Defines an agent's role within the team.

    Attributes:
        name: Unique identifier for this agent in the team
        agent: The SimpleAgent instance
        capabilities: List of skills/tasks this agent can handle
        priority: Higher priority agents are preferred (default: 0)
        description: Human-readable description of the agent's role
    """

    name: str
    agent: Any  # SimpleAgent but avoiding circular import
    capabilities: List[str] = field(default_factory=list)
    priority: int = 0
    description: str = ""

    def can_handle(self, task_keywords: List[str]) -> bool:
        """Check if this agent can handle a task based on keywords."""
        if not self.capabilities:
            return True  # No specific capabilities = general purpose

        task_lower = [k.lower() for k in task_keywords]
        cap_lower = [c.lower() for c in self.capabilities]

        return any(cap in task for cap in cap_lower for task in task_lower)


@dataclass
class TeamResult:
    """Result from a team execution."""

    final_response: str
    """The final consolidated response."""

    agent_responses: Dict[str, str]
    """Responses from each agent that contributed."""

    turns_used: int
    """Number of conversation turns used."""

    strategy: TeamStrategy
    """Strategy that was used."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the execution."""


class AgentTeam:
    """
    Orchestrates multiple agents working together.

    The team can operate in different strategies:
    - SUPERVISOR: A supervisor agent delegates to specialists
    - ROUND_ROBIN: Agents take turns in a round-robin fashion
    - PARALLEL: All agents work simultaneously
    - SEQUENTIAL: Agents work in sequence, passing context
    """

    def __init__(
        self,
        name: str,
        strategy: TeamStrategy = TeamStrategy.SUPERVISOR,
        supervisor: Optional[Any] = None,
        max_rounds: int = 10,
        merge_strategy: Optional[Callable[[List[str]], str]] = None,
    ):
        """
        Initialize an agent team.

        Args:
            name: Name of the team
            strategy: Collaboration strategy
            supervisor: Supervisor agent (required for SUPERVISOR strategy)
            max_rounds: Maximum conversation rounds
            merge_strategy: Function to merge parallel results (default: concatenate)
        """
        self.name = name
        self.strategy = strategy
        self.supervisor = supervisor
        self.max_rounds = max_rounds
        self.merge_strategy = merge_strategy or self._default_merge
        self.agents: Dict[str, AgentRole] = {}

    def add_agent(self, role: AgentRole) -> "AgentTeam":
        """
        Add an agent to the team.

        Args:
            role: AgentRole defining the agent and its capabilities

        Returns:
            Self for chaining
        """
        self.agents[role.name] = role
        logger.info(f"Added agent '{role.name}' to team '{self.name}'")
        return self

    def remove_agent(self, name: str) -> bool:
        """Remove an agent from the team by name."""
        if name in self.agents:
            del self.agents[name]
            return True
        return False

    def get_agent(self, name: str) -> Optional[AgentRole]:
        """Get an agent role by name."""
        return self.agents.get(name)

    def list_agents(self) -> List[str]:
        """List all agent names in the team."""
        return list(self.agents.keys())

    def run(self, task: str, **kwargs) -> TeamResult:
        """
        Execute the team on a task.

        Args:
            task: The task/question to process
            **kwargs: Additional arguments passed to strategy

        Returns:
            TeamResult with final response and metadata
        """
        logger.info(f"Team '{self.name}' starting task: {task[:100]}...")

        if self.strategy == TeamStrategy.SUPERVISOR:
            return self._run_supervisor(task, **kwargs)
        elif self.strategy == TeamStrategy.ROUND_ROBIN:
            return self._run_round_robin(task, **kwargs)
        elif self.strategy == TeamStrategy.PARALLEL:
            return self._run_parallel(task, **kwargs)
        elif self.strategy == TeamStrategy.SEQUENTIAL:
            return self._run_sequential(task, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _run_supervisor(self, task: str, **kwargs) -> TeamResult:
        """
        Supervisor delegates to specialists.

        The supervisor decides which agent should handle each sub-task
        and synthesizes the final response.
        """
        if not self.supervisor:
            raise ValueError("Supervisor strategy requires a supervisor agent")

        agent_responses = {}
        turns = 0

        # Build agent info for supervisor
        agent_info = "\n".join(
            [
                f"- {name}: {role.description or ', '.join(role.capabilities) or 'general assistant'}"
                for name, role in self.agents.items()
            ]
        )

        # Initial delegation prompt
        delegation_prompt = f"""You are coordinating a team of specialists. Your job is to:
1. Analyze the task
2. Delegate to the appropriate specialist(s)
3. Synthesize their responses into a final answer

Available specialists:
{agent_info}

Instructions:
- To delegate, respond with: DELEGATE:<agent_name>|<sub_task>
- To provide final answer after gathering info: FINAL:<your_synthesized_answer>
- You can delegate multiple times before providing final answer

Task: {task}

What would you like to do first?"""

        self.supervisor.history.append({"role": "user", "content": delegation_prompt})

        for round_num in range(self.max_rounds):
            turns += 1
            response = self.supervisor.run("")  # Continue conversation

            if response.startswith("DELEGATE:"):
                # Parse delegation
                try:
                    _, rest = response.split(":", 1)
                    if "|" in rest:
                        agent_name, sub_task = rest.split("|", 1)
                    else:
                        agent_name = rest.strip()
                        sub_task = task  # Use original task

                    agent_name = agent_name.strip()

                    if agent_name in self.agents:
                        # Execute specialist task
                        specialist = self.agents[agent_name].agent
                        specialist_response = specialist.run(sub_task.strip())
                        agent_responses[agent_name] = specialist_response

                        # Feed back to supervisor
                        feedback = f"Specialist '{agent_name}' responded:\n{specialist_response}\n\nWould you like to delegate to another specialist or provide a FINAL: answer?"
                        self.supervisor.history.append(
                            {"role": "assistant", "content": response}
                        )
                        self.supervisor.history.append(
                            {"role": "user", "content": feedback}
                        )
                    else:
                        # Agent not found
                        feedback = f"Agent '{agent_name}' not found. Available: {list(self.agents.keys())}. Try again."
                        self.supervisor.history.append(
                            {"role": "assistant", "content": response}
                        )
                        self.supervisor.history.append(
                            {"role": "user", "content": feedback}
                        )
                except ValueError:
                    # Malformed delegation
                    self.supervisor.history.append(
                        {"role": "assistant", "content": response}
                    )
                    self.supervisor.history.append(
                        {
                            "role": "user",
                            "content": "Invalid delegation format. Use DELEGATE:<agent_name>|<sub_task>",
                        }
                    )

            elif response.startswith("FINAL:"):
                # Final answer
                final = response.split(":", 1)[1].strip()
                return TeamResult(
                    final_response=final,
                    agent_responses=agent_responses,
                    turns_used=turns,
                    strategy=self.strategy,
                    metadata={"supervisor": self.supervisor.name},
                )
            else:
                # Non-protocol response - treat as final
                return TeamResult(
                    final_response=response,
                    agent_responses=agent_responses,
                    turns_used=turns,
                    strategy=self.strategy,
                )

        # Max rounds reached
        logger.warning(f"Team '{self.name}' reached max rounds without resolution")
        return TeamResult(
            final_response="Team reached maximum rounds without final answer.",
            agent_responses=agent_responses,
            turns_used=turns,
            strategy=self.strategy,
            metadata={"max_rounds_reached": True},
        )

    def _run_round_robin(self, task: str, **kwargs) -> TeamResult:
        """
        Agents take turns responding.

        Each agent builds on the previous agent's response.
        """
        agent_responses = {}
        turns = 0
        current_context = task

        # Sort by priority (higher first)
        sorted_agents = sorted(
            self.agents.values(), key=lambda r: r.priority, reverse=True
        )

        for round_num in range(self.max_rounds):
            for role in sorted_agents:
                turns += 1
                prompt = (
                    f"Previous context:\n{current_context}\n\nYour turn to contribute:"
                )
                response = role.agent.run(prompt)
                agent_responses[role.name] = response
                current_context = response

                # Check for termination signals
                if "TERMINATE" in response or "FINAL" in response:
                    return TeamResult(
                        final_response=response,
                        agent_responses=agent_responses,
                        turns_used=turns,
                        strategy=self.strategy,
                    )

        # Return last response
        return TeamResult(
            final_response=current_context,
            agent_responses=agent_responses,
            turns_used=turns,
            strategy=self.strategy,
        )

    def _run_parallel(self, task: str, **kwargs) -> TeamResult:
        """
        All agents work simultaneously.

        Results are merged using the merge_strategy function.
        """
        agent_responses = {}
        max_workers = kwargs.get("max_workers", len(self.agents))

        def run_agent(role: AgentRole) -> tuple:
            response = role.agent.run(task)
            return role.name, response

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_agent, role): role.name
                for role in self.agents.values()
            }

            for future in as_completed(futures):
                try:
                    name, response = future.result()
                    agent_responses[name] = response
                except Exception as e:
                    agent_name = futures[future]
                    logger.error(f"Agent '{agent_name}' failed: {e}")
                    agent_responses[agent_name] = f"Error: {e}"

        # Merge responses
        responses_list = list(agent_responses.values())
        final_response = self.merge_strategy(responses_list)

        return TeamResult(
            final_response=final_response,
            agent_responses=agent_responses,
            turns_used=1,  # All run in parallel = 1 turn
            strategy=self.strategy,
        )

    def _run_sequential(self, task: str, **kwargs) -> TeamResult:
        """
        Agents work in sequence, passing context.

        Each agent receives the previous agent's output.
        """
        agent_responses = {}
        turns = 0
        context = task

        # Sort by priority
        sorted_agents = sorted(
            self.agents.values(), key=lambda r: r.priority, reverse=True
        )

        for role in sorted_agents:
            turns += 1
            response = role.agent.run(context)
            agent_responses[role.name] = response
            context = response  # Pass to next agent

        return TeamResult(
            final_response=context,
            agent_responses=agent_responses,
            turns_used=turns,
            strategy=self.strategy,
        )

    def _default_merge(self, responses: List[str]) -> str:
        """Default strategy: concatenate with headers."""
        if len(responses) == 1:
            return responses[0]

        parts = []
        for i, resp in enumerate(responses, 1):
            parts.append(f"=== Response {i} ===\n{resp}")
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"AgentTeam(name='{self.name}', "
            f"strategy={self.strategy.value}, "
            f"agents={list(self.agents.keys())})"
        )
