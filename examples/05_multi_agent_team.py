# ./examples/05_multi_agent_team.py
"""
Multi-Agent Team Example
========================
Demonstrates AgentTeam orchestration with multiple specialized agents.

Run: python examples/05_multi_agent_team.py
"""

from ollamatoolkit import AgentTeam, ModelSelector
from ollamatoolkit.agents.team import AgentRole, TeamStrategy


def main():
    selector = ModelSelector()
    model = selector.get_best_chat_model()

    if not model:
        print("No model found. Run: ollama pull llama3.1")
        return

    print(f"Using model: {model}")

    # Define specialized agent roles
    roles = [
        AgentRole(
            name="researcher",
            description="Gathers and analyzes information",
            skills=["research", "analysis", "fact-checking"],
            model=f"ollama/{model}",
        ),
        AgentRole(
            name="writer",
            description="Creates clear, engaging content",
            skills=["writing", "editing", "formatting"],
            model=f"ollama/{model}",
        ),
        AgentRole(
            name="critic",
            description="Reviews and provides constructive feedback",
            skills=["review", "critique", "improvement"],
            model=f"ollama/{model}",
        ),
    ]

    # Create team with sequential strategy
    team = AgentTeam(
        name="content_team",
        roles=roles,
        strategy=TeamStrategy.SEQUENTIAL,
        max_rounds=3,
    )

    print("\n--- Sequential Team Execution ---")

    # Run the team on a task
    result = team.run(
        task="Write a brief explanation of how solar panels work.",
        context="Target audience: high school students",
    )

    print(f"\nFinal Response:\n{result.final_response}")
    print(f"\nTotal turns: {result.total_turns}")
    print(f"Strategy: {result.strategy_used}")

    # Show individual agent outputs
    print("\n--- Agent Contributions ---")
    for agent_name, output in result.agent_outputs.items():
        print(f"\n{agent_name.upper()}:")
        print(f"  {output[:200]}..." if len(output) > 200 else f"  {output}")


if __name__ == "__main__":
    main()
