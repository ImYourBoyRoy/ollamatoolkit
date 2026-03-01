# ./examples/02_with_tools.py
"""
Agent with Tools Example
========================
Demonstrates how to create an agent with custom function calling.

Run: python examples/02_with_tools.py
"""

from ollamatoolkit import SimpleAgent, ModelSelector


def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Mock implementation
    return f"The weather in {location} is sunny, 72°F"


def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def main():
    selector = ModelSelector()
    model = selector.get_for_capability("tools")

    if not model:
        print("No tool-capable model found. Run: ollama pull llama3.1")
        return

    print(f"Using model: {model}")

    # Define tools as OpenAI-compatible function schemas
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression like '2 + 2'",
                        }
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    # Map function names to actual functions
    function_map = {
        "get_weather": get_weather,
        "calculate": calculate,
    }

    # Create agent with tools
    agent = SimpleAgent(
        name="tool_agent",
        system_message="You are a helpful assistant with access to weather and calculator tools.",
        model_config={"model": f"ollama/{model}"},
        tools=tools,
        function_map=function_map,
    )

    # The agent will automatically call tools as needed
    response = agent.run("What's the weather in Tokyo?")
    print(f"\nWeather query: {response}")

    response = agent.run("What's 15 * 23 + 42?")
    print(f"\nMath query: {response}")


if __name__ == "__main__":
    main()
