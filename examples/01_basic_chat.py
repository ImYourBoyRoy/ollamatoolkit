# ./examples/01_basic_chat.py
"""
Basic Chat Example
==================
Demonstrates the simplest way to create an agent and chat with it.

Run: python examples/01_basic_chat.py
"""

from ollamatoolkit import SimpleAgent, ModelSelector


def main():
    # Auto-select the best available chat model
    selector = ModelSelector()
    model = selector.get_best_chat_model()

    if not model:
        print("No models found. Run: ollama pull llama3.1")
        return

    print(f"Using model: {model}")

    # Create a simple agent
    agent = SimpleAgent(
        name="assistant",
        system_message="You are a helpful assistant. Be concise.",
        model_config={
            "model": f"ollama/{model}",
            "temperature": 0.7,
        },
    )

    # Single response
    response = agent.run("What's the capital of France?")
    print(f"\nResponse: {response}")

    # Continue the conversation
    response = agent.run("What's a famous landmark there?")
    print(f"\nFollow-up: {response}")


if __name__ == "__main__":
    main()
