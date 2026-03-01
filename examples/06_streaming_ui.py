# ./examples/06_streaming_ui.py
"""
Streaming Output Example
========================
Demonstrates real-time token streaming for responsive UIs.

Run: python examples/06_streaming_ui.py
"""

from ollamatoolkit import SimpleAgent, ModelSelector


def main():
    selector = ModelSelector()
    model = selector.get_best_chat_model()

    if not model:
        print("No model found. Run: ollama pull llama3.1")
        return

    print(f"Using model: {model}")

    # Create agent
    agent = SimpleAgent(
        name="storyteller",
        system_message="You are a creative storyteller. Write engaging short stories.",
        model_config={
            "model": f"ollama/{model}",
            "temperature": 0.8,
        },
    )

    print("\n--- Streaming Response ---")
    print("Assistant: ", end="", flush=True)

    # Stream tokens as they arrive
    for token in agent.run_streaming(
        "Tell me a very short story about a robot learning to paint."
    ):
        print(token, end="", flush=True)

    print("\n")  # New line after streaming completes

    # You can also use streaming with callbacks
    print("--- Streaming with Callback ---")

    tokens_received = []

    def on_token(token: str):
        tokens_received.append(token)
        print(token, end="", flush=True)

    print("Assistant: ", end="", flush=True)

    for token in agent.run_streaming(
        "What do you think the robot's first painting looked like?"
    ):
        on_token(token)

    print(f"\n\n[Received {len(tokens_received)} tokens]")


if __name__ == "__main__":
    main()
