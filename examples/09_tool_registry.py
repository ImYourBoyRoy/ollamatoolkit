# ./examples/09_tool_registry.py
"""
Tool Registry Example - LLM-First Tools
========================================
Demonstrates how ToolRegistry makes tools trivially easy for small models.

Run: python examples/09_tool_registry.py
"""

from ollamatoolkit import ToolRegistry, SimpleAgent, ModelSelector


def demo_manual_execution():
    """Show how to manually execute tools with LLM-friendly results."""
    print("=" * 60)
    print("MANUAL TOOL EXECUTION")
    print("=" * 60)

    # Create registry with all default tools
    registry = ToolRegistry()

    print(
        f"\nRegistered {registry.tool_count} tools in {len(registry.categories)} categories"
    )
    print(f"Categories: {registry.categories}")

    # List available tools
    print("\n--- Available Tools ---")
    for tool in registry.list_tools()[:8]:
        print(f"  • {tool['name']}: {tool['description'][:50]}...")

    # Execute tools - results are LLM-friendly!
    print("\n--- Tool Execution ---")

    # 1. List files
    result = registry.execute("list_files", {"path": "."})
    print(f"\nlist_files: {result.summary}")
    if result.success:
        print(
            f"  Result: {result.result[:200] if isinstance(result.result, str) else result.result}..."
        )

    # 2. Calculate
    result = registry.execute("calculate", {"expression": "15 * 23 + 42"})
    print(f"\ncalculate: {result.to_llm_string()}")

    # 3. Handle errors gracefully
    result = registry.execute("read_file", {"path": "nonexistent_file.txt"})
    print(f"\nread_file (error): {result.to_llm_string()}")

    # 4. Get help for a tool
    print("\n--- Tool Help ---")
    print(registry.get_tool_help("search_files"))


def demo_with_agent():
    """Show how to use ToolRegistry with SimpleAgent."""
    print("\n" + "=" * 60)
    print("AGENT WITH TOOL REGISTRY")
    print("=" * 60)

    # Check for model
    selector = ModelSelector()
    model = selector.get_for_capability("tools")

    if not model:
        print("No tool-capable model found. Run: ollama pull llama3.1")
        return

    print(f"Using model: {model}")

    # Create registry
    registry = ToolRegistry()

    # Get schemas and function map - THAT'S IT!
    schemas = registry.get_schemas()

    print(f"\nGenerated {len(schemas)} OpenAI-compatible function schemas")

    # Show a sample schema
    print("\n--- Sample Schema (calculate) ---")
    for schema in schemas:
        if schema["function"]["name"] == "calculate":
            import json

            print(json.dumps(schema, indent=2))
            break

    # Create agent with tools
    tool_agent = SimpleAgent(
        name="tool_agent",
        system_message="""You are a helpful assistant with access to tools.
When asked to perform a task, use the appropriate tool.
Always explain what you're doing and interpret the results for the user.""",
        model_config={"model": f"ollama/{model}"},
        tools=schemas,
        function_map={
            name: (lambda args, n=name: registry.execute(n, args).to_llm_string())
            for name in [t["function"]["name"] for t in schemas]
        },
    )

    print("\n--- Agent Ready ---")
    print(f"Initialized agent: {tool_agent.name}")
    print("The agent now has access to all tools via function calling.")
    print("Tools are executed with LLM-friendly results and error handling.")


def demo_error_handling():
    """Show how errors are handled with helpful suggestions."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING FOR DUMB MODELS")
    print("=" * 60)

    registry = ToolRegistry()

    # Various error cases - all give helpful suggestions
    errors = [
        ("read_file", {"path": "/nonexistent/file.txt"}),
        ("calculate", {"expression": "invalid math"}),
        ("nonexistent_tool", {"arg": "value"}),
    ]

    for tool_name, args in errors:
        result = registry.execute(tool_name, args)
        print(f"\n{tool_name}:")
        print(f"  Success: {result.success}")
        print(f"  Error: {result.error}")
        print(f"  Suggestion: {result.suggestion}")


def main():
    demo_manual_execution()
    demo_with_agent()
    demo_error_handling()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("""
The ToolRegistry makes it trivial for small models to use tools:

1. registry = ToolRegistry()           # All tools auto-registered
2. schemas = registry.get_schemas()    # OpenAI-compatible schemas
3. result = registry.execute(name, args)  # LLM-friendly results

Every result includes:
- ✓ Success status
- ✓ Human-readable summary
- ✓ Truncated output (won't overflow context)
- ✓ Error suggestions (when things fail)

Even a "dumb" 7B model can now use complex tools reliably.
""")


if __name__ == "__main__":
    main()
