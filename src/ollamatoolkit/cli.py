# ./ollamatoolkit/cli.py
"""
Ollama Toolkit - CLI Runner
===========================
Entry point for running agents via command line.
Supports configuration loading, dynamic tool registration, and interactive mode.
"""

import sys
import argparse
import logging

# Add src to path if running directly
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from ollamatoolkit.config import ToolkitConfig
from ollamatoolkit.agents.role import RoleAgent
from ollamatoolkit.tools.files import FileTools
from ollamatoolkit.tools.math import MathTools
from ollamatoolkit.tools.server import OllamaServerTools
from ollamatoolkit.tools.db import SQLDatabaseTool
from ollamatoolkit.tools.mcp import MCPToolManager
from ollamatoolkit.tools.vision import VisionTools
from ollamatoolkit.tools.vector import VectorTools
from ollamatoolkit.tools.system import SystemTools
from ollamatoolkit.tools.web import WebTools


def build_parser():
    parser = argparse.ArgumentParser(description="Ollama Toolkit Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # CONFIG command
    cfg_parser = subparsers.add_parser("config", help="Configuration utilities")
    cfg_parser.add_argument(
        "--init", action="store_true", help="Create sample_config.json"
    )

    # RUN command
    run_parser = subparsers.add_parser("run", help="Run a role-based agent")
    run_parser.add_argument("role_file", help="Path to role definition JSON")
    run_parser.add_argument(
        "--task", "-t", required=False, help="Initial task/instruction"
    )
    run_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Enter interactive chat mode"
    )
    run_parser.add_argument(
        "--dashboard", "-d", action="store_true", help="Enable TUI Dashboard"
    )
    run_parser.add_argument("--model", "-m", help="Override LLM model")
    run_parser.add_argument(
        "--tools",
        choices=["all", "files", "math", "db", "server", "system", "web", "vision"],
        nargs="+",
        help="Enable standard tools",
    )
    run_parser.add_argument("--db-path", help="Path for database tools (if enabled)")
    run_parser.add_argument("--config", "-c", help="Path to configuration file")

    # Batch mode arguments
    run_parser.add_argument(
        "--batch-file",
        "-b",
        help="Path to JSONL file with batch tasks (one JSON object per line)",
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        default="./batch_output",
        help="Output directory for batch results (default: ./batch_output)",
    )
    run_parser.add_argument(
        "--streaming",
        "-s",
        action="store_true",
        help="Enable streaming output (print tokens as they arrive)",
    )

    # MODELS command
    models_parser = subparsers.add_parser("models", help="List and inspect models")
    models_parser.add_argument(
        "--capability",
        "-c",
        choices=["vision", "embedding", "tools", "completion", "reasoning"],
        help="Filter by capability",
    )
    models_parser.add_argument("--json", action="store_true", help="Output as JSON")
    models_parser.add_argument(
        "--base-url", default="http://localhost:11434", help="Ollama server URL"
    )

    # CHAT command (quick interactive mode)
    chat_parser = subparsers.add_parser("chat", help="Quick interactive chat")
    chat_parser.add_argument(
        "--model", "-m", help="Model to use (auto-selects if not specified)"
    )
    chat_parser.add_argument(
        "--base-url", default="http://localhost:11434", help="Ollama server URL"
    )
    chat_parser.add_argument(
        "--system", "-s", default="You are a helpful assistant.", help="System prompt"
    )

    return parser


def run_batch_mode(agent, batch_file: str, output_dir: str):
    """
    Execute tasks from a JSONL batch file.

    Each line in the batch file should be a JSON object with:
    - "task" or "prompt": The task to execute (required)
    - "id": Optional task identifier (defaults to line number)

    Results are saved to output_dir as {id}.json files.

    Args:
        agent: The agent to run tasks with
        batch_file: Path to JSONL file with tasks
        output_dir: Directory to save results
    """
    import json
    from datetime import datetime

    batch_path = Path(batch_file)
    if not batch_path.exists():
        print(f"Error: Batch file '{batch_file}' not found.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    start_time = datetime.now()

    print(f"Starting batch processing from: {batch_file}")
    print(f"Output directory: {output_path}")
    print("-" * 50)

    with open(batch_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                task_data = json.loads(line)
                task_id = task_data.get("id", f"task_{line_num}")
                task_prompt = task_data.get("task") or task_data.get("prompt", "")

                if not task_prompt:
                    print(f"[{line_num}] Skipping: No task/prompt found")
                    results.append(
                        {"id": task_id, "success": False, "error": "No task found"}
                    )
                    continue

                print(f"[{line_num}] Processing: {task_id}")
                print(
                    f"    Task: {task_prompt[:80]}{'...' if len(task_prompt) > 80 else ''}"
                )

                # Run the task
                task_start = datetime.now()
                try:
                    response = agent.run(task_prompt)
                    duration = (datetime.now() - task_start).total_seconds()

                    # Save result
                    result_data = {
                        "id": task_id,
                        "task": task_prompt,
                        "response": response,
                        "duration_seconds": duration,
                        "timestamp": datetime.now().isoformat(),
                        "success": True,
                    }

                    result_file = output_path / f"{task_id}.json"
                    with open(result_file, "w", encoding="utf-8") as rf:
                        json.dump(result_data, rf, indent=2, ensure_ascii=False)

                    print(f"    ✓ Completed in {duration:.2f}s -> {result_file.name}")
                    results.append(
                        {"id": task_id, "success": True, "duration": duration}
                    )

                except Exception as task_error:
                    duration = (datetime.now() - task_start).total_seconds()
                    print(f"    ✗ Failed: {task_error}")
                    results.append(
                        {
                            "id": task_id,
                            "success": False,
                            "error": str(task_error),
                            "duration": duration,
                        }
                    )

            except json.JSONDecodeError as e:
                print(f"[{line_num}] Invalid JSON: {e}")
                results.append(
                    {
                        "id": f"line_{line_num}",
                        "success": False,
                        "error": f"Invalid JSON: {e}",
                    }
                )

    # Print summary
    total_time = (datetime.now() - start_time).total_seconds()
    success_count = sum(1 for r in results if r.get("success"))
    fail_count = len(results) - success_count

    print("-" * 50)
    print("Batch Processing Complete")
    print(f"  Total tasks: {len(results)}")
    print(f"  Successful:  {success_count}")
    print(f"  Failed:      {fail_count}")
    print(f"  Total time:  {total_time:.2f}s")

    # Save summary
    summary_file = output_path / "_batch_summary.json"
    with open(summary_file, "w", encoding="utf-8") as sf:
        json.dump(
            {
                "batch_file": str(batch_path),
                "total_tasks": len(results),
                "successful": success_count,
                "failed": fail_count,
                "total_duration_seconds": total_time,
                "completed_at": datetime.now().isoformat(),
                "results": results,
            },
            sf,
            indent=2,
        )
    print(f"  Summary:     {summary_file}")


def handle_models_command(args):
    """Handle the 'models' CLI command for model discovery."""
    import json as json_module
    from ollamatoolkit.models.selector import ModelSelector

    try:
        selector = ModelSelector(base_url=args.base_url)
    except Exception as e:
        print(f"Error connecting to Ollama at {args.base_url}: {e}")
        return

    # Filter by capability if specified
    if args.capability:
        models = selector.get_models_by_capability(args.capability)
        model_names = [m.name for m in models]
    else:
        model_names = selector.model_names

    if args.json:
        # JSON output
        output = {
            "total_models": len(model_names),
            "models": [],
        }
        for name in model_names:
            info = selector.get_model(name)
            if info:
                output["models"].append(
                    {
                        "name": info.name,
                        "family": info.family,
                        "size": info.parameter_size,
                        "capabilities": info.capabilities,
                        "quantization": info.quantization,
                    }
                )
        print(json_module.dumps(output, indent=2))
    else:
        # Table output
        print(f"\n{'Model':<40} {'Family':<15} {'Size':<10} {'Capabilities'}")
        print("-" * 90)
        for name in model_names:
            info = selector.get_model(name)
            if info:
                caps = ", ".join(info.capabilities)
                print(
                    f"{info.name:<40} {info.family:<15} {info.parameter_size:<10} {caps}"
                )
        print(f"\nTotal: {len(model_names)} models")

        # Show summary
        summary = selector.summary()
        print(f"\nCapabilities: {', '.join(summary['by_capability'].keys())}")


def handle_chat_command(args):
    """Handle the 'chat' CLI command for quick interactive chat."""
    from ollamatoolkit import SimpleAgent, ModelSelector

    # Auto-select model if not specified
    if args.model:
        model = args.model
    else:
        try:
            selector = ModelSelector(base_url=args.base_url)
            model = selector.get_best_chat_model()
            if not model:
                print("No chat model found. Run: ollama pull llama3.1")
                return
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return

    print(f"🤖 OllamaToolkit Chat (model: {model})")
    print("Type 'quit' or 'exit' to end the session.\n")

    agent = SimpleAgent(
        name="chat",
        system_message=args.system,
        model_config={
            "model": f"ollama/{model}" if not model.startswith("ollama/") else model,
            "api_base": args.base_url,
        },
    )

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            print("Assistant: ", end="", flush=True)
            for token in agent.run_streaming(user_input):
                print(token, end="", flush=True)
            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 1. Load Configuration
    cfg = ToolkitConfig.load(args.config)  # Loads defaults or file

    # Handle Config Init
    if args.command == "config":
        if args.init:
            cfg.save_sample()
            print("Created sample_config.json")
        return

    # Handle Overrides
    if args.command == "run":
        if args.model:
            cfg.agent.model = args.model

    # Handle MODELS command
    if args.command == "models":
        handle_models_command(args)
        return

    # Handle CHAT command
    if args.command == "chat":
        handle_chat_command(args)
        return

    # Configure Logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        filename=cfg.logging.file_path if cfg.logging.file_path else None,
    )

    path = Path(args.role_file)
    if not path.exists():
        print(f"Error: Role file '{path}' not found.")
        sys.exit(1)

    # 2. Setup Tool Registry
    registry = {}

    # Determine allowed tools
    # If valid args.tools passed, use intersection. Else use config.
    allowed = set(cfg.tools.allowed_tools)
    if args.tools and "all" not in args.tools:
        allowed = allowed.intersection(set(args.tools))
    elif args.tools and "all" in args.tools:
        allowed = {"files", "math", "server", "db", "system", "web"}

    selected_tools = list(allowed)
    print(f"Enabled Tools: {selected_tools}")

    # Initialize Standard Tools
    if "files" in selected_tools:
        ft = FileTools(cfg.tools.root_dir, cfg.tools.read_only)
        registry["read_file"] = ft.read_file
        registry["write_file"] = ft.write_file
        registry["list_dir"] = ft.list_dir
        registry["file_exists"] = ft.file_exists

    if "math" in selected_tools:
        registry["calculate"] = MathTools.calculate

    if "system" in selected_tools:
        registry["get_current_time"] = SystemTools.get_current_time
        registry["get_system_report"] = SystemTools.get_system_report
        registry["monitor_resources"] = SystemTools.monitor_resources
        registry["wait_seconds"] = SystemTools.wait_seconds

    if "web" in selected_tools:
        # Instantiate with config
        web = WebTools(cfg.web)
        registry["fetch_url"] = web.fetch_url
        registry["post_data"] = web.post_data

    if "vision" in selected_tools:
        vis = VisionTools(cfg.vision, cfg.agent.base_url)
        registry["analyze_image"] = vis.analyze_image
        registry["analyze_video"] = vis.analyze_video
        registry["detect_objects"] = vis.detect_objects
        registry["compare_images"] = vis.compare_images
        registry["ocr_image"] = vis.ocr_image
        registry["image_to_code"] = vis.image_to_code
        registry["describe_scene"] = vis.describe_scene

    if "vector" in selected_tools:
        vec = VectorTools(cfg.vector)
        registry["ingest_text"] = vec.ingest_text
        registry["ingest_file"] = vec.ingest_file
        registry["search_memory"] = vec.search_memory

    if "server" in selected_tools:
        srv = OllamaServerTools(cfg.agent.base_url)
        registry["list_models"] = srv.list_models
        registry["pull_model"] = srv.pull_model
        registry["show_model_info"] = srv.show_model_info

    if "db" in selected_tools:
        if not args.db_path:
            print(
                "Warning: DB tools requested but --db-path not provided. Using :memory:"
            )
            db_path = ":memory:"
        else:
            db_path = args.db_path
        db = SQLDatabaseTool(db_path, read_only=False)
        registry["run_query"] = db.run_query
        registry["get_table_info"] = db.get_table_info  # V2
        registry["execute_script"] = db.execute_script  # V2
        registry["list_tables"] = db.list_tables

    # Custom Tools (Dynamic Import)
    if cfg.tools.custom_tools:
        import importlib

        print(f"Loading custom tools: {cfg.tools.custom_tools}")
        for tool_path in cfg.tools.custom_tools:
            try:
                module_name, func_name = tool_path.rsplit(".", 1)
                mod = importlib.import_module(module_name)
                func = getattr(mod, func_name)

                if not callable(func):
                    print(f"Warning: {tool_path} is not callable.")
                    continue

                registry[func.__name__] = func
                print(f"  + Registered '{func.__name__}' from {tool_path}")
            except Exception as e:
                print(f"Error loading custom tool '{tool_path}': {e}")

    if "system" in selected_tools:
        registry["get_current_time"] = SystemTools.get_current_time
        registry["get_system_report"] = SystemTools.get_system_report
        registry["monitor_resources"] = SystemTools.monitor_resources
        registry["wait_seconds"] = SystemTools.wait_seconds

    # Native WebTools disabled in favor of WebScraperToolkit MCP
    # if "web" in selected_tools:
    #     # Instantiate with config
    #     web = WebTools(cfg.web)
    #     registry["fetch_url"] = web.fetch_url
    #     registry["post_data"] = web.post_data

    # MCP Servers
    mcp_manager = None
    if cfg.tools.mcp_servers:
        print(f"Initializing MCP Servers: {list(cfg.tools.mcp_servers.keys())}")
        mcp_manager = MCPToolManager(cfg.tools.mcp_servers)
        mcp_manager.start_all()

        proxies = mcp_manager.get_proxy_functions()
        for name, func in proxies.items():
            registry[name] = func
            print(f"  + Registered MCP tool '{name}'")

    # 3. Create Agent
    try:
        model_config = {
            "model": cfg.agent.model,
            "base_url": cfg.agent.base_url,
            "api_key": cfg.agent.api_key,
            "temperature": cfg.agent.temperature,
            "caching": cfg.agent.caching,
            "fallbacks": cfg.agent.fallbacks,
        }

        agent = RoleAgent.from_role_file(
            role_path=path,
            model_config=model_config,
            tool_registry=registry,
            history_limit=cfg.agent.history_limit,
        )

        print(f"Agent '{agent.name}' initialized.")
        print(f"Model: {model_config['model']}")
        print("-" * 40)

        # 4. Run loop
        if args.dashboard or cfg.dashboard.enabled:
            # Check for Rich
            import importlib.util

            if importlib.util.find_spec("rich") is None:
                print(
                    "Error: Dashboard requires 'rich'. Install with 'pip install ollamatoolkit[full]' or 'pip install rich'."
                )
                sys.exit(1)

            if not args.task:
                print("Error: Dashboard mode currently requires --task.")
                sys.exit(1)

            from ollamatoolkit.dashboard import DashboardRunner

            print("Launching Mission Control...")
            dash = DashboardRunner(agent, cfg.dashboard)
            dash.run(args.task)

        elif hasattr(args, "batch_file") and args.batch_file:
            # Batch Mode - Process JSONL file
            run_batch_mode(agent, args.batch_file, args.output_dir)

        elif args.interactive:
            print("Interactive Mode. Type 'exit' to quit.")
            if args.task:
                print(f"Initial Task: {args.task}")
                if hasattr(args, "streaming") and args.streaming:
                    print("AI: ", end="", flush=True)
                    for token in agent.run_streaming(args.task):
                        print(token, end="", flush=True)
                    print("\n")
                else:
                    resp = agent.run(args.task)
                    print(f"AI: {resp}\n")

            while True:
                try:
                    user_input = input("You: ")
                    if user_input.lower() in ["exit", "quit"]:
                        break

                    if hasattr(args, "streaming") and args.streaming:
                        print("AI: ", end="", flush=True)
                        for token in agent.run_streaming(user_input, max_turns=5):
                            print(token, end="", flush=True)
                        print()
                    else:
                        print("AI: ...", end="\r")
                        resp = agent.run(user_input, max_turns=5)
                        print(f"AI: {resp}")

                except KeyboardInterrupt:
                    break
        else:
            # Single Shot
            if not args.task:
                print(
                    "Error: --task is required unless --interactive or --batch-file is set."
                )
                sys.exit(1)

            print(f"Task: {args.task}")
            if hasattr(args, "streaming") and args.streaming:
                print("-" * 40)
                print("Response:")
                for token in agent.run_streaming(args.task):
                    print(token, end="", flush=True)
                print()
            else:
                resp = agent.run(args.task)
                print("-" * 40)
                print("Final Response:")
                print(resp)

    except Exception as e:
        print(f"Execution failed: {e}")
    finally:
        if mcp_manager:
            print("Shutting down MCP servers...")
            mcp_manager.shutdown()


if __name__ == "__main__":
    main()
