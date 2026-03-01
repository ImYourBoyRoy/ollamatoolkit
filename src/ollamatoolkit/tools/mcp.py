# ./src/ollamatoolkit/tools/mcp.py
"""
Ollama Toolkit - MCP Client
===========================
Client for Model Context Protocol (MCP) servers via Stdio.
Allows connecting to external tools (e.g. webscrapertoolkit).
"""

import json
import logging
import os
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, cast

logger = logging.getLogger(__name__)


class MCPClient:
    """
    Manages a connection to an MCP Server (JSON-RPC 2.0 over Stdio).
    """

    def __init__(
        self, name: str, command: str, args: List[str], env: Optional[Dict] = None
    ):
        self.name = name
        self.command = command
        self.args = args
        self.env = env or os.environ.copy()

        self.process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._msg_id = 1

    def connect(self):
        """Starts the subprocess and performs strict MCP handshake."""
        try:
            logger.info(f"MCP [{self.name}]: Starting {self.command} {self.args}")
            self.process = subprocess.Popen(
                [self.command] + self.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,  # pipe stderr to parent stderr for visibility
                env=self.env,
                text=True,  # text mode for json
                encoding="utf-8",
                bufsize=0,  # unbuffered
            )

            # 1. Initialize
            init_payload = {
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "ollamatoolkit", "version": "0.2.0"},
                },
            }
            resp = self._send_request(init_payload)
            if "error" in resp:
                raise RuntimeError(f"MCP Init Failed: {resp['error']}")

            # 2. Initialized Notification
            self._send_notification("notifications/initialized")
            logger.info(f"MCP [{self.name}]: Connected.")

        except Exception as e:
            logger.error(f"MCP [{self.name}] Connect Error: {e}")
            if self.process:
                self.process.kill()
            raise

    def _send_request(self, payload: Dict) -> Dict:
        """Sends a JSON-RPC request and waits for a response (Sync)."""
        if not self.process:
            raise RuntimeError("MCP Not Connected")

        with self._lock:
            if self.process.stdin is None or self.process.stdout is None:
                raise RuntimeError("MCP process pipes are not available.")

            payload["jsonrpc"] = "2.0"
            payload["id"] = self._msg_id
            self._msg_id += 1

            msg_str = json.dumps(payload) + "\n"
            self.process.stdin.write(msg_str)
            self.process.stdin.flush()

            # Read response
            # Simple blocking read line-by-line.
            # Real MCP might send notifications, we iterate until we find ID match?
            # For this 'Simple' version, assume strict Request-Response sync mostly or ignore notifications.

            while True:
                line = self.process.stdout.readline()
                if not line:
                    raise EOFError("MCP Server closed connection.")

                try:
                    data = json.loads(line)
                    # Ignore notifications/logs
                    if "id" not in data:
                        continue

                    if data["id"] == payload["id"]:
                        return data
                except json.JSONDecodeError:
                    continue

    def _send_notification(self, method: str, params: Optional[Dict] = None):
        """Sends a notification (no response expected)."""
        if not self.process:
            return
        if self.process.stdin is None:
            return
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params:
            payload["params"] = params

        msg_str = json.dumps(payload) + "\n"
        self.process.stdin.write(msg_str)
        self.process.stdin.flush()

    def list_tools(self) -> List[Dict]:
        """Call tools/list and format for Ollama/LiteLLM."""
        resp = self._send_request({"method": "tools/list", "params": {}})
        if "error" in resp:
            return []

        tools = resp.get("result", {}).get("tools", [])

        # Convert MCP schema to Ollama/OpenAI schema if needed?
        # Actually LiteLLM usually wants a function schema.
        # MCP tool: {name, description, inputSchema}
        # Ollama tool: {type: function, function: {name, description, parameters: inputSchema}}

        converted: List[Dict[str, Any]] = []
        for t in tools:
            tool_item = cast(Dict[str, Any], t)
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_item["name"],
                        "description": tool_item.get("description", ""),
                        "parameters": tool_item.get("inputSchema", {}),
                    },
                }
            )
        return converted

    def call_tool(self, name: str, arguments: Dict) -> str:
        """Executes a tool on the server."""
        resp = self._send_request(
            {"method": "tools/call", "params": {"name": name, "arguments": arguments}}
        )

        if "error" in resp:
            return f"Error: {resp['error'].get('message', 'Unknown tool error')}"

        # Result content is usually a list of text/image content
        content_items = resp.get("result", {}).get("content", [])
        texts = [c["text"] for c in content_items if c.get("type") == "text"]
        return "\n".join(texts)

    def close(self):
        if self.process:
            self.process.terminate()
            self.process = None


class MCPToolManager:
    """Helper to load servers from config and generate callables."""

    def __init__(self, servers_config: Dict[str, Dict]):
        self.clients: List[MCPClient] = []
        self.servers_config = servers_config

    def start_all(self):
        for name, cfg in self.servers_config.items():
            cmd = cfg.get("command")
            args = cfg.get("args", [])
            env = cfg.get("env")

            client = MCPClient(name, cmd, args, env)
            try:
                client.connect()
                self.clients.append(client)
            except Exception as e:
                print(f"Failed to start MCP Server '{name}': {e}")

    def get_proxy_functions(self) -> Dict[str, Any]:
        """
        Returns a dictionary of {tool_name: callable} for all tools in all connected servers.
        The callable ensures the client.call_tool is invoked.
        """
        registry: Dict[str, Callable[..., str]] = {}
        for client in self.clients:
            tools = client.list_tools()
            for t_def in tools:
                function_payload = cast(Dict[str, Any], t_def.get("function", {}))
                func_name = str(function_payload.get("name", ""))
                if not func_name:
                    continue

                # Create a closure to capture the client and tool name
                def make_proxy(c: MCPClient, t_name: str):
                    def proxy(**kwargs):
                        return c.call_tool(t_name, kwargs)

                    return proxy

                proxy_func = make_proxy(client, func_name)

                # IMPORTANT: Attach the schema so SimpleAgent/RoleAgent doesn't try to inspect the proxy.
                setattr(proxy_func, "_tool_def", t_def)

                registry[func_name] = proxy_func
        return registry

    def get_tool_schemas(self) -> List[Dict]:
        """Returns the list of tool definitions (JSON schemas) for the LLM."""
        schemas = []
        for client in self.clients:
            schemas.extend(client.list_tools())
        return schemas

    def shutdown(self):
        for c in self.clients:
            c.close()
