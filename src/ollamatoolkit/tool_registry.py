# ./ollamatoolkit/tool_registry.py
"""
OllamaToolkit - LLM Tool Registry
=================================
Makes tools instantly usable by small/local LLMs via function calling.

Key Design Principles:
- Auto-generates OpenAI-compatible function schemas
- Formats results for LLM consumption (not raw data)
- Provides error recovery suggestions when tools fail
- Limits output to avoid context overflow
- Logging and transparency at every step

Usage:
    from ollamatoolkit import ToolRegistry

    # Create registry with default tools
    registry = ToolRegistry()

    # Get OpenAI-compatible tool schemas
    schemas = registry.get_schemas()

    # Execute a tool call from LLM
    result = registry.execute("read_file", {"path": "readme.md"})

    # Or use with SimpleAgent
    agent = SimpleAgent(
        tools=registry.get_schemas(),
        function_map=registry.get_function_map(),
    )
"""

import json
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Result Wrappers
# =============================================================================


@dataclass
class ToolResult:
    """
    LLM-friendly result wrapper.

    Ensures all tool outputs are consistent and interpretable by small models.
    """

    success: bool
    tool_name: str
    result: Any = None
    summary: str = ""  # One-line summary for LLM
    error: Optional[str] = None
    suggestion: Optional[str] = None  # What to try if failed
    truncated: bool = False
    original_length: int = 0

    def to_llm_string(self) -> str:
        """Format result for LLM consumption."""
        if self.success:
            output = f"✓ {self.tool_name}: {self.summary}"
            if self.result is not None:
                if isinstance(self.result, str):
                    output += f"\n\n{self.result}"
                elif isinstance(self.result, (dict, list)):
                    output += f"\n\n{json.dumps(self.result, indent=2, default=str)}"
                else:
                    output += f"\n\n{self.result}"
            if self.truncated:
                output += f"\n\n[Truncated from {self.original_length} chars]"
            return output
        else:
            output = f"✗ {self.tool_name} failed: {self.error}"
            if self.suggestion:
                output += f"\n→ Suggestion: {self.suggestion}"
            return output

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "tool_name": self.tool_name,
            "result": self.result,
            "summary": self.summary,
            "error": self.error,
            "suggestion": self.suggestion,
        }


# =============================================================================
# Tool Definition
# =============================================================================


@dataclass
class ToolDefinition:
    """Definition of a registered tool."""

    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]  # JSON Schema for parameters
    category: str = "general"
    examples: List[str] = field(default_factory=list)
    max_output_chars: int = 4000  # Limit output for small models

    def to_openai_schema(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """
    Central registry for all LLM-callable tools.

    Makes tools easy for small/local models to use effectively.
    """

    def __init__(
        self,
        max_output_chars: int = 4000,
        include_defaults: bool = True,
        root_dir: str = ".",
    ):
        """
        Initialize the tool registry.

        Args:
            max_output_chars: Limit output length for small models
            include_defaults: Register default toolkit tools
            root_dir: Root directory for file operations
        """
        self.max_output_chars = max_output_chars
        self.root_dir = root_dir
        self._tools: Dict[str, ToolDefinition] = {}
        self._instances: Dict[str, Any] = {}  # Cached tool instances

        if include_defaults:
            self._register_default_tools()

    # =========================================================================
    # Registration
    # =========================================================================

    def register(
        self,
        name: str,
        function: Callable,
        description: str,
        parameters: Dict[str, Any],
        category: str = "general",
        examples: List[str] = None,
        max_output_chars: Optional[int] = None,
    ) -> None:
        """
        Register a tool for LLM use.

        Args:
            name: Tool name (used in function calling)
            function: The callable to execute
            description: Clear description for LLM
            parameters: JSON Schema for parameters
            category: Tool category for organization
            examples: Example usage strings
            max_output_chars: Override default output limit
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            function=function,
            parameters=parameters,
            category=category,
            examples=examples or [],
            max_output_chars=max_output_chars or self.max_output_chars,
        )
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def _register_default_tools(self):
        """Register all default toolkit tools."""
        self._register_file_tools()
        self._register_math_tools()
        self._register_web_tools()
        self._register_email_tools()
        self._register_schema_tools()

    # =========================================================================
    # File Tools
    # =========================================================================

    def _register_file_tools(self):
        """Register file operation tools."""
        from ollamatoolkit.tools.files import FileTools

        files = FileTools(root_dir=self.root_dir)
        self._instances["files"] = files

        # read_file
        self.register(
            name="read_file",
            function=lambda path: files.read_text(path),
            description="Read the contents of a text file. Returns the file content as a string.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read (relative to working directory)",
                    }
                },
                "required": ["path"],
            },
            category="files",
            examples=["read_file('readme.md')", "read_file('src/main.py')"],
        )

        # write_file
        self.register(
            name="write_file",
            function=lambda path, content: files.write_text(
                path, content, overwrite=True
            ),
            description="Write content to a text file. Creates the file if it doesn't exist.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write",
                    },
                },
                "required": ["path", "content"],
            },
            category="files",
        )

        # list_files
        self.register(
            name="list_files",
            function=lambda path=".": files.list_dir(path),
            description="List files and directories in a path. Returns names and types.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)",
                        "default": ".",
                    }
                },
                "required": [],
            },
            category="files",
        )

        # search_files
        self.register(
            name="search_files",
            function=lambda pattern, path=".": files.find_files(
                pattern, path, max_results=20
            ),
            description="Find files matching a glob pattern (e.g., '*.py', '**/*.md').",
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match (e.g., '*.py', '**/*.json')",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
            category="files",
        )

        # search_content
        self.register(
            name="search_content",
            function=lambda query, path=".": files.grep(query, path, max_results=10),
            description="Search for text content within files. Returns matching lines.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text or regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search",
                        "default": ".",
                    },
                },
                "required": ["query"],
            },
            category="files",
        )

    # =========================================================================
    # Math Tools
    # =========================================================================

    def _register_math_tools(self):
        """Register math operation tools."""
        from ollamatoolkit.tools.math import MathTools

        # MathTools methods are static
        # calculate
        self.register(
            name="calculate",
            function=lambda expression: MathTools.calculate(expression),
            description="Safely evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, sin, cos, mean, median, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to calculate (e.g., '2 + 2', 'sqrt(16)', 'mean([1,2,3])')",
                    }
                },
                "required": ["expression"],
            },
            category="math",
            examples=[
                "calculate('2 + 2')",
                "calculate('sqrt(144)')",
                "calculate('mean([1, 2, 3, 4, 5])')",
            ],
        )

        # statistics (analyze_list)
        self.register(
            name="statistics",
            function=lambda numbers: MathTools.analyze_list(numbers),
            description="Get statistical summary (count, sum, mean, median, min, max, stdev) for a list of numbers.",
            parameters={
                "type": "object",
                "properties": {
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to analyze",
                    }
                },
                "required": ["numbers"],
            },
            category="math",
        )

    # =========================================================================
    # Web Tools
    # =========================================================================

    def _register_web_tools(self):
        """Register web operation tools."""
        # Use a simple fetch function that doesn't require config
        import httpx

        def fetch_url(url: str) -> str:
            """Fetch URL content and return as text."""
            try:
                with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.text[:10000]  # Limit to 10k chars
            except Exception as e:
                raise RuntimeError(f"Failed to fetch {url}: {e}")

        # fetch_url
        self.register(
            name="fetch_url",
            function=fetch_url,
            description="Fetch content from a URL. Returns the page content as text.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch (must start with http:// or https://)",
                    }
                },
                "required": ["url"],
            },
            category="web",
            max_output_chars=6000,  # Web pages can be long
        )

    # =========================================================================
    # Email Tools
    # =========================================================================

    def _register_email_tools(self):
        """Register email tools."""
        try:
            from ollamatoolkit.tools.email import EmailTools

            email = EmailTools()
            if not email.available:
                logger.debug("emailtoolkit not available, skipping email tools")
                return

            self._instances["email"] = email

            # validate_email
            self.register(
                name="validate_email",
                function=lambda email_addr: email.is_valid(email_addr),
                description="Check if an email address is valid.",
                parameters={
                    "type": "object",
                    "properties": {
                        "email_addr": {
                            "type": "string",
                            "description": "Email address to validate",
                        }
                    },
                    "required": ["email_addr"],
                },
                category="email",
            )

            # extract_emails
            self.register(
                name="extract_emails",
                function=lambda text: [
                    e.to_dict() for e in email.extract(text, max_results=10)
                ],
                description="Extract email addresses from text. Returns list of found emails.",
                parameters={
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "Text to extract emails from",
                        }
                    },
                    "required": ["text"],
                },
                category="email",
            )

        except ImportError:
            logger.debug("Email tools not available")

    # =========================================================================
    # Schema Tools
    # =========================================================================

    def _register_schema_tools(self):
        """Register schema validation tools."""
        from ollamatoolkit.tools.schema import SchemaTools

        schema = SchemaTools()
        self._instances["schema"] = schema

        # validate_json
        self.register(
            name="validate_json",
            function=lambda data, json_schema: (
                schema.validate(data, json_schema).to_dict()
                if hasattr(schema.validate(data, json_schema), "to_dict")
                else {
                    "valid": schema.validate(data, json_schema).valid,
                    "errors": schema.validate(data, json_schema).errors,
                }
            ),
            description="Validate JSON data against a JSON schema. Returns validity and errors.",
            parameters={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "object",
                        "description": "JSON data to validate",
                    },
                    "json_schema": {
                        "type": "object",
                        "description": "JSON Schema to validate against",
                    },
                },
                "required": ["data", "json_schema"],
            },
            category="schema",
        )

    # =========================================================================
    # Execution
    # =========================================================================

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> ToolResult:
        """
        Execute a tool with LLM-friendly result handling.

        This is the main entry point for tool execution. It:
        1. Validates the tool exists
        2. Executes with proper error handling
        3. Formats results for LLM consumption
        4. Truncates long outputs
        5. Provides suggestions on failure

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            ToolResult with LLM-friendly output
        """
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")

        # Check tool exists
        if tool_name not in self._tools:
            available = list(self._tools.keys())
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Tool '{tool_name}' not found",
                suggestion=f"Available tools: {', '.join(available[:10])}",
            )

        tool = self._tools[tool_name]

        try:
            # Execute the tool
            result = tool.function(**arguments)

            # Format the result
            formatted = self._format_result(result, tool)

            logger.info(f"Tool {tool_name} succeeded: {formatted.summary}")
            return formatted

        except TypeError as e:
            # Wrong arguments
            logger.warning(f"Tool {tool_name} type error: {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Invalid arguments: {e}",
                suggestion=self._get_argument_help(tool),
            )

        except FileNotFoundError as e:
            logger.warning(f"Tool {tool_name} file not found: {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"File not found: {e}",
                suggestion="Check the file path. Use 'list_files' to see available files.",
            )

        except PermissionError as e:
            logger.warning(f"Tool {tool_name} permission denied: {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Permission denied: {e}",
                suggestion="Check file permissions or try a different path.",
            )

        except json.JSONDecodeError as e:
            logger.warning(f"Tool {tool_name} JSON error: {e}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=f"Invalid JSON: {e}",
                suggestion="Ensure the data is valid JSON format.",
            )

        except Exception as e:
            # Generic error with stack trace for debugging
            logger.error(f"Tool {tool_name} failed: {e}\n{traceback.format_exc()}")
            return ToolResult(
                success=False,
                tool_name=tool_name,
                error=str(e),
                suggestion=self._get_error_suggestion(tool_name, e),
            )

    def _format_result(self, result: Any, tool: ToolDefinition) -> ToolResult:
        """Format a successful result for LLM consumption."""
        # Convert to string for length checking
        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, indent=2, default=str)
        else:
            result_str = str(result)

        original_length = len(result_str)
        truncated = False

        # Truncate if too long
        if original_length > tool.max_output_chars:
            result_str = result_str[: tool.max_output_chars] + "\n...[truncated]"
            truncated = True

        # Generate summary
        summary = self._generate_summary(result, tool.name)

        return ToolResult(
            success=True,
            tool_name=tool.name,
            result=result_str if truncated else result,
            summary=summary,
            truncated=truncated,
            original_length=original_length,
        )

    def _generate_summary(self, result: Any, tool_name: str) -> str:
        """Generate a one-line summary of the result."""
        if result is None:
            return "Operation completed"

        if isinstance(result, bool):
            return "True" if result else "False"

        if isinstance(result, str):
            lines = result.count("\n") + 1
            chars = len(result)
            return f"Returned {chars} chars ({lines} lines)"

        if isinstance(result, list):
            return f"Returned {len(result)} items"

        if isinstance(result, dict):
            keys = list(result.keys())[:3]
            return f"Returned object with keys: {keys}"

        return f"Returned {type(result).__name__}"

    def _get_argument_help(self, tool: ToolDefinition) -> str:
        """Generate help text for tool arguments."""
        props = tool.parameters.get("properties", {})
        required = tool.parameters.get("required", [])

        parts = []
        for name, schema in props.items():
            req = "(required)" if name in required else "(optional)"
            desc = schema.get("description", "")
            parts.append(f"{name} {req}: {desc}")

        return "Expected arguments: " + "; ".join(parts)

    def _get_error_suggestion(self, tool_name: str, error: Exception) -> str:
        """Generate a helpful suggestion based on the error."""
        error_str = str(error).lower()

        if "connection" in error_str or "network" in error_str:
            return "Check network connection and try again."

        if "timeout" in error_str:
            return "Operation timed out. Try with a simpler request."

        if "memory" in error_str:
            return "Out of memory. Try with smaller data."

        if "not found" in error_str:
            return "Resource not found. Verify it exists first."

        return "Check the arguments and try again. Use list_tools to see available options."

    # =========================================================================
    # Schema Generation
    # =========================================================================

    def get_schemas(self, categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible function schemas for all tools.

        Args:
            categories: Filter by category (None = all)

        Returns:
            List of tool schemas for function calling
        """
        schemas = []
        for tool in self._tools.values():
            if categories is None or tool.category in categories:
                schemas.append(tool.to_openai_schema())
        return schemas

    def get_function_map(self) -> Dict[str, Callable]:
        """
        Get function map for use with SimpleAgent.

        Returns:
            Dict mapping tool names to their execute wrappers
        """
        return {
            name: lambda args, n=name: self.execute(n, args).to_llm_string()
            for name in self._tools.keys()
        }

    def list_tools(self, category: str = None) -> List[Dict[str, str]]:
        """
        List available tools with descriptions.

        Args:
            category: Filter by category

        Returns:
            List of tool info dicts
        """
        tools = []
        for tool in self._tools.values():
            if category is None or tool.category == category:
                tools.append(
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "category": tool.category,
                    }
                )
        return tools

    def get_tool_help(self, tool_name: str) -> str:
        """
        Get detailed help for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Formatted help string
        """
        if tool_name not in self._tools:
            return f"Tool '{tool_name}' not found"

        tool = self._tools[tool_name]
        lines = [
            f"Tool: {tool.name}",
            f"Category: {tool.category}",
            f"Description: {tool.description}",
            "",
            "Parameters:",
        ]

        for name, schema in tool.parameters.get("properties", {}).items():
            req = (
                "required"
                if name in tool.parameters.get("required", [])
                else "optional"
            )
            desc = schema.get("description", "")
            ptype = schema.get("type", "any")
            lines.append(f"  - {name} ({ptype}, {req}): {desc}")

        if tool.examples:
            lines.append("")
            lines.append("Examples:")
            for ex in tool.examples:
                lines.append(f"  {ex}")

        return "\n".join(lines)

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    @property
    def categories(self) -> List[str]:
        """List of tool categories."""
        return list({t.category for t in self._tools.values()})
