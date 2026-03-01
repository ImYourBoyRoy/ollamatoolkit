# ./src/ollamatoolkit/agents/simple.py
"""Simple agent runtime for synchronous/async LiteLLM conversations with tool execution.

Run via code import: ``from ollamatoolkit.agents import SimpleAgent``.
Inputs: model_config dict (model/base_url/api_key/temperature/caching/fallbacks),
optional tool schemas + function map, optional ConversationMemory, and user prompts.
Outputs: assistant text responses, streaming token iterator, structured Pydantic objects,
and in-memory conversation history entries suitable for chat-completions APIs.
Side effects: invokes remote model APIs, executes registered tool callables, and may
persist/modify external state through those tools. Keep tool functions idempotent when possible.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
    cast,
)

from litellm import acompletion, completion
from pydantic import BaseModel

if TYPE_CHECKING:
    from .memory import ConversationMemory

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT", bound=BaseModel)
MessageDict = Dict[str, Any]
ToolCallable = Callable[..., Any]
HookCallable = Callable[..., None]


@dataclass
class AgentHooks:
    """Lifecycle hooks for integrating UI, telemetry, and orchestration callbacks."""

    on_start: Optional[Callable[[str], None]] = None
    on_token: Optional[Callable[[str], None]] = None
    on_thinking_start: Optional[Callable[[], None]] = None
    on_thinking_end: Optional[Callable[[str], None]] = None
    on_tool_call: Optional[Callable[[str, Dict[str, Any]], None]] = None
    on_tool_result: Optional[Callable[[str, str], None]] = None
    on_end: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None

    def as_dict(self) -> Dict[str, HookCallable]:
        """Expose non-null hooks in legacy dict format."""
        return {
            name: cast(HookCallable, hook)
            for name, hook in self.__dict__.items()
            if hook is not None
        }


class SimpleAgent:
    """Minimal conversational agent with tool-calls, streaming, and structured output."""

    def __init__(
        self,
        name: str,
        system_message: str,
        model_config: Dict[str, Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        function_map: Optional[Dict[str, ToolCallable]] = None,
        mock_responses: Optional[List[Any]] = None,
        hooks: Optional[Union[AgentHooks, Dict[str, HookCallable]]] = None,
        history_limit: int = 50,
        memory: Optional["ConversationMemory"] = None,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.config = model_config
        self.tools: List[Dict[str, Any]] = tools or []
        self.function_map: Dict[str, ToolCallable] = function_map or {}
        self.mock_responses: List[Any] = mock_responses or []
        self.history_limit = history_limit
        self.memory = memory

        if isinstance(hooks, AgentHooks):
            self._hooks = hooks
        elif isinstance(hooks, Mapping):
            self._hooks = AgentHooks(
                **{
                    name: hook
                    for name, hook in hooks.items()
                    if hasattr(AgentHooks, name)
                }
            )
        else:
            self._hooks = AgentHooks()

        # Backwards-compatible property used by existing code paths.
        self.hooks: Dict[str, HookCallable] = self._hooks.as_dict()

        if self.memory is not None:
            self.memory.add_message("system", system_message)
            self.history = self.memory.messages
        else:
            self.history = [{"role": "system", "content": system_message}]

    def _get_hook(self, name: str) -> Optional[HookCallable]:
        hook = getattr(self._hooks, name, None)
        if hook is None and name == "on_tool_call":
            # Legacy hook name support.
            return self.hooks.get("on_tool")
        return hook

    def _call_hook(self, name: str, *args: Any) -> None:
        hook = self._get_hook(name)
        if hook is None:
            return
        try:
            hook(*args)
        except Exception as exc:  # pragma: no cover - defensive callback isolation
            logger.debug("Hook '%s' failed: %s", name, exc)

    # ------------------------------------------------------------------
    # Tool registration and schema generation
    # ------------------------------------------------------------------
    def register_tool(self, func: ToolCallable) -> ToolCallable:
        """Register a callable as an LLM-invokable tool."""
        schema = getattr(func, "_tool_def", None)
        if not isinstance(schema, dict):
            schema = self._generate_tool_schema(func)

        function_section = schema.get("function", {})
        tool_name = str(function_section.get("name", func.__name__))
        self.tools.append(schema)
        self.function_map[tool_name] = func
        return func

    def tool(self) -> Callable[[ToolCallable], ToolCallable]:
        """Decorator form of ``register_tool``."""

        def wrapper(func: ToolCallable) -> ToolCallable:
            return self.register_tool(func)

        return wrapper

    def _generate_tool_schema(self, func: ToolCallable) -> Dict[str, Any]:
        signature = inspect.signature(func)
        properties: Dict[str, Dict[str, str]] = {}
        required: List[str] = []

        for param_name, parameter in signature.parameters.items():
            if param_name == "self":
                continue
            properties[param_name] = {
                "type": self._annotation_to_json_type(parameter.annotation)
            }
            if parameter.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": inspect.getdoc(func) or "No description provided.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    @staticmethod
    def _annotation_to_json_type(annotation: Any) -> str:
        if annotation is int:
            return "integer"
        if annotation is float:
            return "number"
        if annotation is bool:
            return "boolean"
        return "string"

    # ------------------------------------------------------------------
    # Conversation helpers
    # ------------------------------------------------------------------
    def _prune_history(self) -> None:
        """Trim legacy in-memory history when memory module is not managing context."""
        if self.memory is not None:
            return
        if len(self.history) <= self.history_limit:
            return

        keep = max(1, self.history_limit - 1)
        self.history = [self.history[0], *self.history[-keep:]]

    @staticmethod
    def _read(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, Mapping):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _normalize_message(
        self, raw: Any, default_role: str = "assistant"
    ) -> MessageDict:
        message: MessageDict = {
            "role": str(self._read(raw, "role", default_role) or default_role),
            "content": self._coerce_content_to_text(self._read(raw, "content", "")),
        }

        tool_call_id = self._read(raw, "tool_call_id")
        if isinstance(tool_call_id, str) and tool_call_id:
            message["tool_call_id"] = tool_call_id

        tool_calls = self._normalize_tool_calls(self._read(raw, "tool_calls"))
        if tool_calls:
            message["tool_calls"] = tool_calls

        return message

    def _normalize_tool_calls(self, raw: Any) -> List[MessageDict]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raw = list(raw) if isinstance(raw, tuple) else [raw]

        tool_calls: List[MessageDict] = []
        for fallback_index, call in enumerate(raw):
            function_data = self._read(call, "function", {})
            name = str(self._read(function_data, "name", "") or "")
            arguments = str(self._read(function_data, "arguments", "") or "")

            normalized: MessageDict = {
                "type": str(self._read(call, "type", "function") or "function"),
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            }

            call_id = self._read(call, "id")
            if isinstance(call_id, str) and call_id:
                normalized["id"] = call_id

            index_value = self._read(call, "index")
            if isinstance(index_value, int):
                normalized["index"] = index_value
            else:
                normalized["index"] = fallback_index

            tool_calls.append(normalized)
        return tool_calls

    @staticmethod
    def _coerce_content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, (dict, list)):
            try:
                return json.dumps(content, ensure_ascii=False)
            except (TypeError, ValueError):
                return str(content)
        return str(content)

    def _append_history(self, message: MessageDict) -> None:
        self.history.append(message)

    def _build_completion_kwargs(
        self,
        *,
        stream: bool = False,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model": self.config["model"],
            "api_base": self.config.get("base_url"),
            "api_key": self.config.get("api_key"),
            "messages": self.history,
            "temperature": self.config.get("temperature", 0.0),
        }
        if self.tools:
            kwargs["tools"] = self.tools
        if stream:
            kwargs["stream"] = True
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.config.get("caching", False):
            kwargs["caching"] = True
        if self.config.get("fallbacks"):
            kwargs["fallbacks"] = self.config["fallbacks"]
        return kwargs

    def _get_mock_message(self) -> Optional[MessageDict]:
        if not self.mock_responses:
            return None

        mock = self.mock_responses.pop(0)
        if hasattr(mock, "choices"):
            choices = self._read(mock, "choices", [])
            if choices:
                mock = self._read(choices[0], "message", mock)

        return self._normalize_message(mock)

    @staticmethod
    def _extract_response_message(response: Any) -> Any:
        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Completion response missing choices")
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            raise ValueError("Completion response missing choice.message")
        return message

    @staticmethod
    def _parse_tool_args(args_str: str) -> Dict[str, Any]:
        if not args_str.strip():
            return {}
        try:
            parsed = json.loads(args_str)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _run_awaitable_sync(value: Awaitable[Any]) -> Any:
        async def _await_value(awaitable: Awaitable[Any]) -> Any:
            return await awaitable

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if asyncio.iscoroutine(value):
                return asyncio.run(value)
            return asyncio.run(_await_value(value))
        raise RuntimeError(
            "Async tool returned in sync execution while an event loop is already running. "
            "Use run_async/step_async for async tools."
        )

    def _execute_tool_sync(self, tool_call: MessageDict) -> MessageDict:
        function_data = cast(Dict[str, Any], tool_call.get("function", {}))
        func_name = str(function_data.get("name", "") or "")
        args_str = str(function_data.get("arguments", "") or "")
        call_id = str(tool_call.get("id", "") or "")
        args = self._parse_tool_args(args_str)

        self._call_hook("on_tool_call", func_name, args)

        try:
            handler = self.function_map.get(func_name)
            if handler is None:
                result = f"Error: Tool {func_name} not found."
            else:
                value = handler(**args)
                if inspect.isawaitable(value):
                    value = self._run_awaitable_sync(value)
                result = str(value)
        except Exception as exc:
            result = f"Error executing tool {func_name}: {exc}"

        self._call_hook("on_tool_result", func_name, result)

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }

    async def _execute_tool_async(self, tool_call: MessageDict) -> MessageDict:
        function_data = cast(Dict[str, Any], tool_call.get("function", {}))
        func_name = str(function_data.get("name", "") or "")
        args_str = str(function_data.get("arguments", "") or "")
        call_id = str(tool_call.get("id", "") or "")
        args = self._parse_tool_args(args_str)

        self._call_hook("on_tool_call", func_name, args)

        try:
            handler = self.function_map.get(func_name)
            if handler is None:
                result = f"Error: Tool {func_name} not found."
            else:
                value = handler(**args)
                if inspect.isawaitable(value):
                    value = await value
                result = str(value)
        except Exception as exc:
            result = f"Error executing tool {func_name}: {exc}"

        self._call_hook("on_tool_result", func_name, result)

        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }

    # ------------------------------------------------------------------
    # Sync execution
    # ------------------------------------------------------------------
    def step(self) -> Optional[MessageDict]:
        """Execute one agent step and return the produced message payload."""
        try:
            message = self._get_mock_message()
            if message is None:
                response = completion(**self._build_completion_kwargs())
                raw_message = self._extract_response_message(response)
                message = self._normalize_message(raw_message)

            self._append_history(message)

            tool_calls = cast(List[MessageDict], message.get("tool_calls", []))
            if tool_calls:
                tool_result_msg: Optional[MessageDict] = None
                for tool_call in tool_calls:
                    tool_result_msg = self._execute_tool_sync(tool_call)
                    self._append_history(tool_result_msg)
                return tool_result_msg

            content = str(message.get("content", "") or "")
            if content:
                self._call_hook("on_end", content)
            return message

        except Exception as exc:
            logger.error("Agent %s step error: %s", self.name, exc)
            self._call_hook("on_error", exc)
            return None

    def run(self, user_message: str, max_turns: int = 10) -> str:
        """Run the loop until a final assistant message is produced or turns are exhausted."""
        self._prune_history()
        self._append_history({"role": "user", "content": user_message})
        self._call_hook("on_start", user_message)

        final_content = ""
        for _ in range(max_turns):
            message = self.step()
            if not message:
                break

            role = str(message.get("role", "") or "")
            content = str(message.get("content", "") or "")

            if role == "tool":
                continue
            if role == "assistant" and not message.get("tool_calls"):
                final_content = content
                if "TERMINATE" in content or "Mission Complete" in content:
                    break
                return content

        return final_content

    def run_streaming(self, user_message: str, max_turns: int = 10) -> Iterator[str]:
        """Run with streamed token output while still supporting tool-call turns."""
        self._prune_history()
        self._append_history({"role": "user", "content": user_message})
        self._call_hook("on_start", user_message)

        for _ in range(max_turns):
            self._call_hook("on_thinking_start")

            mock_message = self._get_mock_message()
            if mock_message is not None:
                content = str(mock_message.get("content", "") or "")
                self._append_history(mock_message)
                for char in content:
                    self._call_hook("on_token", char)
                    yield char
                self._call_hook("on_thinking_end", content)
                self._call_hook("on_end", content)
                return

            try:
                full_content = ""
                stream_tool_calls: List[MessageDict] = []

                for chunk in completion(**self._build_completion_kwargs(stream=True)):
                    choices = self._read(chunk, "choices", [])
                    if not choices:
                        continue
                    delta = self._read(choices[0], "delta")
                    if delta is None:
                        continue

                    token = self._coerce_content_to_text(self._read(delta, "content"))
                    if token:
                        full_content += token
                        self._call_hook("on_token", token)
                        yield token

                    new_calls = self._normalize_tool_calls(
                        self._read(delta, "tool_calls")
                    )
                    if new_calls:
                        self._merge_stream_tool_calls(stream_tool_calls, new_calls)

                self._call_hook("on_thinking_end", full_content)

                assistant_message: MessageDict = {
                    "role": "assistant",
                    "content": full_content,
                }
                if stream_tool_calls:
                    assistant_message["tool_calls"] = stream_tool_calls
                self._append_history(assistant_message)

                if stream_tool_calls:
                    for tool_call in stream_tool_calls:
                        tool_result = self._execute_tool_sync(tool_call)
                        self._append_history(tool_result)
                    continue

                if full_content:
                    if (
                        "TERMINATE" in full_content
                        or "Mission Complete" in full_content
                    ):
                        break
                    self._call_hook("on_end", full_content)
                return

            except Exception as exc:
                logger.error("Agent %s streaming error: %s", self.name, exc)
                self._call_hook("on_error", exc)
                return

    def _merge_stream_tool_calls(
        self,
        accumulated: List[MessageDict],
        updates: List[MessageDict],
    ) -> None:
        for update in updates:
            index_value = update.get("index")
            if not isinstance(index_value, int) or index_value < 0:
                index_value = len(accumulated)

            while len(accumulated) <= index_value:
                accumulated.append(
                    {
                        "index": len(accumulated),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                )

            current = accumulated[index_value]

            update_id = update.get("id")
            if isinstance(update_id, str) and update_id:
                current["id"] = update_id

            current["type"] = str(update.get("type", "function") or "function")
            current["index"] = index_value

            current_function = cast(Dict[str, Any], current.setdefault("function", {}))
            update_function = cast(Dict[str, Any], update.get("function", {}))

            update_name = str(update_function.get("name", "") or "")
            if update_name:
                current_function["name"] = update_name

            update_args = str(update_function.get("arguments", "") or "")
            if update_args:
                current_function["arguments"] = (
                    str(current_function.get("arguments", "")) + update_args
                )

    # ------------------------------------------------------------------
    # Async execution
    # ------------------------------------------------------------------
    async def step_async(self) -> Optional[MessageDict]:
        """Async variant of ``step`` that also awaits async tool callables."""
        try:
            message = self._get_mock_message()
            if message is None:
                response = await acompletion(**self._build_completion_kwargs())
                raw_message = self._extract_response_message(response)
                message = self._normalize_message(raw_message)

            self._append_history(message)

            tool_calls = cast(List[MessageDict], message.get("tool_calls", []))
            if tool_calls:
                tool_result_msg: Optional[MessageDict] = None
                for tool_call in tool_calls:
                    tool_result_msg = await self._execute_tool_async(tool_call)
                    self._append_history(tool_result_msg)
                return tool_result_msg

            content = str(message.get("content", "") or "")
            if content:
                self._call_hook("on_end", content)
            return message

        except Exception as exc:
            logger.error("Agent %s async step error: %s", self.name, exc)
            self._call_hook("on_error", exc)
            return None

    async def run_async(self, user_message: str, max_turns: int = 10) -> str:
        """Async full loop equivalent of ``run``."""
        self._prune_history()
        self._append_history({"role": "user", "content": user_message})
        self._call_hook("on_start", user_message)

        final_content = ""
        for _ in range(max_turns):
            message = await self.step_async()
            if not message:
                break

            role = str(message.get("role", "") or "")
            content = str(message.get("content", "") or "")
            if role == "tool":
                continue

            if role == "assistant" and not message.get("tool_calls"):
                final_content = content
                return content

        return final_content

    # ------------------------------------------------------------------
    # Structured output
    # ------------------------------------------------------------------
    def run_structured(
        self,
        user_message: str,
        response_model: Type[ModelT],
        max_retries: int = 3,
    ) -> ModelT:
        """Return a validated Pydantic object produced by the model."""
        schema = response_model.model_json_schema()
        self._append_history(
            {
                "role": "system",
                "content": f"Return JSON matching this schema:\n{json.dumps(schema)}",
            }
        )
        self._append_history({"role": "user", "content": user_message})

        for _ in range(max_retries):
            content = ""
            mock_message = self._get_mock_message()
            if mock_message is not None:
                content = str(mock_message.get("content", "") or "")

            if not content:
                response = completion(
                    **self._build_completion_kwargs(
                        response_format={"type": "json_object"}
                    )
                )
                raw_message = self._extract_response_message(response)
                content = self._coerce_content_to_text(
                    self._read(raw_message, "content")
                )

            try:
                payload = json.loads(content)
                result = response_model.model_validate(payload)
                self._call_hook("on_end", result)
                return result
            except Exception as exc:
                logger.warning("Structured validation failed: %s", exc)
                self._append_history(
                    {
                        "role": "user",
                        "content": f"Validation error: {exc}. Return corrected JSON only.",
                    }
                )

        raise ValueError("Failed to obtain valid structured output")

    async def run_structured_async(
        self,
        user_message: str,
        response_model: Type[ModelT],
        max_retries: int = 3,
    ) -> ModelT:
        """Async variant of ``run_structured``."""
        schema = response_model.model_json_schema()
        self._append_history(
            {
                "role": "system",
                "content": f"Return JSON matching this schema:\n{json.dumps(schema)}",
            }
        )
        self._append_history({"role": "user", "content": user_message})

        for _ in range(max_retries):
            content = ""
            mock_message = self._get_mock_message()
            if mock_message is not None:
                content = str(mock_message.get("content", "") or "")

            if not content:
                response = await acompletion(
                    **self._build_completion_kwargs(
                        response_format={"type": "json_object"}
                    )
                )
                raw_message = self._extract_response_message(response)
                content = self._coerce_content_to_text(
                    self._read(raw_message, "content")
                )

            try:
                payload = json.loads(content)
                result = response_model.model_validate(payload)
                self._call_hook("on_end", result)
                return result
            except Exception as exc:
                logger.warning("Async structured validation failed: %s", exc)
                self._append_history(
                    {
                        "role": "user",
                        "content": f"Validation error: {exc}. Return corrected JSON only.",
                    }
                )

        raise ValueError("Failed to obtain valid structured output")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def load_mocks_from_log(log_path: Path) -> List[Dict[str, str]]:
        """Build mock response entries from a JSON telemetry log file."""
        if not log_path.exists():
            return []

        try:
            data = json.loads(log_path.read_text(encoding="utf-8"))
            records = data if isinstance(data, list) else [data]
            mocks: List[Dict[str, str]] = []
            for item in records:
                if not isinstance(item, Mapping):
                    continue
                response_text = str(item.get("response", "") or "")
                mocks.append({"role": "assistant", "content": response_text})
            return mocks
        except Exception as exc:
            logger.error("Failed to load mocks from %s: %s", log_path, exc)
            return []
