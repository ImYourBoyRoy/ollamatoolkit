# ./tests/test_types.py
"""
Tests for Pydantic type definitions.
"""

import pytest
from ollamatoolkit.types import (
    Message,
    Tool,
    Options,
    GenerateRequest,
    GenerateResponse,
    ChatRequest,
    ChatResponse,
    EmbedRequest,
    EmbedResponse,
    ListResponse,
    StreamEvent,
    ResponseError,
    RequestError,
)


class TestSubscriptableBaseModel:
    """Test dict-like access on models."""

    def test_getitem(self):
        msg = Message(role="user", content="Hello")
        assert msg["role"] == "user"
        assert msg["content"] == "Hello"

    def test_getitem_missing_raises(self):
        msg = Message(role="user")
        with pytest.raises(KeyError):
            _ = msg["nonexistent"]

    def test_setitem(self):
        msg = Message(role="user")
        msg["content"] = "Updated"
        assert msg.content == "Updated"

    def test_contains(self):
        msg = Message(role="user", content="Hello")
        assert "role" in msg
        assert "content" in msg
        # unset optional field
        msg2 = Message(role="system")
        assert "thinking" not in msg2

    def test_get_method(self):
        msg = Message(role="assistant")
        assert msg.get("role") == "assistant"
        assert msg.get("nonexistent") is None
        assert msg.get("nonexistent", "default") == "default"


class TestMessage:
    """Test Message model."""

    def test_basic_message(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_with_images(self):
        msg = Message(role="user", content="Check this", images=["base64data"])
        assert msg.images == ["base64data"]

    def test_message_with_tool_calls(self):
        tool_call = Message.ToolCall(
            id="call_1",
            index=0,
            function=Message.ToolCall.Function(name="search", arguments={"q": "test"}),
        )
        msg = Message(role="assistant", tool_calls=[tool_call])
        assert msg.tool_calls[0].id == "call_1"
        assert msg.tool_calls[0].index == 0
        assert msg.tool_calls[0].function.name == "search"
        assert msg.tool_calls[0].function.arguments == {"q": "test"}

    def test_message_tool_call_id_field(self):
        msg = Message(role="tool", content="{}", tool_call_id="call_123")
        assert msg.tool_call_id == "call_123"

    def test_serialization(self):
        msg = Message(role="user", content="Test")
        data = msg.model_dump(exclude_none=True)
        assert data == {"role": "user", "content": "Test"}


class TestOptions:
    """Test Options model."""

    def test_default_options(self):
        opts = Options()
        assert opts.temperature is None
        assert opts.num_ctx is None

    def test_custom_options(self):
        opts = Options(temperature=0.7, num_ctx=4096, top_p=0.9)
        assert opts.temperature == 0.7
        assert opts.num_ctx == 4096
        assert opts.top_p == 0.9


class TestTool:
    """Test Tool model."""

    def test_basic_tool(self):
        tool = Tool(
            function=Tool.Function(
                name="get_weather",
                description="Get weather for a location",
                parameters=Tool.Function.Parameters(
                    properties={
                        "location": Tool.Function.Parameters.Property(
                            type="string", description="City name"
                        )
                    },
                    required=["location"],
                ),
            )
        )
        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert "location" in tool.function.parameters.properties


class TestGenerateRequest:
    """Test GenerateRequest model."""

    def test_basic_request(self):
        req = GenerateRequest(model="llama2", prompt="Hello")
        assert req.model == "llama2"
        assert req.prompt == "Hello"

    def test_request_serialization(self):
        req = GenerateRequest(model="llama2", prompt="Hi", temperature=0.5, stream=True)
        # Options should be nested properly if passed
        data = req.model_dump(exclude_none=True)
        assert data["model"] == "llama2"
        assert data["stream"] is True


class TestGenerateResponse:
    """Test GenerateResponse model."""

    def test_basic_response(self):
        resp = GenerateResponse(
            model="llama2",
            response="Hello back!",
            done=True,
            total_duration=1000000,
        )
        assert resp.response == "Hello back!"
        assert resp.done is True


class TestChatRequest:
    """Test ChatRequest model."""

    def test_chat_request(self):
        req = ChatRequest(
            model="llama2",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert req.model == "llama2"
        assert len(req.messages) == 1


class TestChatResponse:
    """Test ChatResponse model."""

    def test_chat_response(self):
        resp = ChatResponse(
            model="llama2",
            message=Message(role="assistant", content="Hello!"),
            done=True,
        )
        assert resp.message.role == "assistant"
        assert resp.message.content == "Hello!"


class TestEmbedRequest:
    """Test EmbedRequest model."""

    def test_single_input(self):
        req = EmbedRequest(model="nomic-embed-text", input="Hello world")
        assert req.input == "Hello world"

    def test_multiple_inputs(self):
        req = EmbedRequest(model="nomic-embed-text", input=["Hello", "World"])
        assert len(req.input) == 2


class TestEmbedResponse:
    """Test EmbedResponse model."""

    def test_embeddings(self):
        resp = EmbedResponse(
            model="nomic-embed-text", embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        assert len(resp.embeddings) == 2
        assert len(resp.embeddings[0]) == 3


class TestStreamEvent:
    """Test StreamEvent model."""

    def test_stream_event_token(self):
        event = StreamEvent(event="token", chunk_index=0, text="Hello")
        assert event.event == "token"
        assert event.text == "Hello"


class TestListResponse:
    """Test ListResponse model."""

    def test_list_models(self):
        resp = ListResponse(
            models=[
                ListResponse.Model(model="llama2", name="llama2"),
                ListResponse.Model(model="mistral", name="mistral"),
            ]
        )
        assert len(resp.models) == 2
        assert resp.models[0].model == "llama2"


class TestErrors:
    """Test error classes."""

    def test_request_error(self):
        err = RequestError("Invalid model")
        assert str(err) == "Invalid model"
        assert err.error == "Invalid model"

    def test_response_error(self):
        err = ResponseError("Not found", 404)
        assert "Not found" in str(err)
        assert "404" in str(err)
        assert err.status_code == 404

    def test_response_error_parses_json(self):
        err = ResponseError('{"error": "Model not found"}', 404)
        assert err.error == "Model not found"
