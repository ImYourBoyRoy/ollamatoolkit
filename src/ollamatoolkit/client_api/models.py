# ./src/ollamatoolkit/client_api/models.py
"""
Model-management endpoint adapters for OllamaToolkit clients.
Run: imported by sync/async client façades.
Inputs: model identifiers and model management request options.
Outputs: typed model metadata, progress, and status responses.
Side effects: may pull/push/create/delete/copy models on Ollama server.
Operational notes: capability/context helpers are heuristic and backward compatible.
"""

from __future__ import annotations

import logging
from typing import (
    Any,
    AsyncIterator,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Union,
    cast,
)

from ollamatoolkit.client_api.transport import AsyncTransport, SyncTransport
from ollamatoolkit.exceptions import ModelNotFoundError
from ollamatoolkit.types import (
    CopyRequest,
    CreateRequest,
    DeleteRequest,
    ListResponse,
    Message,
    ModelDetails,
    Options,
    ProcessResponse,
    ProgressResponse,
    PullRequest,
    PushRequest,
    ResponseError,
    ShowRequest,
    ShowResponse,
    StatusResponse,
    VersionResponse,
)

logger = logging.getLogger(__name__)


class SyncModelAPI:
    """Synchronous adapter for model-management endpoints."""

    def __init__(self, transport: SyncTransport) -> None:
        self._transport = transport

    def list(self) -> ListResponse:
        """List available models from `/api/tags`."""
        return self._transport.request(ListResponse, "GET", "/api/tags")

    def version(self) -> VersionResponse:
        """Get Ollama server version from `/api/version`."""
        return self._transport.request(VersionResponse, "GET", "/api/version")

    def show(self, model: str) -> ShowResponse:
        """Show detailed model metadata from `/api/show`."""
        request = ShowRequest(model=model)
        return self._transport.request(
            ShowResponse,
            "POST",
            "/api/show",
            json=request.model_dump(exclude_none=True),
        )

    def ps(self) -> ProcessResponse:
        """List running models from `/api/ps`."""
        return self._transport.request(ProcessResponse, "GET", "/api/ps")

    def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Pull model data from registry via `/api/pull`."""
        request = PullRequest(model=model, insecure=insecure, stream=stream)
        payload = request.model_dump(exclude_none=True)
        if stream:
            return self._transport.stream(
                ProgressResponse,
                "POST",
                "/api/pull",
                json=payload,
            )
        return self._transport.request(
            ProgressResponse,
            "POST",
            "/api/pull",
            json=payload,
        )

    def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Push model data to registry via `/api/push`."""
        request = PushRequest(model=model, insecure=insecure, stream=stream)
        payload = request.model_dump(exclude_none=True)
        if stream:
            return self._transport.stream(
                ProgressResponse,
                "POST",
                "/api/push",
                json=payload,
            )
        return self._transport.request(
            ProgressResponse,
            "POST",
            "/api/push",
            json=payload,
        )

    def create(
        self,
        model: str,
        *,
        quantize: Optional[str] = None,
        from_: Optional[str] = None,
        files: Optional[dict[str, str]] = None,
        adapters: Optional[dict[str, str]] = None,
        template: Optional[str] = None,
        license: Optional[Union[str, List[str]]] = None,
        system: Optional[str] = None,
        parameters: Optional[Union[Mapping[str, Any], Options]] = None,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        stream: bool = False,
    ) -> Union[ProgressResponse, Iterator[ProgressResponse]]:
        """Create a model via `/api/create`."""
        request = CreateRequest(
            model=model,
            stream=stream,
            quantize=quantize,
            from_=from_,
            files=files,
            adapters=adapters,
            template=template,
            license=license,
            system=system,
            parameters=parameters,
            messages=messages,
        )
        payload = request.model_dump(exclude_none=True)
        if stream:
            return self._transport.stream(
                ProgressResponse,
                "POST",
                "/api/create",
                json=payload,
            )
        return self._transport.request(
            ProgressResponse,
            "POST",
            "/api/create",
            json=payload,
        )

    def delete(self, model: str) -> StatusResponse:
        """Delete a model via `/api/delete`."""
        response = self._transport.request_raw(
            "DELETE",
            "/api/delete",
            json=DeleteRequest(model=model).model_dump(exclude_none=True),
        )
        return StatusResponse(
            status="success" if response.status_code == 200 else "error"
        )

    def copy(self, source: str, destination: str) -> StatusResponse:
        """Copy a model via `/api/copy`."""
        response = self._transport.request_raw(
            "POST",
            "/api/copy",
            json=CopyRequest(source=source, destination=destination).model_dump(
                exclude_none=True
            ),
        )
        return StatusResponse(
            status="success" if response.status_code == 200 else "error"
        )

    def ensure_model_available(
        self,
        model: str,
        auto_pull: bool = True,
        stream_progress: bool = False,
    ) -> bool:
        """Ensure a model exists locally, optionally pulling missing models."""
        try:
            self.show(model)
            return True
        except ResponseError as exc:
            if "not found" not in str(exc).lower():
                raise

            if not auto_pull:
                raise ModelNotFoundError(model) from None

            logger.info("Model '%s' not found locally. Pulling...", model)
            try:
                if stream_progress:
                    progress_stream = cast(
                        Iterator[ProgressResponse],
                        self.pull(model, stream=True),
                    )
                    for progress in progress_stream:
                        status = progress.status or ""
                        if progress.completed and progress.total:
                            pct = (progress.completed / progress.total) * 100
                            print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
                        else:
                            print(f"\r  {status}", end="", flush=True)
                    print()
                else:
                    self.pull(model, stream=False)
                logger.info("Successfully pulled model '%s'", model)
                return True
            except Exception as pull_error:
                logger.error("Failed to pull model '%s': %s", model, pull_error)
                raise ModelNotFoundError(
                    model,
                    f"Model '{model}' not found and pull failed: {pull_error}",
                ) from pull_error

    def get_model_details(self, model: str) -> dict[str, Any]:
        """Get merged detail/modelinfo metadata for a model."""
        show_response = self.show(model)
        details = (
            show_response.details.model_dump(exclude_none=True)
            if isinstance(show_response.details, ModelDetails)
            else {}
        )
        model_info = dict(show_response.modelinfo or {})
        return {**details, **model_info}

    def get_model_capabilities(self, model: str) -> List[str]:
        """Infer capabilities for a model using template and metadata heuristics."""
        try:
            show_response = self.show(model)
        except Exception:
            return []

        caps: set[str] = set()
        details = (
            show_response.details.model_dump(exclude_none=True)
            if isinstance(show_response.details, ModelDetails)
            else {}
        )
        model_info = dict(show_response.modelinfo or {})
        full_meta = {**details, **model_info}
        str_meta = str(full_meta).lower()
        families = details.get("families", []) or [details.get("family")]
        families_str = str(families).lower()

        if "bert" in families_str or "nomic-bert" in families_str:
            caps.add("embedding")

        is_embedding_only = "bert" in families_str and "llama" not in families_str
        if not is_embedding_only:
            caps.add("completion")
            template = (show_response.template or "").lower()
            if (
                "{{ .messages" in template
                or "message" in template
                or "[inst]" in template
                or "<|im_start|>" in template
            ):
                caps.add("chat")

        if "clip" in families or "mclip" in families or "vision" in str_meta:
            caps.add("vision")

        template = (show_response.template or "").lower()
        if "tool" in template:
            caps.add("tools")

        return sorted(caps)

    def get_model_context_length(self, model: str) -> int:
        """Return model context length from metadata, falling back to 4096."""
        try:
            show_response = self.show(model)
            model_info = dict(show_response.modelinfo or {})

            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    return int(value)

            if show_response.parameters:
                for line in show_response.parameters.split("\n"):
                    if "num_ctx" in line or "context_length" in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[-1].isdigit():
                            return int(parts[-1])
        except Exception:
            return 4096

        return 4096


class AsyncModelAPI:
    """Asynchronous adapter for model-management endpoints."""

    def __init__(self, transport: AsyncTransport) -> None:
        self._transport = transport

    async def list(self) -> ListResponse:
        """List available models from `/api/tags`."""
        return await self._transport.request(ListResponse, "GET", "/api/tags")

    async def version(self) -> VersionResponse:
        """Get Ollama server version from `/api/version`."""
        return await self._transport.request(VersionResponse, "GET", "/api/version")

    async def show(self, model: str) -> ShowResponse:
        """Show detailed model metadata from `/api/show`."""
        request = ShowRequest(model=model)
        return await self._transport.request(
            ShowResponse,
            "POST",
            "/api/show",
            json=request.model_dump(exclude_none=True),
        )

    async def ps(self) -> ProcessResponse:
        """List running models from `/api/ps`."""
        return await self._transport.request(ProcessResponse, "GET", "/api/ps")

    async def pull(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
        """Pull model data from registry via `/api/pull`."""
        request = PullRequest(model=model, insecure=insecure, stream=stream)
        payload = request.model_dump(exclude_none=True)
        if stream:
            return self._transport.stream(
                ProgressResponse,
                "POST",
                "/api/pull",
                json=payload,
            )
        return await self._transport.request(
            ProgressResponse,
            "POST",
            "/api/pull",
            json=payload,
        )

    async def push(
        self,
        model: str,
        *,
        insecure: bool = False,
        stream: bool = False,
    ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
        """Push model data to registry via `/api/push`."""
        request = PushRequest(model=model, insecure=insecure, stream=stream)
        payload = request.model_dump(exclude_none=True)
        if stream:
            return self._transport.stream(
                ProgressResponse,
                "POST",
                "/api/push",
                json=payload,
            )
        return await self._transport.request(
            ProgressResponse,
            "POST",
            "/api/push",
            json=payload,
        )

    async def create(
        self,
        model: str,
        *,
        quantize: Optional[str] = None,
        from_: Optional[str] = None,
        files: Optional[dict[str, str]] = None,
        adapters: Optional[dict[str, str]] = None,
        template: Optional[str] = None,
        license: Optional[Union[str, List[str]]] = None,
        system: Optional[str] = None,
        parameters: Optional[Union[Mapping[str, Any], Options]] = None,
        messages: Optional[Sequence[Union[Mapping[str, Any], Message]]] = None,
        stream: bool = False,
    ) -> Union[ProgressResponse, AsyncIterator[ProgressResponse]]:
        """Create a model via `/api/create`."""
        request = CreateRequest(
            model=model,
            stream=stream,
            quantize=quantize,
            from_=from_,
            files=files,
            adapters=adapters,
            template=template,
            license=license,
            system=system,
            parameters=parameters,
            messages=messages,
        )
        payload = request.model_dump(exclude_none=True)
        if stream:
            return self._transport.stream(
                ProgressResponse,
                "POST",
                "/api/create",
                json=payload,
            )
        return await self._transport.request(
            ProgressResponse,
            "POST",
            "/api/create",
            json=payload,
        )

    async def delete(self, model: str) -> StatusResponse:
        """Delete a model via `/api/delete`."""
        response = await self._transport.request_raw(
            "DELETE",
            "/api/delete",
            json=DeleteRequest(model=model).model_dump(exclude_none=True),
        )
        return StatusResponse(
            status="success" if response.status_code == 200 else "error"
        )

    async def copy(self, source: str, destination: str) -> StatusResponse:
        """Copy a model via `/api/copy`."""
        response = await self._transport.request_raw(
            "POST",
            "/api/copy",
            json=CopyRequest(source=source, destination=destination).model_dump(
                exclude_none=True
            ),
        )
        return StatusResponse(
            status="success" if response.status_code == 200 else "error"
        )

    async def ensure_model_available(
        self,
        model: str,
        auto_pull: bool = True,
        stream_progress: bool = False,
    ) -> bool:
        """Ensure a model exists locally, optionally pulling missing models."""
        try:
            await self.show(model)
            return True
        except ResponseError as exc:
            if "not found" not in str(exc).lower():
                raise

            if not auto_pull:
                raise ModelNotFoundError(model) from None

            logger.info("Model '%s' not found locally. Pulling...", model)
            try:
                if stream_progress:
                    progress_stream = cast(
                        AsyncIterator[ProgressResponse],
                        await self.pull(model, stream=True),
                    )
                    async for progress in progress_stream:
                        status = progress.status or ""
                        if progress.completed and progress.total:
                            pct = (progress.completed / progress.total) * 100
                            print(f"\r  {status}: {pct:.1f}%", end="", flush=True)
                        else:
                            print(f"\r  {status}", end="", flush=True)
                    print()
                else:
                    await self.pull(model, stream=False)
                logger.info("Successfully pulled model '%s'", model)
                return True
            except Exception as pull_error:
                logger.error("Failed to pull model '%s': %s", model, pull_error)
                raise ModelNotFoundError(
                    model,
                    f"Model '{model}' not found and pull failed: {pull_error}",
                ) from pull_error

    async def get_model_details(self, model: str) -> dict[str, Any]:
        """Get merged detail/modelinfo metadata for a model."""
        show_response = await self.show(model)
        details = (
            show_response.details.model_dump(exclude_none=True)
            if isinstance(show_response.details, ModelDetails)
            else {}
        )
        model_info = dict(show_response.modelinfo or {})
        return {**details, **model_info}

    async def get_model_capabilities(self, model: str) -> List[str]:
        """Infer capabilities for a model using template and metadata heuristics."""
        try:
            show_response = await self.show(model)
        except Exception:
            return []

        caps: set[str] = set()
        details = (
            show_response.details.model_dump(exclude_none=True)
            if isinstance(show_response.details, ModelDetails)
            else {}
        )
        model_info = dict(show_response.modelinfo or {})
        full_meta = {**details, **model_info}
        str_meta = str(full_meta).lower()
        families = details.get("families", []) or [details.get("family")]
        families_str = str(families).lower()

        if "bert" in families_str or "nomic-bert" in families_str:
            caps.add("embedding")

        is_embedding_only = "bert" in families_str and "llama" not in families_str
        if not is_embedding_only:
            caps.add("completion")
            template = (show_response.template or "").lower()
            if (
                "{{ .messages" in template
                or "message" in template
                or "[inst]" in template
                or "<|im_start|>" in template
            ):
                caps.add("chat")

        if "clip" in families or "mclip" in families or "vision" in str_meta:
            caps.add("vision")

        template = (show_response.template or "").lower()
        if "tool" in template:
            caps.add("tools")

        return sorted(caps)

    async def get_model_context_length(self, model: str) -> int:
        """Return model context length from metadata, falling back to 4096."""
        try:
            show_response = await self.show(model)
            model_info = dict(show_response.modelinfo or {})

            for key, value in model_info.items():
                if key.endswith(".context_length"):
                    return int(value)

            if show_response.parameters:
                for line in show_response.parameters.split("\n"):
                    if "num_ctx" in line or "context_length" in line:
                        parts = line.split()
                        if len(parts) >= 2 and parts[-1].isdigit():
                            return int(parts[-1])
        except Exception:
            return 4096

        return 4096
