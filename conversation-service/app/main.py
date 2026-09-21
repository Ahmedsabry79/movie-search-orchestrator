from __future__ import annotations

import asyncio
import hmac
import json
import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .agent import MovieAgent
from .config import get_settings
from .mcp_client import MCPGateway
from .model_client import ModelGateway
from .runtime_client import RuntimeStoreClient
from .schemas import ConversationCreateRequest, UserMessageRequest
from .service import ConversationService

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    runtime = RuntimeStoreClient(settings)
    mcp = MCPGateway(settings)
    model = ModelGateway(settings)
    agent = MovieAgent(settings, mcp, model)
    app.state.service = ConversationService(settings, runtime, agent)
    app.state.runtime = runtime
    app.state.mcp = mcp
    app.state.model = model
    yield
    await runtime.close()
    await model.close()


app = FastAPI(
    title="Movie Conversation Agent",
    version="1.0.0",
    description="Stateful movie agent orchestrator using Gemma through the model gateway and MCP retrieval tools.",
    lifespan=lifespan,
)


def authorize(x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_key and (x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key)):
        raise HTTPException(status_code=401, detail="Invalid API key")


def service(request: Request) -> ConversationService:
    return request.app.state.service


def _translate_http_error(exc: httpx.HTTPStatusError) -> HTTPException:
    if exc.response.status_code == 404:
        return HTTPException(status_code=404, detail="Conversation not found")
    return HTTPException(status_code=503, detail="Agent persistence service is unavailable")


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    async def check(url: str, headers: dict[str, str] | None = None) -> bool:
        try:
            async with httpx.AsyncClient(timeout=4.0, headers=headers) as client:
                response = await client.get(url)
            return response.status_code == 200
        except Exception:
            return False

    db_headers = {"X-API-Key": settings.api_key} if settings.api_key else None
    model_headers = {"Authorization": f"Bearer {settings.model_api_key}"} if settings.model_api_key else None
    db_ok, mcp_ok, model_ok = await asyncio.gather(
        check(f"{settings.db_app_url.rstrip('/')}/health/ready", db_headers),
        check(settings.mcp_url.rsplit("/mcp", 1)[0] + "/health/ready"),
        check(f"{settings.model_base_url.rsplit('/v1', 1)[0]}/health/ready", model_headers),
    )
    ready = db_ok and mcp_ok and model_ok
    return JSONResponse(
        {"status": "ready" if ready else "unavailable", "db_app": db_ok, "mcp": mcp_ok, "model": model_ok},
        status_code=200 if ready else 503,
    )


@app.post("/conversations", dependencies=[Depends(authorize)])
async def create_conversation(payload: ConversationCreateRequest, request: Request) -> dict:
    try:
        return await service(request).create_conversation(title=payload.title, metadata=payload.metadata)
    except httpx.HTTPStatusError as exc:
        raise _translate_http_error(exc) from exc


@app.get("/conversations/{conversation_id}", dependencies=[Depends(authorize)])
async def get_conversation(conversation_id: str, request: Request) -> dict:
    try:
        return await service(request).get_conversation(conversation_id)
    except httpx.HTTPStatusError as exc:
        raise _translate_http_error(exc) from exc


@app.post("/conversations/{conversation_id}/messages", dependencies=[Depends(authorize)])
async def send_message(conversation_id: str, payload: UserMessageRequest, request: Request) -> dict:
    try:
        result = await service(request).run_message(
            conversation_id=conversation_id,
            content=payload.content,
        )
        return result.model_dump(mode="json")
    except httpx.HTTPStatusError as exc:
        raise _translate_http_error(exc) from exc
    except Exception as exc:
        logger.exception("Conversation message failed")
        raise HTTPException(status_code=503, detail="The movie agent could not complete this request") from exc


@app.post("/conversations/{conversation_id}/messages/stream", dependencies=[Depends(authorize)])
async def stream_message(conversation_id: str, payload: UserMessageRequest, request: Request) -> StreamingResponse:
    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    async def on_event(event_type: str, data: dict) -> None:
        await queue.put((event_type, data))

    async def producer() -> None:
        try:
            await service(request).run_message(
                conversation_id=conversation_id,
                content=payload.content,
                on_event=on_event,
            )
        except Exception:
            logger.exception("Streaming agent run failed")
            await queue.put(("error", {"message": "The movie agent could not complete this request."}))
        finally:
            await queue.put(None)

    async def event_stream():
        task = asyncio.create_task(producer())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                event_type, data = item
                yield f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
