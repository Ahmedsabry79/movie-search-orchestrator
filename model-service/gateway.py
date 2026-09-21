from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

EMBEDDING_UPSTREAM = os.getenv("EMBEDDING_UPSTREAM", "http://127.0.0.1:8001")
GENERATOR_UPSTREAM = os.getenv("GENERATOR_UPSTREAM", "http://127.0.0.1:8002")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "movie-agent-local")

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "content-length",
    "content-encoding",
}


def _forward_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=10.0))
    yield
    await app.state.http.aclose()


app = FastAPI(title="Movie Agent Model Gateway", version="1.1.0", lifespan=lifespan)


async def _is_ready(base_url: str) -> bool:
    try:
        response = await app.state.http.get(f"{base_url}/health", timeout=3.0)
        return response.status_code == 200
    except Exception:
        return False


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
async def ready() -> Response:
    embedding_ok, generator_ok = await asyncio.gather(
        _is_ready(EMBEDDING_UPSTREAM),
        _is_ready(GENERATOR_UPSTREAM),
    )
    payload = {
        "status": "ready" if embedding_ok and generator_ok else "starting",
        "embedding": embedding_ok,
        "generator": generator_ok,
    }
    return JSONResponse(payload, status_code=200 if embedding_ok and generator_ok else 503)


def _authorized(request: Request) -> bool:
    if not MODEL_API_KEY:
        return True
    return request.headers.get("authorization") == f"Bearer {MODEL_API_KEY}"


def _auth_error() -> JSONResponse:
    return JSONResponse(
        {"error": {"message": "Unauthorized", "type": "authentication_error"}},
        status_code=401,
    )


def _upstream_headers(request: Request) -> dict[str, str]:
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
    }
    # Both internal vLLM processes use the same API key. Reassert it rather than
    # trusting arbitrary caller header forwarding.
    if MODEL_API_KEY:
        headers["authorization"] = f"Bearer {MODEL_API_KEY}"
    return headers


async def _proxy(request: Request, upstream: str, path: str) -> Response:
    if not _authorized(request):
        return _auth_error()

    url = httpx.URL(f"{upstream}/v1/{path}", query=request.url.query.encode())
    body = await request.body()
    upstream_request = app.state.http.build_request(
        request.method,
        url,
        headers=_upstream_headers(request),
        content=body,
    )
    upstream_response = await app.state.http.send(upstream_request, stream=True)

    return StreamingResponse(
        upstream_response.aiter_raw(),
        status_code=upstream_response.status_code,
        headers=_forward_headers(upstream_response.headers),
        media_type=upstream_response.headers.get("content-type"),
        background=BackgroundTask(upstream_response.aclose),
    )


@app.get("/v1/models")
async def models(request: Request) -> Response:
    if not _authorized(request):
        return _auth_error()
    headers = {"Authorization": f"Bearer {MODEL_API_KEY}"} if MODEL_API_KEY else {}

    async def fetch(base_url: str) -> list[dict]:
        try:
            response = await app.state.http.get(f"{base_url}/v1/models", headers=headers)
            response.raise_for_status()
            return response.json().get("data", [])
        except Exception:
            return []

    embedding_models, generator_models = await asyncio.gather(
        fetch(EMBEDDING_UPSTREAM), fetch(GENERATOR_UPSTREAM)
    )
    return JSONResponse({"object": "list", "data": embedding_models + generator_models})


# Explicit OpenAI-compatible generation endpoints. These make the public
# contract obvious while preserving streaming and every vLLM-specific field
# (tools, tool_choice, reasoning_effort, chat_template_kwargs, etc.).
@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    return await _proxy(request, GENERATOR_UPSTREAM, "chat/completions")


@app.post("/v1/completions")
async def completions(request: Request) -> Response:
    return await _proxy(request, GENERATOR_UPSTREAM, "completions")


@app.post("/v1/responses")
async def responses(request: Request) -> Response:
    return await _proxy(request, GENERATOR_UPSTREAM, "responses")


@app.post("/v1/embeddings")
async def embeddings(request: Request) -> Response:
    return await _proxy(request, EMBEDDING_UPSTREAM, "embeddings")


def _upstream_for(path: str) -> str:
    if path == "embeddings" or path.startswith("embeddings/"):
        return EMBEDDING_UPSTREAM
    return GENERATOR_UPSTREAM


# Keep a generic /v1 passthrough for vLLM/OpenAI-compatible endpoints we do not
# need to hard-code. Explicit routes above take precedence.
@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def proxy_v1(path: str, request: Request) -> Response:
    return await _proxy(request, _upstream_for(path), path)


if __name__ == "__main__":
    uvicorn.run(
        "gateway:app",
        host="0.0.0.0",
        port=int(os.getenv("GATEWAY_PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )