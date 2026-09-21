from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .config import Settings, get_settings
from .embedding import EmbeddingClient
from .indexer import MovieIndexer
from .repository import MovieRepository
from .schemas import IndexStatus, ReindexResponse, SearchRequest, SearchResponse
from .vector_store import MovieVectorStore


def require_api_key(
    settings: Settings = Depends(get_settings),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    repository = MovieRepository(
        settings.db_app_url,
        api_key=settings.api_key,
        page_size=settings.index_source_page_size,
    )
    embeddings = EmbeddingClient(
        base_url=settings.model_base_url,
        api_key=settings.model_api_key,
        model=settings.embedding_model,
        expected_dimension=settings.embedding_dimension,
    )
    store = MovieVectorStore(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection,
        dimension=settings.embedding_dimension,
        rrf_k=settings.rrf_k,
        candidate_multiplier=settings.search_candidate_multiplier,
    )
    indexer = MovieIndexer(repository, embeddings, store, settings.index_batch_size)

    app.state.settings = settings
    app.state.repository = repository
    app.state.embeddings = embeddings
    app.state.store = store
    app.state.indexer = indexer

    await asyncio.to_thread(store.ensure_collection)

    if settings.auto_index_on_start:
        existing = await asyncio.to_thread(store.row_count)
        should_index = not settings.index_if_empty_only or existing in (None, 0)
        if should_index:
            await indexer.reindex()

    yield


app = FastAPI(
    title="Movie Hybrid Retrieval API",
    description="Three-field BGE-M3 dense retrieval + Milvus BM25 for the movie LLM agent.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health/live")
async def health_live() -> dict[str, str]:
    return {"status": "alive"}


@app.get("/health/ready")
async def health_ready(request: Request) -> JSONResponse:
    repository: MovieRepository = request.app.state.repository
    embeddings: EmbeddingClient = request.app.state.embeddings
    store: MovieVectorStore = request.app.state.store

    db_ok, model_ok, milvus_ok = await asyncio.gather(
        asyncio.to_thread(repository.ping),
        embeddings.ping(),
        asyncio.to_thread(store.ping),
    )
    ready = db_ok and model_ok and milvus_ok and store.exists()
    return JSONResponse(
        {
            "status": "ready" if ready else "unavailable",
            "db_app": db_ok,
            "embedding_model": model_ok,
            "milvus": milvus_ok,
            "collection": store.exists() if milvus_ok else False,
        },
        status_code=200 if ready else 503,
    )


@app.get("/index/status", response_model=IndexStatus, dependencies=[Depends(require_api_key)])
async def index_status(request: Request) -> IndexStatus:
    store: MovieVectorStore = request.app.state.store
    exists = await asyncio.to_thread(store.exists)
    rows = await asyncio.to_thread(store.row_count) if exists else 0
    return IndexStatus(collection=store.collection, exists=exists, rows=rows)


@app.post("/index/rebuild", response_model=ReindexResponse, dependencies=[Depends(require_api_key)])
async def rebuild_index(request: Request) -> ReindexResponse:
    indexer: MovieIndexer = request.app.state.indexer
    store: MovieVectorStore = request.app.state.store
    indexed = await indexer.reindex(recreate=True)
    return ReindexResponse(indexed=indexed, collection=store.collection)


@app.post("/search", response_model=SearchResponse, dependencies=[Depends(require_api_key)])
async def search_movies(payload: SearchRequest, request: Request) -> SearchResponse:
    embeddings: EmbeddingClient = request.app.state.embeddings
    store: MovieVectorStore = request.app.state.store

    query_vector = None
    if payload.mode in ("hybrid", "dense"):
        query_vector = (await embeddings.embed([payload.query]))[0]

    results = await asyncio.to_thread(store.search, payload, query_vector)
    return SearchResponse(
        query=payload.query,
        mode=payload.mode,
        count=len(results),
        results=results,
    )
