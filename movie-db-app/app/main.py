import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from .agent_contracts import AgentAnalyticsRequest, AgentMovieSearchRequest, ReferenceResolveRequest
from .agent_search import agent_movie_search, resolve_reference, vector_index_page
from .analytics import analyze_movies
from .bootstrap import bootstrap
from .config import settings as default_settings
from .contracts import SearchRequest
from .details import movie_details
from .schema import schema_details
from .runtime_contracts import (
    ConversationCreateRequest, ConversationMemoryUpdateRequest, EventCreateRequest, MessageCreateRequest,
    RunCreateRequest, RunUpdateRequest,
)
from .runtime_store import (
    append_event, append_message, create_conversation, create_run,
    get_conversation_snapshot, update_conversation_memory, update_run_record,
)
from .search import ResolutionRequired, structured_search

logger = logging.getLogger(__name__)


def create_app(settings=default_settings, engine=None):
    if not 0 < settings.fuzzy_threshold <= 1 or not 0 <= settings.fuzzy_min_gap <= 1:
        raise ValueError("Invalid fuzzy matching settings")
    if not 1 <= settings.query_timeout_ms <= 120000:
        raise ValueError("QUERY_TIMEOUT_MS must be between 1 and 120000")
    database = engine if engine is not None else create_engine(settings.database_url, pool_pre_ping=True, connect_args={"connect_timeout": 10})

    @asynccontextmanager
    async def lifespan(app):
        app.state.ready = False
        await run_in_threadpool(bootstrap, database, settings)
        app.state.ready = True
        yield
        app.state.ready = False
        if engine is None:
            database.dispose()

    app = FastAPI(title="TMDB Database API", version="1.0.0", lifespan=lifespan,
                  description="Schema discovery, validated structured search with automatic fuzzy matching, and movie details.")
    app.state.ready = False

    def authorize(x_api_key: str | None = Header(default=None)):
        if settings.api_key and (x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_key)):
            raise HTTPException(status_code=401, detail="Invalid API key")

    def connection():
        with database.connect() as db:
            with db.begin():
                db.execute(text("SET TRANSACTION READ ONLY"))
                db.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {"timeout": str(settings.query_timeout_ms)})
                db.execute(text("SELECT set_config('pg_trgm.similarity_threshold', :threshold, true)"), {"threshold": str(settings.fuzzy_threshold)})
                db.execute(text("SELECT set_config('pg_trgm.strict_word_similarity_threshold', :threshold, true)"), {"threshold": str(settings.fuzzy_threshold)})
                yield db

    def runtime_session():
        with Session(database) as db:
            with db.begin():
                db.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {"timeout": str(settings.query_timeout_ms)})
                yield db

    @app.exception_handler(SQLAlchemyError)
    async def database_error(request: Request, exc: SQLAlchemyError):
        logger.exception("Database request failed", exc_info=exc)
        return JSONResponse(status_code=503, content={"detail": "Database request failed or timed out. Retry or narrow the query."})

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready():
        if not app.state.ready:
            raise HTTPException(status_code=503, detail="Dataset is not ready")
        with database.connect() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ready"}

    @app.get("/schemas", dependencies=[Depends(authorize)])
    def schemas(tables: list[str] | None = Query(default=None), db=Depends(connection)):
        try:
            return schema_details(db, tables)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/search", dependencies=[Depends(authorize)])
    def search(payload: SearchRequest, db=Depends(connection)):
        try:
            return structured_search(db, payload, settings)
        except ResolutionRequired as exc:
            return {**exc.payload, "results": []}
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


    @app.post("/agent/search", dependencies=[Depends(authorize)])
    def agent_search(payload: AgentMovieSearchRequest, db=Depends(connection)):
        try:
            return agent_movie_search(db, payload, settings)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/agent/references/resolve", dependencies=[Depends(authorize)])
    def agent_reference_resolve(payload: ReferenceResolveRequest, db=Depends(connection)):
        try:
            return resolve_reference(db, payload, settings)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/agent/analyze", dependencies=[Depends(authorize)])
    def agent_analyze(payload: AgentAnalyticsRequest, db=Depends(connection)):
        try:
            return analyze_movies(db, payload, settings)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/internal/vector/movies", dependencies=[Depends(authorize)])
    def vector_movies(
        offset: int = Query(default=0, ge=0, le=10_000_000),
        limit: int = Query(default=500, ge=1, le=1000),
        db=Depends(connection),
    ):
        return vector_index_page(db, offset=offset, limit=limit)


    @app.post("/internal/agent-runtime/conversations", dependencies=[Depends(authorize)])
    def runtime_create_conversation(payload: ConversationCreateRequest, db=Depends(runtime_session)):
        return create_conversation(db, payload)

    @app.get("/internal/agent-runtime/conversations/{conversation_id}", dependencies=[Depends(authorize)])
    def runtime_get_conversation(
        conversation_id: str,
        message_limit: int = Query(default=50, ge=1, le=500),
        event_limit: int = Query(default=200, ge=1, le=2000),
        db=Depends(runtime_session),
    ):
        result = get_conversation_snapshot(
            db, conversation_id, message_limit=message_limit, event_limit=event_limit
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result

    @app.post("/internal/agent-runtime/conversations/{conversation_id}/messages", dependencies=[Depends(authorize)])
    def runtime_append_message(conversation_id: str, payload: MessageCreateRequest, db=Depends(runtime_session)):
        result = append_message(db, conversation_id, payload)
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result

    @app.put("/internal/agent-runtime/conversations/{conversation_id}/memory", dependencies=[Depends(authorize)])
    def runtime_update_memory(
        conversation_id: str, payload: ConversationMemoryUpdateRequest, db=Depends(runtime_session)
    ):
        result = update_conversation_memory(db, conversation_id, payload)
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result

    @app.post("/internal/agent-runtime/conversations/{conversation_id}/runs", dependencies=[Depends(authorize)])
    def runtime_create_run(conversation_id: str, payload: RunCreateRequest, db=Depends(runtime_session)):
        result = create_run(db, conversation_id, payload)
        if result is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return result

    @app.patch("/internal/agent-runtime/runs/{run_id}", dependencies=[Depends(authorize)])
    def runtime_update_run(run_id: str, payload: RunUpdateRequest, db=Depends(runtime_session)):
        result = update_run_record(db, run_id, payload)
        if result is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        return result

    @app.post("/internal/agent-runtime/runs/{run_id}/events", dependencies=[Depends(authorize)])
    def runtime_append_event(run_id: str, payload: EventCreateRequest, db=Depends(runtime_session)):
        result = append_event(db, run_id, payload)
        if result is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
        return result

    @app.get("/movies/{movie_id}", dependencies=[Depends(authorize)])
    def details(movie_id: int, db=Depends(connection)):
        if not 0 < movie_id < 2**63:
            raise HTTPException(status_code=422, detail="movie_id must be a positive 64-bit integer")
        result = movie_details(db, movie_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Movie not found")
        return result

    return app


app = create_app()
