import hmac
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from .bootstrap import bootstrap
from .config import settings as default_settings
from .contracts import SearchRequest
from .details import movie_details
from .schema import schema_details
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
