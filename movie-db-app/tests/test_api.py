import os
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.bootstrap import bootstrap
from app.config import Settings
from app.main import create_app
from conftest import write_sources

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def database(tmp_path_factory):
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set TEST_DATABASE_URL to a disposable PostgreSQL database; its movie_catalog schema is replaced")
    engine = create_engine(url, poolclass=NullPool, connect_args={"prepare_threshold": None, "connect_timeout": 10})
    settings = Settings(data_dir=tmp_path_factory.mktemp("catalog"))
    write_sources(settings.data_dir / "raw")
    with engine.begin() as db:
        db.execute(text("DROP SCHEMA IF EXISTS movie_catalog CASCADE"))
        db.execute(text("DROP SCHEMA IF EXISTS agent_runtime CASCADE"))
    yield engine, settings
    with engine.begin() as db:
        db.execute(text("DROP SCHEMA IF EXISTS movie_catalog CASCADE"))
        db.execute(text("DROP SCHEMA IF EXISTS agent_runtime CASCADE"))
    engine.dispose()


@pytest.fixture(scope="module")
def client(database):
    engine, settings = database
    with TestClient(create_app(settings, engine)) as client:
        yield client


def test_startup_details_and_current_schema(client):
    assert client.get("/health/ready").status_code == 200
    details = client.get("/movies/1")
    assert details.status_code == 200
    details = details.json()
    assert details["title"] == "Interstellar"
    assert len(details["cast"]) == 2
    assert details["keywords"][1]["name"] == "space"
    assert len(details["crew"]) == 2
    schema = client.get("/schemas?tables=movies").json()
    assert len(schema["tables"]) == 1
    assert schema["tables"][0]["orm_relationships"]
    assert client.get("/movies/99999").status_code == 404


def test_startup_reuses_committed_seed(client, database, monkeypatch):
    from app import bootstrap as module
    monkeypatch.setattr(module, "ensure_raw_files", lambda *_: pytest.fail("Restart must not download"))
    monkeypatch.setattr(module, "preprocess", lambda *_: pytest.fail("Restart must not preprocess"))
    engine, settings = database
    bootstrap(engine, settings)
    assert client.post("/search", json={"operation": "count"}).json()["count"] == 4


def test_combined_filters(client):
    response = client.post("/search", json={"filters": [
        {"field": "genre", "value": "science fiction"},
        {"field": "release_year", "operator": "gt", "value": 2010},
        {"field": "vote_average", "operator": "gt", "value": 7.5},
    ]})
    assert response.status_code == 200, response.text
    assert [row["id"] for row in response.json()["results"]] == [1]


@pytest.mark.parametrize("query,expected", [("Intersteler", 1), ("Avatr", 2), ("avatar", 2)])
def test_automatic_fuzzy_title(client, query, expected):
    response = client.post("/search", json={"text_match": {"query": query}})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"][0]["id"] == expected
    assert body["text_resolution"]["selected_movie_id"] == expected


def test_exact_title_ambiguity_even_with_one_result_limit(client):
    body = client.post("/search", json={"text_match": {"query": "The Matrix"}, "limit": 1}).json()
    assert body["text_resolution"]["requires_clarification"] is True
    assert body["text_resolution"]["selected_movie_id"] is None
    assert len(body["text_resolution"]["candidates"]) == 2


def test_partial_title_and_empty_search(client):
    body = client.post("/search", json={"text_match": {"query": "Matrix"}}).json()
    assert {row["id"] for row in body["results"]} == {3, 4}
    body = client.post("/search", json={"text_match": {"query": "zzzzzzzxq"}}).json()
    assert body["status"] == "no_matches"


def test_director_resolution_and_counts(client):
    response = client.post("/search", json={"operation": "count", "filters": [{"field": "director", "value": "Christopher Nolen"}]})
    assert response.status_code == 200, response.text
    assert response.json()["count"] == 4


def test_aggregates_count_movies_once_per_group(client):
    body = client.post("/search", json={"operation": "aggregate", "group_by": "genre"}).json()
    assert len(body["results"]) == 2
    assert all(row["movie_count"] == 4 for row in body["results"])
    body = client.post("/search", json={"operation": "aggregate", "group_by": "actor", "metrics": [
        {"function": "count", "alias": "movies"}, {"function": "sum", "field": "revenue", "alias": "revenue"}
    ]}).json()
    actor = next(row for row in body["results"] if row["group_id"] == 101)
    assert actor["movies"] == 1 and actor["revenue"] == 100
    assert client.post("/search", json={"operation": "aggregate", "group_by": "director"}).json()["results"][0]["movie_count"] == 4


def test_refinement_preserves_order_and_scope(client):
    body = client.post("/search", json={"movie_ids": [1, 2], "filters": [{"field": "vote_average", "operator": "gt", "value": 8}]}).json()
    assert [row["id"] for row in body["results"]] == [1]
    assert client.post("/search", json={"movie_ids": []}).json()["matched_movie_count"] == 0


@pytest.mark.parametrize("payload", [
    {"python": "__import__('os').system('id')"},
    {"filters": [{"field": "id; DROP TABLE movies", "value": 1}]},
    {"filters": [{"field": "vote_count", "value": "2000"}]},
    {"filters": [{"field": "vote_count", "value": True}]},
    {"limit": 1000}, {"sort": [{"field": "secret"}]},
    {"filters": [{"field": "release_date", "value": "bad date"}]},
    {"operation": "list", "group_by": "genre"},
])
def test_invalid_requests_are_rejected(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_api_key(client, database):
    engine, settings = database
    with TestClient(create_app(replace(settings, api_key="test-key"), engine)) as protected:
        assert protected.post("/search", json={}).status_code == 401
        assert protected.post("/search", json={}, headers={"X-API-Key": "test-key"}).status_code == 200


def test_failed_seed_rolls_back_and_recovers(database, monkeypatch):
    # This test runs after the shared client tests and rebuilds their disposable schema.
    engine, settings = database
    from app import bootstrap as module
    original = module.csv_rows
    with engine.begin() as db:
        db.execute(text("DROP SCHEMA movie_catalog CASCADE"))
    def fail_mid_seed(path, table):
        if table.name == "movie_genres":
            raise ValueError("Simulated interrupted seed")
        yield from original(path, table)
    monkeypatch.setattr(module, "csv_rows", fail_mid_seed)
    with pytest.raises(ValueError, match="interrupted"):
        bootstrap(engine, settings)
    with engine.connect() as db:
        assert db.scalar(text("SELECT to_regclass('movie_catalog.movies')")) is None
    monkeypatch.setattr(module, "csv_rows", original)
    bootstrap(engine, settings)
    with engine.connect() as db:
        assert db.scalar(text("SELECT count(*) FROM movie_catalog.movies")) == 4


def test_agent_search_uses_full_relations_and_fuzzy_names(client):
    response = client.post("/agent/search", json={
        "titles": ["Interstelar"],
        "genres": {"values": ["science fiction"], "match": "all"},
        "production_companies": {"values": ["Example Studio"], "match": "all"},
        "production_countries": {"values": ["Namibia"], "match": "all"},
        "spoken_languages": {"values": ["English"], "match": "all"},
        "cast": {"values": ["Actor On"], "match": "all"},
        "crew": {"values": ["Christopher Nolen"], "match": "all"},
        "crew_jobs": {"values": ["Writer"], "match": "all"},
        "runtime": {"min": 150},
        "vote_average": {"min": 8.0},
        "release_year": {"eq": 2014},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["count"] == 1
    assert body["returned_count"] == 1
    assert body["results"][0]["movie_id"] == 1
    assert "space" in body["results"][0]["keywords"]
    assert "Actor One" in body["results"][0]["cast"]


def test_agent_search_no_match_recommends_semantic_fallback(client):
    body = client.post("/agent/search", json={"titles": ["zzzzzzzxq"]}).json()
    assert body["status"] == "no_matches"
    assert body["semantic_fallback_recommended"] is True
    assert body["results"] == []


def test_agent_reference_resolver_and_vector_source(client):
    resolved = client.post("/agent/references/resolve", json={
        "reference_type": "genre",
        "query": "science fiction",
        "limit": 5,
    })
    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["results"][0]["value"] == "Science Fiction"

    page = client.get("/internal/vector/movies?offset=0&limit=2")
    assert page.status_code == 200, page.text
    body = page.json()
    assert body["total"] == 4
    assert len(body["items"]) == 2
    assert body["next_offset"] == 2
    first = body["items"][0]
    assert first["movie_id"] == 1
    assert "Science Fiction" in first["genres"]
    assert "space" in first["keywords"]


def test_agent_search_sorting(client):
    body = client.post("/agent/search", json={
        "genres": {"values": ["Science Fiction"], "match": "all"},
        "sort": [{"field": "vote_average", "direction": "desc"}],
        "limit": 2,
    }).json()
    assert [row["movie_id"] for row in body["results"]] == [3, 1]


def test_agent_runtime_persistence_is_owned_by_db_app(client):
    conversation = client.post("/internal/agent-runtime/conversations", json={"title": "Movie search", "metadata": {}})
    assert conversation.status_code == 200, conversation.text
    conversation_id = conversation.json()["id"]

    message = client.post(
        f"/internal/agent-runtime/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "Find Interstellar", "metadata": {}},
    )
    assert message.status_code == 200, message.text

    run = client.post(
        f"/internal/agent-runtime/conversations/{conversation_id}/runs",
        json={"user_message_id": message.json()["id"], "plan": {"strategy": "structured_first"}},
    )
    assert run.status_code == 200, run.text
    run_id = run.json()["id"]

    event = client.post(
        f"/internal/agent-runtime/runs/{run_id}/events",
        json={"event_type": "tool_started", "payload": {"tool": "search_movies_db"}},
    )
    assert event.status_code == 200, event.text

    snapshot = client.get(f"/internal/agent-runtime/conversations/{conversation_id}")
    assert snapshot.status_code == 200, snapshot.text
    body = snapshot.json()
    assert body["messages"][0]["content"] == "Find Interstellar"
    assert body["runs"][0]["id"] == run_id
    assert body["events"][0]["event_type"] == "tool_started"
