# TMDB Database App

PostgreSQL and FastAPI backend for the TMDB 5000 movie assignment. It includes preprocessing, automatic first-start download and seeding, current schema discovery, structured search with automatic fuzzy matching, and movie details.

Semantic documents, embeddings, semantic search, conversation management, and a Streamlit interface are not implemented in this step. No preprocessing report is generated; the data decisions are explained in `preprocessing.py` comments.

## Run with Docker Compose

Requires Docker with the Compose v2 plugin.

```bash
cp .env.example .env
docker compose up --build -d
docker compose logs -f app
```

Open the interactive API documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

The initial download and seed can take a few minutes. The app accepts requests only after the dataset is ready. The API binds to localhost by default, and PostgreSQL is only exposed inside the Compose network.

Startup order:

1. PostgreSQL starts and passes `pg_isready`.
2. The app waits for that health check, then takes a PostgreSQL transaction advisory lock.
3. It creates the `movie_catalog` schema, `pg_trgm`, missing tables and their indexes.
4. If a completed seed exists, it verifies the version and table counts and reuses it.
5. Otherwise, it obtains both original CSV files, preprocesses them, and reads the written CSVs to seed tables in foreign-key dependency order.
6. Data and the completion marker commit together. The API becomes ready.

A failure rolls back database changes. On the next start it reuses downloaded source files and retries preprocessing and seeding. Concurrent app startups serialize through the advisory lock. Existing nonempty tables without a valid seed marker are not automatically overwritten.

`create_all` initializes tables; it is not a schema migration system. An incompatible existing schema or dataset version requires an explicit migration. This package will not silently replace an existing database.

## Dataset download and local files

Default source: [TMDB 5000 Movie Dataset on Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

The app downloads the public dataset archive automatically and keeps the two original files under `/app/data/raw`. If Kaggle requires authentication in your environment, set `KAGGLE_API_TOKEN`, or `KAGGLE_USERNAME` and `KAGGLE_KEY`, in `.env`. Restart the app after correcting a failed download.

You can also supply the original files yourself:

```text
data/raw/tmdb_5000_movies.csv
data/raw/tmdb_5000_credits.csv
```

```bash
docker compose -f compose.yaml -f compose.local-data.yaml up --build -d
```

Both files must be present and readable when using this read-only mount. Source data is not bundled in the project archive. PostgreSQL data and downloaded/processed files persist in separate named volumes.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/schemas` | Current catalog columns, types, keys, indexes, checks, ORM relationships and search contract |
| GET | `/schemas?tables=movies&tables=movie_genres` | Details for selected tables |
| POST | `/search` | Validated structured query; never arbitrary SQL or Python |
| GET | `/movies/{movie_id}` | Movie fields, overview, keywords, genres, companies, countries, languages, cast and crew |
| GET | `/health/live` | Process health |
| GET | `/health/ready` | Successful startup and database connectivity |

When `API_KEY` is set, supply `X-API-Key` on schema/search/detail requests. Health checks remain unauthenticated. The default is intended for local development; keep the API behind an authenticated boundary when deployed.

The schema endpoint reads PostgreSQL metadata on each call, so it reflects the current catalog. It also exposes the accepted JSON request schema and field semantics. Callers can cache it for a session.

## Structured search

Requests support `operation: list`, `count`, or `aggregate`. Filters are ANDed. Values inside an `in` filter are ORed. Filter names and operations are allowlisted, values are bound as parameters, and API query transactions are read-only with a configurable timeout.

### Filters and sorting

```bash
curl -s http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{
    "operation": "list",
    "filters": [
      {"field": "genre", "operator": "eq", "value": "Science Fiction"},
      {"field": "release_year", "operator": "gt", "value": 2010},
      {"field": "vote_average", "operator": "gt", "value": 7.5},
      {"field": "vote_count", "operator": "gte", "value": 1000}
    ],
    "sort": [{"field": "vote_average", "direction": "desc"}],
    "limit": 10
  }'
```

Scalar filter fields: `id`, `budget`, `revenue`, `runtime`, `vote_average`, `vote_count`, `popularity`, `release_date`, `release_year`, `original_language`, `status`.

Numeric/date operators: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `is_null`, `not_null`. `original_language` and `status` support `eq`, `ne`, `in` and null checks. Null checks take no value. SQL null semantics apply: for example, `revenue > 0` excludes unknown revenue.

Name filters: `genre`, `keyword`, `production_company`, `production_country`, `spoken_language`, `actor`, `director`. They resolve exact normalized names first, then approximate names. If resolution is ambiguous, the response contains candidates and no search results; ask for clarification and retry with the chosen ID. A name `in` list requires every supplied name to resolve confidently.

ID filters append `_id`, for example `genre_id`, `director_id`, `actor_id`. `production_country_id` accepts an ISO country code; `spoken_language_id` accepts an ISO language code. Name and ID filters support `eq` and `in`.

Sorting accepts scalar fields, `title`, and `similarity` when a text match is present. A movie-ID tie-breaker keeps ordering stable. Missing sort values appear last.

`limit` is 1–100. `offset` enables pagination. Responses include `matched_movie_count`, `has_more`, applied filters and any resolved entities.

### Automatic fuzzy title matching

```json
{
  "text_match": {
    "fields": ["title", "original_title"],
    "query": "Intersteler"
  },
  "limit": 5
}
```

There is no fuzzy mode to choose. The server normalizes the query with the same NFKC/casefold/whitespace rules as preprocessing. It looks for exact matches in the filtered scope first. If there are none, it uses PostgreSQL trigram and strict-word similarity to handle spelling mistakes and partial titles. Each movie appears once, using its strongest field score.

`tagline` is available when explicitly included in `fields`; it is not part of default title identification. One- and two-character queries require an exact match to avoid very broad approximate results.

The default minimum similarity is `0.30`. If the top two candidates are within `0.10`, resolution is ambiguous. Multiple exact matches are also ambiguous. These are initial tunable thresholds, not calibrated probabilities. Inspect `text_resolution.requires_clarification` and its candidate list before selecting a movie. The candidate check is independent of the requested result page size.

Search still returns matching candidates, even when identity is ambiguous. `selected_movie_id` identifies the suggested best candidate only when resolution is sufficiently distinct; it does not silently narrow list/count/aggregate results to that one candidate. To query one resolved movie, use its ID or `/movies/{id}`.

### Counts

```json
{
  "operation": "count",
  "filters": [{"field": "director", "value": "Christopher Nolan"}]
}
```

### Grouped aggregations

```json
{
  "operation": "aggregate",
  "group_by": "genre",
  "metrics": [
    {"function": "count", "alias": "movie_count"},
    {"function": "avg", "field": "vote_average", "alias": "average_rating"}
  ],
  "sort": [{"field": "movie_count", "direction": "desc"}],
  "limit": 10
}
```

Groups: `genre`, `keyword`, `production_company`, `production_country`, `spoken_language`, `actor`, `director`, `release_year`, `original_language`. Omit `group_by` for global aggregates. Omit `metrics` for a movie count.

Metrics: `count` counts distinct movies; `sum`, `avg`, `min`, `max` operate on `budget`, `revenue`, `runtime`, `vote_average`, `vote_count`, `popularity`. Each movie contributes once per group, even with multiple acting roles. A movie may legitimately contribute to several genre groups. Empty global counts are zero; other empty aggregates are null. Unknown scalar grouping values remain a null group; missing associations cannot be assigned to an entity group.

Aggregate sorting accepts metric aliases, `group_value`, and `group_id` when present. Aggregate responses include `total_groups` and `has_more`.

### Later conversation-manager integration

The API is stateless. A caller can refine a known set by supplying `movie_ids`, or preserve and extend earlier filters to re-run the full search. An empty `movie_ids` list matches nothing. Keep displayed result order separately for references such as "the first movie"; a paginated page is not the full matching set.

## Run preprocessing separately

```bash
python -m pip install -r requirements.txt
python preprocessing.py --input-dir data/raw --output-dir data/processed
```

Outputs are one `*_table.csv` per table. Null values use the explicit `\N` marker. The seeder restores integers, datetimes, JSON and nulls from these files instead of relying on CSV type inference. Ratings keep decimals. Malformed JSON and conflicting entity identities stop processing; missing/empty metadata is preserved as documented in the code. Missing or unmatched credits generate log warnings, not a standalone report.

## Tests

```bash
python -m pip install -r requirements-dev.txt
pytest tests/test_preprocessing.py -q
```

Integration tests require a **disposable** PostgreSQL database with permission to create `pg_trgm` and schemas. They drop and recreate its `movie_catalog` schema.

```bash
TEST_DATABASE_URL='postgresql+psycopg://user:password@localhost:5432/movie_tests' pytest -q
```

Without `TEST_DATABASE_URL`, integration tests are skipped. Covered behaviors include null rules, ID integrity, multi-role credits, title ambiguity, fuzzy resolution, counts/aggregations, request validation, authentication, startup reuse and transaction rollback/recovery.

## Code layout

```text
preprocessing.py          Original preprocessing flow, corrected and made reusable
app/models.py            Tables, ORM relationships, constraints and indexes
app/bootstrap.py         Download, schema checks, CSV seeding and completion marker
app/contracts.py         Validated request models
app/search.py            Deterministic query construction and text resolution
app/details.py           Complete movie records
app/schema.py            Live database schema introspection
app/main.py              FastAPI endpoints and lifecycle
compose.yaml             PostgreSQL and app, with readiness dependency
compose.local-data.yaml  Optional local CSV mount
```

PostgreSQL `pg_trgm` matching: [official documentation](https://www.postgresql.org/docs/current/pgtrgm.html). Compose health dependency: [official documentation](https://docs.docker.com/compose/how-tos/startup-order/).
