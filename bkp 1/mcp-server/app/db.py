from __future__ import annotations

import unicodedata
from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .schemas import (
    DatabaseSearchCriteria,
    MovieSearchHit,
    ReferenceMatch,
    ReferenceType,
    StringSetFilter,
)


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def json_safe(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class MovieDatabaseRepository:
    """Structured/fuzzy query layer over the normalized TMDB PostgreSQL schema."""

    def __init__(self, dsn: str, *, fuzzy_threshold: float, query_timeout_ms: int):
        self._dsn = dsn
        self._fuzzy_threshold = fuzzy_threshold
        self._query_timeout_ms = query_timeout_ms

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(
            self._dsn,
            connect_timeout=5,
            options=f"-c statement_timeout={self._query_timeout_ms}",
            row_factory=dict_row,
        )

    def ping(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 AS ok")
                return cur.fetchone()["ok"] == 1
        except Exception:
            return False

    @staticmethod
    def _has_pg_trgm(conn: psycopg.Connection) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm') AS ok")
            return bool(cur.fetchone()["ok"])

    @staticmethod
    def _add_numeric_range(
        where: list[str], params: list[Any], column: str, value: Any | None
    ) -> None:
        if value is None:
            return
        if value.eq is not None:
            where.append(f"{column} = %s")
            params.append(value.eq)
        if value.min is not None:
            where.append(f"{column} >= %s")
            params.append(value.min)
        if value.max is not None:
            where.append(f"{column} <= %s")
            params.append(value.max)

    @staticmethod
    def _fuzzy_clause(
        column: str,
        value: str,
        threshold: float,
        has_trgm: bool,
    ) -> tuple[str, list[Any]]:
        normalized = normalize_text(value)
        if has_trgm:
            return (
                f"({column} = %s OR {column} LIKE %s OR similarity({column}, %s) >= %s)",
                [normalized, f"%{normalized}%", normalized, threshold],
            )
        return f"({column} = %s OR {column} LIKE %s)", [normalized, f"%{normalized}%"]

    def _add_fuzzy_group(
        self,
        where: list[str],
        params: list[Any],
        column: str,
        values: list[str],
        threshold: float,
        has_trgm: bool,
    ) -> None:
        if not values:
            return
        clauses: list[str] = []
        for value in values:
            clause, clause_params = self._fuzzy_clause(column, value, threshold, has_trgm)
            clauses.append(clause)
            params.extend(clause_params)
        where.append("(" + " OR ".join(clauses) + ")")

    @staticmethod
    def _relation_exact_predicate(
        *,
        relation_table: str,
        relation_alias: str,
        lookup_table: str,
        lookup_alias: str,
        relation_fk: str,
        lookup_pk: str,
        value_sql: str,
        movie_column: str = "movie_id",
        extra_join: str = "",
    ) -> str:
        return (
            f"EXISTS (SELECT 1 FROM {relation_table} {relation_alias} "
            f"JOIN {lookup_table} {lookup_alias} ON {lookup_alias}.{lookup_pk} = {relation_alias}.{relation_fk} "
            f"{extra_join} WHERE {relation_alias}.{movie_column} = m.id AND ({value_sql}))"
        )

    def _add_exact_relation_filter(
        self,
        where: list[str],
        params: list[Any],
        value_filter: StringSetFilter | None,
        *,
        relation_table: str,
        relation_alias: str,
        lookup_table: str,
        lookup_alias: str,
        relation_fk: str,
        lookup_pk: str,
        name_column: str = "name_search",
        code_column: str | None = None,
    ) -> None:
        if value_filter is None:
            return

        predicates: list[tuple[str, list[Any]]] = []
        for value in value_filter.values:
            normalized = normalize_text(value)
            if code_column:
                sql = f"{lookup_alias}.{name_column} = %s OR lower({lookup_alias}.{code_column}) = %s"
                value_params = [normalized, normalized]
            else:
                sql = f"{lookup_alias}.{name_column} = %s"
                value_params = [normalized]
            predicates.append(
                (
                    self._relation_exact_predicate(
                        relation_table=relation_table,
                        relation_alias=relation_alias,
                        lookup_table=lookup_table,
                        lookup_alias=lookup_alias,
                        relation_fk=relation_fk,
                        lookup_pk=lookup_pk,
                        value_sql=sql,
                    ),
                    value_params,
                )
            )

        if value_filter.match == "all":
            for sql, sql_params in predicates:
                where.append(sql)
                params.extend(sql_params)
        else:
            where.append("(" + " OR ".join(sql for sql, _ in predicates) + ")")
            for _, sql_params in predicates:
                params.extend(sql_params)

    def _add_people_filter(
        self,
        where: list[str],
        params: list[Any],
        value_filter: StringSetFilter | None,
        *,
        kind: str,
        threshold: float,
        has_trgm: bool,
    ) -> None:
        if value_filter is None:
            return
        relation = "movie_cast" if kind == "cast" else "movie_crew"
        alias = "mc" if kind == "cast" else "mw"
        predicates: list[tuple[str, list[Any]]] = []
        for value in value_filter.values:
            normalized = normalize_text(value)
            if has_trgm:
                name_sql = "(p.name_search = %s OR p.name_search LIKE %s OR similarity(p.name_search, %s) >= %s)"
                name_params = [normalized, f"%{normalized}%", normalized, threshold]
            else:
                name_sql = "(p.name_search = %s OR p.name_search LIKE %s)"
                name_params = [normalized, f"%{normalized}%"]
            sql = (
                f"EXISTS (SELECT 1 FROM {relation} {alias} "
                f"JOIN people p ON p.id = {alias}.person_id "
                f"WHERE {alias}.movie_id = m.id AND {name_sql})"
            )
            predicates.append((sql, name_params))
        if value_filter.match == "all":
            for sql, sql_params in predicates:
                where.append(sql)
                params.extend(sql_params)
        else:
            where.append("(" + " OR ".join(sql for sql, _ in predicates) + ")")
            for _, sql_params in predicates:
                params.extend(sql_params)

    @staticmethod
    def _add_relation_text_filter(
        where: list[str],
        params: list[Any],
        value_filter: StringSetFilter | None,
        *,
        relation_table: str,
        alias: str,
        column: str,
    ) -> None:
        if value_filter is None:
            return
        predicates: list[tuple[str, list[Any]]] = []
        for value in value_filter.values:
            normalized = normalize_text(value)
            sql = (
                f"EXISTS (SELECT 1 FROM {relation_table} {alias} "
                f"WHERE {alias}.movie_id = m.id "
                f"AND lower(coalesce({alias}.{column}, '')) LIKE %s)"
            )
            predicates.append((sql, [f"%{normalized}%"]))
        if value_filter.match == "all":
            for sql, sql_params in predicates:
                where.append(sql)
                params.extend(sql_params)
        else:
            where.append("(" + " OR ".join(sql for sql, _ in predicates) + ")")
            for _, sql_params in predicates:
                params.extend(sql_params)

    @staticmethod
    def _add_scalar_string_filter(
        where: list[str], params: list[Any], value_filter: StringSetFilter | None, column: str
    ) -> None:
        if value_filter is None:
            return
        normalized = [normalize_text(v) for v in value_filter.values]
        clauses = [f"lower(coalesce({column}, '')) = %s" for _ in normalized]
        # A scalar movie column can only equal one value, so `all` with multiple
        # distinct values is impossible. Interpret the user's values as allowed alternatives.
        where.append("(" + " OR ".join(clauses) + ")")
        params.extend(normalized)

    def search(self, criteria: DatabaseSearchCriteria) -> list[MovieSearchHit]:
        threshold = criteria.fuzzy_threshold or self._fuzzy_threshold
        with self._connect() as conn:
            has_trgm = self._has_pg_trgm(conn)
            where: list[str] = []
            params: list[Any] = []

            if criteria.movie_ids:
                where.append("m.id = ANY(%s)")
                params.append(criteria.movie_ids)

            self._add_fuzzy_group(where, params, "m.title_search", criteria.titles, threshold, has_trgm)
            self._add_fuzzy_group(
                where, params, "m.original_title_search", criteria.original_titles, threshold, has_trgm
            )
            self._add_fuzzy_group(where, params, "m.tagline_search", criteria.taglines, threshold, has_trgm)

            self._add_exact_relation_filter(
                where,
                params,
                criteria.genres,
                relation_table="movie_genres",
                relation_alias="mg",
                lookup_table="genres",
                lookup_alias="g",
                relation_fk="genre_id",
                lookup_pk="id",
            )
            self._add_exact_relation_filter(
                where,
                params,
                criteria.keywords,
                relation_table="movie_keywords",
                relation_alias="mk",
                lookup_table="keywords",
                lookup_alias="k",
                relation_fk="keyword_id",
                lookup_pk="id",
            )
            self._add_exact_relation_filter(
                where,
                params,
                criteria.production_companies,
                relation_table="movie_production_companies",
                relation_alias="mpc",
                lookup_table="production_companies",
                lookup_alias="pc",
                relation_fk="company_id",
                lookup_pk="id",
            )
            self._add_exact_relation_filter(
                where,
                params,
                criteria.production_countries,
                relation_table="movie_production_countries",
                relation_alias="mcountry",
                lookup_table="production_countries",
                lookup_alias="country",
                relation_fk="country_iso_3166_1",
                lookup_pk="iso_3166_1",
                code_column="iso_3166_1",
            )
            self._add_exact_relation_filter(
                where,
                params,
                criteria.spoken_languages,
                relation_table="movie_spoken_languages",
                relation_alias="mlang",
                lookup_table="spoken_languages",
                lookup_alias="lang",
                relation_fk="language_iso_639_1",
                lookup_pk="iso_639_1",
                code_column="iso_639_1",
            )

            self._add_people_filter(
                where,
                params,
                criteria.cast,
                kind="cast",
                threshold=threshold,
                has_trgm=has_trgm,
            )
            self._add_people_filter(
                where,
                params,
                criteria.crew,
                kind="crew",
                threshold=threshold,
                has_trgm=has_trgm,
            )
            self._add_relation_text_filter(
                where,
                params,
                criteria.cast_characters,
                relation_table="movie_cast",
                alias="mc_char",
                column="character",
            )
            self._add_relation_text_filter(
                where,
                params,
                criteria.crew_departments,
                relation_table="movie_crew",
                alias="crew_dep",
                column="department",
            )
            self._add_relation_text_filter(
                where,
                params,
                criteria.crew_jobs,
                relation_table="movie_crew",
                alias="crew_job",
                column="job",
            )

            self._add_numeric_range(where, params, "m.budget", criteria.budget)
            self._add_numeric_range(where, params, "m.popularity", criteria.popularity)
            self._add_numeric_range(where, params, "m.revenue", criteria.revenue)
            self._add_numeric_range(where, params, "m.runtime", criteria.runtime)
            self._add_numeric_range(where, params, "m.vote_average", criteria.vote_average)
            self._add_numeric_range(where, params, "m.vote_count", criteria.vote_count)
            self._add_scalar_string_filter(where, params, criteria.original_languages, "m.original_language")
            self._add_scalar_string_filter(where, params, criteria.statuses, "m.status")

            if criteria.release_date is not None:
                if criteria.release_date.eq is not None:
                    where.append("m.release_date::date = %s")
                    params.append(criteria.release_date.eq)
                if criteria.release_date.from_date is not None:
                    where.append("m.release_date::date >= %s")
                    params.append(criteria.release_date.from_date)
                if criteria.release_date.to_date is not None:
                    where.append("m.release_date::date <= %s")
                    params.append(criteria.release_date.to_date)

            if criteria.homepage_contains:
                where.append("coalesce(m.homepage, '') ILIKE %s")
                params.append(f"%{criteria.homepage_contains.strip()}%")
            if criteria.overview_contains:
                where.append("coalesce(m.overview, '') ILIKE %s")
                params.append(f"%{criteria.overview_contains.strip()}%")

            score_exprs: list[str] = []
            score_params: list[Any] = []
            if has_trgm:
                for column, values in (
                    ("m.title_search", criteria.titles),
                    ("m.original_title_search", criteria.original_titles),
                    ("m.tagline_search", criteria.taglines),
                ):
                    for value in values:
                        score_exprs.append(f"similarity({column}, %s)")
                        score_params.append(normalize_text(value))
            fuzzy_score_sql = (
                "GREATEST(" + ", ".join(score_exprs) + ")" if score_exprs else "NULL::double precision"
            )

            sql = f"""
                SELECT
                    m.id AS movie_id,
                    m.title,
                    m.original_title,
                    m.tagline,
                    m.release_date::date AS release_date,
                    m.status,
                    m.original_language,
                    m.runtime,
                    m.budget,
                    m.revenue,
                    m.popularity,
                    m.vote_average,
                    m.vote_count,
                    {fuzzy_score_sql} AS fuzzy_score
                FROM movies m
                WHERE {' AND '.join(where) if where else 'TRUE'}
                ORDER BY fuzzy_score DESC NULLS LAST,
                         m.vote_average DESC NULLS LAST,
                         m.popularity DESC NULLS LAST,
                         m.id ASC
                LIMIT %s
            """
            all_params = score_params + params + [criteria.limit]
            with conn.cursor() as cur:
                cur.execute(sql, all_params)
                rows = cur.fetchall()

            if not rows:
                return []

            ids = [int(row["movie_id"]) for row in rows]
            related = self._relation_summaries(conn, ids)
            results: list[MovieSearchHit] = []
            for row in rows:
                movie_id = int(row["movie_id"])
                results.append(
                    MovieSearchHit(
                        movie_id=movie_id,
                        title=row["title"],
                        original_title=row["original_title"],
                        tagline=row["tagline"],
                        release_date=row["release_date"],
                        status=row["status"],
                        original_language=row["original_language"],
                        runtime=float(row["runtime"]) if row["runtime"] is not None else None,
                        budget=int(row["budget"]) if row["budget"] is not None else None,
                        revenue=int(row["revenue"]) if row["revenue"] is not None else None,
                        popularity=float(row["popularity"]) if row["popularity"] is not None else None,
                        vote_average=float(row["vote_average"]) if row["vote_average"] is not None else None,
                        vote_count=int(row["vote_count"]) if row["vote_count"] is not None else None,
                        fuzzy_score=float(row["fuzzy_score"]) if row["fuzzy_score"] is not None else None,
                        genres=related["genres"].get(movie_id, []),
                        production_companies=related["production_companies"].get(movie_id, []),
                        production_countries=related["production_countries"].get(movie_id, []),
                        spoken_languages=related["spoken_languages"].get(movie_id, []),
                        cast=related["cast"].get(movie_id, []),
                        crew=related["crew"].get(movie_id, []),
                    )
                )
            return results

    @staticmethod
    def _relation_summaries(conn: psycopg.Connection, ids: list[int]) -> dict[str, dict[int, list[str]]]:
        output: dict[str, dict[int, list[str]]] = {
            "genres": defaultdict(list),
            "production_companies": defaultdict(list),
            "production_countries": defaultdict(list),
            "spoken_languages": defaultdict(list),
            "cast": defaultdict(list),
            "crew": defaultdict(list),
        }
        queries = {
            "genres": """
                SELECT mg.movie_id, g.name
                FROM movie_genres mg JOIN genres g ON g.id = mg.genre_id
                WHERE mg.movie_id = ANY(%s) ORDER BY mg.movie_id, g.name
            """,
            "production_companies": """
                SELECT mpc.movie_id, pc.name
                FROM movie_production_companies mpc
                JOIN production_companies pc ON pc.id = mpc.company_id
                WHERE mpc.movie_id = ANY(%s) ORDER BY mpc.movie_id, pc.name
            """,
            "production_countries": """
                SELECT mpc.movie_id, pc.name
                FROM movie_production_countries mpc
                JOIN production_countries pc ON pc.iso_3166_1 = mpc.country_iso_3166_1
                WHERE mpc.movie_id = ANY(%s) ORDER BY mpc.movie_id, pc.name
            """,
            "spoken_languages": """
                SELECT msl.movie_id, sl.name
                FROM movie_spoken_languages msl
                JOIN spoken_languages sl ON sl.iso_639_1 = msl.language_iso_639_1
                WHERE msl.movie_id = ANY(%s) ORDER BY msl.movie_id, sl.name
            """,
            "cast": """
                SELECT mc.movie_id, p.name
                FROM movie_cast mc JOIN people p ON p.id = mc.person_id
                WHERE mc.movie_id = ANY(%s)
                ORDER BY mc.movie_id, mc.cast_order NULLS LAST, p.name
            """,
            "crew": """
                SELECT mc.movie_id, p.name
                FROM movie_crew mc JOIN people p ON p.id = mc.person_id
                WHERE mc.movie_id = ANY(%s)
                ORDER BY mc.movie_id, p.name
            """,
        }
        with conn.cursor() as cur:
            for key, query in queries.items():
                cur.execute(query, (ids,))
                seen: dict[int, set[str]] = defaultdict(set)
                for row in cur.fetchall():
                    movie_id = int(row["movie_id"])
                    name = row["name"]
                    if name and name not in seen[movie_id]:
                        output[key][movie_id].append(name)
                        seen[movie_id].add(name)
        return output

    def movie_details(self, movie_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM movies WHERE id = %s", (movie_id,))
                movie = cur.fetchone()
                if movie is None:
                    return None
                movie = dict(movie)
                # Search-helper columns are implementation details, not movie metadata.
                for key in ("title_search", "original_title_search", "tagline_search"):
                    movie.pop(key, None)

                relations: dict[str, Any] = {}
                relation_queries = {
                    "genres": """
                        SELECT g.* FROM movie_genres mg JOIN genres g ON g.id = mg.genre_id
                        WHERE mg.movie_id = %s ORDER BY g.name
                    """,
                    "keywords": """
                        SELECT k.* FROM movie_keywords mk JOIN keywords k ON k.id = mk.keyword_id
                        WHERE mk.movie_id = %s ORDER BY k.name
                    """,
                    "production_companies": """
                        SELECT pc.* FROM movie_production_companies mpc
                        JOIN production_companies pc ON pc.id = mpc.company_id
                        WHERE mpc.movie_id = %s ORDER BY pc.name
                    """,
                    "production_countries": """
                        SELECT pc.* FROM movie_production_countries mpc
                        JOIN production_countries pc ON pc.iso_3166_1 = mpc.country_iso_3166_1
                        WHERE mpc.movie_id = %s ORDER BY pc.name
                    """,
                    "spoken_languages": """
                        SELECT sl.* FROM movie_spoken_languages msl
                        JOIN spoken_languages sl ON sl.iso_639_1 = msl.language_iso_639_1
                        WHERE msl.movie_id = %s ORDER BY sl.name
                    """,
                    "cast": """
                        SELECT mc.credit_id, mc.person_id, p.name, p.gender, mc.character, mc.cast_order
                        FROM movie_cast mc JOIN people p ON p.id = mc.person_id
                        WHERE mc.movie_id = %s
                        ORDER BY mc.cast_order NULLS LAST, p.name
                    """,
                    "crew": """
                        SELECT mc.credit_id, mc.person_id, p.name, p.gender, mc.department, mc.job
                        FROM movie_crew mc JOIN people p ON p.id = mc.person_id
                        WHERE mc.movie_id = %s
                        ORDER BY mc.department NULLS LAST, mc.job NULLS LAST, p.name
                    """,
                }
                for key, query in relation_queries.items():
                    cur.execute(query, (movie_id,))
                    rows = [dict(row) for row in cur.fetchall()]
                    for row in rows:
                        row.pop("name_search", None)
                    relations[key] = rows

            return json_safe({"movie": movie, **relations})

    def resolve_reference(
        self,
        reference_type: ReferenceType,
        query: str | None,
        *,
        limit: int,
    ) -> list[ReferenceMatch]:
        with self._connect() as conn:
            has_trgm = self._has_pg_trgm(conn)
            normalized = normalize_text(query) if query else None
            threshold = self._fuzzy_threshold

            if reference_type in {"status", "original_language"}:
                column = "status" if reference_type == "status" else "original_language"
                params: list[Any] = []
                where = ""
                if normalized:
                    where = f"WHERE (lower(coalesce({column}, '')) = %s OR lower(coalesce({column}, '')) LIKE %s)"
                    params.extend([normalized, f"%{normalized}%"])
                params.append(limit)
                sql = f"""
                    SELECT DISTINCT {column} AS value
                    FROM movies
                    {where}
                    AND {column} IS NOT NULL
                    ORDER BY value
                    LIMIT %s
                """ if where else f"""
                    SELECT DISTINCT {column} AS value
                    FROM movies
                    WHERE {column} IS NOT NULL
                    ORDER BY value
                    LIMIT %s
                """
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return [ReferenceMatch(value=row["value"], score=1.0) for row in cur.fetchall()]

            table_meta = {
                "genre": ("genres", "id", None, None),
                "keyword": ("keywords", "id", None, None),
                "production_company": ("production_companies", "id", None, None),
                "production_country": ("production_countries", None, "iso_3166_1", None),
                "spoken_language": ("spoken_languages", None, "iso_639_1", None),
                "person": ("people", "id", None, None),
                "cast": ("people", "id", None, "movie_cast"),
                "crew": ("people", "id", None, "movie_crew"),
            }
            table, id_col, code_col, relation = table_meta[reference_type]
            alias = "x"
            joins = ""
            if relation:
                joins = f"JOIN {relation} rel ON rel.person_id = {alias}.id"

            select_id = f", {alias}.{id_col} AS entity_id" if id_col else ", NULL::bigint AS entity_id"
            select_code = f", {alias}.{code_col} AS code" if code_col else ", NULL::text AS code"
            params = []
            where_parts: list[str] = []
            score_sql = "1.0::double precision"
            if normalized:
                exact_parts = [f"{alias}.name_search = %s", f"{alias}.name_search LIKE %s"]
                params.extend([normalized, f"%{normalized}%"])
                if code_col:
                    exact_parts.append(f"lower({alias}.{code_col}) = %s")
                    params.append(normalized)
                if has_trgm:
                    exact_parts.append(f"similarity({alias}.name_search, %s) >= %s")
                    params.extend([normalized, threshold])
                    score_sql = f"similarity({alias}.name_search, %s)"
                    score_params = [normalized]
                else:
                    score_params = []
                where_parts.append("(" + " OR ".join(exact_parts) + ")")
            else:
                score_params = []
            params.append(limit)

            sql = f"""
                SELECT DISTINCT {alias}.name AS value
                    {select_id}
                    {select_code},
                    {score_sql} AS score
                FROM {table} {alias}
                {joins}
                {'WHERE ' + ' AND '.join(where_parts) if where_parts else ''}
                ORDER BY score DESC, value ASC
                LIMIT %s
            """
            with conn.cursor() as cur:
                cur.execute(sql, score_params + params)
                rows = cur.fetchall()
                return [
                    ReferenceMatch(
                        value=row["value"],
                        id=int(row["entity_id"]) if row["entity_id"] is not None else None,
                        code=row["code"],
                        score=float(row["score"] or 0.0),
                    )
                    for row in rows
                    if row["value"]
                ]
