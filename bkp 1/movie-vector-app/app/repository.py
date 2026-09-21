from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row


@dataclass(slots=True)
class MovieRecord:
    movie_id: int
    title: str
    original_title: str = ""
    tagline: str = ""
    overview: str = ""
    genres: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    release_year: int | None = None
    vote_average: float | None = None
    popularity: float | None = None

    def grouped_document(self) -> str:
        """Full movie representation used for grouped dense + BM25 retrieval."""
        sections: list[str] = []
        if self.title:
            sections.append(f"Title: {self.title}")
        if self.original_title and self.original_title != self.title:
            sections.append(f"Original title: {self.original_title}")
        if self.tagline:
            sections.append(f"Tagline: {self.tagline}")
        if self.genres:
            sections.append("Genres: " + ", ".join(self.genres))
        if self.keywords:
            sections.append("Keywords: " + ", ".join(self.keywords))
        if self.overview:
            sections.append(f"Overview: {self.overview}")
        return "\n".join(sections)

    def overview_document(self) -> str:
        """Overview-only dense representation.

        Keep the text unprefixed so the vector captures the synopsis itself rather
        than metadata labels. TMDB can contain an empty overview, in which case
        the embedder receives an empty string for that field.
        """
        return self.overview.strip()

    def title_tagline_document(self) -> str:
        """Short title + tagline dense representation."""
        sections: list[str] = []
        if self.title:
            sections.append(f"Title: {self.title}")
        if self.tagline:
            sections.append(f"Tagline: {self.tagline}")
        return "\n".join(sections)

    # Backward-compatible alias for any code that still calls the previous name.
    def retrieval_document(self) -> str:
        return self.grouped_document()


class MovieRepository:
    """Read retrieval-ready movie records from the existing Postgres DB.

    The loader intentionally tolerates either normalized genre/keyword tables
    or raw JSON/text columns on `movies`, so it fits the common TMDB seed
    layouts without coupling the vector service to the ORM of movie-db-app.
    """

    def __init__(self, dsn: str):
        self._dsn = dsn

    def ping(self) -> bool:
        try:
            with psycopg.connect(self._dsn, connect_timeout=3) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone() == (1,)
        except Exception:
            return False

    def _table_exists(self, conn: psycopg.Connection, table: str) -> bool:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM information_schema.tables
                    WHERE table_schema = 'public' AND table_name = %s
                )
                """,
                (table,),
            )
            return bool(cur.fetchone()[0])

    def _columns(self, conn: psycopg.Connection, table: str) -> set[str]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                """,
                (table,),
            )
            return {row[0] for row in cur.fetchall()}

    @staticmethod
    def _first(columns: set[str], *candidates: str) -> str | None:
        return next((name for name in candidates if name in columns), None)

    @staticmethod
    def _parse_names(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, dict):
                    name = item.get("name")
                    if name:
                        out.append(str(name).strip())
                elif item is not None:
                    out.append(str(item).strip())
            return [x for x in out if x]
        if isinstance(value, dict):
            name = value.get("name")
            return [str(name).strip()] if name else []
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw[0:1] in ("[", "{"):
                try:
                    return MovieRepository._parse_names(json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    pass
            separator = "|" if "|" in raw else ","
            return [part.strip() for part in raw.split(separator) if part.strip()]
        return [str(value).strip()]

    @staticmethod
    def _year(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, (date, datetime)):
            return value.year
        text = str(value).strip()
        if len(text) >= 4 and text[:4].isdigit():
            return int(text[:4])
        return None

    def _load_relation_names(
        self,
        conn: psycopg.Connection,
        relation_table: str,
        lookup_table: str,
        movie_fk_candidates: Iterable[str],
        lookup_fk_candidates: Iterable[str],
    ) -> dict[int, list[str]]:
        if not self._table_exists(conn, relation_table) or not self._table_exists(conn, lookup_table):
            return {}

        relation_columns = self._columns(conn, relation_table)
        lookup_columns = self._columns(conn, lookup_table)
        movie_fk = self._first(relation_columns, *movie_fk_candidates)
        lookup_fk = self._first(relation_columns, *lookup_fk_candidates)
        lookup_id = self._first(lookup_columns, "id", f"{lookup_table[:-1]}_id")
        name_col = self._first(lookup_columns, "name", "title")
        if not all((movie_fk, lookup_fk, lookup_id, name_col)):
            return {}

        query = f'''SELECT r."{movie_fk}" AS movie_id, l."{name_col}" AS name
                    FROM "{relation_table}" r
                    JOIN "{lookup_table}" l ON l."{lookup_id}" = r."{lookup_fk}"'''
        result: dict[int, list[str]] = defaultdict(list)
        with conn.cursor() as cur:
            cur.execute(query)
            for movie_id, name in cur.fetchall():
                if movie_id is not None and name:
                    result[int(movie_id)].append(str(name).strip())
        return dict(result)

    def fetch_all(self) -> list[MovieRecord]:
        with psycopg.connect(self._dsn) as conn:
            if not self._table_exists(conn, "movies"):
                raise RuntimeError("Expected a public.movies table but none was found")

            columns = self._columns(conn, "movies")
            id_col = self._first(columns, "id", "movie_id")
            title_col = self._first(columns, "title", "name")
            if not id_col or not title_col:
                raise RuntimeError("movies table must contain an id/movie_id and title/name column")

            mapping = {
                "movie_id": id_col,
                "title": title_col,
                "original_title": self._first(columns, "original_title"),
                "tagline": self._first(columns, "tagline"),
                "overview": self._first(columns, "overview", "description"),
                "release_date": self._first(columns, "release_date", "released_at", "year"),
                "vote_average": self._first(columns, "vote_average", "rating"),
                "popularity": self._first(columns, "popularity"),
                "genres_raw": self._first(columns, "genres", "genre_names"),
                "keywords_raw": self._first(columns, "keywords", "keyword_names"),
            }

            selections: list[str] = []
            for alias, column in mapping.items():
                if column:
                    selections.append(f'"{column}" AS "{alias}"')
                else:
                    selections.append(f'NULL AS "{alias}"')

            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(f'SELECT {", ".join(selections)} FROM "movies" ORDER BY "{id_col}"')
                movie_rows = cur.fetchall()

            genre_map = self._load_relation_names(
                conn,
                relation_table="movie_genres",
                lookup_table="genres",
                movie_fk_candidates=("movie_id", "id_movie"),
                lookup_fk_candidates=("genre_id", "id_genre"),
            )
            keyword_map = self._load_relation_names(
                conn,
                relation_table="movie_keywords",
                lookup_table="keywords",
                movie_fk_candidates=("movie_id", "id_movie"),
                lookup_fk_candidates=("keyword_id", "id_keyword"),
            )

            records: list[MovieRecord] = []
            for row in movie_rows:
                movie_id = int(row["movie_id"])
                genres = genre_map.get(movie_id) or self._parse_names(row["genres_raw"])
                keywords = keyword_map.get(movie_id) or self._parse_names(row["keywords_raw"])
                records.append(
                    MovieRecord(
                        movie_id=movie_id,
                        title=str(row["title"] or "").strip(),
                        original_title=str(row["original_title"] or "").strip(),
                        tagline=str(row["tagline"] or "").strip(),
                        overview=str(row["overview"] or "").strip(),
                        genres=sorted(set(genres)),
                        keywords=sorted(set(keywords)),
                        release_year=self._year(row["release_date"]),
                        vote_average=float(row["vote_average"]) if row["vote_average"] is not None else None,
                        popularity=float(row["popularity"]) if row["popularity"] is not None else None,
                    )
                )
            return records
