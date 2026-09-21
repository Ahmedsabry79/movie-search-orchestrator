from __future__ import annotations

from dataclasses import dataclass, field

import httpx


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
        return self.overview.strip()

    def title_tagline_document(self) -> str:
        sections: list[str] = []
        if self.title:
            sections.append(f"Title: {self.title}")
        if self.tagline:
            sections.append(f"Tagline: {self.tagline}")
        return "\n".join(sections)

    def retrieval_document(self) -> str:
        return self.grouped_document()


class MovieRepository:
    """HTTP client for retrieval-ready movie records owned by movie-db-app.

    The vector service intentionally has no PostgreSQL credentials or SQL driver.
    All source metadata used to build the Milvus index comes through the DB app's
    internal, authenticated vector-index endpoint.
    """

    def __init__(self, base_url: str, api_key: str = "", page_size: int = 500):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._page_size = max(1, min(page_size, 1000))

    @property
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self._api_key} if self._api_key else {}

    def ping(self) -> bool:
        try:
            with httpx.Client(timeout=3.0) as client:
                response = client.get(f"{self._base_url}/health/ready")
                return response.status_code == 200
        except Exception:
            return False

    def fetch_all(self) -> list[MovieRecord]:
        records: list[MovieRecord] = []
        offset = 0
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=5.0), headers=self._headers) as client:
            while True:
                response = client.get(
                    f"{self._base_url}/internal/vector/movies",
                    params={"offset": offset, "limit": self._page_size},
                )
                response.raise_for_status()
                payload = response.json()
                items = payload.get("items", [])
                for item in items:
                    records.append(
                        MovieRecord(
                            movie_id=int(item["movie_id"]),
                            title=str(item.get("title") or "").strip(),
                            original_title=str(item.get("original_title") or "").strip(),
                            tagline=str(item.get("tagline") or "").strip(),
                            overview=str(item.get("overview") or "").strip(),
                            genres=[str(value) for value in item.get("genres", []) if value],
                            keywords=[str(value) for value in item.get("keywords", []) if value],
                            release_year=int(item["release_year"]) if item.get("release_year") is not None else None,
                            vote_average=float(item["vote_average"]) if item.get("vote_average") is not None else None,
                            popularity=float(item["popularity"]) if item.get("popularity") is not None else None,
                        )
                    )
                next_offset = payload.get("next_offset")
                if next_offset is None:
                    break
                offset = int(next_offset)
        return records
