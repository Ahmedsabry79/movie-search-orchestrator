import json

import pandas as pd
import pytest

from preprocessing import CREDITS_FILE, MOVIES_FILE


def write_sources(folder, movies=None, credits=None):
    folder.mkdir(parents=True, exist_ok=True)
    if movies is None:
        movies = []
        for movie_id, title, rating, revenue, year in [
            (1, "Interstellar", 8.6, 100, 2014), (2, "Avatar", 7.9, 200, 2009),
            (3, "The Matrix", 8.7, 300, 1999), (4, "The Matrix", 6.0, 0, 2020),
        ]:
            movies.append({
                "id": movie_id, "title": title, "original_title": title,
                "overview": f"Overview for {title}", "tagline": "A journey beyond", "homepage": "",
                "original_language": "en", "status": "Released", "budget": 50,
                "revenue": revenue, "runtime": 160 if movie_id == 1 else 110,
                "vote_average": rating, "vote_count": 2000, "popularity": 0,
                "release_date": f"{year}-01-01",
                "genres": json.dumps([{"id": 10, "name": "Science Fiction"}, {"id": 20, "name": "Drama"}]),
                "keywords": json.dumps([{"id": 30, "name": "space"}, {"id": 31, "name": ""}]),
                "production_companies": json.dumps([{"id": 40, "name": "Example Studio"}]),
                "production_countries": json.dumps([{"iso_3166_1": "NA", "name": "Namibia"}]),
                "spoken_languages": json.dumps([{"iso_639_1": "en", "name": "English"}]),
            })
    if credits is None:
        credits = []
        for movie_id in [1, 2, 3, 4]:
            cast = [{"id": 101 if movie_id == 1 else 102, "cast_id": 1, "name": "Actor One" if movie_id == 1 else "Actor Two", "gender": 2, "credit_id": f"cast-{movie_id}", "character": "Hero", "order": 0}]
            if movie_id == 1:
                cast.append({**cast[0], "credit_id": "cast-1-second-role", "character": "Other role", "order": 1})
            crew = [{"id": 201, "name": "Christopher Nolan", "gender": 2, "credit_id": f"director-{movie_id}", "department": "Directing", "job": "Director"}]
            if movie_id == 1:
                crew.append({**crew[0], "credit_id": "writer-1", "department": "Writing", "job": "Writer"})
            credits.append({"movie_id": movie_id, "title": "unused", "cast": json.dumps(cast), "crew": json.dumps(crew)})
    pd.DataFrame(movies).to_csv(folder / MOVIES_FILE, index=False)
    pd.DataFrame(credits).to_csv(folder / CREDITS_FILE, index=False)
    return movies, credits


@pytest.fixture
def raw_sources(tmp_path):
    raw = tmp_path / "raw"
    write_sources(raw)
    return raw
