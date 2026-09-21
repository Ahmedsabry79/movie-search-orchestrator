import json

import pandas as pd
import pytest

from conftest import write_sources
from preprocessing import CREDITS_FILE, MOVIES_FILE, normalize_text, preprocess


def test_cleaning_and_null_csv_roundtrip(raw_sources, tmp_path):
    movies = pd.read_csv(raw_sources / MOVIES_FILE, keep_default_na=False).astype(object)
    movies.loc[0, ["release_date", "budget", "revenue", "vote_count", "runtime"]] = ["not-a-date", 0, -1, 1.5, 0]
    movies.loc[1, "vote_average"] = -1
    movies.loc[2, "release_date"] = ""
    movies.loc[3, "vote_count"] = 0
    movies.to_csv(raw_sources / MOVIES_FILE, index=False)
    tables = preprocess(raw_sources, tmp_path / "processed")
    first = tables["movies"].iloc[0]
    for field in ["release_date", "budget", "revenue", "vote_count", "runtime"]:
        assert pd.isna(first[field])
    assert first.vote_average == 8.6
    assert pd.isna(tables["movies"].iloc[1].vote_average)
    assert pd.isna(tables["movies"].iloc[2].release_date)
    assert pd.isna(tables["movies"].iloc[3].vote_count)
    assert tables["production_countries"].iloc[0].iso_3166_1 == "NA"
    saved = pd.read_csv(tmp_path / "processed/movies_table.csv", keep_default_na=False)
    assert saved.iloc[0].release_date == "\\N"
    assert "Overview" in first.overview
    assert pd.isna(tables["keywords"].iloc[1]["name"])


def test_person_ids_and_multiple_roles(raw_sources, tmp_path):
    tables = preprocess(raw_sources, tmp_path / "processed")
    assert set(tables["people"].id) == {101, 102, 201}
    cast = tables["movie_cast"]
    assert set(cast[cast.movie_id == 1].person_id) == {101}
    assert set(cast[cast.movie_id == 2].person_id) == {102}
    assert len(cast[cast.movie_id == 1]) == 2
    assert set(tables["movie_crew"].person_id) == {201}


def test_missing_credits_preserve_movies(raw_sources, tmp_path):
    credits = pd.read_csv(raw_sources / CREDITS_FILE).iloc[1:]
    credits.to_csv(raw_sources / CREDITS_FILE, index=False)
    tables = preprocess(raw_sources, tmp_path / "processed")
    assert len(tables["movies"]) == 4
    assert {"cast", "crew"}.issubset(json.loads(tables["movies"].iloc[0].missing_fields))


def test_missing_json_is_different_from_empty(raw_sources, tmp_path):
    movies = pd.read_csv(raw_sources / MOVIES_FILE, keep_default_na=False)
    movies.loc[0, "keywords"] = ""
    movies.loc[1, "keywords"] = "[]"
    movies.to_csv(raw_sources / MOVIES_FILE, index=False)
    tables = preprocess(raw_sources, tmp_path / "processed")
    assert "keywords" in json.loads(tables["movies"].iloc[0].missing_fields)
    assert "keywords" not in json.loads(tables["movies"].iloc[1].missing_fields)


@pytest.mark.parametrize("problem", ["duplicate_movie", "duplicate_credits", "invalid_json", "conflicting_genre"])
def test_bad_data_fails_before_csv_export(raw_sources, tmp_path, problem):
    movies = pd.read_csv(raw_sources / MOVIES_FILE, keep_default_na=False)
    credits = pd.read_csv(raw_sources / CREDITS_FILE, keep_default_na=False)
    if problem == "duplicate_movie":
        movies.loc[1, "id"] = 1
    elif problem == "duplicate_credits":
        credits.loc[1, "movie_id"] = 1
    elif problem == "invalid_json":
        movies.loc[0, "genres"] = "broken"
    else:
        movies.loc[1, "genres"] = '[{"id":10,"name":"Different name"}]'
    movies.to_csv(raw_sources / MOVIES_FILE, index=False)
    credits.to_csv(raw_sources / CREDITS_FILE, index=False)
    with pytest.raises(ValueError):
        preprocess(raw_sources, tmp_path / "processed")
    assert not (tmp_path / "processed").exists()


def test_normalization_is_shared():
    assert normalize_text("  THE\u00a0  Matrix  ") == "the matrix"
    assert normalize_text("Ａvatar") == "avatar"


def test_repeated_links_and_credits_are_removed(tmp_path):
    raw = tmp_path / "raw"
    movies, credits = write_sources(raw)
    movies[0]["genres"] = json.dumps(json.loads(movies[0]["genres"]) * 2)
    credits[0]["cast"] = json.dumps(json.loads(credits[0]["cast"]) * 2)
    write_sources(raw, movies, credits)
    tables = preprocess(raw, tmp_path / "processed")
    assert len(tables["movie_genres"]) == 8
    assert len(tables["movie_cast"]) == 5
