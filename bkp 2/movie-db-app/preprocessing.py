import argparse
import json
import logging
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
NULL_VALUE = "\\N"
MOVIES_FILE = "tmdb_5000_movies.csv"
CREDITS_FILE = "tmdb_5000_credits.csv"


def clean_text(value):
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def normalize_text(value):
    ## Keep the original text for display, use this value only for searching
    value = clean_text(value)
    if value is None:
        return None
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def positive_numbers(values, integer=False):
    ## Missing, invalid and non positive values are unknown, not actual zero values
    values = pd.to_numeric(values, errors="coerce")
    values = values.where(np.isfinite(values) & (values > 0))
    if integer:
        ## Do not round a fractional budget, revenue or vote count into an integer
        values = values.where((values % 1 == 0) & (values < 2**63))
        return values.astype("Int64")
    return values.astype("Float64")


def required_id(value, description):
    try:
        value = str(value).strip()
        if not value.isdecimal() or int(value) <= 0 or int(value) >= 2**63:
            raise ValueError
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid {description}: {value!r}") from None


def parse_json_list(value, movie_id, column):
    ## Missing JSON is unknown; [] means the source has a known empty list
    value = clean_text(value)
    if value is None:
        return None
    try:
        result = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON in movie {movie_id}, column {column}") from exc
    if result is None:
        return None
    if not isinstance(result, list) or any(not isinstance(item, dict) for item in result):
        raise ValueError(f"Expected a JSON list of objects in movie {movie_id}, column {column}")
    return result


def add_entity(records, key, values, table_name):
    ## A repeated id must still refer to the same entity
    previous = records.get(key)
    if previous is None:
        records[key] = values
        return
    for column, value in values.items():
        old = previous[column]
        if old is None:
            previous[column] = value
        elif value is not None and old != value:
            raise ValueError(f"Conflicting {table_name} id {key}, column {column}: {old!r} / {value!r}")


def preprocess(input_dir=".", output_dir="processed"):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    ## Read strings first so names like NA and original missing values are preserved
    movies_data = pd.read_csv(input_dir / MOVIES_FILE, dtype=str, keep_default_na=False)
    credits_data = pd.read_csv(input_dir / CREDITS_FILE, dtype=str, keep_default_na=False)

    movies_columns = [
        "id", "title", "budget", "homepage", "original_language", "original_title",
        "overview", "popularity", "release_date", "revenue", "runtime", "status",
        "tagline", "vote_average", "vote_count",
    ]
    json_columns = ["genres", "keywords", "production_companies", "production_countries", "spoken_languages"]
    for frame, columns, label in [
        (movies_data, movies_columns + json_columns, "movies"),
        (credits_data, ["movie_id", "cast", "crew"], "credits"),
    ]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {label} columns: {sorted(missing)}")

    ## Validate both ids before joining, equal row counts alone are not enough
    movies_data["id"] = movies_data["id"].map(lambda value: required_id(value, "movie id"))
    credits_data["movie_id"] = credits_data["movie_id"].map(lambda value: required_id(value, "credits movie id"))
    if movies_data.empty:
        raise ValueError("Movies input is empty")
    if movies_data["id"].duplicated().any() or credits_data["movie_id"].duplicated().any():
        raise ValueError("Duplicate movie ids in movies or credits")
    missing_credits = set(movies_data["id"]) - set(credits_data["movie_id"])
    unmatched_credits = set(credits_data["movie_id"]) - set(movies_data["id"])
    if missing_credits:
        logger.warning("Movies without credits: %s", sorted(missing_credits))
    if unmatched_credits:
        logger.warning("Credits without movies, excluded from the join: %s", sorted(unmatched_credits))
    credits_data = credits_data.drop(columns=["title"], errors="ignore")
    data = pd.merge(movies_data, credits_data, left_on="id", right_on="movie_id", how="left", validate="one_to_one")
    data = data.drop(columns=["movie_id"]).sort_values("id").reset_index(drop=True)
    if len(data) != len(movies_data):
        raise ValueError("Data length mismatch after merge")

    ## Parse each JSON field once, keep missing fields separate from empty lists
    for column in json_columns + ["cast", "crew"]:
        data[column] = [parse_json_list(value, movie_id, column) for movie_id, value in zip(data["id"], data[column])]

    ## Prepare the movies table, keep overview and tagline for later retrieval
    movies_table = data[movies_columns].copy()
    for column in ["title", "homepage", "original_language", "original_title", "overview", "status", "tagline"]:
        movies_table[column] = movies_table[column].map(clean_text)
    if movies_table["title"].isna().any():
        raise ValueError("A movie has no title")

    ## Convert non-null dates into datetime; missing and invalid dates become null
    movies_table["release_date"] = pd.to_datetime(movies_table["release_date"], format="mixed", errors="coerce", utc=True).dt.tz_localize(None)
    ## Budget, revenue and vote count must be positive integers, otherwise null
    for column in ["budget", "revenue", "vote_count"]:
        movies_table[column] = positive_numbers(movies_table[column], integer=True)
    ## Rating and runtime can contain decimals; keep them, but make non positive values null
    movies_table["vote_average"] = positive_numbers(movies_table["vote_average"])
    movies_table["runtime"] = positive_numbers(movies_table["runtime"])
    ## Zero popularity is allowed, negative and invalid values are unknown
    popularity = pd.to_numeric(movies_table["popularity"], errors="coerce")
    movies_table["popularity"] = popularity.where(np.isfinite(popularity) & (popularity >= 0)).astype("Float64")
    movies_table["title_search"] = movies_table["title"].map(normalize_text)
    movies_table["original_title_search"] = movies_table["original_title"].map(normalize_text)
    movies_table["tagline_search"] = movies_table["tagline"].map(normalize_text)
    ## Preserve whether relation data was missing, without writing a separate report
    movies_table["missing_fields"] = [
        json.dumps([column for column in json_columns + ["cast", "crew"] if row[column] is None])
        for _, row in data.iterrows()
    ]
    tables = {"movies": movies_table}
    print("Movies Table: Ok")

    ## Prepare the entity tables and their associative tables
    ## Keywords are saved here as well, semantic documents will be prepared later
    definitions = [
        ("genres", "id", "genre_id", "movie_genres"),
        ("keywords", "id", "keyword_id", "movie_keywords"),
        ("production_companies", "id", "company_id", "movie_production_companies"),
        ("production_countries", "iso_3166_1", "country_iso_3166_1", "movie_production_countries"),
        ("spoken_languages", "iso_639_1", "language_iso_639_1", "movie_spoken_languages"),
    ]
    for column, key_column, foreign_column, link_name in definitions:
        entities, links = {}, set()
        for _, row in data.iterrows():
            for entry in row[column] or []:
                key = required_id(entry.get(key_column), f"{column} id") if key_column == "id" else clean_text(entry.get(key_column))
                name = clean_text(entry.get("name"))
                if key is None:
                    raise ValueError(f"Missing id in movie {row['id']}, column {column}")
                ## Some source keywords have an id but no name; keep the id and a null name
                add_entity(entities, key, {key_column: key, "name": name, "name_search": normalize_text(name)}, column)
                links.add((row["id"], key))
        tables[column] = pd.DataFrame([entities[key] for key in sorted(entities)], columns=[key_column, "name", "name_search"])
        tables[link_name] = pd.DataFrame(sorted(links), columns=["movie_id", foreign_column])
        print(f"{column} and {link_name} Tables: Ok")

    ## Prepare one people table for both cast and crew
    ## The JSON id identifies the person; cast_id is only local to a movie
    people, cast_rows, crew_rows = {}, {}, {}
    for _, row in data.iterrows():
        movie_id = row["id"]
        for kind, records in [("cast", cast_rows), ("crew", crew_rows)]:
            for entry in row[kind] or []:
                person_id = required_id(entry.get("id"), "person id")
                name, credit_id = clean_text(entry.get("name")), clean_text(entry.get("credit_id"))
                if credit_id is None:
                    raise ValueError(f"Missing credit_id in movie {movie_id}, {kind}")
                ## Zero gender means unknown in this source
                gender = entry.get("gender")
                gender = int(gender) if str(gender) in {"1", "2", "3"} else None
                add_entity(people, person_id, {"id": person_id, "name": name, "name_search": normalize_text(name), "gender": gender}, "people")
                credit = {"credit_id": credit_id, "movie_id": movie_id, "person_id": person_id}
                if kind == "cast":
                    ## Character and billing order belong to the movie, not the person
                    order = entry.get("order")
                    if order is not None and (isinstance(order, bool) or not str(order).isdecimal()):
                        raise ValueError(f"Invalid cast order in movie {movie_id}")
                    credit.update(character=clean_text(entry.get("character")), cast_order=int(order) if order is not None else None)
                else:
                    credit.update(department=clean_text(entry.get("department")), job=clean_text(entry.get("job")))
                ## Repeated credits are removed, conflicting credits stop preprocessing
                add_entity(records, credit_id, credit, f"movie_{kind}")
    tables["people"] = pd.DataFrame([people[key] for key in sorted(people)], columns=["id", "name", "name_search", "gender"])
    tables["people"]["gender"] = tables["people"]["gender"].astype("Int64")
    tables["movie_cast"] = pd.DataFrame([cast_rows[key] for key in sorted(cast_rows)], columns=["credit_id", "movie_id", "person_id", "character", "cast_order"])
    tables["movie_cast"]["cast_order"] = tables["movie_cast"]["cast_order"].astype("Int64")
    tables["movie_crew"] = pd.DataFrame([crew_rows[key] for key in sorted(crew_rows)], columns=["credit_id", "movie_id", "person_id", "department", "job"])
    print("People, Movie Cast and Movie Crew Tables: Ok")

    ## Validate references before saving any output
    movie_ids, person_ids = set(tables["movies"]["id"]), set(tables["people"]["id"])
    for name, frame in tables.items():
        if "movie_id" in frame and not set(frame["movie_id"]).issubset(movie_ids):
            raise ValueError(f"Unknown movie id in {name}")
        if "person_id" in frame and not set(frame["person_id"]).issubset(person_ids):
            raise ValueError(f"Unknown person id in {name}")
    for column, key_column, foreign_column, link_name in definitions:
        if not set(tables[link_name][foreign_column]).issubset(set(tables[column][key_column])):
            raise ValueError(f"Unknown {column} id in {link_name}")

    ## Finally, save the tables to CSV files for database seeding
    ## Use \\N for null so empty strings and country codes like NA remain unchanged
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        path = output_dir / f"{name}_table.csv"
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, na_rep=NULL_VALUE, date_format="%Y-%m-%d %H:%M:%S")
        temporary.replace(path)
    print("All tables saved to CSV files.")
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--output-dir", default="processed")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    preprocess(args.input_dir, args.output_dir)
