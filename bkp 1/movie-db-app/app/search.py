import math
from datetime import datetime

from sqlalchemy import and_, cast, distinct, exists, extract, func, Integer, literal, or_, select

from preprocessing import normalize_text
from .contracts import Metric
from .models import (
    CastCredit, Company, Country, CrewCredit, Genre, Keyword, Language, Movie, Person,
    movie_genres, movie_keywords, movie_production_companies,
    movie_production_countries, movie_spoken_languages,
)

SCALAR_FIELDS = {
    name: getattr(Movie, name) for name in [
        "id", "budget", "revenue", "runtime", "vote_average", "vote_count",
        "popularity", "release_date", "original_language", "status",
    ]
}
SCALAR_FIELDS["release_year"] = cast(extract("year", Movie.release_date), Integer)
INTEGER_FIELDS = {"id", "budget", "revenue", "vote_count", "release_year"}
NUMBER_FIELDS = INTEGER_FIELDS | {"runtime", "vote_average", "popularity"}
RELATIONS = {
    "genre": (Genre.__table__, movie_genres, "genre_id", "id"),
    "keyword": (Keyword.__table__, movie_keywords, "keyword_id", "id"),
    "production_company": (Company.__table__, movie_production_companies, "company_id", "id"),
    "production_country": (Country.__table__, movie_production_countries, "country_iso_3166_1", "iso_3166_1"),
    "spoken_language": (Language.__table__, movie_spoken_languages, "language_iso_639_1", "iso_639_1"),
    "actor": (Person.__table__, CastCredit.__table__, "person_id", "id"),
    "director": (Person.__table__, CrewCredit.__table__, "person_id", "id"),
}
ID_RELATIONS = {f"{name}_id": name for name in RELATIONS}
LIST_COLUMNS = [Movie.id, Movie.title, Movie.original_title, Movie.release_date,
                Movie.vote_average, Movie.vote_count, Movie.runtime, Movie.budget, Movie.revenue]


class ResolutionRequired(Exception):
    def __init__(self, field, query, candidates):
        self.payload = {
            "status": "ambiguous" if candidates else "no_matches",
            "field": field, "query": query, "candidates": candidates,
            "message": "Choose an entity id before applying this filter." if candidates else "No sufficiently similar entity was found.",
        }


def similarity_expression(column, query):
    return func.greatest(func.similarity(column, query), func.strict_word_similarity(query, column))


def fuzzy_condition(column, query):
    # Both operators use the request-local pg_trgm thresholds and support GIN indexes.
    return or_(column.bool_op("%")(query), column.bool_op("%>>")(query))


def classify(candidates, exact, gap):
    if not candidates:
        return "no_matches"
    if len(candidates) == 1:
        return "matched"
    if exact or candidates[0]["similarity"] - candidates[1]["similarity"] < gap:
        return "ambiguous"
    return "matched"


def resolve_entity(connection, field, query, settings):
    entity, link, foreign, primary = RELATIONS[field]
    normalized = normalize_text(query)
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    column = entity.c.name_search
    statement = select(entity.c[primary].label("id"), entity.c.name)
    if field in {"actor", "director"}:
        scope = [link.c[foreign] == entity.c[primary]]
        if field == "director":
            scope.append(link.c.job == "Director")
        statement = statement.where(exists(select(1).select_from(link).where(*scope)))
    exact = connection.execute(statement.add_columns(literal(1.0).label("similarity")).where(column == normalized).order_by(entity.c[primary]).limit(6)).mappings().all()
    rows = exact
    if not rows and len(normalized) >= 3:
        score = similarity_expression(column, normalized)
        rows = connection.execute(statement.add_columns(score.label("similarity")).where(fuzzy_condition(column, normalized)).order_by(score.desc(), entity.c[primary]).limit(6)).mappings().all()
    candidates = [dict(row) for row in rows]
    if classify(candidates, bool(exact), settings.fuzzy_min_gap) != "matched":
        raise ResolutionRequired(field, query, candidates)
    return candidates[0]["id"], candidates[0]


def typed_value(field, value):
    if field in NUMBER_FIELDS:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} requires a finite number")
        if isinstance(value, int) and not -(2**63) <= value < 2**63:
            raise ValueError(f"{field} requires a number within the 64-bit integer range")
        if not math.isfinite(value):
            raise ValueError(f"{field} requires a finite number")
        if field in INTEGER_FIELDS and (int(value) != value or not -(2**63) <= value < 2**63):
            raise ValueError(f"{field} requires a 64-bit integer")
        return int(value) if field in INTEGER_FIELDS else value
    if not isinstance(value, str) or not value.strip() or len(value) > 300:
        raise ValueError(f"{field} requires a nonempty string of at most 300 characters")
    if field == "release_date":
        try:
            result = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("release_date requires an ISO date or datetime") from exc
        if result.tzinfo:
            raise ValueError("release_date uses a timezone-free calendar date/datetime")
        return result
    return value.strip()


def scalar_condition(item):
    field, operator, value = item.field, item.operator, item.value
    column = SCALAR_FIELDS[field]
    if operator in {"is_null", "not_null"}:
        if value is not None:
            raise ValueError(f"{operator} does not accept a value")
        return column.is_(None) if operator == "is_null" else column.is_not(None)
    if operator == "in":
        if not isinstance(value, list) or not 1 <= len(value) <= 100:
            raise ValueError("in requires 1 to 100 values")
        return column.in_([typed_value(field, part) for part in value])
    value = typed_value(field, value)
    if field in {"original_language", "status"} and operator not in {"eq", "ne"}:
        raise ValueError(f"{field} supports only eq, ne, in and null checks")
    return {"eq": column.__eq__, "ne": column.__ne__, "gt": column.__gt__, "gte": column.__ge__, "lt": column.__lt__, "lte": column.__le__}[operator](value)


def relation_condition(connection, item, settings):
    field = ID_RELATIONS.get(item.field, item.field)
    entity, link, foreign, primary = RELATIONS[field]
    if item.operator not in {"eq", "in"}:
        raise ValueError(f"{item.field} supports eq and in only")
    values = item.value if item.operator == "in" else [item.value]
    if not isinstance(values, list) or not 1 <= len(values) <= 100:
        raise ValueError("Relation filter requires 1 to 100 values")
    ids, resolutions = [], []
    for value in values:
        if item.field in ID_RELATIONS:
            if primary == "id":
                value = typed_value("id", value)
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f"{item.field} requires a country/language code")
            ids.append(value)
        else:
            if not isinstance(value, str) or len(value) > 300:
                raise ValueError(f"{field} requires an entity name")
            entity_id, resolution = resolve_entity(connection, field, value, settings)
            ids.append(entity_id)
            resolutions.append({"field": field, "input": value, **resolution})
    predicates = [link.c.movie_id == Movie.id, link.c[foreign].in_(ids)]
    if field == "director":
        predicates.append(link.c.job == "Director")
    return exists(select(1).select_from(link).where(*predicates)), resolutions


def apply_text_match(connection, base, request, settings):
    match = request.text_match
    if match is None:
        return base, None, None
    query = normalize_text(match.query)
    columns = [getattr(Movie, f"{field}_search") for field in match.fields]
    exact_condition = or_(*(column == query for column in columns))
    exact = bool(connection.scalar(select(exists(base.where(exact_condition)))))
    score = literal(1.0) if exact else func.greatest(*(similarity_expression(column, query) for column in columns))
    if exact:
        base = base.where(exact_condition)
    elif len(query) < 3:
        base = base.where(literal(False))
    else:
        base = base.where(or_(*(fuzzy_condition(column, query) for column in columns)))
    candidates = [dict(row) for row in connection.execute(
        base.with_only_columns(Movie.id, Movie.title, Movie.release_date, score.label("similarity"))
        .order_by(score.desc(), Movie.id).limit(6)
    ).mappings()]
    status = classify(candidates, exact, settings.fuzzy_min_gap)
    resolution = {
        "status": status, "query": match.query, "fields": match.fields,
        "method": "normalized_exact" if exact else "trigram_and_word_similarity",
        "threshold": settings.fuzzy_threshold, "candidates": candidates,
        "selected_movie_id": candidates[0]["id"] if status == "matched" else None,
        "requires_clarification": status == "ambiguous",
    }
    return base, score, resolution


def aggregate_statement(base, request):
    # Starting from unique movie ids prevents filters across many-to-many relations
    # from multiplying movies, revenues or averages.
    eligible = base.with_only_columns(Movie.id).subquery()
    dimensions = []
    source = Movie.__table__.join(eligible, Movie.id == eligible.c.id)
    group = request.group_by
    if group in RELATIONS:
        entity, link, foreign, primary = RELATIONS[group]
        source = source.join(link, link.c.movie_id == Movie.id).join(entity, entity.c[primary] == link.c[foreign])
        dimensions = [entity.c[primary].label("group_id"), entity.c.name.label("group_value")]
    elif group:
        dimensions = [SCALAR_FIELDS[group].label("group_value")]
    metrics = request.metrics or [Metric(function="count", alias="movie_count")]
    numeric = {metric.field for metric in metrics if metric.function != "count"}
    if any(field not in NUMBER_FIELDS - {"id", "release_year"} for field in numeric):
        raise ValueError("Numeric aggregates require budget, revenue, runtime, vote_average, vote_count or popularity")
    for metric in metrics:
        if metric.function == "count" and metric.field not in {None, "id"}:
            raise ValueError("count counts movies; omit its field or use id")
    flattened = select(Movie.id, *dimensions, *(SCALAR_FIELDS[field].label(field) for field in sorted(numeric))).select_from(source)
    if group == "director":
        flattened = flattened.where(CrewCredit.job == "Director")
    rows = flattened.distinct().subquery()
    group_columns = [rows.c[column.name] for column in dimensions]
    expressions = []
    for metric in metrics:
        expression = func.count(distinct(rows.c.id)) if metric.function == "count" else getattr(func, metric.function)(rows.c[metric.field])
        expressions.append(expression.label(metric.alias))
    statement = select(*group_columns, *expressions).select_from(rows)
    if group_columns:
        statement = statement.group_by(*group_columns)
    allowed_sort = {column.name: column for column in [*group_columns, *expressions]}
    order = []
    for spec in request.sort:
        if spec.field not in allowed_sort:
            raise ValueError(f"Unknown aggregate sort field: {spec.field}")
        column = allowed_sort[spec.field]
        order.append((column.asc() if spec.direction == "asc" else column.desc()).nulls_last())
    if not order:
        order = [expressions[0].desc().nulls_last()]
    return statement, [*order, *(column.asc().nulls_last() for column in group_columns)]


def structured_search(connection, request, settings):
    predicates, resolved = [], []
    for item in request.filters:
        if item.field in SCALAR_FIELDS:
            predicates.append(scalar_condition(item))
        elif item.field in RELATIONS or item.field in ID_RELATIONS:
            predicate, resolutions = relation_condition(connection, item, settings)
            predicates.append(predicate)
            resolved.extend(resolutions)
        else:
            raise ValueError(f"Unsupported filter field: {item.field}. Use text_match for title, original_title or tagline.")
    if request.movie_ids is not None:
        predicates.append(Movie.id.in_(request.movie_ids))
    base = select(Movie.id).where(*predicates)
    base, score, resolution = apply_text_match(connection, base, request, settings)
    total = connection.scalar(select(func.count()).select_from(base.subquery()))
    response = {
        "status": "ok" if total else "no_matches", "operation": request.operation,
        "matched_movie_count": total, "applied_filters": [item.model_dump() for item in request.filters],
        "resolved_entities": resolved, "text_resolution": resolution,
    }
    if request.operation == "count":
        return {**response, "count": total}
    if request.operation == "aggregate":
        statement, order = aggregate_statement(base, request)
        total_groups = connection.scalar(select(func.count()).select_from(statement.subquery()))
        results = [dict(row) for row in connection.execute(statement.order_by(*order).offset(request.offset).limit(request.limit)).mappings()]
        return {**response, "results": results, "total_groups": total_groups, "has_more": request.offset + len(results) < total_groups}
    columns = LIST_COLUMNS + ([score.label("similarity")] if score is not None else [])
    order = []
    for spec in request.sort:
        column = Movie.title_search if spec.field == "title" else SCALAR_FIELDS.get(spec.field)
        if spec.field == "similarity":
            column = score
        if column is None:
            raise ValueError(f"Unsupported sort field: {spec.field}")
        order.append((column.asc() if spec.direction == "asc" else column.desc()).nulls_last())
    if not order and score is not None:
        order.append(score.desc())
    results = [dict(row) for row in connection.execute(base.with_only_columns(*columns).order_by(*order, Movie.id).offset(request.offset).limit(request.limit)).mappings()]
    return {**response, "results": results, "limit": request.limit, "offset": request.offset,
            "has_more": request.offset + len(results) < total}
