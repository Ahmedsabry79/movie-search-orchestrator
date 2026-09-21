from __future__ import annotations

from sqlalchemy import Integer, cast, distinct, func, select

from .agent_contracts import AgentAnalyticsRequest, AgentAnalyticsResponse
from .agent_search import build_movie_predicates
from .models import (
    CastCredit,
    Company,
    Country,
    CrewCredit,
    Genre,
    Keyword,
    Language,
    Movie,
    Person,
    movie_genres,
    movie_keywords,
    movie_production_companies,
    movie_production_countries,
    movie_spoken_languages,
)


NAMED_DIMENSIONS = {
    "genre": (Genre.__table__, movie_genres, "genre_id", "id"),
    "keyword": (Keyword.__table__, movie_keywords, "keyword_id", "id"),
    "production_company": (
        Company.__table__, movie_production_companies, "company_id", "id"
    ),
    "production_country": (
        Country.__table__, movie_production_countries, "country_iso_3166_1", "iso_3166_1"
    ),
    "spoken_language": (
        Language.__table__, movie_spoken_languages, "language_iso_639_1", "iso_639_1"
    ),
}

NUMERIC_METRICS = {
    "budget": Movie.__table__.c.budget,
    "revenue": Movie.__table__.c.revenue,
    "runtime": Movie.__table__.c.runtime,
    "vote_average": Movie.__table__.c.vote_average,
    "vote_count": Movie.__table__.c.vote_count,
    "popularity": Movie.__table__.c.popularity,
}


def _add_dimension(source, dimension: str, index: int):
    """Return (updated source, output columns, additional predicates).

    Every join is selected from a fixed allow-list. No caller-provided table/column/join
    expression reaches SQLAlchemy.
    """
    movie = Movie.__table__
    predicates = []

    if dimension == "original_language":
        return source, [movie.c.original_language.label("original_language")], predicates
    if dimension == "status":
        return source, [movie.c.status.label("status")], predicates
    if dimension == "release_year":
        return source, [cast(func.extract("year", movie.c.release_date), Integer).label("release_year")], predicates

    if dimension in NAMED_DIMENSIONS:
        entity, link, foreign, primary = NAMED_DIMENSIONS[dimension]
        link_alias = link.alias(f"analytics_{dimension}_link_{index}")
        entity_alias = entity.alias(f"analytics_{dimension}_{index}")
        source = source.join(link_alias, link_alias.c.movie_id == movie.c.id).join(
            entity_alias, entity_alias.c[primary] == link_alias.c[foreign]
        )
        return source, [
            entity_alias.c[primary].label(f"{dimension}_id"),
            entity_alias.c.name.label(dimension),
        ], predicates

    if dimension in {"cast_member", "cast_character"}:
        credit = CastCredit.__table__.alias(f"analytics_cast_{index}")
        source = source.join(credit, credit.c.movie_id == movie.c.id)
        if dimension == "cast_character":
            return source, [credit.c.character.label("cast_character")], predicates
        person = Person.__table__.alias(f"analytics_cast_person_{index}")
        source = source.join(person, person.c.id == credit.c.person_id)
        return source, [
            person.c.id.label("cast_member_id"),
            person.c.name.label("cast_member"),
        ], predicates

    if dimension in {"crew_member", "director", "crew_department", "crew_job"}:
        credit = CrewCredit.__table__.alias(f"analytics_crew_{index}")
        source = source.join(credit, credit.c.movie_id == movie.c.id)
        if dimension == "crew_department":
            return source, [credit.c.department.label("crew_department")], predicates
        if dimension == "crew_job":
            return source, [credit.c.job.label("crew_job")], predicates
        person = Person.__table__.alias(f"analytics_crew_person_{index}")
        source = source.join(person, person.c.id == credit.c.person_id)
        if dimension == "director":
            predicates.append(credit.c.job == "Director")
            return source, [
                person.c.id.label("director_id"),
                person.c.name.label("director"),
            ], predicates
        return source, [
            person.c.id.label("crew_member_id"),
            person.c.name.label("crew_member"),
        ], predicates

    raise ValueError(f"Unsupported analytics dimension: {dimension}")


def analyze_movies(connection, request: AgentAnalyticsRequest, settings) -> AgentAnalyticsResponse:
    threshold = request.filters.fuzzy_threshold or settings.fuzzy_threshold
    predicates = build_movie_predicates(request.filters, threshold)

    eligible = select(Movie.id).where(*predicates).distinct().subquery("analytics_eligible")
    matched_movie_count = connection.scalar(select(func.count()).select_from(eligible)) or 0
    if not matched_movie_count:
        return AgentAnalyticsResponse(
            status="no_matches",
            matched_movie_count=0,
            group_by=request.group_by,
            metrics=[metric.model_dump(mode="json") for metric in request.metrics],
            total_groups=0,
            returned_count=0,
            has_more=False,
            filters=request.filters.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            results=[],
        )

    movie = Movie.__table__
    source = movie.join(eligible, movie.c.id == eligible.c.id)
    dimension_columns = []
    dimension_predicates = []
    for index, dimension in enumerate(request.group_by):
        source, columns, extra = _add_dimension(source, dimension, index)
        dimension_columns.extend(columns)
        dimension_predicates.extend(extra)

    metric_fields = sorted({metric.field for metric in request.metrics if metric.field})
    flattened = (
        select(
            movie.c.id.label("movie_id"),
            *dimension_columns,
            *(NUMERIC_METRICS[field].label(field) for field in metric_fields),
        )
        .select_from(source)
        .where(*dimension_predicates)
        .distinct()
        .subquery("analytics_rows")
    )

    group_columns = [flattened.c[column.name] for column in dimension_columns]
    metric_expressions = []
    for metric in request.metrics:
        if metric.function == "count":
            expression = func.count(distinct(flattened.c.movie_id))
        else:
            expression = getattr(func, metric.function)(flattened.c[metric.field])
        metric_expressions.append(expression.label(metric.alias))

    statement = select(*group_columns, *metric_expressions).select_from(flattened)
    if group_columns:
        statement = statement.group_by(*group_columns)

    total_groups = connection.scalar(select(func.count()).select_from(statement.subquery())) or 0

    sortable = {column.name: column for column in group_columns}
    sortable.update({column.name: column for column in metric_expressions})
    order = []
    for spec in request.sort:
        column = sortable.get(spec.field)
        if column is None:
            raise ValueError(
                f"Unknown analytics sort field {spec.field!r}. Use a group-by output field or metric alias."
            )
        order.append((column.asc() if spec.direction == "asc" else column.desc()).nulls_last())
    if not order:
        order.append(metric_expressions[0].desc().nulls_last())
        order.extend(column.asc().nulls_last() for column in group_columns)

    rows = [
        dict(row)
        for row in connection.execute(
            statement.order_by(*order).offset(request.offset).limit(request.limit)
        ).mappings()
    ]
    return AgentAnalyticsResponse(
        status="ok",
        matched_movie_count=int(matched_movie_count),
        group_by=request.group_by,
        metrics=[metric.model_dump(mode="json") for metric in request.metrics],
        total_groups=int(total_groups),
        returned_count=len(rows),
        has_more=request.offset + len(rows) < int(total_groups),
        filters=request.filters.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
        results=rows,
    )
