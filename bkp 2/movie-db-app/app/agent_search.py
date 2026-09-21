from __future__ import annotations

from sqlalchemy import Date, Integer, and_, case, cast, exists, func, literal, or_, select
from sqlalchemy.orm import Session, selectinload

from preprocessing import normalize_text
from .agent_contracts import (
    AgentMovieSearchHit,
    AgentMovieSearchRequest,
    AgentMovieSearchResponse,
    ReferenceMatch,
    ReferenceResolveRequest,
    ReferenceResolutionResponse,
    StringSetFilter,
    VectorIndexMovie,
    VectorIndexPage,
)
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


NAMED_RELATIONS = {
    "genres": (Genre.__table__, movie_genres, "genre_id", "id", None),
    "keywords": (Keyword.__table__, movie_keywords, "keyword_id", "id", None),
    "production_companies": (
        Company.__table__,
        movie_production_companies,
        "company_id",
        "id",
        None,
    ),
    "production_countries": (
        Country.__table__,
        movie_production_countries,
        "country_iso_3166_1",
        "iso_3166_1",
        "iso_3166_1",
    ),
    "spoken_languages": (
        Language.__table__,
        movie_spoken_languages,
        "language_iso_639_1",
        "iso_639_1",
        "iso_639_1",
    ),
}

REFERENCE_TABLES = {
    "genre": (Genre.__table__, "id", None),
    "keyword": (Keyword.__table__, "id", None),
    "production_company": (Company.__table__, "id", None),
    "production_country": (Country.__table__, None, "iso_3166_1"),
    "spoken_language": (Language.__table__, None, "iso_639_1"),
    "person": (Person.__table__, "id", None),
    "cast": (Person.__table__, "id", None),
    "crew": (Person.__table__, "id", None),
}


def _similarity(column, normalized: str):
    return func.greatest(
        func.similarity(column, normalized),
        func.strict_word_similarity(normalized, column),
    )


def _fuzzy_predicate(column, normalized: str, threshold: float):
    parts = [column == normalized, column.contains(normalized)]
    if len(normalized) >= 3:
        parts.append(_similarity(column, normalized) >= threshold)
    return or_(*parts)


def _text_group(column, values: list[str], threshold: float):
    if not values:
        return None
    return or_(*[_fuzzy_predicate(column, normalize_text(value), threshold) for value in values])


def _combine_relation_values(value_filter: StringSetFilter, predicates: list):
    return and_(*predicates) if value_filter.match == "all" else or_(*predicates)


def _named_relation_condition(value_filter: StringSetFilter, relation_name: str, threshold: float):
    entity, link, relation_fk, entity_pk, code_column = NAMED_RELATIONS[relation_name]
    join = link.join(entity, entity.c[entity_pk] == link.c[relation_fk])
    predicates = []
    for value in value_filter.values:
        normalized = normalize_text(value)
        name_match = _fuzzy_predicate(entity.c.name_search, normalized, threshold)
        if code_column:
            name_match = or_(name_match, func.lower(entity.c[code_column]) == normalized)
        predicates.append(
            exists(
                select(1)
                .select_from(join)
                .where(link.c.movie_id == Movie.id, name_match)
            )
        )
    return _combine_relation_values(value_filter, predicates)


def _people_condition(value_filter: StringSetFilter, *, cast_credit: bool, threshold: float):
    credit = CastCredit.__table__ if cast_credit else CrewCredit.__table__
    join = credit.join(Person.__table__, Person.id == credit.c.person_id)
    predicates = []
    for value in value_filter.values:
        normalized = normalize_text(value)
        predicates.append(
            exists(
                select(1)
                .select_from(join)
                .where(
                    credit.c.movie_id == Movie.id,
                    _fuzzy_predicate(Person.name_search, normalized, threshold),
                )
            )
        )
    return _combine_relation_values(value_filter, predicates)


def _credit_text_condition(value_filter: StringSetFilter, *, table, column):
    predicates = []
    for value in value_filter.values:
        # Credit text does not have a materialized normalized helper column, so use
        # case-insensitive containment. Person names are handled separately with pg_trgm.
        query = " ".join(value.split())
        predicates.append(
            exists(
                select(1).select_from(table).where(
                    table.c.movie_id == Movie.id,
                    func.coalesce(column, "").ilike(f"%{query}%"),
                )
            )
        )
    return _combine_relation_values(value_filter, predicates)


def _range_conditions(column, value):
    if value is None:
        return []
    conditions = []
    if value.eq is not None:
        conditions.append(column == value.eq)
    if value.min is not None:
        conditions.append(column >= value.min)
    if value.max is not None:
        conditions.append(column <= value.max)
    return conditions


def _scalar_string_condition(column, value_filter: StringSetFilter | None):
    if value_filter is None:
        return None
    values = [normalize_text(value) for value in value_filter.values]
    lowered = func.lower(func.coalesce(column, ""))
    # A scalar column cannot simultaneously equal several distinct values. Treat
    # multiple supplied values as allowed alternatives regardless of match mode.
    return or_(*[lowered == value for value in values])


def _movie_options():
    return (
        selectinload(Movie.genres),
        selectinload(Movie.keywords),
        selectinload(Movie.production_companies),
        selectinload(Movie.production_countries),
        selectinload(Movie.spoken_languages),
        selectinload(Movie.cast).selectinload(CastCredit.person),
        selectinload(Movie.crew).selectinload(CrewCredit.person),
    )


def _names(items):
    return sorted({item.name for item in items if item.name})


def _movie_to_hit(movie: Movie, fuzzy_score: float | None) -> AgentMovieSearchHit:
    return AgentMovieSearchHit(
        movie_id=movie.id,
        title=movie.title,
        original_title=movie.original_title,
        tagline=movie.tagline,
        release_date=movie.release_date.date() if movie.release_date else None,
        status=movie.status,
        original_language=movie.original_language,
        runtime=movie.runtime,
        budget=movie.budget,
        revenue=movie.revenue,
        popularity=movie.popularity,
        vote_average=movie.vote_average,
        vote_count=movie.vote_count,
        fuzzy_score=fuzzy_score,
        genres=_names(movie.genres),
        keywords=_names(movie.keywords),
        production_companies=_names(movie.production_companies),
        production_countries=_names(movie.production_countries),
        spoken_languages=_names(movie.spoken_languages),
        cast=sorted({credit.person.name for credit in movie.cast if credit.person and credit.person.name}),
        crew=sorted({credit.person.name for credit in movie.crew if credit.person and credit.person.name}),
    )


def agent_movie_search(connection, criteria: AgentMovieSearchRequest, settings) -> AgentMovieSearchResponse:
    threshold = criteria.fuzzy_threshold or settings.fuzzy_threshold
    predicates = []

    if criteria.movie_ids:
        predicates.append(Movie.id.in_(criteria.movie_ids))

    for column, values in (
        (Movie.title_search, criteria.titles),
        (Movie.original_title_search, criteria.original_titles),
        (Movie.tagline_search, criteria.taglines),
    ):
        condition = _text_group(column, values, threshold)
        if condition is not None:
            predicates.append(condition)

    for attribute, relation_name in (
        (criteria.genres, "genres"),
        (criteria.keywords, "keywords"),
        (criteria.production_companies, "production_companies"),
        (criteria.production_countries, "production_countries"),
        (criteria.spoken_languages, "spoken_languages"),
    ):
        if attribute is not None:
            predicates.append(_named_relation_condition(attribute, relation_name, threshold))

    if criteria.cast is not None:
        predicates.append(_people_condition(criteria.cast, cast_credit=True, threshold=threshold))
    if criteria.crew is not None:
        predicates.append(_people_condition(criteria.crew, cast_credit=False, threshold=threshold))

    if criteria.cast_characters is not None:
        predicates.append(
            _credit_text_condition(
                criteria.cast_characters,
                table=CastCredit.__table__,
                column=CastCredit.__table__.c.character,
            )
        )
    if criteria.crew_departments is not None:
        predicates.append(
            _credit_text_condition(
                criteria.crew_departments,
                table=CrewCredit.__table__,
                column=CrewCredit.__table__.c.department,
            )
        )
    if criteria.crew_jobs is not None:
        predicates.append(
            _credit_text_condition(
                criteria.crew_jobs,
                table=CrewCredit.__table__,
                column=CrewCredit.__table__.c.job,
            )
        )

    predicates.extend(_range_conditions(Movie.budget, criteria.budget))
    predicates.extend(_range_conditions(Movie.popularity, criteria.popularity))
    predicates.extend(_range_conditions(Movie.revenue, criteria.revenue))
    predicates.extend(_range_conditions(Movie.runtime, criteria.runtime))
    predicates.extend(_range_conditions(Movie.vote_average, criteria.vote_average))
    predicates.extend(_range_conditions(Movie.vote_count, criteria.vote_count))

    original_language = _scalar_string_condition(Movie.original_language, criteria.original_languages)
    if original_language is not None:
        predicates.append(original_language)
    status = _scalar_string_condition(Movie.status, criteria.statuses)
    if status is not None:
        predicates.append(status)

    if criteria.release_date is not None:
        date_column = cast(Movie.release_date, Date)
        if criteria.release_date.eq is not None:
            predicates.append(date_column == criteria.release_date.eq)
        if criteria.release_date.from_date is not None:
            predicates.append(date_column >= criteria.release_date.from_date)
        if criteria.release_date.to_date is not None:
            predicates.append(date_column <= criteria.release_date.to_date)

    if criteria.release_year is not None:
        year_column = cast(func.extract("year", Movie.release_date), Integer)
        predicates.extend(_range_conditions(year_column, criteria.release_year))

    if criteria.homepage_contains:
        predicates.append(Movie.homepage.ilike(f"%{criteria.homepage_contains.strip()}%"))
    if criteria.overview_contains:
        predicates.append(Movie.overview.ilike(f"%{criteria.overview_contains.strip()}%"))

    score_parts = []
    for column, values in (
        (Movie.title_search, criteria.titles),
        (Movie.original_title_search, criteria.original_titles),
        (Movie.tagline_search, criteria.taglines),
    ):
        for value in values:
            score_parts.append(_similarity(column, normalize_text(value)))
    score = func.greatest(*score_parts) if score_parts else literal(None)

    eligible = select(Movie.id, score.label("fuzzy_score")).where(*predicates)
    total = connection.scalar(select(func.count()).select_from(eligible.subquery())) or 0
    if not total:
        return AgentMovieSearchResponse(
            status="no_matches",
            count=0,
            returned_count=0,
            has_more=False,
            criteria=criteria.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
            semantic_fallback_recommended=True,
            results=[],
        )

    order = []
    if criteria.sort:
        sort_fields = {
            "title": Movie.title_search,
            "release_date": Movie.release_date,
            "release_year": cast(func.extract("year", Movie.release_date), Integer),
            "budget": Movie.budget,
            "revenue": Movie.revenue,
            "runtime": Movie.runtime,
            "vote_average": Movie.vote_average,
            "vote_count": Movie.vote_count,
            "popularity": Movie.popularity,
            "fuzzy_score": score if score_parts else None,
        }
        for spec in criteria.sort:
            column = sort_fields[spec.field]
            if column is None:
                raise ValueError("fuzzy_score sorting requires title/original-title/tagline criteria")
            order.append((column.asc() if spec.direction == "asc" else column.desc()).nulls_last())
    else:
        if score_parts:
            order.append(score.desc().nulls_last())
        order.extend(
            [
                Movie.vote_average.desc().nulls_last(),
                Movie.popularity.desc().nulls_last(),
            ]
        )
    order.append(Movie.id.asc())
    rows = connection.execute(
        select(Movie.id, score.label("fuzzy_score"))
        .where(*predicates)
        .order_by(*order)
        .limit(criteria.limit)
    ).mappings().all()
    ordered_ids = [row["id"] for row in rows]
    scores = {row["id"]: row["fuzzy_score"] for row in rows}

    with Session(bind=connection) as session:
        movies = session.scalars(
            select(Movie).where(Movie.id.in_(ordered_ids)).options(*_movie_options())
        ).all()
        by_id = {movie.id: movie for movie in movies}
        results = [
            _movie_to_hit(by_id[movie_id], float(scores[movie_id]) if scores[movie_id] is not None else None)
            for movie_id in ordered_ids
            if movie_id in by_id
        ]

    return AgentMovieSearchResponse(
        status="ok",
        count=int(total),
        returned_count=len(results),
        has_more=int(total) > len(results),
        criteria=criteria.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
        semantic_fallback_recommended=False,
        results=results,
    )


def _reference_scope(reference_type: str, table):
    if reference_type == "cast":
        return exists(select(1).select_from(CastCredit.__table__).where(CastCredit.person_id == table.c.id))
    if reference_type == "crew":
        return exists(select(1).select_from(CrewCredit.__table__).where(CrewCredit.person_id == table.c.id))
    return None


def _status_reference_rows(connection, query: str | None, limit: int, threshold: float):
    column = Movie.status
    raw = column.label("value")
    statement = select(raw).where(column.is_not(None)).distinct()
    score = literal(1.0)
    if query:
        normalized = normalize_text(query)
        lowered = func.lower(column)
        score = case(
            (lowered == normalized, 1.0),
            else_=func.similarity(lowered, normalized),
        )
        filters = [lowered == normalized, lowered.contains(normalized)]
        if len(normalized) >= 3:
            filters.append(func.similarity(lowered, normalized) >= threshold)
        statement = statement.where(or_(*filters))
    rows = connection.execute(
        statement.add_columns(score.label("score")).order_by(score.desc(), raw.asc()).limit(limit)
    ).mappings().all()
    return [ReferenceMatch(value=row["value"], score=float(row["score"] or 0.0)) for row in rows]


def _original_language_reference_rows(connection, query: str | None, limit: int, threshold: float):
    used = (
        select(Movie.original_language.label("code"))
        .where(Movie.original_language.is_not(None))
        .distinct()
        .subquery()
    )
    language = Language.__table__
    join = used.outerjoin(language, func.lower(language.c.iso_639_1) == func.lower(used.c.code))
    code = used.c.code
    name = language.c.name
    score = literal(1.0)
    predicates = []
    if query:
        normalized = normalize_text(query)
        name_score = _similarity(language.c.name_search, normalized)
        score = case(
            (func.lower(code) == normalized, 1.0),
            (language.c.name_search == normalized, 1.0),
            else_=name_score,
        )
        matches = [
            func.lower(code) == normalized,
            language.c.name_search == normalized,
            language.c.name_search.contains(normalized),
        ]
        if len(normalized) >= 3:
            matches.append(name_score >= threshold)
        predicates.append(or_(*matches))
    rows = connection.execute(
        select(code.label("value"), code.label("code"), name.label("name"), score.label("score"))
        .select_from(join)
        .where(*predicates)
        .order_by(score.desc(), code.asc())
        .limit(limit)
    ).mappings().all()
    return [
        ReferenceMatch(
            value=row["value"],
            code=row["code"],
            score=float(row["score"] or 0.0),
            extra={"name": row["name"]} if row["name"] else {},
        )
        for row in rows
    ]


def resolve_reference(connection, request: ReferenceResolveRequest, settings) -> ReferenceResolutionResponse:
    reference_type = request.reference_type
    if reference_type == "status":
        results = _status_reference_rows(
            connection, request.query, request.limit, settings.fuzzy_threshold
        )
        return ReferenceResolutionResponse(
            reference_type=reference_type, query=request.query, count=len(results), results=results
        )
    if reference_type == "original_language":
        results = _original_language_reference_rows(
            connection, request.query, request.limit, settings.fuzzy_threshold
        )
        return ReferenceResolutionResponse(
            reference_type=reference_type, query=request.query, count=len(results), results=results
        )

    table, id_column, code_column = REFERENCE_TABLES[reference_type]
    raw_name = table.c.name
    name_search = table.c.name_search
    entity_id = table.c[id_column] if id_column else literal(None)
    entity_code = table.c[code_column] if code_column else literal(None)
    score = literal(1.0)
    predicates = [raw_name.is_not(None)]
    scope = _reference_scope(reference_type, table)
    if scope is not None:
        predicates.append(scope)

    if request.query:
        normalized = normalize_text(request.query)
        similarity = _similarity(name_search, normalized)
        score = case((name_search == normalized, 1.0), else_=similarity)
        matches = [name_search == normalized, name_search.contains(normalized)]
        if code_column:
            matches.append(func.lower(table.c[code_column]) == normalized)
        if len(normalized) >= 3:
            matches.append(similarity >= settings.fuzzy_threshold)
        predicates.append(or_(*matches))

    rows = connection.execute(
        select(
            raw_name.label("value"),
            entity_id.label("entity_id"),
            entity_code.label("code"),
            score.label("score"),
        )
        .where(*predicates)
        .distinct()
        .order_by(score.desc(), raw_name.asc())
        .limit(request.limit)
    ).mappings().all()
    results = [
        ReferenceMatch(
            value=row["value"],
            id=int(row["entity_id"]) if row["entity_id"] is not None else None,
            code=row["code"],
            score=float(row["score"] or 0.0),
        )
        for row in rows
    ]
    return ReferenceResolutionResponse(
        reference_type=reference_type,
        query=request.query,
        count=len(results),
        results=results,
    )


def vector_index_page(connection, *, offset: int, limit: int) -> VectorIndexPage:
    total = connection.scalar(select(func.count(Movie.id))) or 0
    with Session(bind=connection) as session:
        movies = session.scalars(
            select(Movie)
            .order_by(Movie.id)
            .offset(offset)
            .limit(limit)
            .options(selectinload(Movie.genres), selectinload(Movie.keywords))
        ).all()
        items = [
            VectorIndexMovie(
                movie_id=movie.id,
                title=movie.title,
                original_title=movie.original_title or "",
                tagline=movie.tagline or "",
                overview=movie.overview or "",
                genres=_names(movie.genres),
                keywords=_names(movie.keywords),
                release_year=movie.release_date.year if movie.release_date else None,
                vote_average=movie.vote_average,
                popularity=movie.popularity,
            )
            for movie in movies
        ]
    next_offset = offset + len(items)
    if next_offset >= total:
        next_offset = None
    return VectorIndexPage(total=int(total), offset=offset, limit=limit, next_offset=next_offset, items=items)
