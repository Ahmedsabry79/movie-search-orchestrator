from datetime import datetime, timezone

from sqlalchemy import inspect
from sqlalchemy.orm import configure_mappers

from .agent_contracts import AgentAnalyticsRequest, AgentMovieSearchRequest, ReferenceResolveRequest
from .contracts import SearchRequest
from .models import Base, SCHEMA
from .search import ID_RELATIONS, RELATIONS, SCALAR_FIELDS


STRUCTURED_MOVIE_COLUMNS = {
    "fuzzy": ["title", "original_title", "tagline"],
    "exact_or_range": [
        "id",
        "budget",
        "original_language",
        "popularity",
        "release_date",
        "release_year",
        "revenue",
        "runtime",
        "status",
        "vote_average",
        "vote_count",
    ],
    "literal_contains": ["homepage"],
    "semantic_only": ["overview"],
}

STRUCTURED_RELATIONS = {
    "fuzzy": [
        "genre",
        "keyword",
        "production_company",
        "production_country",
        "spoken_language",
        "cast",
        "crew",
        "cast_character",
    ],
    "literal_contains": ["crew_department", "crew_job"],
}

ANALYTICS_DIMENSIONS = [
    "genre",
    "keyword",
    "production_company",
    "production_country",
    "spoken_language",
    "cast_member",
    "crew_member",
    "director",
    "cast_character",
    "crew_department",
    "crew_job",
    "original_language",
    "status",
    "release_year",
]


def schema_details(connection, selected=None):
    inspector = inspect(connection)
    allowed = {table.name for table in Base.metadata.tables.values() if table.name != "dataset_state"}
    requested = set(selected or allowed)
    if requested - allowed:
        raise ValueError(f"Unknown tables: {sorted(requested - allowed)}")
    configure_mappers()
    models = {mapper.local_table.name: mapper for mapper in Base.registry.mappers}
    result = []
    for name in sorted(requested):
        columns = inspector.get_columns(name, schema=SCHEMA)
        relationships = []
        if name in models:
            for relation in models[name].relationships:
                relationships.append({
                    "name": relation.key,
                    "target": relation.mapper.local_table.name,
                    "direction": relation.direction.name,
                    "association_table": relation.secondary.name if relation.secondary is not None else None,
                    "collection": relation.uselist,
                })
        result.append({
            "name": name,
            "columns": [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                    "default": column.get("default"),
                }
                for column in columns
            ],
            "primary_key": inspector.get_pk_constraint(name, schema=SCHEMA),
            "foreign_keys": inspector.get_foreign_keys(name, schema=SCHEMA),
            "unique_constraints": inspector.get_unique_constraints(name, schema=SCHEMA),
            "check_constraints": inspector.get_check_constraints(name, schema=SCHEMA),
            "indexes": inspector.get_indexes(name, schema=SCHEMA),
            "orm_relationships": relationships,
        })
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc),
        "tables": result,
        "search_contract": SearchRequest.model_json_schema(),
        "agent_search_contract": AgentMovieSearchRequest.model_json_schema(),
        "analytics_contract": AgentAnalyticsRequest.model_json_schema(),
        "reference_resolve_contract": ReferenceResolveRequest.model_json_schema(),
        "search_fields": {
            "legacy_scalar_filters": sorted(SCALAR_FIELDS),
            "legacy_entity_names": sorted(RELATIONS),
            "legacy_entity_ids": sorted(ID_RELATIONS),
            "structured_movie_columns": STRUCTURED_MOVIE_COLUMNS,
            "structured_relations": STRUCTURED_RELATIONS,
            "analytics_dimensions": ANALYTICS_DIMENSIONS,
            "analytics_metric_fields": [
                "budget", "revenue", "runtime", "vote_average", "vote_count", "popularity"
            ],
            "analytics_functions": ["count", "sum", "avg", "min", "max"],
        },
        "semantics": {
            "filters": "Structured filters across fields are ANDed. Multi-value relation filters explicitly choose match=all or match=any.",
            "unknowns": "Nonpositive budget, revenue, vote_count, vote_average and runtime are NULL; ratings and runtime retain decimals.",
            "dates": "Invalid or missing release dates are NULL. release_year is derived from release_date.",
            "counts": "Movie counts and analytics count distinct movie ids per group, preventing many-to-many joins from multiplying movies.",
            "fuzzy": "Fuzzy-capable fields use normalized exact/containment plus pg_trgm similarity. Agent search returns per-result fuzzy_matches with query, matched DB value, field and similarity (0..1). Similarity is not a probability.",
            "overview": "overview is intentionally not SQL-searchable by the agent. Any overview/plot/theme meaning must use semantic retrieval through movie-vector-app.",
            "analytics": "Analytics uses only allow-listed logical dimensions/metrics. movie-db-app owns all table joins; callers never provide raw SQL or join expressions.",
            "missing_fields": "Names of relation fields missing in source; an empty related list without that flag is known empty.",
        },
    }
