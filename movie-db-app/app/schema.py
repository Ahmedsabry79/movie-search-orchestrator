from datetime import datetime, timezone

from sqlalchemy import inspect
from sqlalchemy.orm import configure_mappers

from .agent_contracts import AgentMovieSearchRequest, ReferenceResolveRequest
from .contracts import SearchRequest
from .models import Base, SCHEMA
from .search import ID_RELATIONS, RELATIONS, SCALAR_FIELDS


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
                    "name": relation.key, "target": relation.mapper.local_table.name,
                    "direction": relation.direction.name,
                    "association_table": relation.secondary.name if relation.secondary is not None else None,
                    "collection": relation.uselist,
                })
        result.append({
            "name": name,
            "columns": [{"name": column["name"], "type": str(column["type"]), "nullable": column["nullable"], "default": column.get("default")} for column in columns],
            "primary_key": inspector.get_pk_constraint(name, schema=SCHEMA),
            "foreign_keys": inspector.get_foreign_keys(name, schema=SCHEMA),
            "unique_constraints": inspector.get_unique_constraints(name, schema=SCHEMA),
            "check_constraints": inspector.get_check_constraints(name, schema=SCHEMA),
            "indexes": inspector.get_indexes(name, schema=SCHEMA),
            "orm_relationships": relationships,
        })
    return {
        "schema": SCHEMA, "generated_at": datetime.now(timezone.utc), "tables": result,
        "search_contract": SearchRequest.model_json_schema(),
        "agent_search_contract": AgentMovieSearchRequest.model_json_schema(),
        "reference_resolve_contract": ReferenceResolveRequest.model_json_schema(),
        "search_fields": {"scalar_filters": sorted(SCALAR_FIELDS), "entity_names": sorted(RELATIONS),
                          "entity_ids": sorted(ID_RELATIONS), "automatic_text_matching": ["title", "original_title", "tagline"]},
        "semantics": {
            "filters": "All filters are ANDed; values within in are ORed. Name filters resolve entities before filtering.",
            "unknowns": "Nonpositive budget, revenue, vote_count, vote_average and runtime are NULL; ratings and runtime retain decimals.",
            "dates": "Invalid or missing release dates are NULL. release_year is derived from release_date.",
            "counts": "Movie counts and numeric aggregates use unique movie ids per group.",
            "fuzzy": "Normalized exact matches first; otherwise trigram/word matching. Similarity scores are not probabilities. Check requires_clarification.",
            "missing_fields": "Names of relation fields missing in source; an empty related list without that flag is known empty.",
        },
    }
