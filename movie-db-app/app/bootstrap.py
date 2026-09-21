import base64
import csv
import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from sqlalchemy import BigInteger, DateTime, Float, Integer, func, inspect, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import OperationalError

from preprocessing import CREDITS_FILE, MOVIES_FILE, NULL_VALUE, preprocess
from .models import Base, DatasetState, SCHEMA
from .runtime_models import AGENT_SCHEMA, AgentBase

logger = logging.getLogger(__name__)
LOCK_ID = 74005301


def ensure_raw_files(settings):
    raw_dir = settings.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    names = (MOVIES_FILE, CREDITS_FILE)
    if all((raw_dir / name).is_file() and (raw_dir / name).stat().st_size > 0 for name in names):
        return raw_dir

    headers = {"User-Agent": "movie-db-app/1.0"}
    token = os.getenv("KAGGLE_API_TOKEN")
    username, key = os.getenv("KAGGLE_USERNAME"), os.getenv("KAGGLE_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username and key:
        encoded = base64.b64encode(f"{username}:{key}".encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
    # Credentials are only sent to Kaggle, never to a custom dataset mirror.
    from urllib.parse import urlparse
    if urlparse(settings.download_url).hostname not in {"kaggle.com", "www.kaggle.com"}:
        headers.pop("Authorization", None)

    logger.info("Downloading the TMDB source archive")
    for attempt in range(3):
        try:
            with tempfile.TemporaryDirectory(dir=raw_dir) as folder:
                archive_path = Path(folder) / "dataset.zip"
                request = urllib.request.Request(settings.download_url, headers=headers)
                with urllib.request.urlopen(request, timeout=60) as response, archive_path.open("wb") as output:
                    total = 0
                    while chunk := response.read(1024 * 1024):
                        total += len(chunk)
                        if total > 256 * 1024 * 1024:
                            raise ValueError("Dataset archive exceeds 256 MiB")
                        output.write(chunk)
                with zipfile.ZipFile(archive_path) as archive:
                    for name in names:
                        candidates = [item for item in archive.infolist() if Path(item.filename).name == name and not item.is_dir()]
                        if len(candidates) != 1 or candidates[0].file_size > 256 * 1024 * 1024:
                            raise ValueError(f"Archive must contain exactly one valid {name}")
                        # Write only the two expected basenames, not archive-supplied paths.
                        with archive.open(candidates[0]) as source, (Path(folder) / name).open("wb") as output:
                            shutil.copyfileobj(source, output)
                for name in names:
                    if not (raw_dir / name).exists() or (raw_dir / name).stat().st_size == 0:
                        (Path(folder) / name).replace(raw_dir / name)
            return raw_dir
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            if attempt == 2:
                raise RuntimeError(
                    "TMDB download failed. Set Kaggle credentials if required, or place both original "
                    "CSV files in data/raw and restart. No movie data was committed."
                ) from exc
            time.sleep(attempt + 1)
    raise RuntimeError("Dataset download failed")


def csv_rows(path, table):
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if set(reader.fieldnames or []) != set(table.columns.keys()):
            raise ValueError(f"Unexpected CSV columns for {table.name}")
        for row in reader:
            result = {}
            for column in table.columns:
                value = row[column.name]
                if value == NULL_VALUE:
                    result[column.name] = None
                elif isinstance(column.type, (Integer, BigInteger)):
                    result[column.name] = int(value)
                elif isinstance(column.type, Float):
                    result[column.name] = float(value)
                elif isinstance(column.type, DateTime):
                    result[column.name] = datetime.fromisoformat(value)
                elif isinstance(column.type, JSONB):
                    result[column.name] = json.loads(value)
                else:
                    result[column.name] = value
            yield result


def validate_existing_columns(connection):
    inspector = inspect(connection)
    for table in Base.metadata.sorted_tables:
        actual = {column["name"] for column in inspector.get_columns(table.name, schema=SCHEMA)}
        if actual != set(table.columns.keys()):
            raise RuntimeError(f"Existing {table.fullname} has a different schema. Apply a migration before starting this version.")


def bootstrap(engine, settings):
    # DB readiness is also checked here for runs outside Docker Compose.
    for attempt in range(30):
        try:
            with engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            break
        except OperationalError:
            if attempt == 29:
                raise
            time.sleep(1)

    # The advisory lock serializes startup across app instances. Data and seed marker
    # commit together; a failed download, preprocessing or insert leaves no partial seed.
    with engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": LOCK_ID})
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {AGENT_SCHEMA}"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
        Base.metadata.create_all(connection)
        AgentBase.metadata.create_all(connection)
        # Lightweight additive migration for the task's conversation-memory fields.
        # `create_all` does not alter an already-existing table, so keep this startup
        # migration idempotent for users upgrading an existing pg_data volume.
        connection.execute(text(
            f"ALTER TABLE {AGENT_SCHEMA}.conversations "
            "ADD COLUMN IF NOT EXISTS memory_summary JSONB"
        ))
        connection.execute(text(
            f"ALTER TABLE {AGENT_SCHEMA}.conversations "
            "ADD COLUMN IF NOT EXISTS memory_compacted_through_message_id VARCHAR(36)"
        ))
        connection.execute(text(
            f"ALTER TABLE {AGENT_SCHEMA}.conversations "
            "ADD COLUMN IF NOT EXISTS memory_updated_at TIMESTAMPTZ"
        ))
        validate_existing_columns(connection)
        state = connection.execute(select(DatasetState.__table__).where(DatasetState.id == 1)).mappings().first()
        data_tables = [table for table in Base.metadata.sorted_tables if table.name != "dataset_state"]
        counts = {table.name: connection.scalar(select(func.count()).select_from(table)) for table in data_tables}
        if state:
            if state["version"] != settings.dataset_version:
                raise RuntimeError("Dataset version changed. Apply an explicit data migration; startup will not overwrite existing data.")
            if counts != state["row_counts"]:
                raise RuntimeError("Seeded table counts changed. Restore or explicitly repair the dataset before restarting.")
            logger.info("Existing dataset is ready: %s movies", counts["movies"])
            return
        if any(counts.values()):
            raise RuntimeError("Tables contain data without a completed seed marker. Startup will not overwrite unmanaged data.")

        raw_dir = ensure_raw_files(settings)
        output_dir = settings.data_dir / "processed"
        preprocess(raw_dir, output_dir)
        for table in data_tables:
            batch = []
            for row in csv_rows(output_dir / f"{table.name}_table.csv", table):
                batch.append(row)
                if len(batch) == 1000:
                    connection.execute(table.insert(), batch)
                    batch.clear()
            if batch:
                connection.execute(table.insert(), batch)
        counts = {table.name: connection.scalar(select(func.count()).select_from(table)) for table in data_tables}
        hashes = {}
        for name in (MOVIES_FILE, CREDITS_FILE):
            with (raw_dir / name).open("rb") as source:
                hashes[name] = hashlib.file_digest(source, "sha256").hexdigest()
        connection.execute(DatasetState.__table__.insert().values(id=1, version=settings.dataset_version, row_counts=counts, source_hashes=hashes))
        logger.info("Database seeded: %s movies", counts["movies"])
