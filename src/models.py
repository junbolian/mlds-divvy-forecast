from sqlalchemy import (
    MetaData,
    Table,
    Column,
    String,
    Integer,
    Float,
    DateTime,
    Numeric,
    Boolean,
    ForeignKey,
    BigInteger,
    JSON,
)
from .db import engine

metadata = MetaData()

# ----------------------------------------------------------------------
# Dimension Table: dim_station
# ----------------------------------------------------------------------
# Contains station-level attributes that change infrequently.
# This stores one row per station, and is updated via UPSERTs.
#   - station_id: unique identifier from the Citybikes API
#   - is_active: marks whether the station should be considered live
#
# This table is joined with fact_station_status to provide context
# for time-series measurements.
dim_station = Table(
    "dim_station",
    metadata,
    Column("station_id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("city", String, nullable=True),
    Column("is_active", Boolean, nullable=False, server_default="true"),
)

# ----------------------------------------------------------------------
# Fact Table: fact_station_status
# ----------------------------------------------------------------------
# Append-only time-series table storing each snapshot pulled from
# the Citybikes live API.
#
# Key columns:
#   - id: surrogate primary key for efficient indexing
#   - station_id: FK to dim_station
#   - timestamp_utc: exact snapshot timestamp (stored in UTC)
#   - free_bikes / empty_slots / capacity: raw station metrics
#   - occupancy_ratio: percentage free bikes (0–1), precision = 5,2
#   - status_label: derived classification (empty/normal/full/offline)
#   - raw_extra: all “extra” fields from API preserved for debugging/future use
fact_station_status = Table(
    "fact_station_status",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("station_id", String, ForeignKey("dim_station.station_id"), nullable=False),
    Column("timestamp_utc", DateTime(timezone=True), nullable=False),
    Column("free_bikes", Integer, nullable=True),
    Column("empty_slots", Integer, nullable=True),
    Column("capacity", Integer, nullable=True),
    Column("occupancy_ratio", Numeric(5, 2), nullable=True),
    Column("status_label", String(16), nullable=True),
    Column("raw_extra", JSON, nullable=True),
)

# ----------------------------------------------------------------------
# Utility: Create all tables
# ----------------------------------------------------------------------
def create_tables() -> None:
    """
    Create all tables in the database if they do not exist.
    """
    metadata.create_all(engine)
