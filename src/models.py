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

# Dimension table: static station information
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

# Fact table: time series of station status snapshots
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


def create_tables() -> None:
    """
    Create all tables in the database if they do not exist.
    """
    metadata.create_all(engine)
