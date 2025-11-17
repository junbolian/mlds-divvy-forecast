import argparse
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from dateutil import parser as date_parser
from sqlalchemy import insert
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .config import DIVVY_API_URL, EMPTY_THRESHOLD, FULL_THRESHOLD
from .db import engine
from .models import dim_station, fact_station_status, create_tables


def fetch_divvy_snapshot() -> Dict[str, Any]:
    """
    Call the Citybikes Divvy endpoint and return the parsed JSON.

    We use the Citybikes v2 'network' endpoint for Divvy:
    https://api.citybik.es/v2/networks/divvy
    """
    response = requests.get(DIVVY_API_URL, timeout=10)
    response.raise_for_status()
    data = response.json()
    return data


def parse_timestamp(ts_str: str) -> datetime:
    """
    Parse timestamp string from API into a timezone-aware datetime.

    Citybikes sometimes returns timestamps with both an offset and a trailing 'Z',
    e.g. '2025-04-17T16:41:02.505032+00:00Z', which breaks dateutil.isoparse.
    We sanitize such strings before parsing and always return a UTC-aware datetime.
    """
    if not ts_str:
        return datetime.now(timezone.utc)

    s = ts_str.strip()

    # If string has both an offset and a trailing 'Z', drop the 'Z'
    if s.endswith("Z") and ("+" in s[:-1] or "-" in s[:-1]):
        s = s[:-1]

    try:
        dt = date_parser.isoparse(s)
    except ValueError:
        # Last-resort: remove any trailing 'Z' and try again, or fall back to "now"
        s2 = s.rstrip("Z")
        try:
            dt = date_parser.isoparse(s2)
        except Exception:
            dt = datetime.now(timezone.utc)

    # Ensure timezone-aware in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)

    return dt



def classify_status(
    free_bikes: int,
    empty_slots: int,
    capacity: int,
    renting: int,
    returning: int,
) -> str:
    """
    Classify station status into:
    'offline', 'empty', 'normal', 'full', or 'unknown'.

    Business logic:
    - If renting == 0 and returning == 0 -> offline
    - Else use occupancy ratio and the 20% / 80% thresholds.
    """
    # Offline: not renting and not returning
    if renting == 0 and returning == 0:
        return "offline"

    if capacity is None or capacity <= 0 or free_bikes is None:
        return "unknown"

    ratio = free_bikes / capacity

    if ratio <= EMPTY_THRESHOLD:
        return "empty"
    if ratio >= FULL_THRESHOLD:
        return "full"
    return "normal"


def transform_stations(raw: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Transform raw API JSON into two lists:
    - stations_dim_rows: for dim_station (upsert)
    - status_fact_rows: for fact_station_status (insert)
    """
    network = raw["network"]
    city = network["location"].get("city", "Chicago, IL")

    stations_dim_rows: List[Dict[str, Any]] = []
    status_fact_rows: List[Dict[str, Any]] = []

    for station in network["stations"]:
        station_id = station["id"]
        name = station.get("name") or "Unknown"
        latitude = station.get("latitude")
        longitude = station.get("longitude")
        ts_str = station.get("timestamp")
        free_bikes = station.get("free_bikes")
        empty_slots = station.get("empty_slots")
        extra = station.get("extra", {}) or {}

        # 1) timestamp
        if ts_str:
            ts = parse_timestamp(ts_str)
        else:
            ts = datetime.utcnow()

        # 2) capacity: prefer extra["slots"]; fallback to free + empty
        slots = extra.get("slots")
        # 2) capacity: always free_bikes + empty_slots
        fb = free_bikes if isinstance(free_bikes, int) and free_bikes >= 0 else 0
        es = empty_slots if isinstance(empty_slots, int) and empty_slots >= 0 else 0
        
        capacity = fb + es if (free_bikes is not None and empty_slots is not None) else None


        # renting / returning flags (default to 1 if missing)
        renting = int(extra.get("renting", 1) or 1)
        returning = int(extra.get("returning", 1) or 1)

        # 3) status label
        status_label = classify_status(
            free_bikes=free_bikes if free_bikes is not None else 0,
            empty_slots=empty_slots if empty_slots is not None else 0,
            capacity=capacity if capacity is not None else 0,
            renting=renting,
            returning=returning,
        )

        # 4) dim_station row
        stations_dim_rows.append(
            {
                "station_id": station_id,
                "name": name,
                "latitude": latitude,
                "longitude": longitude,
                "city": city,
                "is_active": True,
            }
        )

        # 5) fact_station_status row
        
        status_fact_rows.append(
            {
                "station_id": station_id,
                "timestamp_utc": ts,
                "free_bikes": free_bikes,
                "empty_slots": empty_slots,
                "capacity": capacity,  
                "occupancy_ratio": (
                    float(free_bikes) / capacity if capacity not in (None, 0) else None
                ),
                "status_label": status_label,
                # Keep all extra info for future use
                "raw_extra": extra,
            }
        )


    return {
        "stations_dim_rows": stations_dim_rows,
        "status_fact_rows": status_fact_rows,
    }


def load_into_db(
    stations_dim_rows: List[Dict[str, Any]],
    status_fact_rows: List[Dict[str, Any]],
) -> None:
    """
    Upsert dim_station and insert fact_station_status rows.
    """
    with engine.begin() as conn:
        # Upsert dim_station (PostgreSQL ON CONFLICT)
        if stations_dim_rows:
            stmt = pg_insert(dim_station).values(stations_dim_rows)
            upsert_stmt = stmt.on_conflict_do_update(
                index_elements=[dim_station.c.station_id],
                set_={
                    "name": stmt.excluded.name,
                    "latitude": stmt.excluded.latitude,
                    "longitude": stmt.excluded.longitude,
                    "city": stmt.excluded.city,
                    "is_active": True,
                },
            )
            conn.execute(upsert_stmt)

        # Insert fact_station_status
        if status_fact_rows:
            conn.execute(insert(fact_station_status), status_fact_rows)


def run_single_snapshot() -> None:
    """
    Run one ETL snapshot: fetch -> transform -> load.
    """
    print("Creating tables if needed...")
    create_tables()

    print("Fetching Divvy snapshot from API...")
    raw = fetch_divvy_snapshot()

    print("Transforming data...")
    transformed = transform_stations(raw)

    print("Loading into PostgreSQL...")
    load_into_db(
        stations_dim_rows=transformed["stations_dim_rows"],
        status_fact_rows=transformed["status_fact_rows"],
    )

    print("ETL snapshot complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Divvy ETL: fetch live data from Citybikes API into PostgreSQL."
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=1,
        help="Number of snapshots to pull. Use >1 to collect time series.",
    )
    parser.add_argument(
        "--sleep",
        type=int,
        default=300,
        help="Seconds to sleep between snapshots when snapshots > 1.",
    )
    args = parser.parse_args()

    for i in range(args.snapshots):
        print(f"Running snapshot {i + 1}/{args.snapshots}...")
        run_single_snapshot()

        if i < args.snapshots - 1:
            print(f"Sleeping {args.sleep} seconds before next snapshot...")
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
