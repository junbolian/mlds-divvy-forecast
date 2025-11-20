from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from .db import engine
from .config import EMPTY_THRESHOLD, FULL_THRESHOLD


def summarize_current_status() -> None:
    """
    Print summary of the latest status per station:
    counts of empty / normal / full / offline / unknown.
    """
    # Query picks the latest record for each station using DISTINCT ON
    query = text(
        """
        SELECT DISTINCT ON (station_id)
               station_id,
               status_label,
               occupancy_ratio,
               timestamp_utc
        FROM fact_station_status
        ORDER BY station_id, timestamp_utc DESC;
        """
    )

    # Counters for each possible station status
    counts = {
        "empty": 0,
        "normal": 0,
        "full": 0,
        "offline": 0,
        "unknown": 0,
    }

    # Execute query and fetch latest status rows
    with engine.connect() as conn:
        result = conn.execute(query)
        rows = result.fetchall()

    print(f"Latest status for {len(rows)} stations:")

    # Tally the counts by status label
    for row in rows:
        # If status_label is None or unexpected, count it as "unknown"
        label = row.status_label or "unknown"
        if label not in counts:
            label = "unknown"
        counts[label] += 1

    # Pretty-print the aggregated counts
    print("Status counts:")
    for label, cnt in counts.items():
        print(f"  {label:7s}: {cnt}")


def classify_ratio(ratio: float | None) -> str:
    """
    Classify occupancy ratio into empty / normal / full / unknown.
    """
    # Missing or NULL ratio → unknown
    if ratio is None:
        return "unknown"

    # Compare ratio against configurable thresholds
    if ratio <= EMPTY_THRESHOLD:
        return "empty"
    if ratio >= FULL_THRESHOLD:
        return "full"
    return "normal"


def predict_next_by_last_hour(window_minutes: int = 60) -> None:
    """
    Naive 'expected demand' model:
    - For each station, compute the average occupancy_ratio over the last window.
    - Use (1 - avg_occ) as an 'expected demand index'.
    - Also classify the predicted status based on avg_occ.
    """
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(minutes=window_minutes)

    # Aggregate average occupancy per station over the time window
    query = text(
        """
        SELECT station_id,
               AVG(occupancy_ratio) AS avg_occ
        FROM fact_station_status
        WHERE timestamp_utc >= :window_start
        GROUP BY station_id;
        """
    )

    # Execute time-windowed average occupancy query
    with engine.connect() as conn:
        result = conn.execute(query, {"window_start": window_start})
        rows = result.fetchall()

    print(f"\nNaive prediction based on last {window_minutes} minutes:")

    # Compute predicted demand & classify status for each station
    for row in rows:
        station_id = row.station_id
        avg_occ = float(row.avg_occ) if row.avg_occ is not None else None

        if avg_occ is not None:
            # Demand index is inverse of occupancy (bounded 0–1)
            demand_index = max(0.0, min(1.0, 1.0 - avg_occ))
            label = classify_ratio(avg_occ)
        else:
            # Not enough data → unknown
            demand_index = None
            label = "unknown"

        print(
            f"Station {station_id}: "
            f"avg_occ_last_hour={avg_occ}, "
            f"expected_demand_index={demand_index}, "
            f"predicted_status={label}"
        )


def main() -> None:
    # Run both summary and prediction routines
    summarize_current_status()
    predict_next_by_last_hour(window_minutes=60)


if __name__ == "__main__":
    main()
