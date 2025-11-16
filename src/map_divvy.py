import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import folium
import pandas as pd
from sqlalchemy import text

from .db import engine

# Local timezone for display on the map
# We keep timestamps in UTC in the database but convert to this tz for users.
LOCAL_TZ_NAME = "America/Chicago"


def status_to_color(status: str) -> str:
    """
    Map our status_label to a marker color.
    """
    mapping: Dict[str, str] = {
        "empty": "red",
        "full": "green",
        "normal": "yellow",
        "offline": "gray",
        "unknown": "lightgray",
    }
    return mapping.get(status, "lightgray")


def build_latest_status_dataframe() -> pd.DataFrame:
    """
    Query PostgreSQL for the latest status row per station.
    Returns a DataFrame with one row per station.
    """
    query = text(
        """
        SELECT s.station_id,
               s.name,
               s.latitude,
               s.longitude,
               f.free_bikes,
               f.empty_slots,
               f.capacity,
               f.occupancy_ratio,
               f.status_label,
               f.timestamp_utc
        FROM dim_station s
        JOIN LATERAL (
            SELECT *
            FROM fact_station_status fs
            WHERE fs.station_id = s.station_id
            ORDER BY fs.timestamp_utc DESC
            LIMIT 1
        ) f ON TRUE;
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if not df.empty:
        # Convert UTC timestamp to local (Chicago) time for display
        # The column is timezone-aware (UTC), we convert to America/Chicago.
        df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ_NAME)

    return df


def build_last_hour_demand(window_minutes: int = 60) -> pd.DataFrame:
    """
    Compute last-hour average occupancy per station and
    convert it into an expected demand index.
    """
    now_utc = datetime.now(timezone.utc)
    window_start = now_utc - timedelta(minutes=window_minutes)

    query = text(
        """
        SELECT station_id,
               AVG(occupancy_ratio) AS avg_occ_last_hour
        FROM fact_station_status
        WHERE timestamp_utc >= :window_start
        GROUP BY station_id;
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"window_start": window_start})

    if df.empty:
        return df

    # expected demand index = 1 - avg occupancy, clipped to [0, 1]
    df["expected_demand_index"] = (
        1.0 - df["avg_occ_last_hour"].astype(float)
    ).clip(lower=0.0, upper=1.0)

    return df


def create_divvy_map(
    output_path: str = "outputs/divvy_map.html",
    default_zoom: int = 12,
) -> None:
    """
    Build an interactive HTML map of the latest Divvy station status.
    """
    df_latest = build_latest_status_dataframe()
    if df_latest.empty:
        print("No data found in the database. Run the ETL first.")
        return

    df_demand = build_last_hour_demand(window_minutes=60)

    # Merge demand info (if available)
    if not df_demand.empty:
        df = df_latest.merge(
            df_demand[["station_id", "avg_occ_last_hour", "expected_demand_index"]],
            on="station_id",
            how="left",
        )
    else:
        df = df_latest.copy()
        df["avg_occ_last_hour"] = None
        df["expected_demand_index"] = None

    # Center map on the mean latitude/longitude
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    folium_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=default_zoom,
        control_scale=True,
    )

    for _, row in df.iterrows():
        status = row["status_label"] or "unknown"
        color = status_to_color(status)

        occ_ratio = row["occupancy_ratio"]
        if occ_ratio is not None:
            occ_str = f"{float(occ_ratio) * 100:.1f}%"
        else:
            occ_str = "N/A"

        avg_occ = row.get("avg_occ_last_hour")
        if avg_occ is not None:
            avg_occ_str = f"{float(avg_occ) * 100:.1f}%"
        else:
            avg_occ_str = "N/A"

        demand_index = row.get("expected_demand_index")
        if demand_index is not None:
            demand_str = f"{float(demand_index):.2f}"
        else:
            demand_str = "N/A"

        # Local time (Chicago) for display
        ts_local = row.get("timestamp_local")
        if ts_local is not None:
            local_ts_str = ts_local.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            local_ts_str = "N/A"

        tooltip_text = row["name"]

        popup_html = f"""
        <b>{row['name']}</b><br>
        Station ID: {row['station_id']}<br>
        Free bikes: {row['free_bikes']}<br>
        Empty slots: {row['empty_slots']}<br>
        Capacity: {row['capacity']}<br>
        Current occupancy: {occ_str}<br>
        Status: {status}<br>
        Avg occupancy (last hour): {avg_occ_str}<br>
        Expected demand index (0-1): {demand_str}<br>
        Last update (Chicago time): {local_ts_str}<br>
        """

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            tooltip=tooltip_text,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(folium_map)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add legend box
    legend_html = """
    <div style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 180px; z-index: 9999;
        background-color: white;
        border: 2px solid grey;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
    ">
    <b>Station Status Legend</b><br>
    <span style="color:red;">&#9679;</span> Empty<br>
    <span style="color:green;">&#9679;</span> Full<br>
    <span style="color:yellow;">&#9679;</span> Normal<br>
    <span style="color:gray;">&#9679;</span> Offline<br>
    <span style="color:lightgray;">&#9679;</span> Unknown
    </div>
    """

    folium_map.get_root().html.add_child(folium.Element(legend_html))
    
    folium_map.save(output_path)
    print(f"Divvy map saved to: {output_path}")


def main() -> None:
    create_divvy_map()


if __name__ == "__main__":
    main()
