# Chicago Divvy Live Data Engineering Pipeline

This repository implements a live data pipeline for Chicago’s Divvy bike system using the public Citybik.es v2 API and PostgreSQL. The project covers:

- Pulling live station data from the API.
- Cleaning and transforming the data into a simple star schema.
- Classifying station status (EMPTY / NORMAL / FULL / OFFLINE).
- Computing a basic “expected demand index” for each station.
- Visualizing results on an interactive map.

All data comes from the live API at runtime. No CSV files or static datasets are stored in the repo.

---

## 1. Data Source

API endpoint:

- `https://api.citybik.es/v2/networks/divvy`

From the JSON response, the project uses:

From `network`:
- `network.location.city` – city name (Chicago) for station metadata.

From each `station`:
- `id` – station identifier.
- `name` – station name.
- `latitude`, `longitude` – geographic location.
- `timestamp` – last update time.
- `free_bikes` – number of available bikes.
- `empty_slots` – number of free docks.
- `extra.slots` – total number of docks (capacity) when available.
- `extra.renting` – flag for whether bikes can be rented.
- `extra.returning` – flag for whether bikes can be returned.
- full `extra` – stored as JSON for future extensions.

These fields are enough to monitor availability and basic operational status for each station.

---

## 2. Data Model and Processing

The pipeline writes data into a PostgreSQL database using a simple star schema:

### 2.1 Dimension table: `dim_station`

Static station information:

- `station_id` – primary key (from `station.id`).
- `name` – station name.
- `latitude`, `longitude`.
- `city` – city name from the network object.
- `is_active` – boolean flag (currently always true).

These attributes change rarely and are modeled as a dimension table.

### 2.2 Fact table: `fact_station_status`

Time-series status per station and snapshot:

- `station_id` – foreign key to `dim_station`.
- `timestamp_utc` – last update time, stored as UTC and timezone-aware.
- `free_bikes`.
- `empty_slots`.
- `capacity` – derived.
- `occupancy_ratio` – derived.
- `status_label` – derived classification.
- `raw_extra` – full JSON from `station.extra`.

#### Timestamp handling

- API timestamps can be irregular (e.g., mixing `+00:00` and `Z`).
- Timestamps are cleaned and parsed, then stored in UTC as `timestamp_utc`.
- For display on the map, timestamps are converted to America/Chicago.

#### Capacity

Capacity is defined as:

- If `free_bikes + empty_slots` is positive:
  - `capacity = free_bikes + empty_slots`
- Else:
  - `capacity = NULL`

This uses the most reliable field when available and falls back to a reasonable approximation.

#### Occupancy ratio

If capacity is positive:

- `occupancy_ratio = free_bikes / capacity`

This gives a value between 0 and 1, representing the share of docks that currently hold bikes.

---

## 3. Station Status and Expected Demand

### 3.1 Station status classification

Each station at each snapshot is classified into one of:

- `offline` – if `renting == 0` and `returning == 0`.
- `empty` – if `occupancy_ratio <= 0.20`.
- `full` – if `occupancy_ratio >= 0.80`.
- `normal` – if `0.20 < occupancy_ratio < 0.80`.
- `unknown` – if capacity or occupancy ratio cannot be computed.

Reasoning:

- The renting/returning flags indicate if the station is generally in service.
- The 20% and 80% thresholds are simple and interpretable:
  - Low occupancy → few bikes, risk of finding no bike.
  - High occupancy → few empty docks, risk of not being able to return.

### 3.2 Expected demand index

The project defines a basic “expected demand index” per station based on recent occupancy:

1. Over a rolling window (default 60 minutes), compute:

   - `avg_occ_last_hour = AVG(occupancy_ratio)`

2. Define:

   - `expected_demand_index = 1 - avg_occ_last_hour`
   - The value is clipped to `[0, 1]`.

Interpretation:

- If a station is usually full (high `avg_occ_last_hour`), there is strong demand for empty docks.
- If a station is usually empty (low `avg_occ_last_hour`), there is strong demand for bikes.
- This index summarizes recent “pressure” on the station using only the data we collect.

This logic is implemented in `src/analytics.py` and reused in `src/map_divvy.py`.

---

## 4. Interactive Map

The script `src/map_divvy.py` builds an interactive HTML map and saves it as:

- `outputs/divvy_map.html`

Key steps:

1. Query the latest status for each station from `fact_station_status`.
2. Join with last-hour aggregates (average occupancy and expected demand index).
3. Convert `timestamp_utc` to America/Chicago time for display.
4. Plot each station on a Folium map (Leaflet under the hood).

Visualization details:

- Each station is drawn as a `CircleMarker`.
- Color encodes `status_label`:
  - `empty` → red
  - `normal` → green
  - `full` → dark blue
  - `offline` → gray
  - `unknown` → light gray
- Tooltip: station name.
- Popup includes:
  - Station ID and name.
  - Free bikes / empty slots / capacity.
  - Current occupancy percentage.
  - Status label.
  - Average occupancy over the last hour.
  - Expected demand index (0–1).
  - Last update time in Chicago local time.

The map can be regenerated at any time after new snapshots are ingested.

---

## 5. Running the Project with Docker

### 5.1 Requirements

- Docker
- Docker Compose
- Internet connection

### 5.2 Start the stack

From the repository root:

```bash
docker compose up -d
```

This starts:

- `db` – PostgreSQL 16 with database `divvy`.
- `app` – Python 3.11 container with the project code and dependencies.

### 5.3 Collect live snapshots

Run the ETL inside the `app` container. Example:

```bash
# 3 snapshots, 30 seconds apart
docker compose exec app python -m src.etl_divvy --snapshots 3 --sleep 30
```

Each snapshot:

- Calls the Divvy API.
- Cleans and transforms the data.
- Inserts records into `dim_station` and `fact_station_status`.

### 5.4 Run analytics summary

```bash
docker compose exec app python -m src.analytics
```

This prints:

- Counts of stations in each status (EMPTY / NORMAL / FULL / OFFLINE / UNKNOWN).
- For each station, the average occupancy in the last hour and the expected demand index.

### 5.5 Generate the map

```bash
docker compose exec app python -m src.map_divvy
```

Open:

- `outputs/divvy_map.html`

in a browser to see the current station states and demand indices on a map.

---

## 6. Project Structure

Basic layout:

- `Dockerfile` – builds the Python application image.
- `docker-compose.yml` – defines `db` (PostgreSQL) and `app` services.
- `requirements.txt` – Python dependencies.
- `README.md` – project description and instructions.
- `outputs/` – generated artifacts (e.g. `divvy_map.html`, not committed to git).
- `src/`
  - `config.py` – configuration (DB connection, API URL, thresholds).
  - `db.py` – SQLAlchemy engine and database connection.
  - `models.py` – table definitions for `dim_station` and `fact_station_status`.
  - `etl_divvy.py` – ETL pipeline from API to PostgreSQL.
  - `analytics.py` – status summary and expected demand index computation.
  - `map_divvy.py` – interactive map generation.

---

## 7. Design Summary

- API choice: Citybik.es v2 exposes live Divvy station data with availability and operational flags, which is ideal for building a streaming-style monitoring pipeline.
- Schema: one dimension table for station metadata plus one fact table for time-series snapshots keeps the model easy to query and extend.
- Processing: timestamp normalization, capacity estimation, and occupancy computation create consistent inputs for downstream analytics.
- Models: the status rules and expected demand index are transparent and directly tied to the data; they can be replaced or extended later without changing the database design.
- Visualization: the interactive map provides an immediate, intuitive check that the pipeline works and that the definitions of status and demand are reasonable in practice.
