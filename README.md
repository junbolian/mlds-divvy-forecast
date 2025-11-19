# Chicago Divvy Live Data Engineering Pipeline

This repository implements a live data pipeline for the **Chicago Divvy bike-sharing system**, using the public **CityBikes v2 API** and a Postgres warehouse.
It demonstrates an end-to-end modern data engineering workflow:

- Pulling live station data from the API.
- Cleaning and transforming the data into a simple star schema.
- Classifying station status (EMPTY / NORMAL / FULL / OFFLINE / UNKNOWN).
- Computing a basic “expected demand index” for each station.
- Visualizing results on an interactive map.
- Running an EDA pipeline to analyze bike availability and station behavior, and generate EDA plots/

All data comes from the live API at runtime. No CSV files or static datasets are stored in the repo.

---

## 1. Data Source

API endpoint:

- `https://api.citybik.es/v2/networks/divvy`

From the JSON response, the project uses:

From `network`:
- `network.location.city` – city name (Chicago) for station metadata.

From each `station`, the pipeline uses:
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

## 2. Tables and Data Manipulation

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

#### Capacity:

Capacity is defined as:
```yaml
if free_bikes + empty_slots > 0:
    capacity = free_bikes + empty_slots
else:
    capacity = NULL
```
This uses the most reliable field when available and falls back to a reasonable approximation.

#### Occupancy ratio

If capacity is positive:
```yaml
occupancy_ratio = free_bikes / capacity
```
This gives a value between 0 and 1, representing the share of docks that currently hold bikes.

#### Station status classification

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
---

## 3. Interactive Map

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
  - `normal` → yellow
  - `full` → green
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

## 4. Running the Project with Docker

### 4.1 Requirements

- Docker
- Docker Compose
- Internet connection

### 4.2 Start the stack

From the repository root:

```bash
docker compose build
docker compose up -d
```

This launches:

- `db` – PostgreSQL 16 with database `divvy`.
- `app` – Python ETL + analytics environment

### 4.3 Collect live snapshots

**Full 1-hour dataset:**
```bash
docker compose exec app python -m src.etl_divvy --snapshots 12 --sleep 300
```
- 12 snapshots total
- 300 seconds (5 minutes) between snapshots
- 1-hour coverage

**Quick test (3 snapshots, 30 seconds apart):**
```bash
docker compose exec app python -m src.etl_divvy --snapshots 3 --sleep 30
```
This collects **3 snapshots**, 30 seconds apart.

#### What Each Snapshot Does
1. **Extract**  
   Fetches live data from the CityBikes Divvy API.
   
3. **Transform**  
   - Parses timestamps  
   - Computes occupancy ratio  
   - Classifies station status (empty / normal / full / offline / unknown)
   - Derive capacity

4. **Load**  
   Inserts structured records into:  
   - `dim_station` (station metadata, upserted)  
   - `fact_station_status` (time-series snapshot data, append-only)

### 4.4 Analytics summary

```bash
docker compose exec app python -m src.analytics
```

Outputs include:
- Counts of stations in each status (EMPTY / NORMAL / FULL / OFFLINE / UNKNOWN).
- For each station, the average occupancy in the last hour.

### 4.5 Generate the map

```bash
docker compose exec app python -m src.map_divvy
```

Output:
- `outputs/divvy_map.html`

Open the file in a browser to see the current station states and demand indices on a map.

### 4.6 Exploratory Data Analysis (EDA)

```bash
docker compose exec app python -m src.analysis.EDA
```

Outputs saved to:
- `outputs/` (local runs)  
- `/app/outputs/` (inside Docker)

#### 1. **Average Occupancy Ratio by Chicago Region**
**File:** `Average_Occupancy_Ratio_by_Region.png`  
Shows differences in station fullness across the **North Side**, **South Side**, **West Side**, and **East Side**.

#### 2. **Distribution of Free Bikes Across All Stations**
**File:** `Distribution_of_Free_Bikes.png`  
A histogram showing how many bikes are typically available at stations, highlighting **demand imbalance**.

#### 3. **Top 10 Stations by Average Occupancy Ratio**
**File:** `Top_10_Stations_by_Average_Occupancy_Ratio.png`  
Identifies stations that are consistently **near capacity** and may require **rebalancing**.

#### 4. **Most Volatile Stations (Highest Occupancy Variance)**
**File:** `Most_Volatile_Stations.png`  
Shows which stations fluctuate the most in availability—often indicating **commuter hubs**, **tourism areas**, or **high-traffic zones**.

#### 5. **Average Occupancy Over Time**
**File:** `Average_Occupancy_Over_Time.png`   
A time-series line plot showing how the system’s **overall occupancy changes** across the collected snapshots.

---

## 5. Key Insights from Analysis
Based on 1-hour live collection:
#### 1. North Side stations show the highest average occupancy
They tend to have more full docks → high return demand.

#### 2. Distribution of free bikes is highly imbalanced
Some stations often approach zero bikes (empty risk), while others rarely drop below half-full.

#### 3. Top 10 busiest stations cluster near Downtown
Likely commuter hubs, receiving traffic during typical work hours.

#### 4. Volatile stations correspond to transit corridors
High variance suggests:
- industrial commuting
- CTA/Metra connectors
- tourism areas
  
#### 5. System-wide occupancy fluctuates smoothly over the hour
No major spikes → consistent usage pattern in the time window.

---

## 6. Airflow DAG

This project includes an Apache Airflow DAG (`airflow/dags/divvy_dag.py`) that automates the data pipeline.

#### DAG Responsibilities
- Run the ETL pipeline (src/etl_divvy.py)
- Pull 1 hour of live Divvy data (12 snapshots × 5 minutes)
- Insert records into PostgreSQL
- Run every hour to continuously refresh the dataset
- Log execution details (visible in Airflow UI)

#### DAG Schedule
The DAG is scheduled to run **hourly**:
```python
schedule_interval = "@hourly"
```

#### Running the DAG
Start Airflow:
```bash
docker compose up -d
```

Unpause:
```bash
docker compose exec airflow-webserver airflow dags unpause divvy_dag
```

Trigger:
```bash
docker compose exec airflow-webserver airflow dags trigger divvy_dag
```

Open the Airflow UI:
> → http://localhost:8080  
> → login: admin / admin

---

## 7. Project Structure

Basic layout:

```arduino
mlds-divvy-forecast/
│
├── src/                   # Application source code
│   ├── analysis/          # Exploratory Data Analysis scripts
│   │   └── EDA.py
│   ├── analytics.py       # status summary and expected demand index computation.
│   ├── etl_divvy.py       # ETL pipeline from API to PostgreSQL.
│   ├── map_divvy.py       # interactive map generation.
│   ├── models.py          # table definitions for "dim_station" and "fact_station_status".
│   ├── config.py          # configuration (DB connection, API URL, thresholds).
│   ├── db.py              # SQLAlchemy engine and database connection.
│   └── __init__.py
│
├── airflow/               # Airflow pipeline orchestration
│   └── dags/
│       └── divvy_dag.py
│
├── outputs/               # Generated results (ignored by git)
│   ├── *.png              # EDA plots
│   └── divvy_map.html     # Interaction station map
│
├── Dockerfile             # App container definition
├── requirements.txt       # Python dependency list.
├── docker-compose.yml     # Services: app + db + airflow
└── README.md              # Project description and instructions.
```

---

## 8. Design Summary

- API choice: Citybik.es v2 exposes live Divvy station data with availability and operational flags, which is ideal for building a streaming-style monitoring pipeline.
- Schema: one dimension table for station metadata plus one fact table for time-series snapshots keeps the model easy to query and extend.
- Processing: timestamp normalization, capacity estimation, and occupancy computation create consistent inputs for downstream analytics.
- Models: the status rules and expected demand index are transparent and directly tied to the data; they can be replaced or extended later without changing the database design.
- Visualization: the interactive map provides an immediate, intuitive check that the pipeline works and that the definitions of status and demand are reasonable in practice.
