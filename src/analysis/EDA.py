import sys, os

# ----------------------------------------------------------------------
# Setup: Add project root to Python path so we can import config.py
# ----------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Visualization + Analysis Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Database Connector
from sqlalchemy import create_engine

# ----------------------------------------------------------------------
# Connect to PostgreSQL
# ----------------------------------------------------------------------
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Load dimension + fact tables
stations = pd.read_sql("SELECT * FROM dim_station", engine)
statuses = pd.read_sql("SELECT * FROM fact_station_status", engine)

print("\n--- Station Table Info ---")
stations.info()

print("\n--- Status Table Info ---")
statuses.info()

# ----------------------------------------------------------------------
# Detect whether running inside Docker (affects output directory)
# ----------------------------------------------------------------------
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")

if RUNNING_IN_DOCKER:
    OUTPUT_DIR = "/app/outputs"
else:
    # Local path: project_root/outputs
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving plots to: {OUTPUT_DIR}")

# ----------------------------------------------------------------------
# Join fact + dimension tables
# ----------------------------------------------------------------------
df = statuses.merge(stations, on="station_id", how="left")

# ----------------------------------------------------------------------
# EDA 1: Top 10 Stations by Average Occupancy
# ----------------------------------------------------------------------
# Compute mean occupancy per station
station_rank = (
    df.groupby(["station_id", "name"])["occupancy_ratio"]
      .mean()
      .reset_index()
      .rename(columns={"occupancy_ratio": "avg_occupancy"})
      .sort_values("avg_occupancy", ascending=False)
      .reset_index(drop=True)
)

# Drop stations with no occupancy data (NaN)
station_rank = station_rank.dropna(subset=["avg_occupancy"])

# Add rank column
station_rank["rank"] = station_rank["avg_occupancy"].rank(
    ascending=False, method="dense"
).astype(int)

station_rank.head(10)

# Plot Top 10 stations
sns.barplot(
    data=station_rank.head(10),
    x="avg_occupancy",
    y="name",
    hue="name",            # ← add this
    dodge=False,
    legend=False,          # ← avoid redundant legend
    palette="Blues_r"
)

plt.title("Top 10 Stations by Average Occupancy Ratio")
plt.xlabel("Average Occupancy Ratio")
plt.ylabel("Station Name")

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(OUTPUT_DIR, "top_10_Stations_by_Average_Occupancy_Ratio.png"), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# EDA 2: Region Classification (North/South/East/West Chicago)
# ----------------------------------------------------------------------
MADISON_LAT = 41.8820  # Boundary between North Side & South Side
STATE_LON = -87.6278   # Boundary between East Side & West Side

def classify_region(lat, lon):
    """Categorize each station into a Chicago region based on lat/lon."""
    if lat > MADISON_LAT and abs(lon - STATE_LON) <= 0.01:
        return "North Side"
    elif lat < MADISON_LAT and abs(lon - STATE_LON) <= 0.01:
        return "South Side"
    elif lon > STATE_LON:
        return "East Side"
    elif lon < STATE_LON:
        return "West Side"
    else:
        return "Central"  # fallback, e.g. exactly on dividing line

# Add region classification to station table
stations["region"] = stations.apply(
    lambda r: classify_region(r["latitude"], r["longitude"]), axis=1
)

# Re-merge to include region
df = statuses.merge(stations, on="station_id", how="left")

# ----------------------------------------------------------------------
# EDA 3: Time-Series Analysis — Average Occupancy Over the Hour
# ----------------------------------------------------------------------
time_series = (
    df.groupby("timestamp_utc")["occupancy_ratio"]
      .mean()
      .reset_index()
)

plt.figure(figsize=(10, 5))
plt.plot(time_series["timestamp_utc"], time_series["occupancy_ratio"], marker="o")
plt.title("Average Occupancy Ratio Over Time")
plt.xlabel("Time")
plt.ylabel("Average Occupancy Ratio")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Average_Occupancy_Over_Time.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------
# EDA 4: Average Occupancy by Chicago Region
# ----------------------------------------------------------------------
region_stats = (
    df.groupby("region")["occupancy_ratio"]
      .agg(["mean", "std", "count"])
      .reset_index()
      .sort_values("mean", ascending=False)
)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=region_stats,
    x="region",
    y="mean",
    hue="region",         
    dodge=False,         
    legend=False,         
    palette="viridis"
)

plt.title("Average Occupancy Ratio by Chicago Region")
plt.ylabel("Mean Occupancy Ratio")
plt.xlabel("Region")

plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(OUTPUT_DIR, "Average_Occupancy_Ratio_by_Chicago_Region.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------
# EDA 5: Distribution of Free Bikes
# ----------------------------------------------------------------------
sns.histplot(df["free_bikes"].dropna(), bins=30)
plt.title("Distribution of Free Bikes Across All Stations")

plt.subplots_adjust(top=0.90)
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(OUTPUT_DIR, "Distribution_of_Free_Bikes_Across_All_Stations.png"), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# EDA 6: Identify Most Volatile (Unstable) Stations
# ----------------------------------------------------------------------
variability = (
    df.groupby("name")["occupancy_ratio"]
      .std()
      .sort_values(ascending=False)
)

variability.head(10).plot(
    kind="barh",
    title="Most Volatile Stations (Occupancy Std Dev)"
)


plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(OUTPUT_DIR, "Most_Volatile_Stations.png"), dpi=300, bbox_inches='tight')
plt.close()