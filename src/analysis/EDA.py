import sys, os
# Add parent directory (src/) to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
import matplotlib.pyplot as plt
import seaborn as sns

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


engine = create_engine(DATABASE_URL)
stations = pd.read_sql("SELECT * FROM dim_station", engine)
statuses = pd.read_sql("SELECT * FROM fact_station_status", engine)

stations.info()
statuses.info()


# If running in Docker (common for your pipeline)
DOCKER_OUTPUT = "/app/outputs"

if os.path.exists(DOCKER_OUTPUT):
    OUTPUT_DIR = DOCKER_OUTPUT
else:
    # Local fallback for Jupyter / local runs
    OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..", "outputs"))

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving plots to: {OUTPUT_DIR}")


df = statuses.merge(stations, on="station_id", how="left")

# Compute mean occupancy per station
station_rank = (
    df.groupby(["station_id", "name"])["occupancy_ratio"]
      .mean()
      .reset_index()
      .rename(columns={"occupancy_ratio": "avg_occupancy"})
      .sort_values("avg_occupancy", ascending=False)
      .reset_index(drop=True)
)

# Add rank column
station_rank["rank"] = station_rank["avg_occupancy"].rank(ascending=False, method="dense").astype(int)

station_rank.head(10)



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

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "top_10_Stations_by_Average_Occupancy_Ratio.png"), dpi=300)
plt.close()


# Constants for Chicago boundaries
MADISON_LAT = 41.8820   # North of this = North Side, South of this = South Side
STATE_LON = -87.6278    # West of this = West Side, East of this = East Side

def classify_region(lat, lon):
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

stations["region"] = stations.apply(lambda r: classify_region(r["latitude"], r["longitude"]), axis=1)


df = statuses.merge(stations, on="station_id", how="left")

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

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Average_Occupancy_Ratio_by_Chicago_Region.png"), dpi=300)
plt.close()


sns.histplot(df["free_bikes"].dropna(), bins=30)
plt.title("Distribution of Free Bikes Across All Stations")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Distribution_of_Free_Bikes_Across_All_Stations.png"), dpi=300)
plt.close()


# Identifies unstable stations — good for maintenance or rebalancing analysis.
variability = df.groupby("name")["occupancy_ratio"].std().sort_values(ascending=False)
variability.head(10).plot(kind="barh", title="Most Volatile Stations (Occupancy Std Dev)")


plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Most_Volatile_Stations.png"), dpi=300)
plt.close()