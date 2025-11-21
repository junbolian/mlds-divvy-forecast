import sys, os
import textwrap

# ----------------------------------------------------------------------
# Setup: Add project root to Python path so we can import config.py
# ----------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from src.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Visualization + Analysis Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io

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


def info_to_df(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue().splitlines()

    rows = []
    # Skip header lines; extract only column lines
    for line in info_str[5:-2]:
        parts = line.split()
        # Format:
        # #  ColumnName  Non-Null-Count  Dtype
        col = parts[1]
        non_null = parts[2]
        dtype = parts[-1]
        rows.append([col, non_null, dtype])

    return pd.DataFrame(rows, columns=["Column", "Non-Null Count", "Dtype"])

print("\nStations Table Info:")
print(info_to_df(stations))

print("\nStatuses Table Info:")
print(info_to_df(statuses))

# ----------------------------------------------------------------------
# Detect whether running inside Docker (affects output directory)
# ----------------------------------------------------------------------
RUNNING_IN_DOCKER = os.path.exists("/.dockerenv")

if RUNNING_IN_DOCKER:
    OUTPUT_DIR = "/app/data/outputs"
else:
    OUTPUT_DIR = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "outputs"
    )

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Saving plots to: {OUTPUT_DIR}")

# Join fact + dimension tables
df = statuses.merge(stations, on="station_id", how="left")

# ----------------------------------------------------------------------
# Region Classification (North/South/East/West Chicago)
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
# EDA 1: Top 10 Stations by Average Occupancy
# ----------------------------------------------------------------------
# Compute mean occupancy per station
df = df[df["capacity"] > 5]

station_rank = (
    df.groupby(["station_id", "name"])["occupancy_ratio"]
      .mean()
      .reset_index()
      .rename(columns={"occupancy_ratio": "avg_occupancy"})
      .sort_values("avg_occupancy", ascending=False)
)

# Clean up long prefix
station_rank["name"] = station_rank["name"].str.replace(
    r"^Public Rack\s*-\s*", "", regex=True
)

station_rank = station_rank.dropna()
top10 = station_rank.head(10)

# Wrap long names for readability
labels = [textwrap.fill(name, width=30) for name in top10["name"]]
plt.figure(figsize=(13, 7))

# Add numeric labels next to each bar
for i, value in enumerate(top10["avg_occupancy"]):
    plt.text(value + 0.01, i, f"{value:.2f}", va="center", fontsize=10)

# Using a clearer gradient (Blues reversed)
colors = sns.color_palette("Blues_r", n_colors=len(top10))
plt.barh(labels, top10["avg_occupancy"], color=colors)
plt.gca().invert_yaxis()

plt.title("Top 10 Stations by Average Occupancy Ratio\n(Based on Live Snapshot Collection)",
          fontsize=16)
plt.xlim(top10["avg_occupancy"].min() * 0.95, top10["avg_occupancy"].max() * 1.05)
plt.xlabel("Average Occupancy Ratio", fontsize=12)
plt.ylabel("")
plt.tight_layout(pad=2.0)

plt.savefig(os.path.join(OUTPUT_DIR, "top_10_Stations_by_Average_Occupancy_Ratio.png"), dpi=300)
plt.close()

# ----------------------------------------------------------------------
# EDA 2: Average Occupancy by Chicago Region
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
# EDA 3: Distribution of Free Bikes
# ----------------------------------------------------------------------
plt.figure(figsize=(13, 7))

sns.histplot(df["free_bikes"].dropna(), bins=30, kde=False)

plt.title("Distribution of Free Bikes Across All Stations", fontsize=16)
plt.xlabel("Number of Free Bikes", fontsize=12)
plt.ylabel("Count of Stations", fontsize=12)

plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Distribution_of_Free_Bikes_Across_All_Stations.png"),
    dpi=300, bbox_inches='tight'
)

plt.close()

# ----------------------------------------------------------------------
# EDA 4: Identify Most Volatile (Unstable) Stations
# ----------------------------------------------------------------------
# Compute standard deviation of occupancy ratio per station
variability = (
    df.groupby("name")["occupancy_ratio"]
      .std()
      .dropna()
      .sort_values(ascending=False)
      .head(10)
)

# Convert index to list and wrap long station names
labels = [textwrap.fill(station, width=35) for station in variability.index]
plt.figure(figsize=(12, 7))

for i, v in enumerate(variability.values):
    plt.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=10)

# Use a gradient color based on magnitude of volatility
colors = sns.light_palette("red", n_colors=len(variability), reverse=True)
plt.barh(labels, variability.values, color=colors)

plt.gca().invert_yaxis()
plt.title("Most Volatile Stations (Occupancy Std Dev)")
plt.xlabel("Standard Deviation of Occupancy Ratio")
plt.tight_layout(pad=2.0)

plt.savefig(os.path.join(OUTPUT_DIR, "Most_Volatile_Stations.png"), dpi=300, bbox_inches='tight')
plt.close()

# ----------------------------------------------------------------------
# EDA 5: Identify Overall Station Status Distribution
# ----------------------------------------------------------------------
status_counts = df["status_label"].value_counts()
explode = [0.05] + [0]* (len(status_counts)-1)

plt.figure(figsize=(6, 6))
plt.pie(
    status_counts,
    labels=status_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=sns.color_palette("Set2"),
    explode=explode
)

plt.title("Overall Station Status Distribution")
plt.tight_layout()

plt.savefig(os.path.join(OUTPUT_DIR, "Station_Status_Distribution.png"), dpi=300)
plt.close()