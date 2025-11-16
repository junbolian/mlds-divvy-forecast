import os

# Database configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "divvy")
DB_USER = os.getenv("DB_USER", "divvy")
DB_PASSWORD = os.getenv("DB_PASSWORD", "divvy")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# Divvy API endpoint (Citybikes v2)
DIVVY_API_URL = os.getenv(
    "DIVVY_API_URL",
    "https://api.citybik.es/v2/networks/divvy",
)

# Status thresholds
# <= 20% full -> "empty"
# >= 80% full -> "full"
EMPTY_THRESHOLD = float(os.getenv("EMPTY_THRESHOLD", "0.2"))
FULL_THRESHOLD = float(os.getenv("FULL_THRESHOLD", "0.8"))
