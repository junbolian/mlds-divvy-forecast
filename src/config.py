import os

# ----------------------------------------------------------------------
# Database configuration
# ----------------------------------------------------------------------
# These settings are loaded from environment variables when available.
# If no environment variable is set, sensible local defaults are used.
# This makes the configuration flexible for both development and production.

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "divvy")
DB_USER = os.getenv("DB_USER", "divvy")
DB_PASSWORD = os.getenv("DB_PASSWORD", "divvy")

# Construct the SQLAlchemy/Postgres connection URL.
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ----------------------------------------------------------------------
# Divvy API configuration
# ----------------------------------------------------------------------
# Citybikes v2 API endpoint for accessing Divvy (Chicago) bike network data.
# Can be overridden via environment variable for testing or alternative networks.
DIVVY_API_URL = os.getenv(
    "DIVVY_API_URL",
    "https://api.citybik.es/v2/networks/divvy",
)

# ----------------------------------------------------------------------
# Occupancy classification thresholds
# ----------------------------------------------------------------------
# Thresholds control how stations are categorized based on occupancy:
#   - occupancy_ratio <= EMPTY_THRESHOLD → "empty"
#   - occupancy_ratio >= FULL_THRESHOLD → "full"
#
# Default values define:
#   empty  = 20% or less full
#   full   = 80% or more full
#
# Values can be adjusted dynamically through environment variables.
EMPTY_THRESHOLD = float(os.getenv("EMPTY_THRESHOLD", "0.2"))
FULL_THRESHOLD = float(os.getenv("FULL_THRESHOLD", "0.8"))
