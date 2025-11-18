from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# ----------------------------------------------------------------------
# Build the SQLAlchemy database URL
# ----------------------------------------------------------------------
# Using the psycopg2 driver (postgresql+psycopg2).
# Credentials are loaded from config/environment variables.
DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ----------------------------------------------------------------------
# Create the SQLAlchemy engine
# ----------------------------------------------------------------------
# echo=False → suppress SQL logging (set to True for debugging).
# future=True → enables SQLAlchemy 2.0 style engine behaviors.
engine = create_engine(
    DATABASE_URL,
    echo=False,  # set True to see raw SQL in logs
    future=True,
)

# ----------------------------------------------------------------------
# Create a session factory for database interactions
# ----------------------------------------------------------------------
# autocommit=False → explicit commit() required for transactions.
# autoflush=False → prevents automatic flush before queries;
#                   gives more control in complex workflows.
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)
