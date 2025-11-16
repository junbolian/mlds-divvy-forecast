FROM python:3.11-slim

# Install system dependencies required by psycopg2 and other libs
RUN apt-get update && \
    apt-get install -y gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# By default, keep the container alive.
# We will run ETL / analytics / map commands via `docker compose exec app ...`
CMD ["sleep", "infinity"]
