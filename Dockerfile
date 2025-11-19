FROM python:3.11-slim

# Install system dependencies needed for psycopg2, numpy, pandas, etc.
RUN apt-get update && \
    apt-get install -y gcc libpq-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies using Docker cache layers
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project into container
COPY . .

# Keep container alive for interactive ETL/analytics commands
CMD ["tail", "-f", "/dev/null"]
