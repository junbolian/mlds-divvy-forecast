# airflow/dags/divvy_dag.py

"""
Airflow DAG for automating the Divvy ETL pipeline.

This DAG is designed for a DOCKER-BASED project.
Unlike virtualenv workflows, we do NOT activate a .venv
because Airflow runs inside its own container and your project code is mounted
at /app via docker-compose volumes.

Airflow simply executes the ETL script inside its container environment
using the shared /app/src package.
"""

import sys
sys.path.append("/app")  # Allow Airflow to import src/

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="divvy_dag",
    default_args=default_args,
    schedule="@hourly",    # ← modern Airflow syntax
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    run_etl = BashOperator(
        task_id="run_etl_snapshot",
        bash_command="python3 -u -m src.etl_divvy --snapshots 12 --sleep 300",
    )

run_etl
