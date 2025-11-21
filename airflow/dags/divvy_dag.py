import sys
sys.path.append("/app")  # allow importing src/

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

with DAG(
    dag_id="divvy_live_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="@hourly",      # run pipeline every hour
    catchup=False,
    description="Live Divvy ETL Pipeline using CityBikes API",
) as dag:

    # ---------------------------------------------------
    # 1. FETCH Snapshot (live API call)
    # ---------------------------------------------------
    fetch_snapshot = BashOperator(
        task_id="fetch_snapshot",
        bash_command="python3 -u -m src.etl_divvy --mode fetch --snapshots 12 --sleep 300",
    )

    # ---------------------------------------------------
    # 2. TRANSFORM Snapshot (cleaning + occupancy + status)
    # ---------------------------------------------------
    transform_snapshot = BashOperator(
        task_id="transform_snapshot",
        bash_command="python3 -u -m src.etl_divvy --mode transform",
    )

    # ---------------------------------------------------
    # 3. LOAD Snapshot (insert into Postgres)
    # ---------------------------------------------------
    load_snapshot = BashOperator(
        task_id="load_snapshot",
        bash_command="python3 -u -m src.etl_divvy --mode load",
    )

    # FINAL DAG CHAIN
    fetch_snapshot >> transform_snapshot >> load_snapshot
