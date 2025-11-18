# airflow/divvy_dag.py
import sys
sys.path.append("/app")  # Airflow can now import your src/ package

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
    schedule_interval="@hourly",  # runs every hour
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    run_etl = BashOperator(
        task_id="run_etl_snapshot",
        bash_command="python -m src.etl_divvy --snapshots 12 --sleep 300",
    )

run_etl
