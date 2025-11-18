# airflow/divvy_dag.py

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# Import your existing ETL function from src/
# This assumes src is on the PYTHONPATH (in Airflow this is typical)
from src.etl_divvy import run_single_snapshot
from src.analytics import summarize_current_status, predict_next_by_last_hour
from src.map_divvy import create_divvy_map


# ---------------------------------------------------------------------
# Default Airflow settings for the DAG
# ---------------------------------------------------------------------
default_args = {
    "owner": "divvy-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ---------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------
with DAG(
    dag_id="divvy_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,  # or "*/5 * * * *" for every 5 minutes
    catchup=False,
    description="End-to-end ETL + analytics DAG for Divvy bike network",
) as dag:


    # -------------------------
    # 1. Extract & Load
    # -------------------------
    etl_task = PythonOperator(
        task_id="run_etl_snapshot",
        python_callable=run_single_snapshot,
    )


    # -------------------------
    # 2. Analytics summary
    # -------------------------
    summary_task = PythonOperator(
        task_id="summarize_status",
        python_callable=summarize_current_status,
    )

    prediction_task = PythonOperator(
        task_id="predict_demand",
        python_callable=predict_next_by_last_hour,
    )


    # -------------------------
    # 3. Update map visualization
    # -------------------------
    map_task = PythonOperator(
        task_id="generate_map",
        python_callable=create_divvy_map,
    )


    # -----------------------------------------------------------------
    # DAG FLOW — define dependencies
    # -----------------------------------------------------------------
    etl_task >> summary_task >> prediction_task >> map_task
