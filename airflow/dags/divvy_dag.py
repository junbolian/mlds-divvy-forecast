from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

from src.api_client import fetch_divvy_data
from src.cleaning import process_divvy_data
from src.db import load_data_to_sql
from src.map_builder import build_interactive_map

with DAG(
    dag_id='divvy_pipeline',
    start_date=datetime(2025, 1, 1),
    schedule_interval='*/40 * * * *',   # every 40 minutes
    catchup=False,
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    }
) as dag:

    pull_live_api = PythonOperator(
        task_id='pull_live_api',
        python_callable=fetch_divvy_data
    )

    clean_transform = PythonOperator(
        task_id='clean_transform',
        python_callable=process_divvy_data,
        op_kwargs={'raw_data': "{{ ti.xcom_pull(task_ids='pull_live_api') }}"}
    )

    load_to_sql = PythonOperator(
        task_id='load_to_sql',
        python_callable=load_data_to_sql,
        op_kwargs={'cleaned_data': "{{ ti.xcom_pull(task_ids='clean_transform') }}"}
    )

    build_visualizations = PythonOperator(
        task_id='build_interactive_map',
        python_callable=build_interactive_map
    )

    pull_live_api >> clean_transform >> load_to_sql >> build_visualizations
