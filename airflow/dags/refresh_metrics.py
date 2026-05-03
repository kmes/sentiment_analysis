from airflow import DAG
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime

with DAG(
    dag_id="refresh_sentiment_metrics",
    schedule="*/5 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    SimpleHttpOperator(
        task_id="refresh_disagreement_rate",
        method="GET",
        http_conn_id="sentiment_api",
        endpoint="/internal/refresh-metrics",
        log_response=True,
    )
