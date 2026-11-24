"""Airflow DAG equivalent of the GitHub Actions retrain workflow.

ingest -> preprocess -> train_register_promote -> redeploy
Each task runs the same commands as scripts/retrain.sh, so the pipeline logic is
defined once in the package and orchestrated either by Airflow or GitHub Actions.
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

ENV = "cd /opt/trajclf && export PYTHONPATH=src && "

with DAG(
    dag_id="trajclf_training",
    start_date=datetime(2026, 11, 1),
    schedule="0 3 * * 1",
    catchup=False,
    tags=["trajclf", "training"],
) as dag:
    ingest = BashOperator(task_id="ingest", bash_command=ENV + "mc cp --recursive local/data/gbg500/ data/raw/")
    preprocess = BashOperator(task_id="preprocess", bash_command=ENV + "python -m trajclf.data")
    train = BashOperator(task_id="train_register_promote",
                         bash_command=ENV + "python -m trajclf.train | tee /tmp/train.log")
    redeploy = BashOperator(
        task_id="redeploy",
        bash_command="grep -q 'champion=True' /tmp/train.log && "
                     "kubectl -n trajclf rollout restart deploy/trajclf-api || echo 'no promotion'",
    )
    ingest >> preprocess >> train >> redeploy
