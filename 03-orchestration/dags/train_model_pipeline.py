from airflow.decorators import dag, task
from datetime import datetime
import subprocess

# Define el DAG usando la TaskFlow API
@dag(
    dag_id="train_model_pipeline_",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "training"]
)
def train_pipeline():

    @task()
    def get_data():

        context = get_current_context()
        conf = context["dag_run"].conf

        year = conf.get("year", 2023)
        month = conf.get("month", 3)
        # Llama al script get_data.py con los parámetros de año y mes
        subprocess.run(["python", "scripts/get_data.py", str(year), str(month)], check=True)

    @task()
    def preprocess():
        context = get_current_context()
        conf = context["dag_run"].conf

        year = conf.get("year", 2023)
        month = conf.get("month", 3)
        # Llama al script get_data.py con los parámetros de año y mes
        subprocess.run(["python", "scripts/preprocess.py", str(year), str(month)], check=True)

    @task()
    def train():
        context = get_current_context()
        conf = context["dag_run"].conf

        year = conf.get("year", 2023)
        month = conf.get("month", 3)
        # Llama al script get_data.py con los parámetros de año y mes
        subprocess.run(["python", "scripts/train.py", str(year), str(month)], check=True)

    # Define la dependencia entre tareas
    get_data() >> preprocess() >> train()

# Instancia del DAG
dag = train_pipeline()
