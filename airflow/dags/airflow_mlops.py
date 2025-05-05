from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from datetime import datetime
import requests
import json
import os
import pandas as pd


# Fonction pour appeler le endpoint training
def endpoint_training(**kwargs):
    url = "http://host.docker.internal:8000/training/"  # Url de notre endpoint bien mettre host.docker.internal, pas mettre localhost sinon airflow lancera la requête en local depuis le conteneur et ça ne peut donc pas marcher. d'où le host.internal
    headers = {
        'Content-Type': 'application/json',  # Spécifie le type de contenu
    }

    response = requests.post(url, headers=headers)

    if response.status_code == 200:
        print(f"Requête réussie, réponse : {response.json()}")
    else:
        print(f"Erreur lors de la requête, statut : {response.status_code}")
    

# Définition du DAG
default_args = {
    "owner": "airflow",
    "start_date": datetime(2025, 4, 26),
    "retries": 1,
}

with DAG(
    dag_id="projet_mlops",
    default_args=default_args,
    tags=['projet_mlops', 'datascientest'],
    schedule_interval="*/15 * * * *",
    catchup=False,
) as dag_projet_mlops :

    # Tâche de récupération des données météo
    training_task = PythonOperator(
        task_id="training_task",
        python_callable=endpoint_training,
        provide_context=True,
        dag=dag_projet_mlops,
    )
    
    training_task  