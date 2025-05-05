import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
import os
import joblib
from sklearn.preprocessing import LabelEncoder
import mlflow
from mlflow import MlflowClient


# --------------------------------------------------------- MLFLOW --------------------------------------------------------

client = MlflowClient(tracking_uri="http://mlflow:5000")

mlflow.set_tracking_uri("http://mlflow:5000") #Mettre celle là aussi, les 2 sont hypers importants...

experiment_name = "labels_encoding"

# Définir l'expérience active
mlflow.set_experiment(experiment_name)

experiment = mlflow.get_experiment_by_name("labels_encoding")

# --------------------------------------------------------- MLFLOW --------------------------------------------------------

# Configuration PostgreSQL
db_config = {
    'user': 'admin',
    'password': 'secret123',
    'host': 'postgre-db',
    'port': '5432',
    'database': 'mlops'
}

base_dir = os.path.dirname(__file__)

csv_file = os.path.join(base_dir, "raw/X_train_update.csv")  # Fichier CSV original
csv_file_y = os.path.join(base_dir, "raw","Y_train_CVw08PX.csv")  # Fichier CSV original
x_train_path= os.path.join(base_dir, "raw_reduit/X_train.csv")
y_train_path= os.path.join(base_dir, "raw_reduit/y_train.csv")
x_test_path= os.path.join(base_dir, "raw_reduit/X_test.csv")
y_test_path= os.path.join(base_dir, "raw_reduit/y_test.csv")

label_encoder = LabelEncoder() #Pour encoder directement les product code pour y_train et y_test

# Liste des fichiers CSV, des tables et des schémas correspondants
files_and_tables = [
    {'file': csv_file, 'table': 'X_train', 'schema': 'full'},
    {'file': csv_file_y , 'table': 'y_train', 'schema': 'full'},
    {'file': x_train_path, 'table': 'X_train', 'schema': 'reduit'},
    {'file': y_train_path, 'table': 'y_train', 'schema': 'reduit'},
    {'file': x_test_path, 'table': 'X_test', 'schema': 'reduit'},
    {'file': y_test_path, 'table': 'y_test', 'schema': 'reduit'},
]

# Connexion à PostgreSQL
engine = create_engine(
    f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
)

# Setup de la base
with engine.begin() as conn: #Mettre begin car fait le commit ce qui est pas le cas avec le connect
    conn.execute(text('CREATE SCHEMA IF NOT EXISTS "full"'))
    print("Schema 'full' created (or already exists).")
    conn.execute(text('CREATE SCHEMA IF NOT EXISTS "reduit"'))
    print("Schema 'reduit' created (or already exists).")


# Fonction pour créer la table si elle n'existe pas dans un schéma
def create_table_if_not_exists(df, table_name, schema):
    inspector = inspect(engine)  # Utilisation de inspect()
    full_table_name = f"{schema}.{table_name}"
    
    # Vérifier si la table existe dans le schéma
    if table_name not in inspector.get_table_names(schema=schema):
        print(f"ℹ️ Création de la table '{full_table_name}'...")
        df.head(0).to_sql(name=table_name, con=engine, schema=schema, if_exists='replace', index=False)
        print(f"✅ Table '{full_table_name}' créée.")
    else:
        print(f"ℹ️ La table '{full_table_name}' existe déjà.")

# Fonction pour insérer les données dans le bon schéma
def insert_data(df, table_name, schema):
    full_table_name = f"{schema}.{table_name}"
    try:
        df.to_sql(name=table_name, con=engine, schema=schema, if_exists='append', index=False)
        print(f"✅ Données insérées dans la table '{full_table_name}'.")
    except Exception as e:
        print(f"❌ Erreur lors de l'insertion des données dans '{full_table_name}': {e}")

# Processus principal
for item in files_and_tables:
    try:
        # Charger le fichier CSV
        df = pd.read_csv(item['file'])
        if item['table'] == "y_train" and item['schema']=='full':
            label_encoder.fit(df["prdtypecode"].values)
            df["new_prdtypecode"]=label_encoder.transform(df["prdtypecode"].values)
            
            joblib.dump(label_encoder, os.path.join(base_dir, "models/label_encoder.pkl")) # Sauvegarde locale temporaire
            
            with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
                print(run.info.artifact_uri)
                mlflow.log_artifact(os.path.join(base_dir, "models","label_encoder.pkl"), artifact_path="label_encoder")
            
            print(f"📄 Label encoder chargé avec succès.")

        elif ((item['schema']=="reduit" and item['table'] == "y_train") or (item['schema']=="reduit" and item['table'] == "y_test")):
            df["old_prdtypecode"]=df["prdtypecode"].values
            df["prdtypecode"]=label_encoder.transform(df["prdtypecode"].values)

        print(f"📄 Fichier '{item['file']}' chargé avec succès.")
        
        # Créer la table si elle n'existe pas dans le schéma spécifié
        create_table_if_not_exists(df, item['table'], item['schema'])
        
        # Insérer les données dans la table du schéma
        insert_data(df, item['table'], item['schema'])

    except Exception as e:
        print(f"❌ Erreur lors du traitement du fichier '{item['file']}': {e}")

print("✅ Importation terminée pour tous les fichiers.")


