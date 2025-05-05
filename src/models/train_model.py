import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from sqlalchemy import create_engine, inspect
import joblib
import mlflow
from mlflow import MlflowClient
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------- MLFLOW --------------------------------------------------------

client = MlflowClient(tracking_uri="http://mlflow:5000")

mlflow.set_tracking_uri("http://mlflow:5000") #Mettre celle là aussi, les 2 sont hypers importants...

experiment_name = "classifier"

base_run_name = "classifier"

# Vérifier ou créer l'expérience
experiment = mlflow.get_experiment_by_name(experiment_name)

if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Expérience créée : {experiment_name}")
else:
    experiment_id = experiment.experiment_id
    print(f"Expérience trouvée : {experiment_name}")

# Définir l'expérience active
mlflow.set_experiment(experiment_name)

runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{base_run_name}%'",
        order_by=["start_time DESC"],
        max_results=10  # Limite à 10 derniers runs
    )

    # Si on trouve des runs existants
if runs:
    version_numbers = []
    for run in runs:
        run_name = run.info.run_name
            # Extraire le numéro de version de "xgb_v1", "xgb_v2", etc.
        if f"{base_run_name}_v" in run_name:
            try:
                version_number = int(run_name.split(f"{base_run_name}_v")[1])
                version_numbers.append(version_number)
            except ValueError:
                continue
        
        # Si on a des versions, incrémenter la version la plus haute
    next_version = max(version_numbers, default=0) + 1
    run_name = f"{base_run_name}_v{next_version}"
    
else:
    # Si aucun run n'existe, on commence à "v1"
    run_name = f"{base_run_name}_v1"

# --------------------------------------------------------- MLFLOW --------------------------------------------------------

# Configuration PostgreSQL
db_user = "admin"
db_password = "secret123"
db_host = "postgre-db"
db_port = "5432"
db_name = "mlops"

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Chemins des fichiers

y_train = pd.read_sql('SELECT * FROM reduit."y_train" ORDER BY "Unnamed: 0"', engine,index_col="Unnamed: 0")
y_train = y_train["prdtypecode"].values

base_dir = os.path.dirname(__file__) 

# Charger les features sauvegardées
image_features_train = np.load(os.path.join(base_dir, '..' , "data", "processed", "image_features_train_reduced.npz"))
text_features_train = np.load(os.path.join(base_dir, '..', "data", "processed", "text_features_train.npz"))


image_scaler = MinMaxScaler()
text_scaler = MinMaxScaler()

X_train = np.hstack([image_scaler.fit_transform(image_features_train["arr_0"]), text_scaler.fit_transform(text_features_train["arr_0"])])

model_a = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
)


model_b = XGBClassifier(
    n_estimators=300, 
    learning_rate=0.05, 
    max_depth=3,
    tree_method="hist",
    device="cuda",  # Utiliser GPU
    random_state=42
)

# Entraînement + logging automatique
with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
    print(f" Nouveau run créé : {run.info.run_id} avec le nom {run_name}")
    model_b.fit(X_train, y_train)
    model_a.fit(X_train, y_train)

    # Sauvegarde du modèle xgb
    model_xgb = os.path.join(base_dir, "xgb.joblib")
    joblib.dump(model_b, model_xgb)

    #  Sauvegarde du RandomForest
    model_randomforest = os.path.join(base_dir, "rf.joblib")
    joblib.dump(model_a, model_randomforest)

    #  Sauvegarde du VitTransformer dans MLFLOW
    model_path_img = os.path.join(base_dir, ".." , "data" ,"models" ,"vit_model_complete.pth")
    mlflow.log_artifact(model_path_img , artifact_path="model_img")

    #  Sauvegarder et logguer les scalers
    scaler_img_path = os.path.join(base_dir, "scaler_img.joblib")
    scaler_text_path = os.path.join(base_dir, "scaler_text.joblib")
    joblib.dump(image_scaler, scaler_img_path)
    joblib.dump(text_scaler, scaler_text_path)

    mlflow.log_artifact(scaler_img_path, artifact_path="scalers")
    mlflow.log_artifact(scaler_text_path, artifact_path="scalers")

    print(" Model trained and logged successfully with MLflow.")
