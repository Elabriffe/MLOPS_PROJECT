import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sqlalchemy import create_engine
import mlflow
from mlflow import MlflowClient
import shutil

#-----------------------------------MLFLOW---------------------------------------
client = MlflowClient(tracking_uri="http://mlflow:5000")
mlflow.set_tracking_uri("http://mlflow:5000") #Mettre celle là aussi, les 2 sont hypers importants... Absolument à mettre ici 
experiment_name = "classifier"
mlflow.set_experiment(experiment_name)
#-----------------------------------MLFLOW---------------------------------------

def load_and_prepare_features(scaler_img_path, scaler_text_path, image_features_filename, text_features_filename):
    scaler_img = joblib.load(scaler_img_path)
    scaler_text = joblib.load(scaler_text_path)
    image_features = np.load(image_features_filename)["arr_0"]
    text_features = np.load(text_features_filename)["arr_0"]
    return np.hstack([scaler_img.transform(image_features), scaler_text.transform(text_features)])

def load_previous_model(experiment_id, y_test, X_test):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName LIKE 'classifier%'",
        order_by=["start_time DESC"],
        max_results=2
    )

    if len(runs) > 1:
        artifacts = client.list_artifacts(runs[1].info.run_id, path="best_model")
        if not artifacts:
            return None, 0, 0, 0, 0, None

        path = mlflow.artifacts.download_artifacts(
            run_id=runs[1].info.run_id,
            artifact_path=artifacts[0].path
        )
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        return (
            model,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred, average="weighted", zero_division=0),
            recall_score(y_test, y_pred, average="weighted", zero_division=0),
            f1_score(y_test, y_pred, average="weighted", zero_division=0),
            path
        )
    return None, 0, 0, 0, 0, None

def evaluate_model(name, model, X, y):
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y, y_pred, average="weighted", zero_division=0)

    print(f"**** {name} ****")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    mlflow.log_metric(f"accuracy_{name}", acc)
    mlflow.log_metric(f"precision_{name}", prec)
    mlflow.log_metric(f"recall_{name}", rec)
    mlflow.log_metric(f"f1_score_{name}", f1)
    
    return acc, y_pred

def predict_evaluate_model(
    scaler_img_path,
    scaler_text_path,
    image_features_filename,
    text_features_filename,
    run_name,
    run_id,
    experiment_id,
    y_test=None,
    model_filename=None,
    model_filename_2=None):
    X_test = load_and_prepare_features(scaler_img_path ,scaler_text_path ,image_features_filename,text_features_filename)

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id, run_id=run_id):
        if model_filename is not None and model_filename_2 is not None and y_test is not None:
            old_model, acc_old, precision_old, recall_old, f1_old, old_model_path = load_previous_model(experiment_id, y_test, X_test)

            xgb = joblib.load(model_filename)
            xgb.set_params(device='cpu', n_jobs=-1)
            rf = joblib.load(model_filename_2)

            acc_xgb, y_pred_xgb = evaluate_model("xgb", xgb, X_test, y_test)
            acc_rf, y_pred_rf = evaluate_model("rf", rf, X_test, y_test)
            if old_model_path:
                acc_old, y_pred_old = evaluate_model("old", old_model, X_test, y_test)

            if acc_xgb >= acc_rf and (acc_xgb >= acc_old or old_model is None):
                mlflow.log_artifact(model_filename, artifact_path="best_model")
                return y_pred_xgb[0]
            elif acc_rf > acc_xgb and (acc_rf > acc_old or old_model is None):
                mlflow.log_artifact(model_filename_2, artifact_path="best_model")
                return y_pred_rf[0]
            else:
                if old_model_path:
                    dir_name = os.path.dirname(old_model_path)
                    new_old_model_path = os.path.join(dir_name, "old.joblib")
                    os.rename(old_model_path, new_old_model_path)
                    mlflow.log_artifact(new_old_model_path, artifact_path="best_model")
                    shutil.rmtree(os.path.dirname(os.path.dirname(new_old_model_path)))
                return old_model.predict(X_test)[0]
        else:
            artifact = client.list_artifacts(run_id, path="best_model")[0]
            path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=artifact.path)
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            shutil.rmtree(os.path.dirname(os.path.dirname(path)))
            return y_pred[0]

if __name__ == "__main__":

# --------------------------------------------------------- MLFLOW --------------------------------------------------------
    base_run_name = "classifier"

    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.runName LIKE '{base_run_name}%'",
            order_by=["start_time DESC"],
            max_results=1  # Limite à 10 derniers runs
        )

    run_name = run[0].info.run_name
    run_id = run[0].info.run_id
# --------------------------------------------------------- MLFLOW --------------------------------------------------------
    # Récupérer la base de données

    db_config = {
        "db_user": "admin",
        "db_password": "secret123",
        "db_host": "postgre-db",
        "db_port": "5432",
        "db_name": "mlops"
    }

    db_user = db_config.get("db_user", "admin")
    db_password = db_config.get("db_password", "secret123")
    db_host = db_config.get("db_host", "postgre-db")
    db_port = db_config.get("db_port", "5432")
    db_name = db_config.get("db_name", "mlops")

    engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
    
    # Définir les chemins
    base_dir = os.path.dirname(__file__)

    model_filename = os.path.join(base_dir, "xgb.joblib")
    model_filename_2 = os.path.join(base_dir, "rf.joblib")
    scaler_img_path = os.path.join(base_dir, "scaler_img.joblib")
    scaler_text_path = os.path.join(base_dir, "scaler_text.joblib")

    image_features_filename = os.path.join(base_dir, '..' , "data", "processed", "image_features_test_reduced.npz")
    text_features_filename = os.path.join(base_dir, '..', "data", "processed", "text_features_test.npz")
    
    # Charger les labels de test
    y_test = pd.read_sql('SELECT * FROM reduit."y_test" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")
    y_test = y_test["prdtypecode"].values

    # Évaluer le modèle
    pred = predict_evaluate_model(scaler_img_path, 
    scaler_text_path, 
    image_features_filename, 
    text_features_filename, 
    run_name, run_id, 
    experiment_id, 
    y_test, 
    model_filename, 
    model_filename_2)

    print(f"Prédiction du premier élément : {pred}")

