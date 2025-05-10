from utils.setup_path import setup_paths
setup_paths()
import asyncio
import sys
from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
from pydantic import BaseModel
import io, os, json, torch, shutil
import pandas as pd
import numpy as np
from minio.error import S3Error
from PIL import Image
from image_feature_extractor import ImageFeatureExtractor
from text_feature_extractor import TextPreprocessor
from predict_model import predict_evaluate_model
from reco import reco
from minio import Minio
import uvicorn
import subprocess
from fastapi.middleware.cors import CORSMiddleware


# Init FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://host.docker.internal:8000"],  # Frontend
    allow_credentials=True,
    allow_methods=["*"],  # Autorise toutes les méthodes HTTP
    allow_headers=["*"],  # Autorise tous les en-têtes
)

# Configuration globale (tu les as déjà dans ton script)
MINIO_URL = "minio:9000"
ACCESS_KEY = "admin"
SECRET_KEY = "secret123"
BUCKET_NAME = "image-api"
bucket_name_test = BUCKET_NAME


device = 'cuda' if torch.cuda.is_available() else 'cpu'
minio_client = Minio(MINIO_URL, access_key=ACCESS_KEY, secret_key=SECRET_KEY, secure=False)

base_dir = os.path.dirname(__file__)
model_path_dir = os.path.join(base_dir, "src", "models")
model_path_dir_img = os.path.join(base_dir, "src", "data", "models", "vit_model_complete.pth")
model_final = os.path.join(model_path_dir, "trained_model.joblib")
model_label_encoder = os.path.join(base_dir, "src" , "data", "models" , "label_encoder.pkl")
scaler_text = os.path.join(model_path_dir, "scaler_text.joblib")
scaler_img = os.path.join(model_path_dir, "scaler_img.joblib")

# Classe Pydantic
class PredictionRequest(BaseModel):
    id: int
    designation: Optional[str] = ""
    description: Optional[str] = ""
    productid: int
    imageid: int

# -------------------------------------------------------
# FONCTION BLOQUANTE POUR LE PIPELINE DE TRAINING
# -------------------------------------------------------
def run_training_pipeline():
    results = {}

    scripts = [
        ("ajout_new_data","./src/data/insert_new_data.py"),
        ("extract_feat_img", "./src/data/processed_img_data.py"),
        ("extract_feat_text", "./src/data/processed_text_data.py"),
        ("train_model", "./src/models/train_model.py"),
        ("predict_model", "./src/models/predict_model.py")
    ]

    for name, path in scripts:
        try:
            result = subprocess.run(
                [sys.executable, path],
                capture_output=True,
                text=True
            )
            results[f"{name}_stdout"] = result.stdout
            results[f"{name}"] = "fully passed"
        except Exception as e:
            results[f"{name}_error"] = str(e)

    return results

# -------------------------------------------------------
# FONCTION BLOQUANTE POUR LA PREDICTION
# -------------------------------------------------------
def run_prediction_pipeline(file_data, filename, form_data):
    # 1. Upload image to MinIO
    if not minio_client.bucket_exists(BUCKET_NAME):
        minio_client.make_bucket(BUCKET_NAME)
    else:
        objects = minio_client.list_objects(BUCKET_NAME, recursive=True)
        for obj in objects:
            minio_client.remove_object(BUCKET_NAME, obj.object_name)

    minio_client.put_object(
        bucket_name=BUCKET_NAME,
        object_name=filename,
        data=io.BytesIO(file_data),
        length=len(file_data),
        content_type="image/jpeg"
    )

    # 2. Convert form data to DataFrame
    prediction = PredictionRequest(**form_data)
    df = pd.DataFrame([prediction.dict()])
    df.columns = ["Unnamed: 0","designation","description","productid","imageid"]

    # 3. Extract image features
    feature_extractor = ImageFeatureExtractor(model_path_dir_img, minio_client, device=device)
    image_features = feature_extractor.extract_features_from_df(df, bucket_name_test)
    feat_path_img = os.path.join(base_dir, "src", "data", "processed", "image_features_pred.npz")
    np.savez(feat_path_img, image_features)

    # 4. Text preprocessing
    text_preprocessor = TextPreprocessor(max_features=100)
    df = text_preprocessor.preprocess_and_clean_data(df)
    df_clean = text_preprocessor.preprocess_and_get_embeddings(df)
    feat_path_text = os.path.join(base_dir, "src", "data", "processed", "text_features_pred.npz")
    text_preprocessor.save_transformed_data(df_clean, feat_path_text)

    # 5. MLflow & model prediction
    from mlflow import MlflowClient
    import mlflow

    client = MlflowClient(tracking_uri="http://mlflow:5000")
    experiment = mlflow.get_experiment_by_name("classifier")
    experiment_id = experiment.experiment_id

    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="tags.mlflow.runName LIKE 'classifier%'",
        order_by=["start_time DESC"],
        max_results=1
    )

    run_name = runs[0].info.run_name
    run_id = runs[0].info.run_id
    
    predi = predict_evaluate_model(
        scaler_img, scaler_text,
        feat_path_img, feat_path_text,
        run_name, run_id, experiment_id,
        y_test=None, model_filename=None, 
        model_filename_2=None
    )

    reco_result = reco(predi, {
        'user': 'admin',
        'password': 'secret123',
        'host': 'postgre-db',
        'port': '5432',
        'database': 'mlops'
    }, 3)

    return {
        "prediction": str(predi),
        "recommandation": reco_result
    }

# --------------------- ENDPOINT ---------------------
@app.post("/prediction/")
async def prediction(
    file: UploadFile = File(...),
    id: int = Form(...),
    designation: str = Form(""),
    description: str = Form(""),
    productid: int = Form(...),
    imageid: int = Form(...)
):
    try:
        file_data = await file.read()
        form_data = {
            "id": id,
            "designation": designation,
            "description": description,
            "productid": productid,
            "imageid": imageid
        }

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: run_prediction_pipeline(file_data, file.filename, form_data)
        )

        return result

    except S3Error as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Erreur générale : {str(e)}"}


# -------------------------------------------------------
# ENDPOINT ASYNC
# -------------------------------------------------------
@app.post("/training/")
async def training():
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, run_training_pipeline)
        return result

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)