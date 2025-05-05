import os
import torch
import pandas as pd
from minio import Minio
import sys
from sqlalchemy import create_engine
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from minio import Minio
from io import BytesIO
import numpy as np

base_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(base_dir, '..' , 'features'))
sys.path.append(utils_dir)

from image_feature_extractor import ImageFeatureExtractor  # Importer la classe que nous avons définie précédemment

# Initialisation du client MinIO
minio_client = Minio(
    "minio:9000",
    access_key="admin",
    secret_key="secret123",
    secure=False
)

db_user = "admin"
db_password = "secret123"
db_host = "postgre-db"
db_port = "5432"
db_name = "mlops"

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Initialiser l'extracteur de caractéristiques avec le modèle pré-entraîné
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models/vit_model_complete.pth")
feature_extractor = ImageFeatureExtractor(model_path, minio_client, device='cuda' if torch.cuda.is_available() else 'cpu')

# Charger les données de train et de test
df_train = pd.read_sql('SELECT * FROM reduit."X_train" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")
df_test = pd.read_sql('SELECT * FROM reduit."X_test" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")

# Extraire les features des images depuis MinIO pour l'ensemble d'entraînement
bucket_name_train = "bucket-images-train"
image_features_train = feature_extractor.extract_features_from_df(df_train, bucket_name_train)

# Extraire les features des images depuis MinIO pour l'ensemble de test
bucket_name_test = "bucket-images-test" #pour le test
image_features_test = feature_extractor.extract_features_from_df(df_test, bucket_name_test)

# Sauvegarder les features extraites
np.savez(os.path.join(base_dir, "processed/image_features_train_reduced.npz"), image_features_train)
np.savez(os.path.join(base_dir, "processed/image_features_test_reduced.npz"), image_features_test)
