import os
import torch
import timm
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
import pandas as pd
from sqlalchemy import create_engine, inspect
from minio import Minio
from io import BytesIO
from torchvision import models
from torch.optim.lr_scheduler import StepLR
import mlflow
from mlflow import MlflowClient

# --------------------------------------------------------- MLFLOW --------------------------------------------------------

client = MlflowClient(tracking_uri="http://mlflow:5000")

mlflow.set_tracking_uri("http://mlflow:5000") #Mettre celle là aussi, les 2 sont hypers importants...

experiment_name = "model_image"

run_name = "model_image"

# Créer une nouvelle expérience si elle n'existe pas déjà
# Si elle existe, tu récupères simplement l'ID de l'expérience existante
experiment_id = mlflow.create_experiment(experiment_name)

# Définir l'expérience active
mlflow.set_experiment(experiment_name)

experiment = mlflow.get_experiment_by_name("model_image")

# --------------------------------------------------------- MLFLOW --------------------------------------------------------



#Dossier du ficher python
base_dir = os.path.dirname(__file__)

config_classe = {
    "nb_type_prod" : 27, #nombre de différent produit dans le set de données
    "nb_epoch" : 25
}

# Configuration MinIO
minio_client = Minio(
    "minio:9000",  
    access_key="admin",  # Remplacez par votre access key
    secret_key="secret123",  # Laisser vide si vous n'avez pas de secret key
    secure=False
)

# Configuration PostgreSQL
db_user = "admin"
db_password = "secret123"
db_host = "postgre-db"
db_port = "5432"
db_name = "mlops"

bucket_name = "bucket-images-train"

image_paths = []
labels = []

# Connexion à la base PostgreSQL
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

# Lecture des données
df_train = pd.read_sql('SELECT * FROM reduit."X_train" ORDER BY "Unnamed: 0"', engine,index_col="Unnamed: 0")

df_labels = pd.read_sql('SELECT * FROM reduit."y_train" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")


for imageid, productid in tqdm(zip(df_train["imageid"], df_train["productid"])):
    # Chemin de l'image dans le bucket MinIO
    object_name = f"image_{imageid}_product_{productid}.jpg"
    
    try:
        # Vérifier si l'image existe dans MinIO
        minio_client.stat_object(bucket_name, object_name)
        
        # Ajouter le chemin de l'image
        image_paths.append(object_name)
        
        # Obtenir le label correspondant
        label = df_labels.loc[df_train[df_train["imageid"]==imageid].index, "prdtypecode"].values[0]
        labels.append(label)
    
    except Exception as e:
        print(f"Image {object_name} not found or error occurred: {e}")

# Charger le modèle pré-entraîné ViT
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)

# Geler toutes les couches sauf la dernière
for param in model.parameters():
    param.requires_grad = False

# Modifier la dernière couche pour l'adapter au nombre de classes de ton problème
# Supposons que tu aies N classes à prédire
N_CLASSES = config_classe['nb_type_prod']  # Le nombre de classes de notre problème

# Remplacer la dernière couche de classification (head)
model.head = nn.Linear(model.head.in_features, N_CLASSES)

# Assurer que les poids de la nouvelle couche sont entraînables
for param in model.head.parameters():
    param.requires_grad = True

  
# Définir un optimiseur pour n'entraîner que la dernière couche
optimizer = Adam(model.head.parameters(), lr=0.01)

# Perte de type Cross-Entropy (classification)
criterion = nn.CrossEntropyLoss()

# Charger tes données (train_loader et test_loader ici doivent être définis auparavant)
# Exemple d'entraînement
device = "cuda" if torch.cuda.is_available() else "cpu"

model.to(device)

# Étape 1 : Créer un Dataset personnalisé
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, minio_config, bucket_name, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.minio_config = minio_config  # Détails de configuration
        self.bucket_name = bucket_name
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Recréer le client MinIO dans chaque processus
        minio_client = Minio(
            self.minio_config['endpoint'],
            access_key=self.minio_config['access_key'],
            secret_key=self.minio_config['secret_key'],
            secure=self.minio_config.get('secure', True)
        )

        # Télécharger l'image depuis MinIO
        img_path = self.image_paths[idx]
        img_data = minio_client.get_object(self.bucket_name, img_path)

        # Lire l'image avec PIL
        img = Image.open(BytesIO(img_data.read())).convert("RGB")

        # Appliquer les transformations
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        label = self.labels[idx]
        return img, label

# Étape 2 : Définir les transformations à appliquer aux images
transform = transforms.Compose([
    transforms.Resize((224,224)),  # Rsize à 224,224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normaliser les images
])



# Réentraîner le modèle


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # Optionnel, mais recommandé sous Windows / Obligatoire pour le mutltithreading sinon ça plante et version 2.4.1 avec cu121 + aussi serialisation pour minio sinon il n'arrive pas à lire

    model.train()

    minio_config = {
        'endpoint':"minio:9000",  
        'access_key':"admin",  # Remplacez par votre access key
        'secret_key':"secret123",  # Laisser vide si vous n'avez pas de secret key
        'secure':False
    }

    dataset = CustomImageDataset(image_paths, labels, minio_config, "bucket-images-train", transform=transform)
    
    train_loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    for epoch in tqdm(range(config_classe['nb_epoch'])):
        running_loss = 0.0
        
        for inputs, labels in train_loader:  
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()  # Réinitialiser les gradients

            # Passage avant
            outputs = model(inputs)
        
            # Calcul de la perte
            loss = criterion(outputs, labels)
        
            # Rétropropagation
            loss.backward()
            optimizer.step()  # Mettre à jour les poids

            running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    
    #Sauvegarde du model
    model.eval()

    with mlflow.start_run(experiment_id=experiment.experiment_id, run_name=run_name) as run:
        torch.save(model, os.path.join(base_dir,"vit_model_complete.pth"))
        print(run.info.artifact_uri)
        mlflow.log_artifact(os.path.join(base_dir, "vit_model_complete.pth"), artifact_path="model_image")
    
    


# Test
# for i, (inputs, labels) in enumerate(dataset):
#    print(f"Batch {i+1} - Inputs: {inputs.shape}, Labels: {labels}")
#    if i==10:
#        break
# print("Lancement préparation du model : ")
# print(f"Nombre d'échantillons dans le DataLoader : {len(train_loader)}")
