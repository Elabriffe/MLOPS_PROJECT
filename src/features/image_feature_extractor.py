import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from minio import Minio
from io import BytesIO
import numpy as np
import random

class FeatureExtractorWithProjection(nn.Module):
    def __init__(self, model, output_dim=128):
        """
        Initialise un extracteur de caractéristiques avec projection.
        
        Parameters:
        - model : Le modèle pré-entraîné utilisé pour l'extraction des caractéristiques.
        - output_dim : La dimension de sortie de la projection (par défaut 128).
        """
        super(FeatureExtractorWithProjection, self).__init__()
        self.model = model
        self.fc = nn.Linear(192, output_dim)  # La dimension d'entrée est 768 (sortie du ViT)
    
    def forward(self, x):
        """
        Effectue l'extraction des caractéristiques en passant par le modèle
        et en appliquant une projection linéaire.
        
        Parameters:
        - x : L'entrée de l'image.
        
        Returns:
        - La sortie après projection.
        """
        x = self.model.forward_features(x)
        x = self.fc(x.mean(dim=1))  # Utiliser la moyenne des tokens
        return x

class ImageFeatureExtractor:
    def __init__(self, model_path, minio_client, device='cpu', seed=42):
        """
        Initialise l'extracteur de caractéristiques pour les images.

        Parameters:
        - model_path : Le chemin vers le modèle pré-entraîné.
        - minio_client : Le client MinIO pour télécharger les images.
        - device : Le périphérique utilisé pour l'extraction (par défaut 'cpu').
        """
        self.set_seed(seed)  # Fixe la seed dès la création de l'objet
        self.device = torch.device(device)
        self.model = torch.load(model_path, weights_only=False)
        self.model_with_projection = FeatureExtractorWithProjection(self.model).eval().to(self.device)
        self.minio_client = minio_client
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def set_seed(self, seed):
        """
        Fixe les seeds pour garantir des résultats reproductibles.

        Essentiel pour ne pas avoir des résultats différents à chaque extraction
        
        Parameters:
        - seed : La valeur de la seed.
        """
        torch.manual_seed(seed)  # Pour le CPU
        torch.cuda.manual_seed_all(seed)  # Pour le GPU si tu utilises CUDA 
        np.random.seed(seed)  # Pour NumPy
        random.seed(seed)  # Pour Python random module
        torch.backends.cudnn.deterministic = True  # Pour avoir des opérations déterministes
        torch.backends.cudnn.benchmark = False  # Pour éviter les optimisations de performance qui introduisent des non-déterminismes

    def extract_features(self, image_path, bucket_name):
        """
        Extrait les caractéristiques d'une image stockée sur MinIO.
        
        Parameters:
        - image_path : Le chemin de l'image dans le bucket MinIO.
        - bucket_name : Le nom du bucket dans MinIO.
        
        Returns:
        - Les caractéristiques extraites sous forme de vecteur numpy.
        """
        # Télécharger l'image depuis MinIO
        img_data = self.minio_client.get_object(bucket_name, image_path)
        img = Image.open(BytesIO(img_data.read())).convert("RGB")
        
        # Appliquer les transformations
        img = self.transform(img).unsqueeze(0).to(self.device)  # Déplacer l'image sur le GPU
        
        with torch.no_grad():
            # Extraire les features avec la projection linéaire
            features = self.model_with_projection(img)  # Passer par la couche de projection
        return features.cpu().numpy().flatten()  # Retourner les caractéristiques sur CPU

    def extract_features_from_df(self, df, bucket_name, prefix="image"):
        """
        Extrait les caractéristiques des images en utilisant les IDs dans un DataFrame.
        
        Parameters:
        - df : DataFrame contenant les informations d'images (imageid, productid).
        - bucket_name : Nom du bucket MinIO.
        - prefix : Le préfixe des noms de fichiers d'image (par défaut 'image').
        
        Returns:
        - Un tableau numpy des caractéristiques extraites de toutes les images.
        """
        image_features = []
        for imageid, productid in zip(df["imageid"], df["productid"]):
            image_path = f"{prefix}_{imageid}_product_{productid}.jpg"
            image_features.append(self.extract_features(image_path, bucket_name))
        return np.array(image_features)