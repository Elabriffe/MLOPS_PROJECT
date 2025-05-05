from transformers import CamembertTokenizer, CamembertModel
import torch
import pandas as pd
import re
import string
import spacy
import numpy as np
import joblib

# Charger SpaCy pour le prétraitement
nlp = spacy.load("fr_core_news_sm")

class TextPreprocessor:
    def __init__(self, max_features=100):
        """
        Initialise le préprocesseur de texte et le modèle CamemBERT.
        
        :param max_features: Nombre maximal de caractéristiques à utiliser
        """
        # Charger le tokenizer et le modèle CamemBERT de Hugging Face
        self.tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.model = CamembertModel.from_pretrained("camembert-base")
        
    def clean_text(self, text):
        """
        Fonction de nettoyage du texte : suppression de la ponctuation, des chiffres,
        et lemmatisation avec SpaCy.
        
        :param text: Texte brut à nettoyer.
        :return: Texte nettoyé.
        """
        text = text.lower()  # Convertir en minuscules
        text = re.sub(r"\d+", "", text)  # Supprimer les chiffres
        text = text.translate(str.maketrans("", "", string.punctuation))  # Supprimer la ponctuation
        
        # Traitement avec SpaCy (lemmatisation et suppression des stopwords)
        doc = nlp(text, disable=["ner", "parser"])
        text = " ".join([token.lemma_ for token in doc if not token.is_stop])  # Lemmatisation et stopwords
        
        return text

    def preprocess_and_clean_data(self, df):
        """
        Applique le nettoyage du texte et génère une colonne 'clean_text' dans le DataFrame.
        
        :param df: DataFrame contenant les colonnes 'designation' et 'description'
        :return: DataFrame avec la colonne 'clean_text' ajoutée
        """
        # Vérification des colonnes nécessaires
        if "designation" in df.columns and "description" in df.columns:
            df["clean_text"] = df["designation"].fillna("") + " " + df["description"].fillna("")  # Concatenate
        # Appliquer la fonction de nettoyage
        df["clean_text"] = df["clean_text"].apply(self.clean_text)
        
        return df

    def get_camembert_embeddings(self, text):
        """
        Extrait les embeddings de texte à l'aide de CamemBERT.
        
        :param text: Texte à transformer en embeddings
        :return: Embeddings de CamemBERT pour le texte
        """
        # Tokenisation du texte avec le tokenizer de CamemBERT
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Passer les entrées dans le modèle CamemBERT
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extraire la représentation du dernier token caché (embeddings de la phrase)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Moyenne des embeddings de tous les tokens
        
        return embeddings

    def preprocess_and_get_embeddings(self, df):
        """
        Applique le nettoyage et génère des embeddings CamemBERT pour chaque texte.
        
        :param df: DataFrame contenant la colonne 'clean_text'
        :return: Liste d'embeddings
        """
        embeddings = []
        for text in df["clean_text"]:
            embedding = self.get_camembert_embeddings(text)
            embeddings.append(embedding.numpy())  # Convertir en numpy array pour utilisation ultérieure
        
        return np.vstack(embeddings)  # Retourner un tableau 2D de tous les embeddings

    def save_transformed_data(self, embeddings, file_path):
        """
        Sauvegarde les embeddings dans un fichier `.npz`.
        
        :param embeddings: Embeddings obtenus
        :param file_path: Chemin du fichier de sauvegarde
        """
        np.savez(file_path, embeddings)


