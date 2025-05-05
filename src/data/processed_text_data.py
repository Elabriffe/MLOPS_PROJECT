import os
import pandas as pd
from sqlalchemy import create_engine
import sys
import pandas as pd
import re
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

base_dir = os.path.dirname(__file__)
utils_dir = os.path.abspath(os.path.join(base_dir, '..' , 'features'))
sys.path.append(utils_dir)

from text_feature_extractor import TextPreprocessor  # Importer la classe du module


# Configuration PostgreSQL
db_user = "admin"
db_password = "secret123"
db_host = "postgre-db"
db_port = "5432"
db_name = "mlops"

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")

base_dir = os.path.dirname(__file__)

#file_model_path = os.path.join(base_dir, "models" , "text_feature_model.joblib")

# Charger les données de train et de test depuis la base de données
df_train = pd.read_sql('SELECT * FROM reduit."X_train" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")
df_test = pd.read_sql('SELECT * FROM reduit."X_test" ORDER BY "Unnamed: 0"', engine, index_col="Unnamed: 0")

# Initialiser le préprocesseur de texte
text_preprocessor = TextPreprocessor(max_features=100)

# Prétraitement et nettoyage des données d'entraînement
df_train_cleaned = text_preprocessor.preprocess_and_clean_data(df_train)
X_train = text_preprocessor.preprocess_and_get_embeddings(df_train_cleaned)
#X_train = text_preprocessor.vectorize_text(df_train_cleaned,file_model_path)  # Vectorisation TF-IDF pour l'entraînement

# Prétraitement et nettoyage des données de test
df_test_cleaned = text_preprocessor.preprocess_and_clean_data(df_test)
X_test = text_preprocessor.preprocess_and_get_embeddings(df_test_cleaned)
#X_test = text_preprocessor.transform_text(df_test_cleaned)  # Transformation TF-IDF pour le test

# Sauvegarder les matrices TF-IDF dans des fichiers .npz
text_preprocessor.save_transformed_data(X_train, os.path.join(base_dir, "processed/text_features_train.npz"))
text_preprocessor.save_transformed_data(X_test, os.path.join(base_dir, "processed/text_features_test.npz"))
