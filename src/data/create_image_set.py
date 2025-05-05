import os
import shutil
import pandas as pd
import numpy as np

# 1️⃣ Demander le nombre total d'images
print("Combien d'images voulez-vous au total pour le train ?")
nb_images_total = int(input())

print("Combien d'images voulez-vous au total pour le test ?")
n_test = int(input())

# 2️⃣ Définir les chemins des fichiers
base_dir = os.path.dirname(__file__)

# Chemins des dossiers d'images
source_dir = os.path.join(base_dir, "raw/image_train")
destination_1 = os.path.join(base_dir, "raw_reduit/image_train")
destination_2 = os.path.join(base_dir, "raw_reduit/image_test")

# Chemins des fichiers CSV
csv_file_X = os.path.join(base_dir, "raw/X_train_update.csv")
csv_file_Y = os.path.join(base_dir, "raw/Y_train_CVw08PX.csv")

x_train_path = os.path.join(base_dir, "raw_reduit/X_train.csv")
y_train_path = os.path.join(base_dir, "raw_reduit/y_train.csv")

x_test_path= os.path.join(base_dir, "raw_reduit/X_test.csv")
y_test_path= os.path.join(base_dir, "raw_reduit/y_test.csv")

# Création des dossiers si besoin
os.makedirs(destination_1, exist_ok=True)
os.makedirs(destination_2, exist_ok=True)

# 3️⃣ Charger les données
df_X = pd.read_csv(csv_file_X, index_col=0)
df_Y = pd.read_csv(csv_file_Y, index_col=0)

# Vérifier si la colonne 'label' est bien dans df_Y
if "prdtypecode" not in df_Y.columns:
    raise ValueError("❌ La colonne 'label' est absente de Y_train ! Vérifiez le fichier.")

# 4️⃣ Calcul de la répartition des images par classe
nb_classes = df_Y["prdtypecode"].nunique()  # Nombre total de classes
images_par_classe = max(1, nb_images_total // nb_classes)  # Nombre d'images à prendre par classe

# 5️⃣ Sélection des images en tenant compte des classes sous-représentées
selected_indices = []
restant = nb_images_total  # Nombre d'images encore à sélectionner

for label, group in df_Y.groupby("prdtypecode"):
    nb_disponible = len(group)  # Nombre d'images disponibles dans cette classe
    nb_a_prendre = min(images_par_classe, nb_disponible)  # Prendre le minimum entre les deux
    
    selected_indices.extend(group.sample(nb_a_prendre).index)
    restant -= nb_a_prendre

    # Si on a atteint le quota total, on arrête
    if restant <= 0:
        break

# Vérifier si on a encore de la place et redistribuer
if restant > 0:
    print(f"🔄 {restant} images encore disponibles, redistribution en cours...")
    available = df_Y.loc[~df_Y.index.isin(selected_indices)]  # Images restantes
    additional_selection = available.sample(min(len(available), restant)).index
    selected_indices.extend(additional_selection)

# 6️⃣ Filtrer X_train et Y_train en fonction des images sélectionnées
df_Y_balanced = df_Y.loc[selected_indices]
df_X_balanced = df_X.loc[selected_indices]

df_X_test = df_X.loc[~df_X.index.isin(df_X_balanced.index)][:n_test]
df_Y_test = df_Y.loc[~df_Y.index.isin(df_Y_balanced.index)][:n_test]

# 7️⃣ Copier les images sélectionnées
def copy_selected_images(df_Y_selected, df_X_balanced, source, destination):
    image_ids = df_X_balanced["imageid"].tolist()  # Récupérer les IDs d'images à copier
    copied_files = 0

    for filename in os.listdir(source):
        if "_" in filename:
            try:
                image_id = int(filename.split("_")[1])  # Extraire l'ID de l'image
                if image_id in image_ids:
                    src_path = os.path.join(source, filename)
                    dst_path = os.path.join(destination, filename)
                    shutil.copy2(src_path, dst_path)
                    copied_files += 1
            except ValueError:
                continue  # Ignorer les fichiers mal formatés

    print(f"✅ {copied_files} fichiers copiés dans {destination}")

# Copie des images
copy_selected_images(df_Y_balanced, df_X_balanced, source_dir, destination_1)
copy_selected_images(df_Y_test, df_X_test, source_dir, destination_2)

# 8️⃣ Sauvegarder les fichiers filtrés
df_X_balanced.to_csv(x_train_path, index=True)
df_Y_balanced.to_csv(y_train_path, index=True)

df_X_test.to_csv(x_test_path, index=True)
df_Y_test.to_csv(y_test_path, index=True)

print(f"✅ Fichiers sauvegardés : X_train ({len(df_X_balanced)}) et y_train ({len(df_Y_balanced)})")

print(f"✅ Fichiers sauvegardés : X_train ({len(df_X_test)}) et y_train ({len(df_Y_test)})")






