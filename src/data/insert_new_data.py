import os
import pandas as pd
import psycopg2
from minio import Minio
from minio.error import S3Error
from sqlalchemy import create_engine
import joblib
from sklearn.preprocessing import LabelEncoder

# Configuration des chemins
cat_dir = '/new_data/catalogue'
img_dir = '/new_data/images'
encoder_path = os.path.join(os.path.dirname(__file__),'models','label_encoder.pkl')

# Configuration PostgreSQL
db_conf = {
    'user': 'admin',
    'password': 'secret123',
    'host': 'postgre-db',
    'port': '5432',
    'dbname': 'mlops'
}

# Nom des tables et schéma PostgreSQL
table_pg = 'X_train'
schema_pg = 'reduit'
tab_pg_full_name = "X_train"
tab_pg_full_name_y = "y_train"

# Configuration MinIO
minio = {
    'endpoint': 'minio:9000',
    'access_key': 'admin',
    'secret_key': 'secret123',
    'bucket': 'bucket-images-train'
}

# Vérifie que le dossier n'est pas vide
def check_dir_not_empty(path):
    return os.path.isdir(path) and len(os.listdir(path)) > 0

# Vérifie que la table PostgreSQL existe
def check_table_exists(conn, table_name):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = %s 
                AND table_name = %s
            );
        """, (schema_pg, table_name))
        return cur.fetchone()[0]

# Vérifie que le bucket MinIO existe
def check_bucket_exists(minio_client, bucket_name):
    return minio_client.bucket_exists(bucket_name)

# Insère les fichiers CSV dans PostgreSQL avec SQLAlchemy
def insert_csv_to_postgres_with_sqlalchemy(csv_file_path, encoder_path, db_conf, table_name, schema, if_exists='append'):
    """
    :param csv_file_path: Chemin vers le fichier CSV
    :param db_conf: Dictionnaire avec les clés 'user', 'password', 'host', 'port', 'dbname'
    :param table_name: Nom de la table cible
    :param if_exists: 'fail', 'replace', 'append'
    """
    df = pd.read_csv(csv_file_path)

    if df.empty:
        print(f"Le fichier CSV '{csv_file_path}' est vide.")
        return

    if('y_train' in csv_file_path):
        label_encoder = joblib.load(encoder_path)
        df["old_prdtypecode"]=df["prdtypecode"].values
        df["prdtypecode"]=label_encoder.transform(df["prdtypecode"].values)
    
    db_url = f"postgresql://{db_conf['user']}:{db_conf['password']}@{db_conf['host']}:{db_conf['port']}/{db_conf['dbname']}"
    engine = create_engine(db_url)

    # Insérer dans la base de données
    df.to_sql(table_name, engine, if_exists=if_exists, index=False,schema=schema)
    print(f"{len(df)} lignes insérées dans la table '{table_name}'.")

# Upload les fichiers images dans MinIO
def upload_images_to_minio(minio_client, folder, bucket_name):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            file_path = os.path.join(folder, filename)
            minio_client.fput_object(bucket_name, filename, file_path)

# Supprime tous les dossiers du répertoires
def clear_directory(folder_path):
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):  # évite de supprimer des sous-dossiers par erreur
                os.remove(file_path)
        print(f"Tous les fichiers de '{folder_path}' ont été supprimés.")
    else:
        print(f"Le dossier '{folder_path}' n'existe pas.")

# === Main script ===
def main():
    if not check_dir_not_empty(cat_dir):
        print("Le dossier contenant des nouvelles données")
        return
    if not check_dir_not_empty(img_dir):
        print("Le dossier contenant des nouvelles données")
        return

    try:
        # Connexion PostgreSQL
        conn = psycopg2.connect(**db_conf)
        if not check_table_exists(conn, table_pg):
            print(f"La table '{table_pg}' n'existe pas.")
            conn.close()
            return

        # Connexion MinIO
        minio_client = Minio(
            minio['endpoint'],
            access_key=minio['access_key'],
            secret_key=minio['secret_key'],
            secure=False
        )

        if not check_bucket_exists(minio_client, minio['bucket']):
            print(f"Le bucket '{minio['bucket']}' n'existe pas.")
            return

        # Insérer les CSV dans PostgreSQL
        insert_csv_to_postgres_with_sqlalchemy(os.path.join(cat_dir, 'New_X_train.csv'), encoder_path, db_conf, tab_pg_full_name, schema_pg, if_exists='append')
        insert_csv_to_postgres_with_sqlalchemy(os.path.join(cat_dir, 'New_y_train.csv'), encoder_path, db_conf, tab_pg_full_name_y, schema_pg, if_exists='append')

        # Upload les images dans MinIO
        upload_images_to_minio(minio_client, img_dir, minio['bucket'])

        clear_directory(cat_dir)
        clear_directory(img_dir)

        print("Fichiers traités avec succès.")

    except Exception as e:
        print("Erreur :", e)
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    main()
