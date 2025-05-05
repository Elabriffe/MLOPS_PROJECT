import boto3
from botocore.client import Config
import os

base_dir = os.path.dirname(__file__)

image_train = os.path.join(base_dir, "raw_reduit/image_train")  # Répertoire pour les 1000 premières images
image_test = os.path.join(base_dir, "raw_reduit/image_test") # Répertoire pour les 200 suivantes


# Configuration MinIO
minio_endpoint = "http://minio:9000"
access_key = "admin"
secret_key = "secret123"
region='eu-west-3'

# Configuration des buckets et dossiers locaux
buckets_and_folders = {
    "bucket-images-train": image_train,  # Remplacez par le chemin réel
    "bucket-images-test": image_test   # Remplacez par le chemin réel
}

# Créer un client S3 compatible avec MinIO
s3 = boto3.client(
    's3',
    endpoint_url=minio_endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    config=Config(signature_version='s3v4'),
    region_name=region
)

# Fonction pour créer un bucket s'il n'existe pas
def create_bucket(bucket_name):
    existing_buckets = [bucket['Name'] for bucket in s3.list_buckets()['Buckets']]
    if bucket_name not in existing_buckets:
        s3.create_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' créé.")
    else:
        print(f"ℹ️  Le bucket '{bucket_name}' existe déjà.")

# Fonction pour uploader un dossier dans un bucket
def upload_folder(bucket_name, folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                file_path = os.path.join(root, file)
                object_name = os.path.relpath(file_path, folder_path).replace("\\", "/")
                s3.upload_file(file_path, bucket_name, object_name)
                print(f"📤 Image '{file}' uploadée dans le bucket '{bucket_name}'.")

# Processus principal
for bucket_name, folder_path in buckets_and_folders.items():
    create_bucket(bucket_name)
    upload_folder(bucket_name, folder_path)

print("✅ Upload terminé pour les deux buckets.")