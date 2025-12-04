from pathlib import Path
import os
import tensorflow as tf


# --- Duplicata managment ---
DATASET_ROOT = Path("./raw_data/classification")  # dossier racine contenant Training/ et Testing/
CLASSES = ["glioma", "pituitary", "meningioma", "notumor"]
SPLITS_ORDER = ["Training", "Testing"]  # Priorité : on garde ce qu'on voit en premier
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")  # adapte si besoin
DRY_RUN = True  # Mettre False pour réellement supprimer les fichiers

# --- data access ---
TRAIN_DIR = Path("./raw_data/classification/Training")
TEST_DIR = Path("./raw_data/classification/Testing")

TARGET_SIZE = (224,224)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
MODEL_TARGET = "local"
BUCKET_NAME = "bucket_name"
