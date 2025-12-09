from pathlib import Path
import os
import tensorflow as tf


MODEL_TARGET = "local"
DATA_TARGET = "local"

BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT= os.environ.get("GCP_PROJECT")
GCP_REGION =os.environ.get("GCP_REGION")
INSTANCE = os.environ.get("INSTANCE")


# --- classification ---
# --- Duplicata managment ---
DATASET_ROOT = Path("./raw_data/classification")  # dossier racine contenant Training/ et Testing/
CLASSES = ["glioma", "pituitary", "meningioma", "notumor"]
SPLITS_ORDER = ["Training", "Testing"]  # Priorité : on garde ce qu'on voit en premier
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")  # adapte si besoin
DRY_RUN = True  # Mettre False pour réellement supprimer les fichiers


TRAIN_DIR = Path("/home/aurore/code/afallo/brain_tumor_detection_project/raw_data/classification/Training")
TEST_DIR = Path("/home/aurore/code/afallo/brain_tumor_detection_project/raw_data/classification/Testing")

TARGET_SIZE = (224,224)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 16

EPOCHS_CLASS = 20

# --- SEG2D ---
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "training_outputs")
MODEL_TARGET = "local"
BUCKET_NAME = "bucket_name"


DATA_DIR_SEG = '/home/aurore/code/afallo/brain_tumor_detection_project/raw_data/segmentation/kaggle_3m/'
IMG_SIZE = 256
EPOCHS_SEG2D = 3

#----- SEG3D ------

RAW_ROOT_3D = Path("/home/aurore/code/afallo/brain_tumor_detection_project/raw_data/segmentation/brats2023_raw")
OUT_ROOT_3D = Path("/home/aurore/code/afallo/brain_tumor_detection_project/raw_data/segmentation/brats2023_preprocessed")


# Taille cible
TARGET_SHAPE_3D = (160, 192, 160)  # (H, W, D)

# On choisit un ordre de canaux cohérent (à garder ensuite dans le Model)
# [T1 = t1 native, T1c, T2w, T2 FLAIR]
MODALITIES_3D = [
    ("t1n", "*-t1n.nii.gz"),
    ("t1c", "*-t1c.nii.gz"),
    ("t2w", "*-t2w.nii.gz"),
    ("t2f", "*-t2f.nii.gz"),
]
