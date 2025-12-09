import numpy as np
import cv2
import pandas as pd
import tensorflow as tf

from brain.params import *



class ColorContrastDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, img_size=256, shuffle=True):
        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))

        # Initialisation du CLAHE
        # On l'appliquera uniquement sur la luminosité (L)
        self.clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # ATTENTION : Ici on remet 3 canaux pour la couleur
        X = np.zeros((self.batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)

        for i, idx in enumerate(indexes):
            img_path = self.df.iloc[idx]['image_path']
            mask_path = self.df.iloc[idx]['mask_path']

            # --- 1. CHARGEMENT & TRAITEMENT COULEUR (LAB) ---
            # Lecture standard en couleur (BGR par défaut dans OpenCV)
            img = cv2.imread(img_path)

            if img is not None:
                # A. Conversion BGR -> LAB
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

                # B. Séparation des canaux
                l, a, b = cv2.split(lab)

                # C. Application du CLAHE sur la Luminosité (L)
                l2 = self.clahe.apply(l)

                # D. Fusion et retour en BGR
                lab = cv2.merge((l2, a, b))
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

                # E. Standardisation
                img = cv2.resize(img, (self.img_size, self.img_size))
                img = img / 255.0  # Normalisation 0-1
                # Pas besoin d'expand_dims ici car l'image a déjà 3 canaux
                X[i] = img

            # --- 2. TRAITEMENT MASQUE ---
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (self.img_size, self.img_size))
                mask = mask / 255.0
                mask = (mask > 0.5).astype(np.float32) # Binarisation stricte
                mask = np.expand_dims(mask, axis=-1)
                y[i] = mask

        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
