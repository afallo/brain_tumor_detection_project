import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt


from brain.params import *
from brain.registry import load_model, save_results, save_model, save_model_seg2D, save_data_gcs

from brain.ml_logic_classification.data import load_path_label_df
from brain.ml_logic_classification.encoders import tumor_encoded
from brain.ml_logic_classification.utils import find_and_erase_duplicates
from brain.ml_logic_classification.preprocess import pipeline_building
from brain.ml_logic_classification.model import compile_model_classification, init_densenet_classification

from brain.ml_logic_segmentation_2D.data import load_path_seg_df
from brain.ml_logic_segmentation_2D.utils import find_and_erase_duplicates
from brain.ml_logic_segmentation_2D.preprocess import ColorContrastDataGenerator
from brain.ml_logic_segmentation_2D.model import compile_model_seg2D, init_model_seg2D, predict_and_plot_seg2D

from brain.ml_logic_segmentation_3D.preprocess import process_case

#============= CLASSIFICATION===============================
def preprocess_classification() :

    find_and_erase_duplicates()

    training_df = load_path_label_df(TRAIN_DIR)
    testing_df = load_path_label_df(TEST_DIR)

    #training_df = training_df.iloc[:DATA_SIZE]

    training_df_encoded = tumor_encoded(training_df)
    test_df_encoded = tumor_encoded(testing_df)

    train_df, val_df = train_test_split(training_df_encoded,
    test_size=0.2,
    stratify=training_df_encoded["tumor_type_encoded"],  # keep class balance
    random_state=42)

    train_ds = pipeline_building(train_df, BATCH_SIZE)
    val_ds = pipeline_building(val_df, BATCH_SIZE)
    test_ds = pipeline_building(test_df_encoded, BATCH_SIZE)

    return (train_ds, val_ds, test_ds)


def train_classification(train_ds, val_ds) :


    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())   # récupérer les labels du batch

    n_classes = len(np.unique(all_labels))
    input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
    model = init_densenet_classification(input_shape, n_classes)
    model = compile_model_classification(model)

    es = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=EPOCHS_CLASS,
                        callbacks=es)


    accuracy  = np.min(history.history['val_accuracy'])




    save_results(metrics=dict(accuracy = accuracy))
    save_model(model=model)

    return history, model

def evaluate_classification(model, test_ds) :
    print(model.evaluate(test_ds))



def main_classification() :

    train_ds, val_ds, test_ds = preprocess_classification()
    history, model = train_classification(train_ds, val_ds)
    evaluate_classification(model, test_ds)









#============= SEGMENTATION 2D ===============================



def preprocess_seg2D():

    df = load_path_seg_df(DATA_DIR_SEG)

    # --- 2. SÉPARATION DES DONNÉES (TRAIN / VAL / TEST) ---
    # On divise d'abord en : 85% (Train+Val) et 15% (Test final)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)

    # On recoupe les 85% restants en : 85% Train et 15% Validation
    train_df, val_df = train_test_split(train_val_df, test_size=0.15, random_state=42)

    train_gen_color = ColorContrastDataGenerator(train_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    val_gen_color = ColorContrastDataGenerator(val_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE)
    test_gen_color = ColorContrastDataGenerator(test_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE, shuffle=False)

    # X_batch, y_batch = train_gen_color[0]   # le premier batch

    # # Afficher quelques images + masques
    # n = 5
    # plt.figure(figsize=(12, 6))
    # for i in range(n):
    #     plt.subplot(2, n, i+1)
    #     plt.imshow(X_batch[i])   # image couleur normalisée
    #     plt.axis("off")
    #     plt.title("Image")

    #     plt.subplot(2, n, i+1+n)
    #     plt.imshow(y_batch[i].squeeze(), cmap="gray")  # masque binaire
    #     plt.axis("off")
    #     plt.title("Masque")
    # plt.tight_layout()
    # plt.show()


    return (train_gen_color, val_gen_color, test_gen_color)




def train_seg2D(train_gen_color, val_gen_color) :


    model = init_model_seg2D(IMG_SIZE, IMG_SIZE, 3)
    model= compile_model_seg2D(model)

    unet_callbacks = [
    # 1. EarlyStopping (Arrêt Précoce)
    # Patiente 10 pour laisser le temps au modèle de dépasser les petits plateaux.
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    ),

    # 2. ReduceLROnPlateau (Ralentissement de l'apprentissage)
    # Si la perte stagne pendant 5 époques, on ralentit pour forcer l'ajustement fin.
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,           # Diviser le Learning Rate par 5
        patience=5,
        min_lr=1e-6,          # Vitesse minimale de sécurité
        verbose=1
    )


    # 3. ModelCheckpoint (Sauvegarde du meilleur modèle)
    # Sauvegarde uniquement si le Dice Score de validation s'améliore.
   # tf.keras.callbacks.ModelCheckpoint(
    #    'unet_final_best.keras',
    #    monitor='val_dice_coef',
    #    save_best_only=True,
    #   mode='max',
    #    verbose=1
    #)
]


    history = model.fit(
    train_gen_color,
    validation_data=val_gen_color,
    epochs=EPOCHS_SEG2D, # On augmente car le modèle est plus gros
    callbacks=unet_callbacks,
    verbose=1
)
    #accuracy  = np.min(history.history['val_accuracy'])

    #save_results(metrics=dict(accuracy = accuracy))
    save_model_seg2D(model=model)

    return (history, model)



def evaluate_seg2D(model, test_gen_color) :
    print(model.evaluate(test_gen_color))



def main_seg2D() :

    train_ds, val_ds, test_ds = preprocess_seg2D()
    history, model = train_seg2D(train_ds, val_ds)
    evaluate_seg2D(model, test_ds)

    df = load_path_seg_df(DATA_DIR_SEG)
    # --- 2. SÉPARATION DES DONNÉES (TRAIN / VAL / TEST) ---
    # On divise d'abord en : 85% (Train+Val) et 15% (Test final)
    train_val_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    predict_and_plot_seg2D(model, test_df, n_samples=3)





#============= SEGMENTATION 3D ===============================



def preprocess_seg3D():

    OUT_ROOT_3D.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted([p for p in RAW_ROOT_3D.iterdir() if p.is_dir()])

    for case_dir in case_dirs:
        case_id = case_dir.name  # ex: BraTS-GLI-00000-000
        out_path = OUT_ROOT_3D / f"{case_id}.npz"

        img, seg = process_case(case_dir)
        np.savez_compressed(out_path, image=img, label=seg)
        if DATA_TARGET == "gcs" :
            destination_blob_name = f"segmentation_3D/processed_data/{case_id}.npz"
            save_data_gcs(BUCKET_NAME, destination_blob_name, out_path, from_file=True)
            os.remove(out_path)
        print("Saved", out_path)

    return None



if __name__ == '__main__':
    try:
        preprocess_classification()

    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
