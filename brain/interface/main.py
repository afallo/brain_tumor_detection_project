import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


from brain.params import *
from brain.ml_logic.data import load_path_label_df
from brain.ml_logic.encoders import tumor_encoded
from brain.ml_logic.utils import find_and_erase_duplicates
from brain.ml_logic.preprocess import pipeline_building
from brain.ml_logic.model import init_model, compile_model, init_densenet
from brain.registry import load_model, save_results, save_model



def preprocess() :

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


def train(train_ds, val_ds) :


    all_labels = []
    for _, labels in train_ds:
        all_labels.extend(labels.numpy())   # récupérer les labels du batch

    n_classes = len(np.unique(all_labels))
    input_shape = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
    model = init_densenet(input_shape, n_classes)
    model = compile_model(model)

    es = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]

    history = model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=1,
                        callbacks=es)


    accuracy  = np.min(history.history['val_accuracy'])




    save_results(metrics=dict(accuracy = accuracy))
    save_model(model=model)

    return history, model

def evaluate(model, test_ds) :
    print(model.evaluate(test_ds))



def main() :

    train_ds, val_ds, test_ds = preprocess()
    history, model = train(train_ds, val_ds)
    evaluate(model, test_ds)

if __name__ == '__main__':
    try:
        preprocess()
        train()
        pred()
    except:
        import sys
        import traceback

        import ipdb
        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
