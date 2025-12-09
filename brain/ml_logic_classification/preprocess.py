import tensorflow as tf
from brain.params import *
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
import numpy as np
from PIL import Image


def load_process_image(path, target_size, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0

    if augment:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    return tf.clip_by_value(img, 0.0, 1.0)


def pipeline_building(df, batch_size) :

    paths = df["image_path"].values.astype(str)
    labels = df["tumor_type_encoded"].values.astype("int32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(len(paths))
    ds = ds.map(
        lambda x, y: (densenet_preprocess_image(x, TARGET_SIZE ,augment=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)

    return ds



def densenet_preprocess_image(path, target_size, augment=True):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32)

    if augment:
        img = tf.image.random_flip_left_right(img)
        # éviter le flip up/down pour les IRM
        img = tf.image.random_brightness(img, max_delta=25.0)
        img = tf.image.random_contrast(img, 0.9, 1.1)

    return densenet_preprocess(img)

def preprocess_for_inference(img: Image.Image, target_size=TARGET_SIZE):
    # Convertir PIL.Image en numpy array
    img = np.array(img)

    print("step1 - conversion numpy:", img.shape)

    # Resize avec TensorFlow
    img = tf.image.resize(img, target_size)

    print("step2 - resize:", img.shape)

    # Cast en float32
    img = tf.cast(img, tf.float32)

    print("step3 - cast:", img.dtype)

    # Normalisation DenseNet
    img = densenet_preprocess(img)

    print("step4 - preprocess DenseNet")

    # Ajouter dimension batch
    img = np.expand_dims(img, axis=0)

    print("step5 - batch shape:", img.shape)

    return img
