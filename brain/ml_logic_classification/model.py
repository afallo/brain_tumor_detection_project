from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121



#inception transfer learning model
def init_model_classification(input_shape, num_classes):

    inception = InceptionV3(input_shape=input_shape,
                            weights='imagenet',
                            include_top=False)

    for layer in inception.layers:
        layer.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = inception(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model



def init_densenet_classification(input_shape, num_classes):
    print("MODEL DENSENET")
    base_model = DenseNet121(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # première phase : backbone gelé
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)

    return model

def compile_model_classification(model):
    '''return a compiled model suited for the CIFAR-10 task'''
    model.compile(optimizer= 'adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    return model
