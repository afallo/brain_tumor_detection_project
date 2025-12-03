# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:55:02 2025

@author: felix
"""

import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers import Input

def initialize_model_with_aug():
    '''Instanciate and return the CNN architecture with light data augmentation'''
    
    model = Sequential()
    
    # Data augmentation légère
    model.add(Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1)
    ]))
    
    # Input layer
    model.add(Input(shape=X.shape[1:]))
    
    # 1st Conv Layer
    model.add(layers.Conv2D(32, (5, 5),
                            padding='same',
                            strides=(1,1),
                            activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(2,2)))
    
    # 2nd Conv Layer
    model.add(layers.Conv2D(32, (5, 5),
                            padding='same',
                            strides=(1,1),
                            activation="relu"))
    model.add(layers.MaxPool2D(pool_size=(4,4)))
    model.add(layers.Dropout(0.25))
    
    # # 3rd Conv Layer + MaxPooling + Dropout
    # model.add(layers.Conv2D(64, (3, 3),
    #                         padding='same',
    #                         strides=(1,1),
    #                         activation="relu"))
    # model.add(layers.MaxPool2D(pool_size=(2,2)))
    
    # Flattening
    model.add(layers.Flatten())
    
    # Output layers
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    
    return model


from tensorflow.keras.metrics import Recall

def compile_model(model):
    '''return a compiled model suited for the CIFAR-10 task'''
    model.compile(optimizer= 'adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])
    return model


model = initialize_model_with_aug()  # <-- utiliser la version avec augmentation
model.summary()

model = compile_model(model)
es = EarlyStopping(patience=3, restore_best_weights=True)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

history = model.fit(X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[es])

    
