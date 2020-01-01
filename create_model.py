#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, Input
import json
from os import path

FFT_BINS = 256
HIDDEN_HEIGHT = FFT_BINS
CLASSES = 2
MODEL_PATH = "model.h5"


def create_model():
    model = Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(256,)),
        keras.layers.Dense(HIDDEN_HEIGHT, activation='relu'),
        keras.layers.Dense(CLASSES, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


if path.exists(MODEL_PATH):
    print("The file", MODEL_PATH, "already exists. Refusing to overwrite")
    exit(1)

model = create_model()
model.save(MODEL_PATH)


## Training can be done in a separate script.

# training_data = json.load(open("traindat.json"))
# assert(max(t["class"] for t in training_data) == CLASSES - 1)
# assert(all(len(t["freqs"]) == FFT_BINS for t in training_data))

# xs = [t["freqs"] for t in training_data]
# ys = [t["class"] for t in training_data]

# model.fit(xs, ys, epochs=5000)
