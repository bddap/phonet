#!/usr/bin/env python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Input
from os import path
import os

FFT_BINS = 256
HIDDEN_HEIGHT = 256
HIDDEN_WIDTH = 1
MODEL_PATH = "model.h5"
CHECKPOINT_PATH = "training_checkpoints/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = path.dirname(CHECKPOINT_PATH)
TRAINING_AUDIO_DIR = "training_audio"

def num_classes():
    return len(os.listdir(TRAINING_AUDIO_DIR))

def create_model():
    classes = num_classes()
    model = Sequential(
        [keras.layers.Dense(256, activation='relu', input_shape=(FFT_BINS,))] +
        [keras.layers.Dense(HIDDEN_HEIGHT, activation='relu') for _ in range(HIDDEN_WIDTH)] +
        [keras.layers.Dense(classes, activation='softmax')]
    )
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    model = create_model()
    model.save(MODEL_PATH)
    model.save_weights(CHECKPOINT_PATH.format(epoch=0))
