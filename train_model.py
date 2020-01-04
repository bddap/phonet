#!/usr/bin/env python3

import tensorflow as tf
import json, gzip
from os import path
from create_model import FFT_BINS, HIDDEN_HEIGHT, MODEL_PATH, CHECKPOINT_PATH
from create_model import CHECKPOINT_DIR, num_classes

def load_model():
    if not path.exists(CHECKPOINT_DIR):
        print("The directory", CHECKPOINT_DIR, "does not exist. Run ./create_model.py to create it.")
        exit(1)
    if not path.exists(MODEL_PATH):
        print("The file", MODEL_PATH, "does not exist. Run ./create_model.py to create it.")
        exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    model.load_weights(latest)
    return model

def load_training_data():
    with gzip.GzipFile("traindat.json.gz") as fd:
        training_data = json.load(fd)
    xs = [t["freqs"] for t in training_data]
    ys = [t["class"] for t in training_data]
    return (xs, ys)

def train(model, xs, ys):
    # load latest training checkpoint if any
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = CHECKPOINT_PATH,
        verbose = 1, 
        save_weights_only = True,
        save_freq = len(xs) * 10,
    )
    model.fit(xs, ys, callbacks=[cp_callback], epochs=5000)

if __name__ == "__main__":
    print("loading model")
    model = load_model()
    print("loading training data")
    (xs, ys) = load_training_data()
    print("starting training")
    train(model, xs, ys)
