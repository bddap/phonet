#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import gzip
import json
from os import path
from create_model import (
    MODEL_PATH, CHECKPOINT_PATH, CHECKPOINT_DIR,
    AUDIO_SLICE_SIZE, KERNEL_SIZE
)


def load_model():
    if not path.exists(CHECKPOINT_DIR):
        print("The directory", CHECKPOINT_DIR,
              "does not exist. Run ./create_model.py to create it.")
        exit(1)
    if not path.exists(MODEL_PATH):
        print("The file", MODEL_PATH,
              "does not exist. Run ./create_model.py to create it.")
        exit(1)

    model = tf.keras.models.load_model(MODEL_PATH)
    latest = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    model.load_weights(latest)
    return model


def slice(xs):
    # slice into AUDIO_SLICE_SIZE length pieces, with KERNEL_SIZE overlap
    # this may end up cutting of the ends of training samples
    assert len(xs) >= AUDIO_SLICE_SIZE

    def e():
        for i in range(0, len(xs), AUDIO_SLICE_SIZE - KERNEL_SIZE):
            yield xs[i:i+AUDIO_SLICE_SIZE]
    return list(e())


def load_training_data():
    with gzip.GzipFile("traindat.json.gz") as fd:
        training_data = json.load(fd)

    rxs = []
    rys = []

    for xs, ys in training_data:
        sxs = slice(xs)
        rxs += sxs
        rys += [ys for ys in range(len(sxs))]

    return (rxs, rys)


def train(model, xs, ys):
    # load latest training checkpoint if any
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        verbose=1,
        save_weights_only=True,
        save_freq=len(xs),
    )
    model.fit(xs, ys, callbacks=[cp_callback], epochs=5000)


if __name__ == "__main__":
    print("loading model")
    model = load_model()
    print("loading training data")
    (xs, ys) = load_training_data()
    print("starting training")
    train(model, xs, ys)
