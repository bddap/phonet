#!/usr/bin/env python3

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D, Dense, Reshape, MaxPooling1D)
import tensorflow as tf
from os import path
import os
import numpy as np
import random


AUDIO_SLICE_SIZE = 256 * 32
KERNEL_SIZE = 256
HIDDEN_HEIGHT = 256
HIDDEN_WIDTH = 8
MODEL_PATH = "model.h5"
CHECKPOINT_PATH = "training_checkpoints/cp-{epoch:04d}.ckpt"
CHECKPOINT_DIR = path.dirname(CHECKPOINT_PATH)
TRAINING_AUDIO_DIR = "training_audio"


def num_classes():
    return len(os.listdir(TRAINING_AUDIO_DIR))


class Rfft(tf.keras.layers.Layer):
    "fft on reals, output is real"

    def call(self, inputs):
        comp = tf.signal.rfft(inputs)
        real = tf.math.real(comp)
        return real


def identity_kernel(shape, dtype):
    ret = [[[1.0 if c == a else 0.0
             for a in range(shape[2])]
            for b in range(shape[1])]
           for c in range(shape[0])]
    return np.array(ret)


class Convolution(tf.keras.layers.Layer):
    def __init__(self, kernel_len, input_len):
        super(Convolution, self).__init__()
        self.kernel_len = kernel_len
        self.input_len = input_len

    def call(self, inputs):
        # TODO: create convolution layer
        print(inputs.shape)
        exit()


def convolution(kernel_len, input_len):
    return [
        Reshape((input_len, 1), input_shape=(input_len,)),
        Conv1D(
            kernel_len,
            kernel_len,
            use_bias=False,
            kernel_initializer=identity_kernel,
            input_shape=(input_len, 1),
            trainable=False,
            activation="linear"
        ),
    ]


def expected_convolution(arr, out_len):
    def e():
        assert len(arr) >= out_len
        for i in range(len(arr) - out_len + 1):
            yield arr[i:i+out_len]
    return list(e())


def test_convolution():
    assert expected_convolution([0, 1, 2], 2) == [[0, 1], [1, 2]]

    kernel_len = 1
    inplen = 100
    s = Sequential(convolution(kernel_len, inplen))
    xs = [[random.random() for i in range(inplen)] for i in range(400)]
    ys = np.array([expected_convolution(x, kernel_len) for x in xs])
    ps = s.predict(xs)
    for (yi, pi) in zip(ys, ps):
        for (y, p) in zip(yi, pi):
            assert abs(p - y) < 1.0e-06


def create_model():
    classes = num_classes()

    lays = []
    lays += [Reshape((AUDIO_SLICE_SIZE,), input_shape=(AUDIO_SLICE_SIZE,))]
    # lays += convolution(KERNEL_SIZE, AUDIO_SLICE_SIZE)
    lays += [Convolution(KERNEL_SIZE, AUDIO_SLICE_SIZE)]
    lays += [Rfft()]
    for _ in range(HIDDEN_WIDTH):
        lays += [Dense(HIDDEN_HEIGHT, activation='relu')]
    lays += [Dense(classes, activation='softmax')]
    lays += [MaxPooling1D(AUDIO_SLICE_SIZE - KERNEL_SIZE + 1)]
    lays += [Reshape((classes,))]
    lays += [Dense(classes, activation='softmax')]

    model = Sequential(lays)
    model.compile(
        optimizer="adadelta",
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    if os.environ.get("TEST_CREATE_MODEL") == "1":
        test_convolution()
        exit()

    model = create_model()
    model.build()
    model.summary()
    model.save(MODEL_PATH)
    model.save_weights(CHECKPOINT_PATH.format(epoch=0))
