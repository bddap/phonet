from tensorflow import keras
from tensorflow.keras import Sequential
from scipy.io import wavfile
import numpy as np
import sys


def create_model():
    FFT_BINS = 480000  # 256
    HIDDEN_HEIGHT = 256
    HIDDEN_WIDTH = 16
    classes = 5
    model = Sequential(
        [keras.layers.Dense(256, activation='relu', input_shape=(FFT_BINS,))] +
        [keras.layers.Dense(HIDDEN_HEIGHT, activation='relu')
         for _ in range(HIDDEN_WIDTH)] +
        [keras.layers.Dense(classes, activation='softmax')]
    )
    model.compile(optimizer="adadelta",
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_training_data():
    ret = []
    for f in [
        "traindat/a.wav",
        "traindat/e.wav",
        "traindat/i.wav",
        "traindat/o.wav",
        "traindat/u.wav",
    ]:
        samplerate, data = wavfile.read(f)
        # TODO: do fft
        ret.append(data)
    return ret


def train(model, training_data):
    xs = np.array(training_data)
    ys = np.array([
        np.array([1, 0, 0, 0, 0]),
        np.array([0, 1, 0, 0, 0]),
        np.array([0, 0, 1, 0, 0]),
        np.array([0, 0, 0, 1, 0]),
        np.array([0, 0, 0, 0, 1]),
    ])
    model.fit(xs, ys, epochs=5000)


def classify(model, wav):
    return model.predict(np.array([wav]))[0]


model = create_model()
training_data = load_training_data()
train(model, training_data)
_, wav = wavfile.read(sys.argv[1])
print(classify(model, wav))
