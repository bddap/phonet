import tensorflow as tf
from tensorflow.keras import Sequential
from scipy.io import wavfile
import numpy as np
import sys
import percache
import os

keras = tf.keras

# loading and fft is expensive and single-threaded so we memoize results
cache = percache.Cache("fftcache", livesync=True)

INPUT_SIZE = 2400  # length of time domain inputs
INPUT_STEP = 99  # length of time domain inputs
WEIGHT_FILE = 'weights'


def create_model():
    FFT_BINS = INPUT_SIZE // 2 + 1
    HIDDEN_HEIGHT = INPUT_SIZE * 2
    HIDDEN_WIDTH = 16
    classes = 5
    model = Sequential(
        [keras.layers.Dense(
            HIDDEN_HEIGHT,
            activation='relu',
            input_shape=(FFT_BINS,))] +
        [keras.layers.Dense(HIDDEN_HEIGHT, activation='relu')
         for _ in range(HIDDEN_WIDTH)] +
        [keras.layers.Dense(classes, activation='softmax')]
    )
    model.compile(optimizer="adadelta",
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_training_data():
    tags = {
        "a": [1, 0, 0, 0, 0],
        "e": [0, 1, 0, 0, 0],
        "i": [0, 0, 1, 0, 0],
        "o": [0, 0, 0, 1, 0],
        "u": [0, 0, 0, 0, 1],
    }
    xs, ys = [], []
    for fn in os.listdir("traindat"):
        for spectrum in load_spectra("traindat/" + fn, INPUT_SIZE, INPUT_STEP):
            xs.append(spectrum)
            ys.append(np.array(tags[fn[0]]))
    return np.array(xs), np.array(ys)


def train(model, training_data):
    xs, ys = training_data
    model.fit(xs, ys, epochs=1)


def try_load_weights(model):
    try:
        model.load_weights(WEIGHT_FILE)
    except tf.errors.NotFoundError:
        pass


def classify(model, wav):
    return model.predict(np.array([wav]))[0]


def rfftr(reals):
    "fft, real to real becuse we dont care about phase"
    # cached because it's expesive
    return np.array([n.real for n in np.fft.rfft(reals)])


@cache
def load_spectra(filename, input_size, input_step):
    ret = []
    samplerate, data = wavfile.read(filename)
    for i in range(0, len(data) - input_size, input_step):
        ret.append(rfftr(np.array(data[i:i+input_size])))
    return np.array(ret)


model = create_model()
training_data = load_training_data()
try_load_weights(model)
for _ in range(1000):
    train(model, training_data)
    model.save_weights(WEIGHT_FILE)
    for testfile in sorted(os.listdir('testdat')):
        spec = load_spectra('testdat/' + testfile, INPUT_SIZE, INPUT_STEP)
        print(testfile, sum(model.predict(spec)).astype(int))
