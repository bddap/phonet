#!/usr/bin/env python3

import tensorflow as tf
import json, sys
from train_model import load_model
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import threading
import random

def accept_input():
    global ys
    model = load_model()
    while True:
        ys = json.loads(next(sys.stdin))
        

ys = [
    0.016049889125306832,
    0.03142627116306363,
    0.061594532020572045,
    0.0011315320898924713,
    0.002835836057313255,
    0.0008936219391059839,
    0.005636701454950392,
    0.015600128276281409,
    0.003431237571616324,
    0.013021746188655826,
    8.902699968196344e-05,
    2.8209029837682697e-14,
    0.0395387970281023,
    0.037104672060727234,
    0.007984950899220602,
    1.9316165837158705e-10,
    0.2621427842633086,
    0.0050053973722182085,
    0.006979027639329366,
    0.009184407449820678,
    0.05485523415030031,
    0.008169812340017032,
    0.0004295294534709773,
    0.002530487933223136,
    0.057628238350075967,
    0.2835967616391093,
    0.018363193224141663,
    0.05477618367464839
]

if __name__ == "__main__":

    threading.Thread(target = accept_input).start()
    
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    
    def animate(_i):
        global ys
        xs = list(range(len(ys)))
        
        ax1.clear()
        ax1.plot(xs, ys)

    ani = animation.FuncAnimation(fig, animate, interval=100)
    plt.show()
