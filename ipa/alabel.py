#!/usr/bin/env python3

import os
import json

# sounds = os.listdir("audio")
# sounds = [
#     "Near-close_near-front_unrounded_vowel.ogg",
#     "Close-mid_central_rounded_vowel.ogg",
#     "Close_front_rounded_vowel.ogg",
#     "Open_back_unrounded_vowel.ogg",
#     "Close-mid_front_unrounded_vowel.ogg",
#     "Near-open_central_unrounded_vowel.ogg",
#     "Close-mid_front_rounded_vowel.ogg",
#     "Open-mid_central_unrounded_vowel.ogg",
#     "Open_front_unrounded_vowel.ogg",
#     "Close-mid_central_unrounded_vowel.ogg",
#     "Near-open_front_unrounded_vowel.ogg",
#     "Open-mid_back_rounded_vowel.ogg",
#     "Close_back_unrounded_vowel.ogg",
#     "Near-close_near-front_rounded_vowel.ogg",
#     "Mid-central_vowel.ogg",
#     "Close-mid_back_rounded_vowel.ogg",
#     "Open-mid_front_unrounded_vowel.ogg",
#     "Open-mid_back_unrounded_vowel.ogg",
#     "Open-mid_central_rounded_vowel.ogg",
#     "Close_back_rounded_vowel.ogg",
#     "Close_central_rounded_vowel.ogg",
#     "Open-mid_front_rounded_vowel.ogg",
#     "Close_front_unrounded_vowel.ogg",
#     "Near-close_near-back_rounded_vowel.ogg",
#     "Open_front_rounded_vowel.ogg",
#     "Open_back_rounded_vowel.ogg",
#     "Close_central_unrounded_vowel.ogg",
#     "Close-mid_back_unrounded_vowel.ogg",
# ]
sounds = [s + '.ogg' for s in json.load(open("ordered_names.json"))]

def introduce(soundname):
    os.system('say ' + str(soundname.replace("_", " ").replace(".ogg", "")))
    os.system('timeout 2 vlc ' + 'audio/' + soundname)

    
for sound in sounds[:]:
    print(sound)
    introduce(sound)
    input()
