#!/usr/bin/env python3
# coding: utf-8

import torch
import scipy.io.wavfile as wav
import numpy as np

data = torch.load("mbira.series")
notes = data["notes"]

print(notes.keys(), len(notes))

for note, series in notes.items():
    filename = "note%d.wav" % note
    wav.write(filename, 44100, (np.hstack(series) * 32768).astype(np.int16))
    print("wrote", filename)
