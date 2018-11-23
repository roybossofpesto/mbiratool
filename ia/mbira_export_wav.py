#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord recorder.')
parser.add_argument('input', metavar='mbira.series', default="mbira.series",
                    help='output series')

args = parser.parse_args()

import torch
import scipy.io.wavfile as wav
import numpy as np

print('loading', args.input)
data = torch.load(args.input)
notes = data["notes"]

print(notes.keys(), len(notes))

for note, series in notes.items():
    filename = "note%d.wav" % (note + 1)
    wav.write(filename, 44100, (np.hstack(series) * 32768).astype(np.int16))
    print("wrote", filename)
