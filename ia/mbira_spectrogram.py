#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord recorder.')
parser.add_argument('input', metavar='mbira.series', default="mbira.series",
                    help='output series')

args = parser.parse_args()

import torch
import numpy as np
from pylab import *

print('loading', args.input)
data = torch.load(args.input)
notes = data["notes"]

print(notes.keys(), len(notes))

for note, series in notes.items():
    series = hstack(series)
    figure()
    title("%s %d" % ("note", note + 1))
    subplot(2, 1, 1)
    print(series.shape)
    ts = arange(series.size, dtype=float) / 44100
    specgram(series, NFFT=940, Fs=44100, sides="onesided")
    ylim(0, 2500)
    subplot(2, 1, 2)
    plot(ts, series)
    xlim(ts.min(), ts.max())

show()
