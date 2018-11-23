#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord recorder.')
parser.add_argument('--output', metavar='mbira.series', default="mbira.series",
                    help='output series')
parser.add_argument('--ntraining', metavar='M', type=int, default=200,
                    help='training set size. minimum number of chunk per class.')
parser.add_argument('--nchunk', type=int, default=18,
                    help='number of consecutive chunk above threshold.')
parser.add_argument('--nnote', metavar='C_out', type=int, default=5,
                    help='number of output class. number of note')
parser.add_argument('--threshold', type=float, default=.1,
                    help='standard deviation threshold')
parser.add_argument('--batch', type=bool, default=False,
                    help='batch mode')

args = parser.parse_args()

import alsaaudio as audio
import numpy as np
import matplotlib.pyplot as plt
import torch

stream = audio.PCM(audio.PCM_CAPTURE,audio.PCM_NONBLOCK)
stream.setchannels(1)
stream.setrate(44100)
stream.setformat(audio.PCM_FORMAT_S16_LE)
chunk_size = stream.setperiodsize(1024)
print(chunk_size)
stream.dumpinfo()


if not args.batch:
    plt.figure()
    line, = plt.plot(np.zeros(chunk_size))
    level = plt.axhline(0)
    plt.axhline(args.threshold, color="r")
    plt.ylim(-1, 1)

def get_series(nserie, nchunk):
    series = []
    chunks = []
    while True:
        ll, chunk = stream.read()
        if ll > 0:
            chunk = np.fromstring(chunk, dtype=np.int16).astype(np.float) / 32768.
            foo = chunk.std()
            if not args.batch:
                level.set_ydata(foo)
                line.set_ydata(chunk)
            if foo > args.threshold:
                chunks.append(chunk)
                print(len(chunks), end='\r')
            else:
                if chunks:
                    print()
                if len(chunks) >= nchunk:
                    series += chunks
                    print('!!!', len(series), nserie)
                    if len(series) >= nserie:
                        return series
                chunks = []
        plt.pause(1e-3)

notes = {}
for note in range(args.nnote):
    print('press RETURN to record note %d' % (note + 1))
    #input()
    series = get_series(args.ntraining, args.nchunk)
    notes[note] = series

print('saving', args.output)
torch.save({
    "notes": notes,
    }, args.output)
print('victory')

