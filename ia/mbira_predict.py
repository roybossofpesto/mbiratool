#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord prediction.')
parser.add_argument('state_filename', metavar='mbira_good.state', type=str, default="mbira_good.state",
                    help='input state filename')
parser.add_argument('--batch', type=bool, default=False,
                    help='batch mode')

args = parser.parse_args()
print(args)

import torch
import model

state = torch.load(args.state_filename)
net = model.Net(state['nsample'], state['nclass_out'])
net.load_state_dict(state['net_state_dict'])
nclass_out = state['nclass_out']
nsample = state['nsample']
print(net, state)

import alsaaudio as audio

stream = audio.PCM(audio.PCM_CAPTURE,audio.PCM_NONBLOCK)
stream.setchannels(1)
stream.setrate(44100)
stream.setformat(audio.PCM_FORMAT_S16_LE)
chunk_size = stream.setperiodsize(1024)
print(nsample, chunk_size)
assert( nsample == chunk_size )
threshold = .1

import numpy as np
import matplotlib.pyplot as plt

plt.figure()
line, = plt.plot(np.zeros(chunk_size))
level = plt.axhline(0)
plt.axhline(threshold, color="r")
plt.ylim(-1, 1)

while True:
    ll, chunk = stream.read()
    if ll > 0:
        chunk = torch.from_numpy(np.fromstring(chunk, dtype=np.int16).astype(np.float) / 32768.).float().unsqueeze(0).unsqueeze(0)
        foo = chunk.std()
        level.set_ydata(foo)
        line.set_ydata(chunk)
        if foo > threshold:
            prediction = net(chunk)
            confidence, note = prediction.max(1)
            print(note, confidence)
        plt.pause(1e-3)

