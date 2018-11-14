#!/usr/bin/env python3
# coding: utf-8

import alsaaudio as audio
import numpy as np
import matplotlib.pyplot as plt

stream = audio.PCM(audio.PCM_CAPTURE,audio.PCM_NONBLOCK)
stream.setchannels(1)
stream.setrate(44100)
stream.setformat(audio.PCM_FORMAT_S16_LE)
chunk_size = stream.setperiodsize(1024)
print(chunk_size)
threshold = .1

plt.figure()
line, = plt.plot(np.zeros(chunk_size))
level = plt.axhline(0)
plt.axhline(threshold)
plt.ylim(-1, 1)

captured = 0
while True:
    ll, chunk = stream.read()
    if ll > 0:
        chunk = np.fromstring(chunk, dtype=np.int16).astype(np.float) / 32768.
        foo = chunk.std()
        level.set_ydata(foo)
        line.set_ydata(chunk)
        if foo > threshold:
            captured += 1
            print(captured)
        else:
            captured = 0
    plt.pause(1e-3)

plt.show()
