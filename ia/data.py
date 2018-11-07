
import numpy as np
import numpy.random as random
import numpy.fft as fft

ts = np.arange(0, 25e-3, 1./44100, dtype=float)

def serie(size, freq, factor=None, phase=None):
    noise = .01 * random.randn(ts.shape[0])
    if freq is None: return np.abs(fft.fft(noise, size))
    if factor is None: factor = 1 + .02 * (random.rand() - .5)
    if phase is None: phase = 2j * np.pi * random.rand()
    freq_ = factor * freq
    ys = np.exp(2j * np.pi * freq_ * ts - phase) + noise
    return np.abs(fft.fft(ys, size))
    #return abs(fft(ys, 512)[256:][::-1])

print("time", ts.shape)

# single class batch
def init_single_class_batch(size, freqs, ntraining):
    samples = []
    targets = []
    for index, freq in enumerate(freqs):
        foo = [serie(size, freq, 1)]
        bar = [index]
        for kk in range(ntraining - 1):
            foo.append(serie(size, freq))
            bar.append(index)
        samples.append(foo)
        targets.append(bar)
    return samples, targets

# multi label batch:
def init_multi_class_batch(size, freqs, ntraining):
    samples = []
    targets = []
    for _ in enumerate(freqs):
        foo = []
        bar = []
        for kk in range(ntraining):
            index = np.random.choice(len(freqs))
            signal = serie(size, freqs[index])
            foo.append(signal)
            bar.append(index)
        samples.append(foo)
        targets.append(bar)
    return samples, targets

