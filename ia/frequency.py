#!/usr/bin/env python3
# coding: utf-8

from pylab import *

ts = arange(0, 25e-3, 1./44100, dtype=float)
freqs = logspace(2.5, 4, 5)
#freqs = array([400])
print("time", ts.shape)
print("freqs", freqs.shape)

def serie(freq, factor=None, phase=None):
    if factor is None: factor = 1 + .1 * (rand() - .5)
    if phase is None: phase = 2j * pi * rand()
    freq_ = factor * freq
    noise = .2 * randn(ts.shape[0])
    ys = exp(2j * pi * freq_ * ts - phase) + noise
    return abs(fft(ys, 512))
    #return abs(fft(ys, 512)[256:][::-1])

nn = 10
samples = []
targets = []
for ident, freq in enumerate(freqs):
    targets.append(ident)
    foo = [serie(freq, 1)]
    for kk in range(nn - 1):
        foo.append(serie(freq))
    samples.append(foo)


print("samples", len(samples))
print("targets", len(targets))

show()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, 16)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        #print('foobar', x.size())
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print(net)
params = list(net.parameters())
print("params", len(params))
print("input_shape", params[0].size())

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=.9)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

for target, sample in zip(targets, samples):
    sample = torch.tensor(sample).unsqueeze(1)
    print("training", target, sample.shape)

sample = torch.randn(10, 1, 47)
target = torch.randn(10, 1, 10)

optimizer.zero_grad()
prediction = net(sample)
print(prediction, prediction.size())
loss = criterion(prediction, target)
print(loss, loss.size())
loss.backward()
optimizer.step()

show()


