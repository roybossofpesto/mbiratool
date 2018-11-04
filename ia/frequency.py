#!/usr/bin/env python3
# coding: utf-8

from pylab import *

ts = arange(0, 5e-2, 1./44100, dtype=float)
freqs = logspace(1.7, 4, 5)
factors = 1 + .2 * (rand(3) - .5)
factors[0] = 1
print("time", ts.shape)
print("freqs/factors", freqs.shape, factors.shape)

data = []
labels = []
for ident, freq in enumerate(freqs):
    labels.append(ident * ones(factors.shape[0]))
    for factor in factors:
        freq_ = factor * freq
        serie = cos(2 * pi * freq_ * ts)+.2*randn(ts.shape[0])
        data.append(abs(fft(serie, 512)[:256]))

data = array(data)
labels = hstack(labels)
print("data", data.shape)
print("labels", labels.shape)

figure()
for serie in data:
    print(serie.shape)
    plot(serie)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(2, 1, 16)
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        print('foobar', x.size())
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

sample = torch.randn(1, 2, 47)
target = torch.randn(1, 1, 10)

optimizer.zero_grad()
prediction = net(sample)
print(prediction, prediction.size())
loss = criterion(prediction, target)
print(loss, loss.size())
loss.backward()
optimizer.step()


show()


