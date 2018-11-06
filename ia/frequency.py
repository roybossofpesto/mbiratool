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
    return abs(fft(ys, 256))
    #return abs(fft(ys, 512)[256:][::-1])

nn = 1000
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
        self.conv1 = nn.Conv1d(1, 1, 128)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 7)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        #print('forward', x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = Net()
print(net)
params = list(net.parameters())
print("params", len(params))
print("input_shape", params[0].size())

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=.9)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()

losses = []
for epoch in range(6):
    for index, sample in zip(targets, samples):
        sample = torch.tensor(sample, dtype=torch.float).unsqueeze(1)
        target = torch.zeros(len(sample), 1, 7)
        target[:, :, index] = 1
        print("training", target.shape, sample.shape)

        optimizer.zero_grad()
        prediction = net(sample)
        print("prediction", prediction.size())

        loss = criterion(prediction, target)
        print("loss", loss)
        losses.append(loss)

        loss.backward()
        optimizer.step()

figure()
plot(losses)

for index, freq in enumerate(freqs):
    print("----", index, freq)
    for kk in range(20):
        tensor = torch.from_numpy(serie(freq)).type(torch.float).unsqueeze(0).unsqueeze(1)
        prediction = net(tensor)
        print(tensor.shape, prediction.shape, prediction.max(2), prediction.min())
        print(prediction)
        confidence, index_ = prediction.max(2)
        #print(confidence, index_)

show()


