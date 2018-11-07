#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord.')
parser.add_argument('--nepoch', metavar='N', type=int, default=4,
                    help='number of training epoch')
parser.add_argument('--ntraining', metavar='M', type=int, default=25,
                    help='training set size')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
print(args)

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

samples = []
targets = []
for ident, freq in enumerate(freqs):
    targets.append(ident)
    foo = [serie(freq, 1)]
    for kk in range(args.ntraining - 1):
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
        #self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(129, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print('forward', x.size())
        x = x.view(-1, 129)
        #x = self.pool(x, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

net = Net()
print(net)
params = list(net.parameters())
print("params", len(params))
print("input_shape", params[0].size())

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=.9)
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()

losses = []
for epoch in range(args.nepoch):
    print("-", epoch, end=" ")
    loss_accum = 0
    for index, sample in zip(targets, samples):
        sample = torch.tensor(sample, dtype=torch.float).unsqueeze(1)
        target = torch.zeros(sample.shape[0], dtype=torch.long)
        target[:] = index
        #print("training", target.shape, sample.shape)
        #print(target)

        optimizer.zero_grad()
        prediction = net(sample)
        #print("prediction", prediction.size())
        #print(prediction)

        loss = criterion(prediction, target)
        loss_accum += loss.item()
        print(loss.item(), end=" ")

        loss.backward()
        optimizer.step()
    print('--', loss_accum)
    losses.append(loss_accum)

figure()
plot(losses)

for index, freq in enumerate(freqs):
    print("#", index, freq, end=' ')
    accum = 0
    for kk in range(100):
        tensor = torch.from_numpy(serie(freq)).type(torch.float).unsqueeze(0).unsqueeze(1)
        prediction = net(tensor)
        #print(tensor.shape, prediction.shape, prediction.min())
        #print(prediction)
        confidence, index_ = prediction.max(1)
        #print(confidence.item(), index_.item())
        accum += index == index_
    print(accum.item())

show()
