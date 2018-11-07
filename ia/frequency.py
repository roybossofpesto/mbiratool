#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord.')
parser.add_argument('--nepoch', metavar='E', type=int, default=50,
                    help='number of training epoch')
parser.add_argument('--ntraining', metavar='M', type=int, default=200,
                    help='training set size')
parser.add_argument('--nclass_in', metavar='C_in', type=int, default=5,
                    help='number of input class')
parser.add_argument('--nclass_out', metavar='C_out', type=int, default=2,
                    help='number of outpu class')
parser.add_argument('--ntest', metavar='N', type=int, default=2000,
                    help='test set size')
parser.add_argument('--sum', dest='accumulate', action='store_const',
                    const=sum, default=max,
                    help='sum the integers (default: find the max)')

args = parser.parse_args()
args.nclass_out = max(args.nclass_out, args.nclass_in+1)
print(args)

from pylab import *

def serie(freq, factor=None, phase=None):
    noise = .2 * randn(ts.shape[0])
    if freq is None: return abs(fft(noise, 256))
    if factor is None: factor = 1 + .1 * (rand() - .5)
    if phase is None: phase = 2j * pi * rand()
    freq_ = factor * freq
    ys = exp(2j * pi * freq_ * ts - phase) + noise
    return abs(fft(ys, 256))
    #return abs(fft(ys, 512)[256:][::-1])

ts = arange(0, 25e-3, 1./44100, dtype=float)
freqs = list(logspace(2.5, 4, args.nclass_in))
freqs.append(None)
#freqs = array([400])
print("time", ts.shape)
print("freqs", len(freqs))


# single class batch
def init_single_class_batch():
    samples = []
    targets = []
    for index, freq in enumerate(freqs):
        foo = [serie(freq, 1)]
        bar = [index]
        for kk in range(args.ntraining - 1):
            foo.append(serie(freq))
            bar.append(index)
        samples.append(foo)
        targets.append(bar)
    return samples, targets

# multi label batch:
def init_multi_class_batch():
    samples = []
    targets = []
    for _ in enumerate(freqs):
        foo = []
        bar = []
        for kk in range(args.ntraining):
            index = np.random.choice(len(freqs))
            signal = serie(freqs[index])
            foo.append(signal)
            bar.append(index)
        samples.append(foo)
        targets.append(bar)
    return samples, targets

#samples, targets = init_single_class_batch()
samples, targets = init_multi_class_batch()

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
        self.fc2 = nn.Linear(64, args.nclass_out)
        self.fc3 = nn.Linear(args.nclass_out, args.nclass_out)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print('forward', x.size())
        x = x.view(-1, 129)
        #x = self.pool(x, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

net = Net()
print(net)
params = list(net.parameters())
print("params", len(params))
print("input_shape", params[0].size())

optimizer = optim.SGD(net.parameters(), lr=0.05, momentum=.1)
criterion = nn.CrossEntropyLoss()
#criterion = nn.MSELoss()

losses = []
for epoch in range(args.nepoch):
    print("-", epoch, "--", end=" ")
    loss_accum = 0
    for target, sample in zip(targets, samples):
        sample = torch.tensor(sample, dtype=torch.float).unsqueeze(1)
        target = torch.tensor(target, dtype=torch.long)
        #print("training", target.shape, target.dtype, sample.shape, sample.dtype)
        #print(target)

        optimizer.zero_grad()
        prediction = net(sample)
        #print("prediction", prediction.size())
        #print(prediction)

        loss = criterion(prediction, target)
        loss_accum += loss.item()
        print(loss.item(), end=" ")

        loss.backward()
        step = optimizer.step()
    print('--', loss_accum)
    losses.append(loss_accum)

figure()
ylabel("loss")
xlabel("epoch")
semilogy(losses)

corr = []
for index, freq in enumerate(freqs):
    print("#", index, freq, end=' ')
    accum = torch.zeros(args.nclass_out)
    for kk in range(args.ntest):
        tensor = torch.from_numpy(serie(freq)).type(torch.float).unsqueeze(0).unsqueeze(1)
        prediction = net(tensor)
        #print(tensor.shape, prediction.shape, prediction.min())
        #print(prediction.shape)
        confidence, index_ = prediction.max(1)
        #print(confidence.item(), index_.item())
        accum[index_] += 1
    accum = 100*accum/args.ntest
    corr.append(accum)
    print(accum)
corr = torch.stack(corr).type(torch.int).numpy()
perfect = np.all(corr == 100 * eye(corr.shape[0], corr.shape[1], dtype=int))

print(corr, corr.shape)
print("PERFECT !!!!!!" if perfect else ":(")

#show()
