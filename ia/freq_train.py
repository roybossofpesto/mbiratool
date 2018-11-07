#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord.')
parser.add_argument('--nepoch', metavar='E', type=int, default=50,
                    help='number of training epoch')
parser.add_argument('--nsample', metavar='S', type=int, default=128,
                    help='dimension of training space')
parser.add_argument('--ntraining', metavar='M', type=int, default=200,
                    help='training set size')
parser.add_argument('--nclass_in', metavar='C_in', type=int, default=5,
                    help='number of input class')
parser.add_argument('--nclass_out', metavar='C_out', type=int, default=2,
                    help='number of outpu class')
parser.add_argument('--ntest', metavar='N', type=int, default=2000,
                    help='test set size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=1,
                    help='learning rate')
parser.add_argument('--batch', type=bool, default=False,
                    help='batch mode')

args = parser.parse_args()
args.nclass_out = max(args.nclass_out, args.nclass_in+1)
print(args)

import numpy as np
import matplotlib.pyplot as plt

freqs = list(np.logspace(2.5, 4, args.nclass_in))
freqs.append(None)
#freqs = array([400])
print("freqs", len(freqs))

import data
#samples, targets = init_single_class_batch()
samples, targets = data.init_multi_class_batch(args.nsample, freqs, args.ntraining)
print("samples", len(samples))
print("targets", len(targets))

import model
import torch

net = model.Net(args.nsample, args.nclass_out)
print("net", net)
params = list(net.parameters())
print("params", len(params))
print("input_shape", params[0].size())

optimizer = model.Optimizer(net.parameters(), lr=args.learning_rate)
criterion = model.Criterion()

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
        #print(loss.item(), end=" ")

        loss.backward()
        step = optimizer.step()
    print(loss_accum)
    losses.append(loss_accum)

if not args.batch:
    plt.figure()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.semilogy(losses)

corr = []
for index, freq in enumerate(freqs):
    print("#", index, freq, end=' ')
    accum = torch.zeros(args.nclass_out)
    for kk in range(args.ntest):
        tensor = torch.from_numpy(data.serie(args.nsample, freq)).type(torch.float).unsqueeze(0).unsqueeze(1)
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
perfect = np.all(corr == 100 * np.eye(corr.shape[0], corr.shape[1], dtype=int))

print(corr, corr.shape)
print("PERFECT !!!!!!" if perfect else ":(")

if perfect:
    torch.save({
        'freqs': freqs,
        'nsample': args.nsample,
        'nclass_in': args.nclass_in,
        'nclass_out': args.nclass_out,
        'net_state_dict': net.state_dict(),
        }, "perfect.state")

if not args.batch:
    plt.show()
