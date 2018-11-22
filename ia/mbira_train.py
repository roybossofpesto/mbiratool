#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='mbira autochord trainer.')
parser.add_argument('--input', metavar='input.series', default="mbira.series",
                    help='input series')
parser.add_argument('--nepoch', metavar='E', type=int, default=50,
                    help='number of training epoch')
parser.add_argument('--nsample', metavar='S', type=int, default=128,
                    help='dimension of training space')
parser.add_argument('--ntraining', metavar='M', type=int, default=200,
                    help='training set size')
parser.add_argument('--nclass_out', metavar='C_out', type=int, default=2,
                    help='number of outpu class')
parser.add_argument('--ntest', metavar='N', type=int, default=2000,
                    help='test set size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=1,
                    help='learning rate')
parser.add_argument('--batch', type=bool, default=False,
                    help='batch mode')

args = parser.parse_args()

import torch

print("loading", args.input)
data = torch.load(args.input)
notes = data["notes"]
args.nclass_in = len(notes)
args.nclass_out = max(args.nclass_out, args.nclass_in)
print(notes.keys())
print(args)

import data

samples, targets = data.init_from_notes(notes)
print("samples", len(samples), sum(list(map(len, samples))))
print("targets", len(targets), sum(list(map(len, targets))))

import model

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

import numpy as np
import matplotlib.pyplot as plt

if not args.batch:
    plt.figure()
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.semilogy(losses)

final_loss = losses[-1]
perfect = final_loss < 1e-3

print(final_loss)
print("PERFECT !!!!!!" if perfect else ":(")

if perfect:
    torch.save({
        'nsample': args.nsample,
        'nclass_in': args.nclass_in,
        'nclass_out': args.nclass_out,
        'net_state_dict': net.state_dict(),
        }, "mbira_good.state")

if not args.batch:
    plt.show()
