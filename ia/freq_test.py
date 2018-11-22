#!/usr/bin/env python3
# coding: utf-8

import argparse

parser = argparse.ArgumentParser(description='freq autochord prediction.')
parser.add_argument('--ntest', metavar='N', type=int, default=2000,
                    help='test set size')
parser.add_argument('state_filename', metavar='perfect.state', type=str, default="perfect.state",
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
freqs = state['freqs']
nclass_out = state['nclass_out']
nsample = state['nsample']

import data
import numpy as np

corr = []
for index, freq in enumerate(freqs):
    print("#", index, freq, end=' ')
    accum = torch.zeros(nclass_out)
    for kk in range(args.ntest):
        tensor = torch.from_numpy(data.serie(nsample, freq)).type(torch.float).unsqueeze(0).unsqueeze(1)
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

