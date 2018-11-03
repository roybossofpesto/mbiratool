#!/usr/bin/env python3
# coding: utf-8

import torch
print('cuda', torch.cuda.is_available())
x = torch.rand(5, 3)
print(x)


