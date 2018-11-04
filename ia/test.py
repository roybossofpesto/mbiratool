#!/usr/bin/env python3
# coding: utf-8

from pylab import *
import torch
from torch.autograd import Variable

print('cuda', torch.cuda.is_available())
if not torch.cuda.is_available():
    print("can't init cuda")
    exit(1)

device = torch.device("cuda")

# views
x = torch.rand(5, 3)
y = x.view(15)
print(x, x.size())
print(y, y.size())

# inplace operations
x.t_()
print(x)

# numpy and cuda tests
z = torch.from_numpy(np.arange(10)).to(device)
z_ = 10 - z
accum = torch.zeros_like(z, device=device)
for kk in range(10):
    accum += z
    accum += z_
accum = accum.to("cpu")
print(accum)


def plot_convergence(current, energy, xs):
    energy_ = torch.from_numpy(energy).requires_grad_(False).to(device, torch.float)
    xs_ = torch.from_numpy(xs).requires_grad_(False).to(device, torch.float)
    current_ = current.to(device).requires_grad_(True)

    positions = []
    converges = []
    for kk in range(1000):
        print('------------', kk)

        foo = current_.repeat(energy.shape[0], 1) - xs_.repeat(current.shape[0], 1).transpose(0, 1)
        foo = torch.exp(-foo * foo/2)
        bar = torch.mv(foo.transpose(0, 1), energy_).mean()
        converges.append(bar.detach().cpu().numpy())
        bar.backward(torch.tensor(.5, device=device))

        positions.append(current_.detach().cpu().numpy())

        grad_norm = current_.grad.norm().cpu().numpy()
        print(grad_norm)
        if grad_norm < 1e-5:
            break

        current_.data.sub_(current_.grad)

        current_.grad.data.zero_()
    converges = array(converges)
    positions = array(positions).transpose()
    print(positions.shape)

    figure()
    subplot(2,1,1)
    plot(xs, energy)

    subplot(2,1,2)
    xlim(xs.min(), xs.max())
    ys = linspace(0, 1, positions.shape[1])
    for position in positions:
        plot(position, ys)

    figure()
    semilogy(converges)



xs = linspace(-20, 20, 512)
def energy_pit(x0):
    print(x0)
    return array([1.5*log(abs(xx-x0)/1.5) if abs(xx-x0) >= 1.5 else abs(xx-x0)-1.5 if abs(xx-x0) >= 1 else (xx-x0)*(xx-x0)/2-1 for xx in xs])

energy_aa = energy_pit(0) + .8 * energy_pit(8) +  .7 * energy_pit(-10)
energy_bb = energy_pit(0)
init_linspace = torch.linspace(-15, 15, 32, device=device)
init_uniform = torch.rand(256, device=device)*30-15

#plot_convergence(init_linspace, energy_aa, xs)
plot_convergence(init_linspace, energy_aa, xs)
plot_convergence(init_uniform, energy_aa, xs)



show()
