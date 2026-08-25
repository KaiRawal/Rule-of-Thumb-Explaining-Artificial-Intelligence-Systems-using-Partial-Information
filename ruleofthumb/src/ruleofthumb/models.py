"""Auxiliary RoT model variants.

Faithful port of ``Linear_regression``, the per-point sub-models,
``RoT_additive`` and ``rand_order`` from the original experiment code.
"""

import torch
from torch import nn

from ruleofthumb.core import RoT


class Linear_regression(RoT):
    def project(self):
        self.b.data[:] = 0


class per_point_NAM(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        width = 128
        self.A = nn.Parameter(torch.randn((width,), requires_grad=True))
        self.A.data /= 2
        self.A.data += 3
        self.a = nn.Parameter(torch.randn((width), requires_grad=True))
        self.B = nn.Parameter(torch.zeros((width, classes), requires_grad=True))
        # self.offset=torch.randn(classes,requires_grad=True)
        self.non_lin = torch.nn.SELU()

    def forward(self, x):
        x = self.non_lin(x[:, None] - self.a[None, :])
        x = x * torch.exp(self.A[None, :])
        x = x.mm(self.B)  # +self.offset[None,:]
        return x


class per_point_RBF(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        width = 32
        self.A = nn.Parameter(torch.ones((width,), requires_grad=True))
        self.A.data[:] = 0.1
        self.a = nn.Parameter(torch.randn((width), requires_grad=True))
        self.a.data *= 5
        self.B = nn.Parameter(torch.zeros((width, classes), requires_grad=True))

    def forward(self, x):
        x = torch.exp(-((x[:, None] - self.a[None, :]) ** 2) / self.A[None] ** 2)
        x = x.mm(self.B)  # +self.offset[None,:]
        return x


class per_point_poly(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.width = 6
        self.A = nn.Parameter(torch.zeros((self.width, classes), requires_grad=True))

    def forward(self, x):
        x = x[:, None].pow(torch.arange(self.width)[None, :])
        x = x.mm(self.A)
        return x


class RoT_additive(RoT):
    def __init__(self, classes, sample_shape, dropout_rate=0.5, sub_model=per_point_NAM):

        super().__init__(classes, sample_shape, dropout_rate)
        assert len(sample_shape) == 1
        self.fns = [sub_model(classes) for i in range(sample_shape[0])]  # Not handling non-vector for now
        for (i, f) in enumerate(self.fns):
            self.add_module("feature " + str(i), f)

    def importance(self, points):
        out = torch.empty(points.shape[0], self.classes, points.shape[1])
        for i in range(points.shape[1]):
            out[:, :, i] = self.fns[i](points[:, i])
        out += super().importance(points)
        return out


def rand_order(points):
    rand = torch.rand_like(points)
    rand = rand.reshape(points.shape[0], -1)
    order = torch.argsort(rand, 1)
    order = order.reshape(points.shape)
    return order
