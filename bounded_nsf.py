"""Rational-quadratic spline flow whose support is exactly the prior box.

The MAF built by build_flow_net has a Gaussian base and unconstrained transforms, so
its support is all of R^24 and some of its mass always falls outside the prior box.
This flow maps the box onto the unit cube, puts a uniform base on the cube, and
composes monotone rational-quadratic splines that map [0,1] onto itself, so every draw
is inside the box by construction. See experiments.md for what motivated it.

nflows' PiecewiseRationalQuadraticCouplingTransform is bounded only with tails=None.
The usual tails="linear" form -- what sbi's posterior_nn(model="nsf") builds -- has
linear tails outside the spline interval and is NOT bounded.

Exposes sbi's calling convention (log_prob(x, condition=), sample(shape, condition=))
so it is a drop-in for the MAF at both training and eval call sites.
"""

import numpy as np
import torch
import torch.nn as nn
from nflows.distributions.base import Distribution
from nflows.flows import Flow
from nflows.nn.nets import ResidualNet
from nflows.transforms import (
    CompositeTransform,
    PiecewiseRationalQuadraticCouplingTransform,
    RandomPermutation,
)
from nflows.transforms.base import Transform

# Keeps u strictly inside the spline's domain; the spline raises on inputs outside
# [0, 1] and (theta - lo) / span can land a few ulp past an endpoint.
_EPS = 1e-6


class BoxToUnit(Transform):
    """Map the prior box onto the unit cube, taking logs on the log-uniform dims.

    Taking the log on those dims makes the uniform base equal the prior exactly, and
    puts the flow in the same space the SMC reference samples in.
    """

    def __init__(self, lo, hi, log_dims):
        super().__init__()
        lo = np.asarray(lo, dtype=np.float64).copy()
        hi = np.asarray(hi, dtype=np.float64).copy()
        mask = np.zeros(len(lo), dtype=bool)
        mask[list(log_dims)] = True
        lo[mask], hi[mask] = np.log(lo[mask]), np.log(hi[mask])
        self.register_buffer("lo", torch.tensor(lo, dtype=torch.float32))
        self.register_buffer("span", torch.tensor(hi - lo, dtype=torch.float32))
        self.register_buffer("log_mask", torch.tensor(mask))

    def forward(self, inputs, context=None):
        safe = inputs.clamp_min(_EPS)
        x = torch.where(self.log_mask, torch.log(safe), inputs)
        u = ((x - self.lo) / self.span).clamp(_EPS, 1.0 - _EPS)
        # d(log theta)/d(theta) = 1/theta on the log dims, 1 elsewhere
        ladj = (-torch.log(self.span).sum()
                - (torch.log(safe) * self.log_mask).sum(-1))
        return u, ladj.expand(inputs.shape[0])

    def inverse(self, inputs, context=None):
        x = self.lo + self.span * inputs
        outputs = torch.where(self.log_mask, torch.exp(x), x)
        ladj = torch.log(self.span).sum() + (x * self.log_mask).sum(-1)
        return outputs, ladj


class UnitCubeUniform(Distribution):
    """Uniform on [0, 1]^dim.

    Written out rather than using BoxUniform so that a sample landing a few ulp outside
    the cube gives density 1 instead of -inf, which would poison the batch's loss.
    """

    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        self.register_buffer("_device_anchor", torch.zeros(1))

    def _log_prob(self, inputs, context):
        return inputs.new_zeros(inputs.shape[0])

    def _sample(self, num_samples, context):
        dev = self._device_anchor.device
        if context is None:
            return torch.rand(num_samples, self._dim, device=dev)
        return torch.rand(context.shape[0], num_samples, self._dim, device=dev)


class BoundedNSF(nn.Module):
    """sbi-style wrapper (condition=, sample_shape) over an nflows Flow."""

    def __init__(self, flow):
        super().__init__()
        self._flow = flow

    def log_prob(self, inputs, condition):
        return self._flow.log_prob(inputs, context=condition)

    def sample(self, sample_shape, condition):
        n = (int(sample_shape[0]) if isinstance(sample_shape, (tuple, list, torch.Size))
             else int(sample_shape))
        # nflows gives (n_context, n, dim); sbi's callers expect (n, n_context, dim)
        return self._flow.sample(n, context=condition).transpose(0, 1)


def build_bounded_nsf(context_features, lo, hi, log_dims, num_bins=10,
                      num_transforms=8, hidden_features=256, num_blocks=2):
    dim = len(lo)

    def make_net(in_features, out_features):
        return ResidualNet(in_features, out_features,
                           hidden_features=hidden_features,
                           context_features=context_features,
                           num_blocks=num_blocks)

    layers = [BoxToUnit(lo, hi, log_dims)]
    for i in range(num_transforms):
        mask = torch.zeros(dim)
        mask[i % 2::2] = 1
        layers.append(PiecewiseRationalQuadraticCouplingTransform(
            mask=mask, transform_net_create_fn=make_net, num_bins=num_bins,
            tails=None, apply_unconditional_transform=False))
        layers.append(RandomPermutation(features=dim))
    return BoundedNSF(Flow(CompositeTransform(layers), UnitCubeUniform(dim)))
