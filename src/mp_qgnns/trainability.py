"""Gradient-variance probe for the HW-preserving ansatz.

Barren plateaus show up as a gradient variance that vanishes exponentially with
the number of qubits. The compound-pyramid (RBS) ansatz preserves Hamming weight
and stays in the C(D, k)-dimensional subspace; at fixed k this subspace grows
polynomially in D, so the gradient variance decays only polynomially.
``gradient_variance`` measures Var[d C / d theta] at random initialisation for a
fixed-observable cost; ``variance_sweep`` runs it across register sizes at fixed
k.
"""
from __future__ import annotations

from math import comb

import numpy as np
import torch

from .core.compound import CompoundPyramidLayer


def gradient_variance(D: int, k: int, num_layers: int = 3, num_samples: int = 200,
                      seed: int = 0) -> float:
    """Variance of dC/dtheta over random initialisations.

    The state starts in the first weight-k basis vector, is evolved by
    ``num_layers`` compound-pyramid layers, and the cost is the probability of
    returning to that basis vector.
    """
    dim = comb(D, k)
    x0 = torch.zeros(1, dim, dtype=torch.float64)
    x0[0, 0] = 1.0
    grads = []
    for s in range(num_samples):
        torch.manual_seed(1000 * seed + s)
        layers = [CompoundPyramidLayer(D, k, init_std=0.3).double() for _ in range(num_layers)]
        x = x0
        for layer in layers:
            x = layer(x)
        cost = (x[0, 0]) ** 2
        cost.backward()
        grads.append(float(layers[0].theta.grad[0]))
    return float(np.var(grads))


def variance_sweep(sizes, k: int = 2, num_layers: int = 3, num_samples: int = 200) -> list[dict]:
    """Gradient variance at fixed ``k`` for each register size D in ``sizes``."""
    out = []
    for D in sizes:
        out.append({"D": D, "k": k, "dim": comb(D, k),
                    "grad_var": gradient_variance(D, k, num_layers, num_samples)})
    return out
