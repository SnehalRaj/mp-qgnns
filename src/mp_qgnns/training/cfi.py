"""Readout and evaluation protocol for the WL-separation task.

The discriminative direction between two CFI graphs sits at a tiny magnitude in
the pooled embedding, dwarfed by a large common mode. We freeze the message
passing, whiten the pooled embeddings on the training split, project onto the
class-mean-difference (Fisher) direction, and train only a one-dimensional
affine. A separability gate zeroes the direction below the WL threshold, so the
model returns chance with exact invariance there.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data_io.cfi import relabel, relabelled_dataset


class FisherReadout(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("std", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("direction", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("midpoint", torch.tensor(0.0, dtype=torch.float64))
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    @torch.no_grad()
    def fit(self, emb: torch.Tensor, y: torch.Tensor) -> None:
        mean = emb.mean(dim=0)
        std = emb.std(dim=0)
        max_std = float(std.max())
        if max_std < 1e-12:
            self.mean.copy_(mean)
            self.std.copy_(torch.full_like(std, float("inf")))
            self.direction.zero_(); self.midpoint.zero_()
            return
        floor = 1e-4 * max_std
        std_eff = torch.clamp(std, min=floor)
        std_eff[std < floor] = float("inf")
        whitened = (emb - mean) / std_eff
        m0 = whitened[y == 0].mean(dim=0)
        m1 = whitened[y == 1].mean(dim=0)
        w = m1 - m0
        norm = torch.linalg.vector_norm(w)
        separable = False
        if float(norm) > 0:
            proj = whitened @ (w / norm)
            within = max(float(proj[y == 0].std()), float(proj[y == 1].std()))
            gap = float((proj[y == 1].mean() - proj[y == 0].mean()).abs())
            separable = gap > 1e3 * (within + 1e-300)
        if separable:
            w = w / norm
            self.direction.copy_(w)
            self.midpoint.copy_(0.5 * (m0 + m1) @ w)
        else:
            self.direction.zero_(); self.midpoint.zero_()
        self.mean.copy_(mean)
        self.std.copy_(std_eff)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        proj = ((emb - self.mean) / self.std) @ self.direction - self.midpoint
        return self.scale * proj + self.bias


def _embed_chunked(model, adj: torch.Tensor, chunk: int = 16) -> torch.Tensor:
    out = []
    for i in range(0, adj.shape[0], chunk):
        with torch.no_grad():
            out.append(model.embed(adj[i:i + chunk]))
    return torch.cat(out, dim=0)


def _run_seed(model_fn, adj0, adj1, n, j, *, seed, n_relabel, epochs, lr, inv_relabel):
    torch.manual_seed(seed)
    A, y = relabelled_dataset(adj0, adj1, n_relabel, seed)
    perm = np.random.default_rng(seed).permutation(A.shape[0])
    A, y = A[perm], y[perm]
    n_tr = int(round(0.6 * A.shape[0]))
    n_va = int(round(0.2 * A.shape[0]))

    model = model_fn()
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    emb_tr = _embed_chunked(model, A[:n_tr])
    emb_va = _embed_chunked(model, A[n_tr:n_tr + n_va])
    emb_te = _embed_chunked(model, A[n_tr + n_va:])
    y_tr, y_va, y_te = y[:n_tr], y[n_tr:n_tr + n_va], y[n_tr + n_va:]

    readout = FisherReadout(model.embed_dim)
    readout.fit(emb_tr, y_tr)
    opt = torch.optim.Adam([readout.scale, readout.bias], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    best_va, best = -1.0, None
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(readout(emb_tr), y_tr)
        loss.backward()
        opt.step()
        if (ep + 1) % max(1, epochs // 5) == 0:
            with torch.no_grad():
                va = ((readout(emb_va) > 0).double() == y_va).double().mean().item()
            if va >= best_va:
                best_va, best = va, {k: v.clone() for k, v in readout.state_dict().items()}
    if best is not None:
        readout.load_state_dict(best)
    with torch.no_grad():
        test_acc = ((readout(emb_te) > 0).double() == y_te).double().mean().item()

    rng = np.random.default_rng(10_000 + seed)
    inv = torch.tensor(np.stack([relabel(np.asarray(adj0), rng) for _ in range(inv_relabel)]),
                       dtype=torch.float64)
    with torch.no_grad():
        logits = readout(_embed_chunked(model, inv))
    spread = float(logits.max() - logits.min())
    return test_acc, spread


def evaluate(model_fn, adj0, adj1, n, j, *, seeds=(0, 1, 2, 3, 4), n_relabel=100,
             epochs=200, lr=1e-2, inv_relabel=50) -> dict:
    """Train and evaluate over seeds. ``model_fn`` builds a fresh model per seed."""
    accs, spreads = [], []
    for s in seeds:
        acc, spread = _run_seed(model_fn, adj0, adj1, n, j, seed=s, n_relabel=n_relabel,
                                epochs=epochs, lr=lr, inv_relabel=inv_relabel)
        accs.append(acc); spreads.append(spread)
    accs = np.array(accs)
    return {
        "j": j,
        "test_acc_mean": float(accs.mean()),
        "test_acc_std": float(accs.std()),
        "invariance_spread_max": float(np.max(spreads)),
        "per_seed_acc": accs.tolist(),
    }
