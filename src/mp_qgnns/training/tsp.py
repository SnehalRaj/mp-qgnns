"""Training, decoding, and tour-ratio evaluation for the TSP model."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..data_io.tsp import tour_length


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Edge BCE reweighted for the tour/non-tour class imbalance."""
    pos = target.sum()
    neg = target.numel() - pos
    weight = torch.where(target > 0.5, neg / (pos + 1e-9), pos / (neg + 1e-9))
    return nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight)


def greedy_tour(edge_probs: np.ndarray) -> np.ndarray:
    """Nearest-by-probability tour starting from city 0."""
    n = edge_probs.shape[0]
    visited = [0]
    while len(visited) < n:
        last = visited[-1]
        order = np.argsort(-edge_probs[last])
        nxt = next(c for c in order if c not in visited)
        visited.append(int(nxt))
    return np.array(visited)


def beam_search(edge_probs: np.ndarray, beam_width: int = 5) -> list[np.ndarray]:
    """Complete tours ranked by joint edge probability, best first."""
    n = edge_probs.shape[0]
    logp = np.log(edge_probs + 1e-12)
    beams = [([0], 0.0)]
    for _ in range(n - 1):
        cand = []
        for path, score in beams:
            last = path[-1]
            for c in range(n):
                if c not in path:
                    cand.append((path + [c], score + logp[last, c]))
        cand.sort(key=lambda x: x[1], reverse=True)
        beams = cand[:beam_width]
    return [np.array(p) for p, _ in beams]


def train(model, coords, heatmaps, *, epochs=200, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = balanced_bce(model(coords), heatmaps)
        loss.backward()
        opt.step()
    return model


def tour_ratio(model, coords, optimal_lengths, beam_width: int = 5) -> float:
    """Mean (decoded tour length / optimal length) over the instances.

    With ``beam_width > 1`` the shortest tour among the beam candidates is taken.
    """
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(coords)).cpu().numpy()
    coords_np = coords.cpu().numpy()
    ratios = []
    for i in range(len(coords_np)):
        if beam_width > 1:
            tours = beam_search(probs[i], beam_width)
            length = min(tour_length(coords_np[i], t) for t in tours)
        else:
            length = tour_length(coords_np[i], greedy_tour(probs[i]))
        ratios.append(length / float(optimal_lengths[i]))
    return float(np.mean(ratios))
