"""Three pre-registered radar MVPs on the real MP-QGNN TSP code path.

Examples
--------
python experiments/radar_2026_11_06.py noise --out results/radar_2026_11_06/noise.json
python experiments/radar_2026_11_06.py depth --out results/radar_2026_11_06/depth.json
python experiments/radar_2026_11_06.py conditional --out results/radar_2026_11_06/conditional.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, beam_search, tour_ratio

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def optimal_lengths(coords, tours):
    return np.asarray([tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))])


def gap(ratio: float) -> float:
    return 100.0 * (float(ratio) - 1.0)


def write_result(path: str, payload: dict):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload["git_sha"] = git_sha()
    payload["elapsed_seconds"] = time.time() - payload.pop("_started")
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_plain(model, coords, heatmaps, epochs=80, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for e in range(epochs):
        opt.zero_grad()
        loss = balanced_bce(model(coords), heatmaps)
        loss.backward()
        opt.step()
        if e in {0, epochs // 2, epochs - 1}:
            losses.append({"epoch": e + 1, "loss": float(loss.detach())})
    return losses


def apply_adjacency(layer, x, coords, noise_std: float):
    """Real unary-register RBS sequence with stochastic angle over-rotation."""
    si, sj = layer.orient_and_sort_edges(coords)
    angles = layer.compute_angles(coords, si, sj)
    if noise_std:
        angles = angles + noise_std * torch.randn_like(angles)
    batch = torch.arange(x.shape[0], device=x.device)
    for e in range(layer.num_edges):
        i, j, theta = si[:, e], sj[:, e], angles[:, e]
        c, s = torch.cos(theta / 2).unsqueeze(-1), torch.sin(theta / 2).unsqueeze(-1)
        xi, xj = x[batch, i, :], x[batch, j, :]
        ie = i.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        je = j.view(-1, 1, 1).expand(-1, 1, x.shape[2])
        x = x.scatter(1, ie, (c * xi - s * xj).unsqueeze(1))
        x = x.scatter(1, je, (s * xi + c * xj).unsqueeze(1))
    return x


class NoisyTSPQGNN(TSPQGNN):
    def __init__(self, *args, angle_noise_std=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.angle_noise_std = angle_noise_std

    def node_embeddings(self, coords):
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](coords)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            x = apply_adjacency(self.adjacency[r], x, coords, self.angle_noise_std)
            x = self.layers[r](x)
        return x


def sampled_tour_ratio(model, coords, lengths, *, angle_noise, total_shots=512, repeats=4,
                       beam_width=1):
    model.eval()
    model.angle_noise_std = angle_noise
    shots = total_shots // repeats
    estimates = []
    with torch.no_grad():
        for _ in range(repeats):
            p = torch.sigmoid(model(coords))
            estimates.append(torch.binomial(torch.full_like(p, shots), p) / shots)
    probs = torch.stack(estimates).mean(0).cpu().numpy()
    c = coords.cpu().numpy()
    ratios = []
    for i in range(len(c)):
        tours = beam_search(probs[i], beam_width)
        ratios.append(min(tour_length(c[i], t) for t in tours) / float(lengths[i]))
    return float(np.mean(ratios))


def run_noise(out):
    payload = {"_started": time.time(), "project": "residual_noise_selection",
               "command": f"python experiments/radar_2026_11_06.py noise --out {out}",
               "data": {"n": 6, "train": 50, "validation": 40, "test": 50},
               "epochs": 100, "shots": 512, "beam_width": 1,
               "target_angle_noise_rad": 0.25,
               "candidate_training_noise_rad": [0.0, 0.08, 0.16, 0.24], "seeds": [0, 1, 2],
               "revision_reason": "Pilot TSP-5 beam-5 gaps were below 1%; use harder TSP-6 greedy decoding.",
               "records": []}
    tr, _, yh = random_instances(6, 50, seed=100)
    va, vt, _ = random_instances(6, 40, seed=101)
    te, tt, _ = random_instances(6, 50, seed=102)
    vl, tl = optimal_lengths(va, vt), optimal_lengths(te, tt)
    for seed in payload["seeds"]:
        candidates = []
        for ns in payload["candidate_training_noise_rad"]:
            torch.manual_seed(seed)
            m = NoisyTSPQGNN(6, num_layers=2, angle_noise_std=ns)
            losses = train_plain(m, tr, yh, epochs=payload["epochs"])
            torch.manual_seed(10_000 + seed)
            val = sampled_tour_ratio(m, va, vl, angle_noise=0.25, beam_width=1)
            candidates.append((val, ns, m, losses))
        val, selected, model, losses = min(candidates, key=lambda x: x[0])
        clean_model = next(x[2] for x in candidates if x[1] == 0.0)
        torch.manual_seed(20_000 + seed)
        base_noisy = sampled_tour_ratio(clean_model, te, tl, angle_noise=0.25, beam_width=1)
        torch.manual_seed(20_000 + seed)
        tuned_noisy = sampled_tour_ratio(model, te, tl, angle_noise=0.25, beam_width=1)
        clean_model.angle_noise_std = 0.0
        model.angle_noise_std = 0.0
        base_clean = tour_ratio(clean_model, te, tl, beam_width=1)
        tuned_clean = tour_ratio(model, te, tl, beam_width=1)
        payload["records"].append({"seed": seed, "selected_training_noise_rad": selected,
            "validation_ratio": val, "baseline_noisy_ratio": base_noisy,
            "selected_noisy_ratio": tuned_noisy, "baseline_clean_ratio": base_clean,
            "selected_clean_ratio": tuned_clean, "baseline_noisy_gap_percent": gap(base_noisy),
            "selected_noisy_gap_percent": gap(tuned_noisy), "loss_trace": losses})
    bg = np.median([r["baseline_noisy_gap_percent"] for r in payload["records"]])
    tg = np.median([r["selected_noisy_gap_percent"] for r in payload["records"]])
    improvement = 100 * (bg - tg) / max(bg, 1e-12)
    clean_delta = np.median([gap(r["selected_clean_ratio"]) - gap(r["baseline_clean_ratio"])
                             for r in payload["records"]])
    payload["summary"] = {"median_baseline_noisy_gap_percent": bg,
                          "median_selected_noisy_gap_percent": tg,
                          "relative_gap_reduction_percent": improvement,
                          "median_clean_gap_delta_points": clean_delta,
                          "decision": "PASS" if improvement >= 10 and clean_delta <= 2 else
                                      ("STOP" if improvement < 3 else "PIVOT")}
    payload["caveat"] = "Gaussian coherent over-rotation plus finite Bernoulli shots; not a hardware noise trace."
    write_result(out, payload)


class StateTSPQGNN(TSPQGNN):
    def forward(self, coords):
        x = self.node_embeddings(coords).reshape(coords.shape[0], -1)
        return x / (torch.linalg.vector_norm(x, dim=-1, keepdim=True) + 1e-12)


def qfim_spectrum(model, coords):
    state_model = StateTSPQGNN(coords.shape[1], num_layers=model.num_layers)
    state_model.load_state_dict(model.state_dict())
    named = dict(state_model.named_parameters())
    qnames = [f"layers.{i}.theta" for i in range(model.num_layers)]
    sizes = [named[n].numel() for n in qnames]
    theta = torch.cat([named[n].detach().reshape(-1) for n in qnames]).requires_grad_(True)
    fixed = {n: p.detach() for n, p in named.items()}
    def states(flat):
        override, pos = dict(fixed), 0
        for n, size in zip(qnames, sizes):
            override[n] = flat[pos:pos + size].view_as(named[n])
            pos += size
        return functional_call(state_model, override, (coords,))
    jac = torch.autograd.functional.jacobian(states, theta).detach()  # B,M,P
    psi = states(theta).detach()
    F = torch.zeros(theta.numel(), theta.numel())
    for b in range(len(coords)):
        J = jac[b]
        proj = J.T @ psi[b]
        F += 4 * (J.T @ J - torch.outer(proj, proj))
    eig = torch.linalg.eigvalsh(F / len(coords)).clamp_min(0).cpu().numpy()[::-1]
    positive = eig[eig > max(eig[0] * 1e-8, 1e-12)] if eig[0] > 0 else np.array([])
    p = eig / max(eig.sum(), 1e-30)
    effective = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    stable = float(eig.sum() / max(eig[0], 1e-30))
    return {"eigenvalues": eig.tolist(), "raw_rank": int(len(positive)),
            "effective_rank": effective, "stable_rank": stable}


def run_depth(out):
    payload = {"_started": time.time(), "project": "data_informed_depth",
               "command": f"python experiments/radar_2026_11_06.py depth --out {out}",
               "data": {"n": 6, "train": 60, "test": 50, "qfim_probe": 4},
               "epochs": 100, "beam_width": 1, "depths": [1, 2, 3], "seeds": [0, 1, 2],
               "revision_reason": "Pilot TSP-5 beam-5 gaps were below 0.4%; use harder TSP-6 greedy decoding.",
               "records": []}
    tr, _, yh = random_instances(6, 60, seed=200)
    te, tt, _ = random_instances(6, 50, seed=201)
    lengths = optimal_lengths(te, tt)
    spectra = {}
    for d in payload["depths"]:
        torch.manual_seed(0)
        probe_model = TSPQGNN(6, num_layers=d)
        spectra[str(d)] = qfim_spectrum(probe_model, tr[:4])
        for seed in payload["seeds"]:
            torch.manual_seed(seed)
            m = TSPQGNN(6, num_layers=d)
            losses = train_plain(m, tr, yh, epochs=payload["epochs"])
            ratio = tour_ratio(m, te, lengths, beam_width=1)
            payload["records"].append({"depth": d, "seed": seed, "tour_ratio": ratio,
                                       "tour_gap_percent": gap(ratio),
                                       "parameters": sum(p.numel() for p in m.parameters()),
                                       "loss_trace": losses})
    payload["qfim"] = spectra
    max_eff = max(v["effective_rank"] for v in spectra.values())
    selected = min(d for d in payload["depths"] if spectra[str(d)]["effective_rank"] >= 0.95 * max_eff)
    med = {d: float(np.median([r["tour_ratio"] for r in payload["records"] if r["depth"] == d]))
           for d in payload["depths"]}
    params = {d: next(r["parameters"] for r in payload["records"] if r["depth"] == d)
              for d in payload["depths"]}
    best = min(med, key=med.get)
    raw_ranks = [spectra[str(d)]["raw_rank"] for d in payload["depths"]]
    passed = med[selected] - med[best] <= 0.02 and params[selected] <= 0.70 * params[best]
    payload["summary"] = {"raw_rank_revision_triggered": raw_ranks == [15, 30, 45],
                          "revision": "raw numerical rank -> entropy effective rank",
                          "selected_depth": selected, "best_depth": best,
                          "median_ratio_by_depth": med,
                          "selected_minus_best_ratio": med[selected] - med[best],
                          "selected_parameter_fraction_of_best": params[selected] / params[best],
                          "decision": "PASS" if passed else "STOP"}
    payload["caveat"] = "Empirical QFIM uses four initialized normalized node states; larger probes may change spectral thresholds."
    write_result(out, payload)


class ConditionalTSPQGNN(TSPQGNN):
    def __init__(self, n_cities, *, condition="none", **kwargs):
        super().__init__(n_cities, **kwargs)
        self.condition = condition
        if condition != "none":
            emb = self.embed[0][-1].out_features
            hidden = self.embed[0][0].out_features
            self.embed = nn.ModuleList(nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, emb))
                                       for _ in range(self.num_layers))

    def conditioned_coords(self, coords):
        if self.condition == "none":
            return coords
        n = coords.shape[1]
        value = float(n) if self.condition == "raw" else 2.0 / float(n)
        return torch.cat([coords, torch.full((*coords.shape[:-1], 1), value,
                                             dtype=coords.dtype, device=coords.device)], -1)

    def node_embeddings(self, coords):
        enc = self.conditioned_coords(coords)
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](enc)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            x = self.adjacency[r](x, coords)
            x = self.layers[r](x)
        return x


def share_learned_modules(source, target):
    target.embed = source.embed
    target.layers = source.layers
    target.head = source.head
    for a, b in zip(source.adjacency, target.adjacency):
        b.angle_mlp = a.angle_mlp


def unique_parameters(models):
    seen, result = set(), []
    for m in models:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p)); result.append(p)
    return result


def joint_train(models, datasets, epochs=80, lr=3e-3):
    opt = torch.optim.Adam(unique_parameters(models), lr=lr)
    trace = []
    for e in range(epochs):
        opt.zero_grad()
        losses = [balanced_bce(m(x), y) for m, (x, y) in zip(models, datasets)]
        loss = torch.stack(losses).mean()
        loss.backward(); opt.step()
        if e in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": e + 1, "loss": float(loss.detach())})
    return trace


def train_family(condition, seed, d5, d6, epochs):
    torch.manual_seed(seed)
    m5 = ConditionalTSPQGNN(5, condition=condition, num_layers=2)
    m6 = ConditionalTSPQGNN(6, condition=condition, num_layers=2)
    share_learned_modules(m5, m6)
    trace = joint_train([m5, m6], [d5, d6], epochs=epochs)
    m7 = ConditionalTSPQGNN(8, condition=condition, num_layers=2)
    share_learned_modules(m5, m7)
    return m5, m6, m7, trace


def run_conditional(out):
    payload = {"_started": time.time(), "project": "conditional_cross_size",
               "command": f"python experiments/radar_2026_11_06.py conditional --out {out}",
               "data": {"train_per_seen_size": 60, "test_sizes": {"5": 35, "6": 35, "8": 25},
                        "train_sizes": [5, 6], "unseen_size": 8},
               "epochs": 100, "beam_width": 1, "seeds": [0, 1, 2],
               "revision_reason": "Pilot TSP-7 beam-5 gaps were near 1%; deploy to TSP-8 with greedy decoding.",
               "records": []}
    tr5, _, y5 = random_instances(5, 60, seed=300)
    tr6, _, y6 = random_instances(6, 60, seed=301)
    tests = {}
    for n, seed, count in [(5, 310, 35), (6, 311, 35), (8, 312, 25)]:
        x, t, _ = random_instances(n, count, seed=seed)
        tests[n] = (x, optimal_lengths(x, t))
    for seed in payload["seeds"]:
        # Transfer baseline: train only on TSP-5, then reuse every learned module at n=7.
        torch.manual_seed(seed)
        b5 = ConditionalTSPQGNN(5, condition="none", num_layers=2)
        btrace = train_plain(b5, tr5, y5, epochs=payload["epochs"])
        b7 = ConditionalTSPQGNN(8, condition="none", num_layers=2); share_learned_modules(b5, b7)
        models = {"tsp5_transfer": (b5, None, b7, btrace)}
        for mode in ["none", "raw", "normalized"]:
            models[f"joint_{mode}"] = train_family(mode, seed, (tr5, y5), (tr6, y6), payload["epochs"])
        for name, (m5, m6, m7, trace) in models.items():
            r5 = tour_ratio(m5, *tests[5], beam_width=1)
            r6 = None if m6 is None else tour_ratio(m6, *tests[6], beam_width=1)
            r7 = tour_ratio(m7, *tests[8], beam_width=1)
            payload["records"].append({"seed": seed, "method": name,
                "ratio_n5": r5, "ratio_n6": r6, "ratio_unseen_n8": r7,
                "gap_n5_percent": gap(r5), "gap_unseen_n8_percent": gap(r7), "loss_trace": trace})
    methods = sorted(set(r["method"] for r in payload["records"]))
    med7 = {m: float(np.median([r["gap_unseen_n8_percent"] for r in payload["records"] if r["method"] == m]))
            for m in methods}
    stronger = min(med7["tsp5_transfer"], med7["joint_none"])
    best_cond = min(["joint_raw", "joint_normalized"], key=lambda m: med7[m])
    improvement = 100 * (stronger - med7[best_cond]) / max(stronger, 1e-12)
    signs = [next(r for r in payload["records"] if r["seed"] == s and r["method"] == best_cond)["gap_unseen_n8_percent"] <
             min(next(r for r in payload["records"] if r["seed"] == s and r["method"] == b)["gap_unseen_n8_percent"]
                 for b in ["tsp5_transfer", "joint_none"]) for s in payload["seeds"]]
    payload["summary"] = {"median_unseen_gap_percent": med7,
                          "first_attempt": "raw graph size", "revision": "bounded 2/n size feature",
                          "best_conditioning": best_cond,
                          "relative_gap_reduction_vs_stronger_baseline_percent": improvement,
                          "seeds_improved": int(sum(signs)),
                          "decision": "PASS" if improvement >= 10 and all(signs) else
                                      ("STOP" if improvement < 3 else "PIVOT")}
    payload["caveat"] = "Small exact TSP splits; conditioned encoder is circuit-compatible input preparation, not hardware execution."
    write_result(out, payload)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", choices=["noise", "depth", "conditional"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    {"noise": run_noise, "depth": run_depth, "conditional": run_conditional}[args.project](args.out)


if __name__ == "__main__":
    main()
