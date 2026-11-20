"""Three pre-registered MP-QGNN radar MVPs for 20 November 2026."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from math import comb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, tour_ratio

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def lengths(coords, tours):
    return np.asarray([tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))])


def gap(ratio):
    return 100.0 * (float(ratio) - 1.0)


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["git_sha"] = git_sha()
    payload["elapsed_seconds"] = time.time() - payload.pop("_started")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train_model(model, x, y, *, epochs=120, lr=3e-3, extra_loss=None):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    trace = []
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(x)
        loss = balanced_bce(logits, y)
        if extra_loss is not None:
            loss = loss + extra_loss(model, x)
        loss.backward()
        opt.step()
        if epoch in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    return trace


def run_trainability(out):
    payload = {
        "_started": time.time(), "project": "structure_trainability_diagnostic",
        "command": f"python experiments/radar_2026_11_20.py trainability --out {out}",
        "gradient_initializations": list(range(12)), "training_seeds": list(range(5)),
        "data": {"n": 6, "gradient_batch": 18, "train": 50, "test": 70},
        "epochs": 130, "records": [], "size_scaling": [],
    }
    configs = [
        {"name": "k1_l1", "k": 1, "layers": 1},
        {"name": "k1_l2", "k": 1, "layers": 2},
        {"name": "k2_l2", "k": 2, "layers": 2},
        {"name": "k3_l2", "k": 3, "layers": 2},
        {"name": "k3_l3", "k": 3, "layers": 3},
    ]
    gx, _, gy = random_instances(6, 18, seed=610)
    tr, _, ytr = random_instances(6, 50, seed=611)
    te, tours, _ = random_instances(6, 70, seed=612)
    opt_lengths = lengths(te, tours)
    for cfg in configs:
        grads, norms = [], []
        for seed in payload["gradient_initializations"]:
            torch.manual_seed(seed)
            m = TSPQGNN(6, D=6, k=cfg["k"], num_layers=cfg["layers"])
            loss = balanced_bce(m(gx), gy)
            g = torch.autograd.grad(loss, m.adjacency[0].angle_mlp[-1].weight)[0].detach().flatten()
            grads.append(g.numpy())
            norms.append(float(torch.linalg.vector_norm(g)))
        arr = np.stack(grads)
        raw_var = float(arr.var(axis=0, ddof=1).mean())
        params = sum(p.numel() for p in m.parameters())
        normalized_signal = float(np.median(norms) / params)
        gaps = []
        for seed in payload["training_seeds"]:
            torch.manual_seed(seed)
            m = TSPQGNN(6, D=6, k=cfg["k"], num_layers=cfg["layers"])
            trace = train_model(m, tr, ytr, epochs=payload["epochs"])
            g = gap(tour_ratio(m, te, opt_lengths, beam_width=1))
            gaps.append(g)
            payload["records"].append({"config": cfg["name"], "seed": seed, "gap_percent": g,
                                       "parameters": sum(p.numel() for p in m.parameters()),
                                       "loss_trace": trace})
        cfg.update({"raw_gradient_variance": raw_var, "median_gradient_norm": float(np.median(norms)),
                    "normalized_gradient_signal": normalized_signal,
                    "median_gap_percent": float(np.median(gaps)), "parameters": params})

    # Size scaling for the representative k=3, two-layer architecture.
    for n in range(4, 9):
        x, _, y = random_instances(n, 14, seed=620 + n)
        gs = []
        for seed in payload["gradient_initializations"]:
            torch.manual_seed(seed)
            m = TSPQGNN(n, D=6, k=3, num_layers=2)
            loss = balanced_bce(m(x), y)
            gs.append(torch.autograd.grad(loss, m.adjacency[0].angle_mlp[-1].weight)[0].detach().flatten().numpy())
        a = np.stack(gs)
        payload["size_scaling"].append({"n": n, "raw_gradient_variance": float(a.var(0, ddof=1).mean())})

    gaps = np.asarray([c["median_gap_percent"] for c in configs])
    raw = np.log10(np.asarray([c["raw_gradient_variance"] for c in configs]) + 1e-30)
    norm = np.log10(np.asarray([c["normalized_gradient_signal"] for c in configs]) + 1e-30)
    rho_raw = float(spearmanr(raw, gaps).statistic)
    rho_norm = float(spearmanr(norm, gaps).statistic)
    size_x = np.asarray([r["n"] for r in payload["size_scaling"]])
    size_y = np.log10(np.asarray([r["raw_gradient_variance"] for r in payload["size_scaling"]]) + 1e-30)
    slope = float(np.polyfit(size_x, size_y, 1)[0])
    selected = configs[int(np.argmax(raw))]
    best_gap = float(gaps.min())
    distance = selected["median_gap_percent"] - best_gap
    go = rho_raw <= -0.6 and distance <= 1 and slope > -0.5
    stop = rho_raw > 0 or distance > 2 or slope < -1
    payload["configs"] = configs
    payload["summary"] = {
        "spearman_raw_variance_vs_gap": rho_raw,
        "spearman_normalized_signal_vs_gap": rho_norm,
        "size_log10_variance_slope_per_city": slope,
        "raw_diagnostic_selection": selected["name"],
        "selection_gap_to_best_points": distance,
        "revision": "raw component variance -> parameter-normalized gradient signal",
        "decision": "PASS" if go else ("STOP" if stop else "PIVOT"),
    }
    payload["caveat"] = "Empirical gradient sampling is not the paper's irrep projector theorem and uses exact reduced-basis gradients rather than QPU estimates."
    save(out, payload)


def graph_descriptors(coords):
    d = torch.cdist(coords, coords)
    n = coords.shape[1]
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=coords.device), diagonal=1)
    e = d[:, mask]
    return torch.stack([e.mean(1), e.std(1), e.min(1).values, e.max(1).values], dim=1)


class FixedBackboneMoE(nn.Module):
    """Classically routed fixed MP-QGNN backbones; only heads and gate train."""
    def __init__(self, n, *, temperature=0.03, uniform=False):
        super().__init__()
        self.temperature = temperature
        self.uniform = uniform
        self.experts = nn.ModuleList([
            TSPQGNN(n, D=6, k=1, num_layers=1),
            TSPQGNN(n, D=6, k=2, num_layers=2),
            TSPQGNN(n, D=6, k=3, num_layers=3),
        ])
        for expert in self.experts:
            for name, p in expert.named_parameters():
                p.requires_grad = name.startswith("head.")
        self.gate = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3))
        if uniform:
            for p in self.gate.parameters():
                p.requires_grad = False
        self.last_weights = None

    def expert_logits(self, coords):
        outs = []
        for e in self.experts:
            with torch.no_grad():
                z = e.node_embeddings(coords)
            outs.append(e.head(z))
        return torch.stack(outs, dim=1)

    def weights(self, coords):
        if self.uniform:
            return torch.full((len(coords), 3), 1 / 3, dtype=coords.dtype, device=coords.device)
        return torch.softmax(self.gate(graph_descriptors(coords)) / self.temperature, dim=-1)

    def forward(self, coords):
        w = self.weights(coords)
        self.last_weights = w
        return (self.expert_logits(coords) * w[:, :, None, None]).sum(1)


class SingleFixedExpert(nn.Module):
    def __init__(self, expert):
        super().__init__(); self.expert = expert

    def forward(self, coords):
        with torch.no_grad():
            z = self.expert.node_embeddings(coords)
        return self.expert.head(z)


def transfer_moe(model, n):
    out = FixedBackboneMoE(n, temperature=model.temperature, uniform=model.uniform)
    state = {k: v for k, v in model.state_dict().items() if "edge_pairs" not in k}
    out.load_state_dict(state, strict=False)
    return out


def effective_experts(model, x):
    with torch.no_grad():
        w = model.weights(x)
        return float(torch.exp(-(w * torch.log(w + 1e-12)).sum(1)).mean())


def run_moe(out):
    payload = {
        "_started": time.time(), "project": "fixed_mpqgnn_soft_moe",
        "command": f"python experiments/radar_2026_11_20.py moe --out {out}",
        "seeds": list(range(5)), "data": {"train_n5": 70, "test_n5": 70, "transfer_test_n6": 80},
        "epochs": 150, "records": [],
    }
    tr, _, ytr = random_instances(5, 70, seed=630)
    te5, tours5, _ = random_instances(5, 70, seed=631); opt5 = lengths(te5, tours5)
    te6, tours6, _ = random_instances(6, 80, seed=632); opt6 = lengths(te6, tours6)
    for seed in payload["seeds"]:
        methods = {}
        torch.manual_seed(seed)
        sharp = FixedBackboneMoE(5, temperature=0.03)
        methods["soft_sharp"] = (sharp, train_model(sharp, tr, ytr, epochs=payload["epochs"]))
        torch.manual_seed(seed)
        revised = FixedBackboneMoE(5, temperature=0.3)
        entropy_bonus = lambda m, x: 0.01 * (m.weights(x) * torch.log(m.weights(x) + 1e-12)).sum(1).mean()
        methods["soft_regularized"] = (revised, train_model(revised, tr, ytr, epochs=payload["epochs"], extra_loss=entropy_bonus))
        torch.manual_seed(seed)
        uniform = FixedBackboneMoE(5, uniform=True)
        methods["uniform"] = (uniform, train_model(uniform, tr, ytr, epochs=payload["epochs"]))

        # Train each fixed-backbone head alone, using identical backbone initialization.
        torch.manual_seed(seed)
        base = FixedBackboneMoE(5)
        for i in range(3):
            single = SingleFixedExpert(copy.deepcopy(base.experts[i]))
            methods[f"single_k{i+1}"] = (single, train_model(single, tr, ytr, epochs=payload["epochs"]))

        for name, (model, trace) in methods.items():
            g5 = gap(tour_ratio(model, te5, opt5, beam_width=1))
            if isinstance(model, FixedBackboneMoE):
                m6 = transfer_moe(model, 6)
                eff = effective_experts(model, te5)
            else:
                k = int(name[-1])
                expert6 = TSPQGNN(6, D=6, k=k, num_layers=k)
                state = {kk: vv for kk, vv in model.expert.state_dict().items() if "edge_pairs" not in kk}
                expert6.load_state_dict(state, strict=False)
                m6 = SingleFixedExpert(expert6); eff = 1.0
            g6 = gap(tour_ratio(m6, te6, opt6, beam_width=1))
            payload["records"].append({"seed": seed, "method": name, "gap_n5_percent": g5,
                                       "gap_transfer_n6_percent": g6, "effective_experts": eff,
                                       "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                       "loss_trace": trace})
    methods = sorted({r["method"] for r in payload["records"]})
    med = {m: {k: float(np.median([r[k] for r in payload["records"] if r["method"] == m]))
               for k in ["gap_n5_percent", "gap_transfer_n6_percent", "effective_experts"]} for m in methods}
    singles = [m for m in methods if m.startswith("single")]
    best_single = min(singles, key=lambda m: med[m]["gap_transfer_n6_percent"])
    revised = "soft_regularized"
    denom = min(med[best_single]["gap_transfer_n6_percent"], med["uniform"]["gap_transfer_n6_percent"])
    rel = 100 * (denom - med[revised]["gap_transfer_n6_percent"]) / max(denom, 1e-12)
    wins = 0
    for seed in payload["seeds"]:
        rr = next(r for r in payload["records"] if r["seed"] == seed and r["method"] == revised)
        ss = min(r["gap_transfer_n6_percent"] for r in payload["records"] if r["seed"] == seed and r["method"] in singles)
        uu = next(r["gap_transfer_n6_percent"] for r in payload["records"] if r["seed"] == seed and r["method"] == "uniform")
        wins += rr["gap_transfer_n6_percent"] < min(ss, uu)
    in_delta = med[revised]["gap_n5_percent"] - med[best_single]["gap_n5_percent"]
    go = rel >= 10 and wins >= 4 and in_delta <= 0.5
    stop = rel <= 0 or wins < 3 or (med[revised]["effective_experts"] < 1.5 and rel < 5)
    payload["summary"] = {"median": med, "best_single": best_single,
                          "relative_transfer_gain_percent": rel, "paired_seed_wins": wins,
                          "in_size_delta_vs_best_single_points": in_delta,
                          "revision": "sharp temperature -> moderate temperature plus entropy bonus",
                          "decision": "PASS" if go else ("STOP" if stop else "PIVOT")}
    payload["caveat"] = "The message backbones are exact reduced-basis simulations and fixed at random initialization; only classical heads and routing are trained."
    save(out, payload)


class FeatureResponseWrapper(nn.Module):
    def __init__(self, model, A, b, inverse=None):
        super().__init__(); self.model = model; self.register_buffer("A", A); self.register_buffer("b", b)
        self.inverse = inverse

    def forward(self, coords):
        x = self.model.node_embeddings(coords)
        y = x @ self.A.T + self.b
        if self.inverse is not None:
            y = (y - self.inverse[1]) @ self.inverse[0].T
        return self.model.head(y)


def response_and_calibration(dim, seed, shots=256):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    spectrum = np.geomspace(0.08, 0.8, dim)
    A = q @ np.diag(spectrum) @ q.T
    b = rng.normal(scale=0.025, size=dim)
    # Zero and unit basis probes; finite calibration noise mimics estimated features.
    sigma = 0.35 / np.sqrt(shots)
    y0 = b + rng.normal(scale=sigma, size=dim)
    cols = []
    for i in range(dim):
        cols.append(A[:, i] + b + rng.normal(scale=sigma, size=dim) - y0)
    Ahat = np.stack(cols, axis=1)
    naive = np.linalg.pinv(Ahat)
    probes = rng.normal(size=(200, dim))
    observed = probes @ A.T + b
    lambdas = np.logspace(-5, 0, 16)
    errors, inverses = [], []
    for lam in lambdas:
        inv = np.linalg.solve(Ahat.T @ Ahat + lam * np.eye(dim), Ahat.T)
        pred = (observed - y0) @ inv.T
        errors.append(float(np.mean((pred - probes) ** 2))); inverses.append(inv)
    j = int(np.argmin(errors))
    return (torch.tensor(A), torch.tensor(b),
            (torch.tensor(naive), torch.tensor(y0)),
            (torch.tensor(inverses[j]), torch.tensor(y0)),
            float(lambdas[j]), float(errors[j]))


def run_calibration(out):
    payload = {
        "_started": time.time(), "project": "reusable_feature_calibration",
        "command": f"python experiments/radar_2026_11_20.py calibration --out {out}",
        "seeds": list(range(5)), "data": {"n": 6, "train": 60, "test": 80},
        "epochs": 150, "calibration": {"feature_dim": 20, "shots_per_probe": 256, "probe_settings": 21},
        "records": [],
    }
    tr, _, ytr = random_instances(6, 60, seed=640)
    te, tours, _ = random_instances(6, 80, seed=641); opt = lengths(te, tours)
    for seed in payload["seeds"]:
        torch.manual_seed(seed)
        model = TSPQGNN(6, D=6, k=3, num_layers=2)
        trace = train_model(model, tr, ytr, epochs=payload["epochs"])
        A, b, naive, ridge, lam, cal_mse = response_and_calibration(comb(6, 3), 700 + seed, 256)
        variants = {
            "ideal": model,
            "noisy": FeatureResponseWrapper(model, A, b),
            "naive_inverse": FeatureResponseWrapper(model, A, b, naive),
            "ridge_inverse": FeatureResponseWrapper(model, A, b, ridge),
        }
        for name, m in variants.items():
            payload["records"].append({"seed": seed, "method": name,
                                       "gap_percent": gap(tour_ratio(m, te, opt, beam_width=1)),
                                       "ridge_lambda": lam if name == "ridge_inverse" else None,
                                       "heldout_calibration_mse": cal_mse if name == "ridge_inverse" else None,
                                       "loss_trace": trace if name == "ideal" else None})
    methods = ["ideal", "noisy", "naive_inverse", "ridge_inverse"]
    med = {m: float(np.median([r["gap_percent"] for r in payload["records"] if r["method"] == m])) for m in methods}
    denom = med["noisy"] - med["ideal"]
    recovery = 100 * (med["noisy"] - med["ridge_inverse"]) / max(denom, 1e-12)
    good, harmful = 0, 0
    for seed in payload["seeds"]:
        row = {r["method"]: r["gap_percent"] for r in payload["records"] if r["seed"] == seed}
        induced = row["noisy"] - row["ideal"]
        recovered = row["noisy"] - row["ridge_inverse"]
        good += induced > 0 and recovered >= 0.5 * induced
        harmful += row["ridge_inverse"] > row["noisy"] + 0.5
    go = recovery >= 50 and good >= 4 and harmful == 0
    stop = recovery < 10 or harmful >= 3
    payload["summary"] = {"median_gap_percent": med, "median_recovery_percent": recovery,
                          "seeds_recovering_at_least_half": good, "seeds_harmed_over_half_point": harmful,
                          "revision": "direct pseudoinverse -> held-out-selected ridge inverse",
                          "decision": "PASS" if go else ("STOP" if stop else "PIVOT")}
    payload["caveat"] = "The response is a controlled synthetic local-feature channel, not a measured IBM calibration; the MP-QGNN model and task path are real."
    save(out, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("project", choices=["trainability", "moe", "calibration"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    {"trainability": run_trainability, "moe": run_moe, "calibration": run_calibration}[args.project](args.out)


if __name__ == "__main__":
    main()
