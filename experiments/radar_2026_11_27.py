"""Three pre-registered, real MP-QGNN radar MVPs for 27 November 2026."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mp_qgnns.core.compound import CompoundPyramidLayer, compound_matrix
from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import EdgeHead, TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, greedy_tour, tour_ratio

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


def train_model(model, x, y, *, epochs=140, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        loss = balanced_bce(model(x), y)
        loss.backward(); opt.step()
        if epoch in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    return trace


def train_head(head, features, y, *, epochs=120, lr=5e-3):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    trace = []
    for epoch in range(epochs):
        opt.zero_grad()
        loss = balanced_bce(head(features), y)
        loss.backward(); opt.step()
        if epoch in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    return trace


def gap_from_logits(logits, coords, optimal_lengths):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    xy = coords.detach().cpu().numpy()
    ratios = []
    for i in range(len(xy)):
        t = greedy_tour(probs[i])
        ratios.append(tour_length(xy[i], t) / float(optimal_lengths[i]))
    return gap(np.mean(ratios))


def transfer_tsp(model, n):
    """Rebuild an n-city model while retaining every size-shared parameter."""
    if isinstance(model, CollectiveTSPQGNN):
        out = CollectiveTSPQGNN(n, D=model.D, k=model.k, num_layers=model.num_layers,
                                collective_init=model.collective_init)
    else:
        out = TSPQGNN(n, D=model.layers[0].D, k=model.layers[0].k,
                      num_layers=model.num_layers)
    state = {k: v for k, v in model.state_dict().items() if "edge_pairs" not in k}
    out.load_state_dict(state, strict=False)
    return out


# ---------------------------------------------------------------------------
# A. Free-fermion / Slater boundary


def slater_vectors(q, subsets):
    """Normalized k-particle Slater coordinates from orthonormal orbitals.

    q: [R, M, D, k], subsets: [C(D,k), k] -> [R, M, C(D,k)].
    """
    blocks = q[:, :, subsets, :]
    v = torch.linalg.det(blocks)
    return v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-12)


def closest_slater(x, D, k, *, steps=90, restarts=3, seed=0):
    """Project a batch of real fixed-k states onto the nearest Slater manifold."""
    shape = x.shape
    target = x.reshape(-1, x.shape[-1])
    target = target / (torch.linalg.vector_norm(target, dim=-1, keepdim=True) + 1e-12)
    if k == 1:
        return target.reshape(shape), torch.ones(shape[:-1], dtype=x.dtype)
    subsets = torch.tensor(list(combinations(range(D), k)), dtype=torch.long)
    g = torch.Generator().manual_seed(seed)
    raw = nn.Parameter(torch.randn(restarts, len(target), D, k, generator=g) * 0.5)
    opt = torch.optim.Adam([raw], lr=0.08)
    for _ in range(steps):
        opt.zero_grad()
        q = torch.linalg.qr(raw, mode="reduced").Q
        v = slater_vectors(q, subsets)
        overlap = torch.einsum("rmc,mc->rm", v, target)
        loss = (1.0 - overlap.square()).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        q = torch.linalg.qr(raw, mode="reduced").Q
        v = slater_vectors(q, subsets)
        overlap = torch.einsum("rmc,mc->rm", v, target)
        best = overlap.square().argmax(dim=0)
        ar = torch.arange(len(target))
        proj = v[best, ar]
        signed = torch.sign((proj * target).sum(-1, keepdim=True) + 1e-12)
        proj = proj * signed
        fidelity = (proj * target).sum(-1).square()
    return proj.reshape(shape), fidelity.reshape(shape[:-1])


def run_fermionic(out):
    payload = {
        "_started": time.time(), "project": "slater_dequantization_boundary",
        "command": f"python experiments/radar_2026_11_27.py fermionic --out {out}",
        "seeds": list(range(5)), "data": {"n": 6, "train": 60, "test": 70},
        "model": {"D": 6, "k": 3, "layers": 2}, "epochs": 140,
        "projection": {"steps": 90, "restarts": 3}, "records": [],
    }
    tr, _, ytr = random_instances(6, 60, seed=2710)
    te, tours, _ = random_instances(6, 70, seed=2711); opt_lengths = lengths(te, tours)
    for seed in payload["seeds"]:
        torch.manual_seed(seed)
        model = TSPQGNN(6, D=6, k=3, num_layers=2)
        trace = train_model(model, tr, ytr, epochs=payload["epochs"])
        model.eval()
        with torch.no_grad():
            ztr = model.node_embeddings(tr)
            zte = model.node_embeddings(te)
        ptr, ftr = closest_slater(ztr, 6, 3, steps=90, restarts=3, seed=100 + seed)
        pte, fte = closest_slater(zte, 6, 3, steps=90, restarts=3, seed=200 + seed)

        raw_head = copy.deepcopy(model.head)
        proj_head = copy.deepcopy(model.head)
        # Evidence-driven revision: refit both heads at the same budget.  The direct
        # projected score is retained to show whether the first attempt saturated.
        direct_gap = gap_from_logits(model.head(pte), te, opt_lengths)
        raw_trace = train_head(raw_head, ztr, ytr)
        proj_trace = train_head(proj_head, ptr, ytr)
        raw_gap = gap_from_logits(raw_head(zte), te, opt_lengths)
        proj_gap = gap_from_logits(proj_head(pte), te, opt_lengths)
        payload["records"].append({
            "seed": seed,
            "original_gap_percent": gap(tour_ratio(model, te, opt_lengths, beam_width=1)),
            "full_representation_refit_gap_percent": raw_gap,
            "direct_slater_projection_gap_percent": direct_gap,
            "slater_projection_refit_gap_percent": proj_gap,
            "projection_penalty_points": proj_gap - raw_gap,
            "median_train_best_slater_fidelity": float(ftr.median()),
            "median_test_best_slater_fidelity": float(fte.median()),
            "median_test_slater_infidelity": float((1 - fte).median()),
            "training_trace": trace, "full_head_refit_trace": raw_trace,
            "projected_head_refit_trace": proj_trace,
        })
    med_inf = float(np.median([r["median_test_slater_infidelity"] for r in payload["records"]]))
    med_pen = float(np.median([r["projection_penalty_points"] for r in payload["records"]]))
    large = sum(r["projection_penalty_points"] >= 1.0 for r in payload["records"])
    close = sum(abs(r["projection_penalty_points"]) <= 0.5 for r in payload["records"])
    go = med_inf >= 0.10 and med_pen >= 1.0 and large >= 4
    stop = med_inf < 0.01 or close >= 4
    payload["summary"] = {
        "median_best_slater_infidelity": med_inf,
        "median_refit_projection_penalty_points": med_pen,
        "seeds_projection_penalty_at_least_one_point": large,
        "seeds_within_half_point": close,
        "k1_mathematical_control_fidelity": 1.0,
        "revision": "direct projection -> equal-budget refit of full and projected edge heads",
        "decision": "PASS" if go else ("STOP" if stop else "PIVOT"),
    }
    payload["caveat"] = (
        "The optimized Slater overlap is a numerical lower bound on the true best fidelity; "
        "three restarts reduce but do not eliminate local-optimum risk. The reduced-basis MP-QGNN path is exact."
    )
    save(out, payload)


# ---------------------------------------------------------------------------
# B. Training-only MP-QGNN prior


def edge_features(coords):
    B, n, _ = coords.shape
    iu = torch.triu_indices(n, n, offset=1, device=coords.device)
    d = torch.cdist(coords, coords)
    eye = torch.eye(n, dtype=torch.bool, device=coords.device)[None]
    nearest = d.masked_fill(eye, float("inf")).min(-1).values
    mean = d.sum(-1) / (n - 1)
    centered = torch.linalg.vector_norm(coords - coords.mean(1, keepdim=True), dim=-1)
    i, j = iu[0], iu[1]
    feat = torch.stack([
        d[:, i, j], centered[:, i] + centered[:, j], (centered[:, i] - centered[:, j]).abs(),
        mean[:, i] + mean[:, j], (mean[:, i] - mean[:, j]).abs(),
        nearest[:, i] + nearest[:, j], (nearest[:, i] - nearest[:, j]).abs(),
    ], dim=-1)
    return iu, feat


class ClassicalEdgeNet(nn.Module):
    """Size-free, permutation-equivariant geometry baseline."""
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(7, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, coords):
        B, n, _ = coords.shape
        iu, feat = edge_features(coords)
        scores = self.net(feat).squeeze(-1)
        logits = torch.zeros(B, n, n, dtype=coords.dtype, device=coords.device)
        logits[:, iu[0], iu[1]] = scores; logits[:, iu[1], iu[0]] = scores
        return logits


def train_student(student, x, y, *, epochs=180, teacher_logits=None, mode="baseline",
                  temperature=2.0, lam=0.5, seed=0):
    opt = torch.optim.Adam(student.parameters(), lr=3e-3)
    rng = np.random.default_rng(seed)
    trace = []
    scale = 0.5
    for epoch in range(epochs):
        opt.zero_grad()
        logits = student(x)
        if mode == "additive":
            mask = float(rng.random() < 0.5)
            loss = balanced_bce(logits + mask * scale * teacher_logits, y)
        elif mode == "distill":
            supervised = balanced_bce(logits, y)
            soft = torch.sigmoid(teacher_logits / temperature)
            distill = nn.functional.binary_cross_entropy_with_logits(logits / temperature, soft)
            loss = supervised + lam * temperature * temperature * distill
        else:
            loss = balanced_bce(logits, y)
        loss.backward(); opt.step()
        if epoch in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    return trace


def run_qpat(out):
    payload = {
        "_started": time.time(), "project": "training_only_mpqgnn_prior",
        "command": f"python experiments/radar_2026_11_27.py qpat --out {out}",
        "seeds": list(range(5)),
        "data": {"train_n5": 80, "test_n5": 70, "transfer_test_n6": 90},
        "teacher_epochs": 130, "student_epochs": 180, "records": [],
    }
    tr, _, ytr = random_instances(5, 80, seed=2720)
    te5, tours5, _ = random_instances(5, 70, seed=2721); opt5 = lengths(te5, tours5)
    te6, tours6, _ = random_instances(6, 90, seed=2722); opt6 = lengths(te6, tours6)
    for seed in payload["seeds"]:
        torch.manual_seed(seed)
        teacher = TSPQGNN(5, D=6, k=3, num_layers=2)
        teacher_trace = train_model(teacher, tr, ytr, epochs=payload["teacher_epochs"])
        teacher.eval()
        with torch.no_grad():
            trained_logits = teacher(tr).detach()
        torch.manual_seed(1000 + seed)
        random_teacher = TSPQGNN(5, D=6, k=3, num_layers=2).eval()
        with torch.no_grad():
            random_logits = random_teacher(tr).detach()

        for method, mode, tlogits in [
            ("classical_baseline", "baseline", None),
            ("additive_drop_prior", "additive", trained_logits),
            ("distilled_trained_mpqgnn", "distill", trained_logits),
            ("distilled_random_mpqgnn", "distill", random_logits),
        ]:
            torch.manual_seed(2000 + seed)
            student = ClassicalEdgeNet()
            trace = train_student(student, tr, ytr, epochs=payload["student_epochs"],
                                  teacher_logits=tlogits, mode=mode, seed=3000 + seed)
            payload["records"].append({
                "seed": seed, "method": method,
                "gap_n5_percent": gap(tour_ratio(student, te5, opt5, beam_width=1)),
                "gap_transfer_n6_percent": gap(tour_ratio(student, te6, opt6, beam_width=1)),
                "inference_quantum_calls": 0,
                "trainable_parameters": sum(p.numel() for p in student.parameters()),
                "student_loss_trace": trace,
                "teacher_loss_trace": teacher_trace if method == "distilled_trained_mpqgnn" else None,
            })
    methods = sorted({r["method"] for r in payload["records"]})
    med = {m: {key: float(np.median([r[key] for r in payload["records"] if r["method"] == m]))
               for key in ["gap_n5_percent", "gap_transfer_n6_percent"]} for m in methods}
    base = med["classical_baseline"]["gap_transfer_n6_percent"]
    revised = med["distilled_trained_mpqgnn"]["gap_transfer_n6_percent"]
    random = med["distilled_random_mpqgnn"]["gap_transfer_n6_percent"]
    rel = 100 * (base - revised) / max(abs(base), 1e-12)
    wins = sum(
        next(r["gap_transfer_n6_percent"] for r in payload["records"] if r["seed"] == s and r["method"] == "distilled_trained_mpqgnn")
        < next(r["gap_transfer_n6_percent"] for r in payload["records"] if r["seed"] == s and r["method"] == "classical_baseline")
        for s in payload["seeds"]
    )
    advantage_random = random - revised
    go = rel >= 10 and wins >= 4 and advantage_random >= 0.5
    stop = rel <= 0 or wins < 3 or advantage_random < 0.25
    payload["summary"] = {
        "median": med, "relative_transfer_gain_percent": rel, "paired_seed_wins": wins,
        "gap_advantage_over_random_teacher_points": advantage_random,
        "revision": "stochastic additive logits -> separate supervised plus temperature-scaled distillation loss",
        "decision": "PASS" if go else ("STOP" if stop else "PIVOT"),
    }
    payload["caveat"] = (
        "The teacher is the real exact reduced-basis MP-QGNN but is simulated, not run on hardware; "
        "the deployed student makes zero quantum calls."
    )
    save(out, payload)


# ---------------------------------------------------------------------------
# C. Contractive-layer no-go and collective revision


class CollectiveCompoundLayer(nn.Module):
    """One-parameter dense number-conserving orbital rotation."""
    def __init__(self, D, k, init=0.0):
        super().__init__(); self.D, self.k = D, k
        self.alpha = nn.Parameter(torch.tensor(float(init)))
        i, j = torch.triu_indices(D, D, offset=1)
        vals = torch.sin((i + 1) * (j + 1)).to(torch.get_default_dtype())
        skew = torch.zeros(D, D, dtype=torch.get_default_dtype())
        skew[i, j] = vals; skew = skew - skew.T
        skew = skew / torch.linalg.matrix_norm(skew)
        self.register_buffer("skew", skew); self.register_buffer("eye", torch.eye(D))

    def orthogonal(self):
        a = self.alpha * self.skew
        return torch.linalg.solve(self.eye + a, self.eye - a)

    def forward(self, x):
        return x @ compound_matrix(self.orthogonal(), self.k).T


class CollectiveTSPQGNN(TSPQGNN):
    def __init__(self, n_cities, D=6, k=3, num_layers=2, collective_init=0.0):
        super().__init__(n_cities, D=D, k=k, num_layers=num_layers)
        self.D, self.k, self.collective_init = D, k, collective_init
        self.collective = nn.ModuleList(
            CollectiveCompoundLayer(D, k, collective_init) for _ in range(num_layers)
        )

    def node_embeddings(self, coords):
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](coords)
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            x = self.adjacency[r](x, coords)
            x = self.layers[r](x)
            x = self.collective[r](x)
        return x


def run_contractive(out):
    payload = {
        "_started": time.time(), "project": "contractive_fixed_particle_audit",
        "command": f"python experiments/radar_2026_11_27.py contractive --out {out}",
        "seeds": list(range(5)), "data": {"n": 6, "train": 60, "test": 70, "transfer_n7": 70},
        "epochs": 140, "records": [],
    }
    tr, _, ytr = random_instances(6, 60, seed=2730)
    te6, tours6, _ = random_instances(6, 70, seed=2731); opt6 = lengths(te6, tours6)
    te7, tours7, _ = random_instances(7, 70, seed=2732); opt7 = lengths(te7, tours7)

    # Literal parity phase on the real trained-feature probability vector.  In a
    # weight-k sector every basis state has eigenvalue (-1)^k.
    torch.manual_seed(0)
    probe = TSPQGNN(6, D=6, k=3, num_layers=2).node_embeddings(te6[:8]).detach()
    alpha = torch.tensor(0.371)
    phase = torch.exp(1j * alpha * ((-1) ** 3))
    p0 = probe.square()
    p1 = (probe.to(torch.complex128) * phase).abs().square()
    literal_delta = float((p0 - p1).abs().max())

    for seed in payload["seeds"]:
        torch.manual_seed(seed)
        base_init = TSPQGNN(6, D=6, k=3, num_layers=2)
        methods = {"baseline": copy.deepcopy(base_init)}
        for name, init in [
            ("collective_random_init", 0.1),
            ("collective_identity_init", 0.0),
            # Evidence-driven second revision: exact identity is a stationary
            # point for probability-only readout, so start close to identity
            # without landing on its zero-gradient symmetry point.
            ("collective_near_identity_init", 0.01),
        ]:
            m = CollectiveTSPQGNN(6, D=6, k=3, num_layers=2, collective_init=init)
            m.load_state_dict(base_init.state_dict(), strict=False)
            for layer in m.collective:
                layer.alpha.data.fill_(init)
            methods[name] = m
        for name, model in methods.items():
            trace = train_model(model, tr, ytr, epochs=payload["epochs"])
            transfer = transfer_tsp(model, 7)
            payload["records"].append({
                "seed": seed, "method": name,
                "gap_n6_percent": gap(tour_ratio(model, te6, opt6, beam_width=1)),
                "gap_transfer_n7_percent": gap(tour_ratio(transfer, te7, opt7, beam_width=1)),
                "trainable_parameters": sum(p.numel() for p in model.parameters()),
                "collective_parameters": sum(p.numel() for n, p in model.named_parameters() if "collective" in n),
                "learned_collective_angles": [float(l.alpha.detach()) for l in model.collective]
                    if isinstance(model, CollectiveTSPQGNN) else [],
                "loss_trace": trace,
            })
    methods = sorted({r["method"] for r in payload["records"]})
    med = {m: {key: float(np.median([r[key] for r in payload["records"] if r["method"] == m]))
               for key in ["gap_n6_percent", "gap_transfer_n7_percent"]} for m in methods}
    revised = "collective_near_identity_init"
    base = med["baseline"]["gap_transfer_n7_percent"]
    new = med[revised]["gap_transfer_n7_percent"]
    rel = 100 * (base - new) / max(abs(base), 1e-12)
    wins = sum(
        next(r["gap_transfer_n7_percent"] for r in payload["records"] if r["seed"] == s and r["method"] == revised)
        < next(r["gap_transfer_n7_percent"] for r in payload["records"] if r["seed"] == s and r["method"] == "baseline")
        for s in payload["seeds"]
    )
    point = base - new
    go = literal_delta < 1e-12 and (point >= 1.0 or rel >= 10) and wins >= 4
    stop = wins < 3 or abs(point) < 0.25
    payload["summary"] = {
        "literal_global_parity_probability_delta": literal_delta,
        "literal_no_go_verified": literal_delta < 1e-12,
        "median": med, "transfer_gain_points": point,
        "relative_transfer_gain_percent": rel, "paired_seed_wins": wins,
        "revision": (
            "random angle -> exact identity (zero-gradient failure) -> "
            "near-identity number-conserving collective angle"
        ),
        "decision": "PASS" if go else ("STOP" if stop else "PIVOT"),
    }
    payload["caveat"] = (
        "The literal no-go is exact for the paper's global Z-string in a fixed-weight sector. "
        "The collective hopping layer is a symmetry-compatible constructive alternative, not the same unitary."
    )
    save(out, payload)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("project", choices=["fermionic", "qpat", "contractive"])
    p.add_argument("--out", required=True)
    a = p.parse_args()
    {"fermionic": run_fermionic, "qpat": run_qpat, "contractive": run_contractive}[a.project](a.out)


if __name__ == "__main__":
    main()

