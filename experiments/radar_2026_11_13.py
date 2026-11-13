"""Three pre-registered MP-QGNN radar MVPs for 13 November 2026."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh

from mp_qgnns.core.adjacency import EquivariantAdjacencyLayer
from mp_qgnns.core.compound import CompoundPyramidLayer
from mp_qgnns.data_io.tsp import random_instances, tour_length
from mp_qgnns.models.tsp import EdgeHead, TSPQGNN
from mp_qgnns.training.tsp import balanced_bce, tour_ratio

torch.set_default_dtype(torch.float64)
torch.set_num_threads(2)


def sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def lengths(coords, tours):
    return np.asarray([tour_length(coords[i].numpy(), tours[i].numpy()) for i in range(len(coords))])


def gap(ratio):
    return 100.0 * (float(ratio) - 1.0)


def save(path, payload):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    payload["git_sha"] = sha(); payload["elapsed_seconds"] = time.time() - payload.pop("_started")
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def train(model, coords, heatmaps, *, epochs=120, lr=3e-3, extra=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    trace = []
    for epoch in range(epochs):
        opt.zero_grad()
        logits = model(coords) if extra is None else model(coords, *extra)
        loss = balanced_bce(logits, heatmaps)
        loss.backward(); opt.step()
        if epoch in {0, epochs // 2, epochs - 1}:
            trace.append({"epoch": epoch + 1, "loss": float(loss.detach())})
    return trace


def sorted_or_random_labels(coords, heatmaps, *, mode, seed=0):
    rng = np.random.default_rng(seed)
    out_x, out_y = [], []
    for i in range(len(coords)):
        if mode == "sorted":
            p = np.lexsort((coords[i, :, 1].numpy(), coords[i, :, 0].numpy()))
        else:
            p = rng.permutation(coords.shape[1])
        out_x.append(coords[i, p])
        out_y.append(heatmaps[i][np.ix_(p, p)])
    return torch.stack(out_x), torch.stack(out_y)


class LexicographicAdjacencyLayer(EquivariantAdjacencyLayer):
    """Same RBS gates and parameters, but gate order follows node labels."""
    def orient_and_sort_edges(self, coords):
        B = coords.shape[0]
        return (self.edge_pairs[:, 0].unsqueeze(0).expand(B, -1),
                self.edge_pairs[:, 1].unsqueeze(0).expand(B, -1))


class LexicographicTSPQGNN(TSPQGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        new = nn.ModuleList()
        for old in self.adjacency:
            layer = LexicographicAdjacencyLayer(old.n, old.angle_mlp[0].out_features)
            layer.angle_mlp.load_state_dict(old.angle_mlp.state_dict())
            new.append(layer)
        self.adjacency = new


def run_invariance(out):
    payload = {"_started": time.time(), "project": "permutation_shortcut",
               "command": f"python experiments/radar_2026_11_13.py invariance --out {out}",
               "data": {"n": 6, "train": 40, "test": 60}, "epochs": 140,
               "seeds": [0, 1, 2], "records": []}
    raw_tr, _, ytr = random_instances(6, 40, seed=410)
    raw_te, tte, yte = random_instances(6, 60, seed=411)
    tr, ytr = sorted_or_random_labels(raw_tr, ytr, mode="sorted")
    te_sorted, _ = sorted_or_random_labels(raw_te, yte, mode="sorted")
    te_random, _ = sorted_or_random_labels(raw_te, yte, mode="random", seed=412)
    opt = lengths(raw_te, tte)
    for seed in payload["seeds"]:
        for name, cls in [("equivariant", TSPQGNN), ("label_ordered", LexicographicTSPQGNN)]:
            torch.manual_seed(seed)
            model = cls(6, num_layers=2)
            trace = train(model, tr, ytr, epochs=payload["epochs"])
            sorted_ratio = tour_ratio(model, te_sorted, opt, beam_width=1)
            random_ratio = tour_ratio(model, te_random, opt, beam_width=1)
            payload["records"].append({"seed": seed, "model": name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "sorted_gap_percent": gap(sorted_ratio), "random_gap_percent": gap(random_ratio),
                "relabel_penalty_points": gap(random_ratio) - gap(sorted_ratio), "loss_trace": trace})
    med = {}
    for name in ["equivariant", "label_ordered"]:
        rows = [r for r in payload["records"] if r["model"] == name]
        med[name] = {k: float(np.median([r[k] for r in rows])) for k in
                     ["sorted_gap_percent", "random_gap_percent", "relabel_penalty_points"]}
    advantage = 100 * (med["label_ordered"]["random_gap_percent"] - med["equivariant"]["random_gap_percent"]) / max(med["label_ordered"]["random_gap_percent"], 1e-12)
    go = (advantage >= 20 and med["equivariant"]["relabel_penalty_points"] <= 1 and
          med["label_ordered"]["relabel_penalty_points"] >= 3 and
          abs(med["equivariant"]["sorted_gap_percent"] - med["label_ordered"]["sorted_gap_percent"]) <= 2)
    stop = advantage < 5 or med["equivariant"]["relabel_penalty_points"] > 2
    payload["summary"] = {"median": med, "relative_random_label_advantage_percent": advantage,
                          "decision": "PASS" if go else ("STOP" if stop else "PIVOT")}
    payload["caveat"] = "Shortcut is node-label ordering on exact Euclidean TSP, not an image watermark; both arms use identical trainable parameter counts."
    save(out, payload)


def rydberg_features(coords, mode="full"):
    """Exact small-n quench features following the QGF Hamiltonian and AFM/FM regimes."""
    arr = coords.detach().cpu().numpy()
    all_nodes, all_edges = [], []
    times = np.asarray([0.25, 0.5, 1.0, 2.0])
    for xy in arr:
        n, dim = len(xy), 1 << len(xy)
        dist = np.linalg.norm(xy[:, None] - xy[None, :], axis=-1)
        np.fill_diagonal(dist, np.inf)
        nn_scale = np.median(np.min(dist, axis=1))
        V = np.clip((nn_scale / np.maximum(dist, 1e-6)) ** 6, 0, 12)
        np.fill_diagonal(V, 0)
        bits = ((np.arange(dim)[:, None] >> np.arange(n)) & 1).astype(float)
        node_regimes, edge_regimes = [], []
        for regime in ["afm", "fm"]:
            omega = 0.3 if regime == "afm" else 0.5
            delta = 0.0 if regime == "afm" else float(V.sum() / (2 * n))
            diag = -delta * bits.sum(1)
            diag += np.einsum("bi,ij,bj->b", bits, np.triu(V, 1), bits)
            H = np.diag(diag)
            for state in range(dim):
                for q in range(n):
                    H[state, state ^ (1 << q)] += omega / 2
            evals, evecs = eigh(H)
            init_coeff = evecs[0].conj()
            ns, cs = [], []
            for t in times:
                psi = evecs @ (np.exp(-1j * evals * t) * init_coeff)
                prob = np.abs(psi) ** 2
                occ = prob @ bits
                pair = np.einsum("b,bi,bj->ij", prob, bits, bits)
                conn = pair - np.outer(occ, occ)
                ns.append(np.diag(conn)); cs.append(conn)
            node_regimes.append(np.asarray(ns)); edge_regimes.append(np.asarray(cs))
        node_raw = np.concatenate(node_regimes, axis=0).T
        edge_raw = np.concatenate(edge_regimes, axis=0).transpose(1, 2, 0)
        if mode == "stat":
            node_raw = np.stack([x.mean(0) for x in node_regimes] + [x.std(0) for x in node_regimes], -1)
            edge_raw = np.stack([x.mean(0) for x in edge_regimes] + [x.std(0) for x in edge_regimes], -1)
        all_nodes.append(node_raw); all_edges.append(edge_raw)
    return torch.tensor(np.stack(all_nodes)), torch.tensor(np.stack(all_edges))


class QGFAdjacencyLayer(EquivariantAdjacencyLayer):
    def __init__(self, n, qdim, mlp_hidden=16):
        super().__init__(n, mlp_hidden)
        self.angle_mlp = nn.Sequential(nn.Linear(3 + qdim, mlp_hidden), nn.ReLU(), nn.Linear(mlp_hidden, 1))

    def compute_angles_with_qgf(self, coords, si, sj, qedge):
        ci = torch.gather(coords, 1, si.unsqueeze(-1).expand(-1, -1, 2))
        cj = torch.gather(coords, 1, sj.unsqueeze(-1).expand(-1, -1, 2))
        diff = cj - ci
        batch = torch.arange(coords.shape[0]).unsqueeze(1).expand_as(si)
        q = qedge[batch, si, sj]
        feats = torch.cat([torch.norm(diff, dim=-1, keepdim=True), diff, q], -1)
        return (torch.tanh(self.angle_mlp(feats.reshape(-1, feats.shape[-1]))) * (np.pi / 2)).view(coords.shape[0], self.num_edges)

    def forward_with_qgf(self, x, coords, qedge):
        si, sj = self.orient_and_sort_edges(coords)
        angles = self.compute_angles_with_qgf(coords, si, sj, qedge)
        batch = torch.arange(x.shape[0])
        for e in range(self.num_edges):
            i, j, theta = si[:, e], sj[:, e], angles[:, e]
            c, s = torch.cos(theta/2).unsqueeze(-1), torch.sin(theta/2).unsqueeze(-1)
            xi, xj = x[batch, i], x[batch, j]
            ie = i[:, None, None].expand(-1, 1, x.shape[2]); je = j[:, None, None].expand_as(ie)
            x = x.scatter(1, ie, (c*xi-s*xj).unsqueeze(1))
            x = x.scatter(1, je, (s*xi+c*xj).unsqueeze(1))
        return x


class QGFTSPQGNN(nn.Module):
    def __init__(self, n, qdim, D=6, k=3, num_layers=2, hidden=64):
        super().__init__(); from math import comb
        emb = comb(D, k); self.num_layers = num_layers
        self.embed = nn.ModuleList(nn.Sequential(nn.Linear(2+qdim, hidden), nn.ReLU(), nn.Linear(hidden, emb)) for _ in range(num_layers))
        self.adjacency = nn.ModuleList(QGFAdjacencyLayer(n, qdim) for _ in range(num_layers))
        self.layers = nn.ModuleList(CompoundPyramidLayer(D, k) for _ in range(num_layers))
        self.head = EdgeHead(emb, 64)

    def node_embeddings(self, coords, qnode, qedge):
        x = None
        for r in range(self.num_layers):
            z = self.embed[r](torch.cat([coords, qnode], -1))
            z = z / (torch.linalg.vector_norm(z, dim=-1, keepdim=True) + 1e-12)
            x = z if x is None else x + z
            x = self.adjacency[r].forward_with_qgf(x, coords, qedge)
            x = self.layers[r](x)
        return x

    def forward(self, coords, qnode, qedge):
        return self.head(self.node_embeddings(coords, qnode, qedge))


def qgf_ratio(model, coords, opt, qnode, qedge):
    model.eval()
    class Wrap(nn.Module):
        def __init__(self, m, qn, qe): super().__init__(); self.m=m; self.qn=qn; self.qe=qe
        def forward(self, x): return self.m(x, self.qn, self.qe)
    return tour_ratio(Wrap(model, qnode, qedge), coords, opt, beam_width=1)


def run_qgf(out):
    payload = {"_started": time.time(), "project": "neutral_atom_qgf",
               "command": f"python experiments/radar_2026_11_13.py qgf --out {out}",
               "data": {"n": 6, "train": 30, "test": 60}, "epochs": 150,
               "seeds": [0,1,2], "feature_modes": ["full", "stat"], "records": []}
    tr, _, ytr = random_instances(6, 30, seed=420)
    te, tte, _ = random_instances(6, 60, seed=421); opt = lengths(te, tte)
    feats = {}
    for mode in payload["feature_modes"]:
        feats[mode] = (*rydberg_features(tr, mode), *rydberg_features(te, mode))
    for seed in payload["seeds"]:
        torch.manual_seed(seed); base = TSPQGNN(6, num_layers=2)
        trace = train(base, tr, ytr, epochs=payload["epochs"])
        ratio = tour_ratio(base, te, opt, beam_width=1)
        payload["records"].append({"seed":seed,"model":"baseline","gap_percent":gap(ratio),
            "parameters":sum(p.numel() for p in base.parameters()),"loss_trace":trace})
        for mode in payload["feature_modes"]:
            qntr,qetr, qnte,qete = feats[mode]
            torch.manual_seed(seed); m = QGFTSPQGNN(6, qntr.shape[-1])
            trace = train(m, tr, ytr, epochs=payload["epochs"], extra=(qntr,qetr))
            ratio = qgf_ratio(m, te, opt, qnte,qete)
            payload["records"].append({"seed":seed,"model":f"qgf_{mode}","gap_percent":gap(ratio),
                "parameters":sum(p.numel() for p in m.parameters()),"loss_trace":trace})
    models = ["baseline","qgf_full","qgf_stat"]
    med = {m:float(np.median([r["gap_percent"] for r in payload["records"] if r["model"]==m])) for m in models}
    best = min(["qgf_full","qgf_stat"], key=med.get)
    rel = 100*(med["baseline"]-med[best])/max(med["baseline"],1e-12)
    abs_gain = med["baseline"]-med[best]
    per_seed = []
    for s in payload["seeds"]:
        b=next(r for r in payload["records"] if r["seed"]==s and r["model"]=="baseline")["gap_percent"]
        q=next(r for r in payload["records"] if r["seed"]==s and r["model"]==best)["gap_percent"]
        per_seed.append(q<b)
    pbase=next(r["parameters"] for r in payload["records"] if r["model"]=="baseline")
    pbest=next(r["parameters"] for r in payload["records"] if r["model"]==best)
    go=rel>=15 and abs_gain>=2 and pbest/pbase<1.5
    stop=rel<5 or sum(per_seed)<=1 or pbest/pbase>=1.5
    payload["summary"]={"median_gap_percent":med,"best_qgf":best,"relative_gain_percent":rel,
                        "absolute_gain_points":abs_gain,"seeds_improved":sum(per_seed),
                        "parameter_ratio":pbest/pbase,"revision":"full time series -> mean/std statistics",
                        "decision":"PASS" if go else ("STOP" if stop else "PIVOT")}
    payload["caveat"]="Exact dimensionless Rydberg simulation on TSP coordinates; not Pasqal hardware and not the paper's unit-disk register optimizer."
    save(out,payload)


def per_graph_bce(model, coords, targets):
    with torch.no_grad():
        loss = nn.functional.binary_cross_entropy_with_logits(model(coords), targets, reduction="none")
    return loss.mean((1,2))


def transfer_model(model, n):
    out=TSPQGNN(n,num_layers=model.num_layers)
    state={k:v for k,v in model.state_dict().items() if "edge_pairs" not in k}
    out.load_state_dict(state,strict=False); return out


def run_vinfo(out):
    payload={"_started":time.time(),"project":"v_usable_selection",
             "command":f"python experiments/radar_2026_11_13.py vinfo --out {out}",
             "data":{"pool_n5":120,"selected":60,"test_n5":60,"transfer_test_n6":60},
             "probe_epochs":40,"final_epochs":140,"seeds":[0,1,2],"records":[]}
    pool,_,ypool=random_instances(5,120,seed=430)
    te5,t5,_=random_instances(5,60,seed=431); opt5=lengths(te5,t5)
    te6,t6,_=random_instances(6,60,seed=432); opt6=lengths(te6,t6)
    null_x=torch.full_like(pool,0.5)
    for seed in payload["seeds"]:
        torch.manual_seed(seed); probe=TSPQGNN(5,num_layers=2); train(probe,pool,ypool,epochs=40)
        torch.manual_seed(seed); null=TSPQGNN(5,num_layers=2); train(null,null_x,ypool,epochs=40)
        scores=(per_graph_bce(null,null_x,ypool)-per_graph_bce(probe,pool,ypool)).numpy()
        order=np.argsort(-scores); top=order[:60]
        rng=np.random.default_rng(500+seed); uniform=rng.choice(len(pool),60,replace=False)
        top30=order[:30]; remaining=order[30:]
        bins=np.array_split(remaining,4)
        mixed=list(top30)
        for b in bins:
            mixed.extend(rng.choice(b, 8 if len(mixed)<54 else 6, replace=False).tolist())
        mixed=np.asarray(mixed[:60])
        selections={"full":np.arange(120),"uniform":uniform,"top_pvi":top,"mixed_pvi":mixed}
        for name,idx in selections.items():
            torch.manual_seed(seed); m=TSPQGNN(5,num_layers=2)
            trace=train(m,pool[idx],ypool[idx],epochs=payload["final_epochs"])
            r5=tour_ratio(m,te5,opt5,beam_width=1)
            m6=transfer_model(m,6); r6=tour_ratio(m6,te6,opt6,beam_width=1)
            payload["records"].append({"seed":seed,"selection":name,"samples":len(idx),
                "gap_n5_percent":gap(r5),"gap_transfer_n6_percent":gap(r6),"loss_trace":trace,
                "score_mean":None if name in {"full","uniform"} else float(scores[idx].mean())})
    methods=["full","uniform","top_pvi","mixed_pvi"]
    med={m:{k:float(np.median([r[k] for r in payload["records"] if r["selection"]==m])) for k in
            ["gap_n5_percent","gap_transfer_n6_percent"]} for m in methods}
    best=min(["top_pvi","mixed_pvi"],key=lambda m:med[m]["gap_transfer_n6_percent"])
    transfer_gain=100*(med["uniform"]["gap_transfer_n6_percent"]-med[best]["gap_transfer_n6_percent"])/max(med["uniform"]["gap_transfer_n6_percent"],1e-12)
    in_size_delta=med[best]["gap_n5_percent"]-med["full"]["gap_n5_percent"]
    signs=[]
    for s in payload["seeds"]:
        u=next(r for r in payload["records"] if r["seed"]==s and r["selection"]=="uniform")["gap_transfer_n6_percent"]
        b=next(r for r in payload["records"] if r["seed"]==s and r["selection"]==best)["gap_transfer_n6_percent"]
        signs.append(b<u)
    go=transfer_gain>=10 and in_size_delta<=1 and all(signs)
    stop=transfer_gain<3 or sum(signs)<=1 or in_size_delta>2
    payload["summary"]={"median":med,"best_selection":best,
                        "relative_transfer_gain_percent":transfer_gain,
                        "in_size_delta_vs_full_points":in_size_delta,
                        "seeds_transfer_improved":sum(signs),
                        "revision":"top pointwise V-information -> high-score plus quantile coverage",
                        "decision":"PASS" if go else ("STOP" if stop else "PIVOT")}
    payload["caveat"]="Pointwise score is an empirical cross-entropy V-information proxy from finite probes; it is not a hardware-shot cost estimate."
    save(out,payload)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("project",choices=["invariance","qgf","vinfo"]); ap.add_argument("--out",required=True)
    a=ap.parse_args(); {"invariance":run_invariance,"qgf":run_qgf,"vinfo":run_vinfo}[a.project](a.out)


if __name__=="__main__": main()
