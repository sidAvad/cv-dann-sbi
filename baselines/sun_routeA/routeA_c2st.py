"""Phase 5: C2ST of Route A's synthesized pairs against genuine simulator pairs.

At beta=1 the power posterior is the posterior, so the synthesized joint must match the
simulator's own joint -- the chains are even initialised from it. A classifier that can tell
the two apart is therefore a direct, assumption-free measure of how much the annealed
Langevin distorted the joint. 0.5 means indistinguishable; 1.0 means trivially separable.

Genuine rows are taken from the beta's own disjoint init slice, so the comparison is against
exactly the rows those chains started from -- paired, not merely same-distribution. The
classifier never sees a row in both classes.

Reported per beta, since beta=1 is the control and the off-one betas quantify how far the
tempering moved the joint (if at all).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def beta_tag(b):
    return f"{b:g}".replace(".", "p")


def c2st(a_rows, b_rows, device, epochs=30, hidden=256, batch=512, seed=0, log=print):
    """Train/test accuracy of a 2-layer MLP separating a_rows from b_rows.

    Inputs are z-scored using statistics from the pooled training half only, so the test
    accuracy cannot be inflated by normalisation leakage.
    """
    g = torch.Generator().manual_seed(seed)
    n = min(a_rows.shape[0], b_rows.shape[0])
    a_rows, b_rows = a_rows[:n], b_rows[:n]
    X = torch.cat([a_rows, b_rows], 0).float()
    y = torch.cat([torch.zeros(n), torch.ones(n)], 0)

    perm = torch.randperm(X.shape[0], generator=g)
    X, y = X[perm], y[perm]
    n_tr = int(0.7 * X.shape[0])
    Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr, ytr = Xtr.to(device), ytr.to(device)
    Xte, yte = Xte.to(device), yte.to(device)

    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(),
                        nn.Linear(hidden, hidden), nn.ReLU(),
                        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        order = torch.randperm(Xtr.shape[0], device=device)
        for i in range(0, Xtr.shape[0] - batch + 1, batch):
            sel = order[i:i + batch]
            opt.zero_grad()
            lossf(net(Xtr[sel]).squeeze(-1), ytr[sel]).backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        acc_te = float((((net(Xte).squeeze(-1) > 0).float()) == yte).float().mean())
        acc_tr = float((((net(Xtr).squeeze(-1) > 0).float()) == ytr).float().mean())
    return acc_te, acc_tr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="/home/sa4604/outputs/cv-dann-sbi/routeA_base")
    p.add_argument("--synth", required=True)
    p.add_argument("--n", type=int, default=20000, help="rows per class")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--betas", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = torch.load(Path(a.base) / "base.pt", weights_only=False)
    td, xd = base["theta_dim"], base["x_dim"]
    betas = ([float(b) for b in a.betas.split(",")] if a.betas else list(base["betas"]))

    print(f"C2ST: synthesized vs genuine, {a.n} rows/class, {td+xd} features, {device}")
    print(f"{'beta':>5} {'acc_test':>9} {'acc_train':>10} {'acc_theta':>10} {'acc_x':>8}")
    rows = []
    for beta in betas:
        f = Path(a.synth) / f"synth_beta_{beta_tag(beta)}.pt"
        if not f.exists():
            print(f"{beta:5g}  (missing {f.name})")
            continue
        d = torch.load(f, weights_only=False)
        syn = d["rows"][:a.n]
        gen = base["pairs"][torch.from_numpy(d["init_idx"][:a.n])]

        acc, acc_tr = c2st(syn, gen, device, epochs=a.epochs)
        acc_th, _ = c2st(syn[:, :td], gen[:, :td], device, epochs=a.epochs)
        acc_x, _ = c2st(syn[:, td:], gen[:, td:], device, epochs=a.epochs)
        print(f"{beta:5g} {acc:9.4f} {acc_tr:10.4f} {acc_th:10.4f} {acc_x:8.4f}")
        rows.append({"beta": float(beta), "acc_test": acc, "acc_train": acc_tr,
                     "acc_theta": acc_th, "acc_x": acc_x})
        del d, syn, gen

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({"n_per_class": a.n, "rows": rows}, fh, indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
