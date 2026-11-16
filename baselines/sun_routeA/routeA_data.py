"""Base dataset for the Sun et al. Route A baseline (phase 1).

Produces the `[theta | x]` row layout upstream's `score_matching_loss`, `train`,
`validate` and `Lgmcmc` all assume (`theta_and_x[:, :theta_dim]` / `[:, theta_dim:]`),
built from the same lofi bank slice Gibbs NPE trains on, and partitions the rows one
disjoint slice per beta for Langevin initialisation.

theta is carried in unit-cube coordinates (prior box -> [0,1]^24, log on the three
log-uniform dims) via the repo's own `BoxToUnit`, so the Langevin runs in the space where
the prior is exactly uniform and every dimension is O(1). See REPRESENTATION.md section 3.
x is the untouched 809-dim vector `load_sim_tensors` returns.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from bounded_nsf import BoxToUnit                                    # noqa: E402
from dataset import (PARAM_KEYS_INFER, N_SCALARS,                    # noqa: E402
                     load_manifest, load_stats)
from train_gibbs_npe import WAVE_LEN, load_sim_tensors               # noqa: E402

X_DIM = WAVE_LEN + N_SCALARS          # 809
THETA_DIM = len(PARAM_KEYS_INFER)     # 24


def theta_box(nested_sim_root):
    """(lo, hi, log_dims) for PARAM_KEYS_INFER, in native units."""
    sys.path.insert(0, str(nested_sim_root))
    from nested_sim.params import THETA_BOUNDS, LOG_SAMPLED, to_binary_names
    bb = to_binary_names(THETA_BOUNDS)
    lo = [bb[k][0] for k in PARAM_KEYS_INFER]
    hi = [bb[k][1] for k in PARAM_KEYS_INFER]
    log_dims = [PARAM_KEYS_INFER.index(k)
                for k in to_binary_names({k: 0 for k in LOG_SAMPLED})]
    return lo, hi, sorted(log_dims)


def box_to_unit(nested_sim_root):
    lo, hi, log_dims = theta_box(nested_sim_root)
    return BoxToUnit(lo, hi, log_dims), log_dims


def beta_partition(n_rows, n_betas, equal=True):
    """Disjoint contiguous-free index slices, one per beta.

    Rows are permuted once with a fixed seed so a slice is not a contiguous block of the
    manifest (bank order is not guaranteed to be exchangeable w.r.t. anything downstream).
    With equal=True every beta gets the same count and the remainder is left unused, so
    per-beta results stay directly comparable.
    """
    rng = np.random.RandomState(0)
    perm = rng.permutation(n_rows)
    per = n_rows // n_betas
    slices = [np.sort(perm[i * per:(i + 1) * per]) for i in range(n_betas)]
    unused = n_rows - per * n_betas if equal else 0
    return slices, per, unused


def build_base(sim_root, stats_path, n_sims, n_val_sims, nested_sim_root, log=print):
    """Mirror train_gibbs_npe.py's data selection exactly, then add unit-cube theta."""
    sim_root, stats_path = Path(sim_root), Path(stats_path)
    stats = load_stats(stats_path)
    manifest = load_manifest(sim_root / "manifest_train.json")

    log(f"Loading {n_sims} sim observations (same slice as Gibbs NPE)...")
    theta, x = load_sim_tensors(sim_root / "train", manifest["index"][:n_sims], stats, log)

    val_idx = manifest["index"][n_sims:n_sims + n_val_sims]
    if val_idx:
        log(f"Loading {len(val_idx)} held-out sims (manifest {n_sims}..{n_sims + len(val_idx) - 1})...")
        val_theta, val_x = load_sim_tensors(sim_root / "train", val_idx, stats, log)
    else:
        val_theta = val_x = None

    tf, log_dims = box_to_unit(nested_sim_root)
    with torch.no_grad():
        theta_u, _ = tf.forward(theta)
        val_theta_u = None if val_theta is None else tf.forward(val_theta)[0]

    log(f"theta -> unit cube: {THETA_DIM} dims, log on {log_dims} "
        f"({[PARAM_KEYS_INFER[i] for i in log_dims]})")
    return {"theta_native": theta, "theta_unit": theta_u, "x": x,
            "val_theta_native": val_theta, "val_theta_unit": val_theta_u, "val_x": val_x,
            "log_dims": log_dims, "n_sims": int(theta.shape[0])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-data-root", default="/home/sa4604/outputs/nested-sim/data/lofi_train_for_gibbs")
    p.add_argument("--stats-path", default="/home/sa4604/outputs/nested-sim/data/lofi_train/norm_stats.json")
    p.add_argument("--nested-sim-root", default="/home/sa4604/projects/nested-sim")
    p.add_argument("--n-sims", type=int, default=980_000)
    p.add_argument("--n-val-sims", type=int, default=4096)
    p.add_argument("--betas", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.3,1.5")
    p.add_argument("--out", required=True, help="output directory")
    a = p.parse_args()

    betas = [float(b) for b in a.betas.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = build_base(a.sim_data_root, a.stats_path, a.n_sims, a.n_val_sims,
                   a.nested_sim_root)

    pairs = torch.cat([d["theta_unit"], d["x"]], dim=1)
    assert pairs.shape == (d["n_sims"], THETA_DIM + X_DIM), pairs.shape
    slices, per, unused = beta_partition(d["n_sims"], len(betas))

    torch.save({"pairs": pairs, "theta_native": d["theta_native"],
                "val_pairs": (None if d["val_x"] is None else
                              torch.cat([d["val_theta_unit"], d["val_x"]], dim=1)),
                "theta_dim": THETA_DIM, "x_dim": X_DIM,
                "log_dims": d["log_dims"], "betas": betas,
                "init_slices": [s.astype(np.int64) for s in slices]},
               out / "base.pt")

    meta = {"n_sims": d["n_sims"], "theta_dim": THETA_DIM, "x_dim": X_DIM,
            "theta_coords": "unit_cube", "log_dims": d["log_dims"],
            "log_dim_names": [PARAM_KEYS_INFER[i] for i in d["log_dims"]],
            "betas": betas, "n_per_beta": per, "rows_unused": unused,
            "sim_data_root": a.sim_data_root, "stats_path": a.stats_path,
            "n_val_sims": 0 if d["val_x"] is None else int(d["val_x"].shape[0])}
    with open(out / "base_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {out/'base.pt'}  pairs={tuple(pairs.shape)}")
    print(f"{len(betas)} betas x {per} chains = {per*len(betas)} of {d['n_sims']} rows "
          f"({unused} unused)")


if __name__ == "__main__":
    main()
