"""
Select the N sims from the training pool whose cath-lab waveforms (Prv/Pra/Pvp/Pap, 804-dim)
are nearest to the 802 real patients in PCA space.

Produces a filtered manifest JSON with the same structure as manifest_train.json.

Usage:
    python scripts/filter_sims_pca.py \\
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \\
        --real-data /home/sa4604/real_data/onebeat_300patients \\
        --n-pool 1000000 \\
        --n-select 300000 \\
        --n-pca-components 50 \\
        --out manifest_train_v3.3.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import load_stats, load_manifest, WAVE_KEYS_REDUCED, T


def load_sim_waveforms(data_dir: Path, index: list, stats: dict, n: int) -> np.ndarray:
    """Load z-scored Prv/Pra/Pvp/Pap waveforms for n sims. Returns (n, 4*T) float32."""
    from dataset import ReducedCVDataset
    ds = ReducedCVDataset(str(data_dir), index[:n], stats)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=4, pin_memory=False)
    wave_len = len(WAVE_KEYS_REDUCED) * T  # 804
    parts = []
    loaded = 0
    for _, x_b in loader:
        parts.append(x_b[:, :wave_len].numpy())
        loaded += len(x_b)
        print(f"\r  loading {loaded}/{n}", end="", flush=True)
    print()
    ds.close()
    return np.concatenate(parts, axis=0)


def load_real_waveforms(real_data_path: Path) -> np.ndarray:
    """Load real patient waveforms (first 804 dims of real_beats tensor)."""
    wave_len = len(WAVE_KEYS_REDUCED) * T  # 804
    beats = torch.load(real_data_path, map_location="cpu", weights_only=False)
    if isinstance(beats, dict):
        beats = beats["x"]
    return beats[:, :wave_len].float().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-data-root",    required=True)
    p.add_argument("--real-data",        required=True)
    p.add_argument("--n-pool",           type=int, default=1_000_000,
                   help="Number of sims to consider as the candidate pool")
    p.add_argument("--n-select",         type=int, default=300_000,
                   help="Number of sims to select")
    p.add_argument("--n-pca-fit",        type=int, default=200_000,
                   help="Number of sims used to fit PCA (random subsample of pool)")
    p.add_argument("--n-pca-components", type=int, default=50)
    p.add_argument("--out",              required=True,
                   help="Output manifest filename, saved alongside manifest_train.json")
    args = p.parse_args()

    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    sim_root = Path(args.sim_data_root)
    stats    = load_stats(Path(__file__).parent.parent / "norm_stats.json")
    manifest = load_manifest(sim_root / "manifest_train.json")
    index    = manifest["index"]

    print(f"Pool: {args.n_pool} sims  Select: {args.n_select}  "
          f"PCA fit: {args.n_pca_fit}  Components: {args.n_pca_components}")

    # ── Load sim waveforms ────────────────────────────────────────────────────
    print(f"\nLoading {args.n_pool} sim waveforms (Prv/Pra/Pvp/Pap, 804-dim)...")
    X_sims = load_sim_waveforms(sim_root / "train", index, stats, args.n_pool)
    print(f"Sim waveforms: {X_sims.shape}  ({X_sims.nbytes / 1e9:.2f} GB)")

    # ── Load real patient waveforms ───────────────────────────────────────────
    print(f"\nLoading real patient waveforms from {args.real_data}...")
    X_real = load_real_waveforms(Path(args.real_data))
    print(f"Real waveforms: {X_real.shape}")

    # ── Fit PCA on random subsample of sims ───────────────────────────────────
    print(f"\nFitting PCA ({args.n_pca_components} components) on {args.n_pca_fit} sims...")
    rng     = np.random.default_rng(42)
    fit_idx = rng.choice(args.n_pool, size=args.n_pca_fit, replace=False)
    pca     = PCA(n_components=args.n_pca_components, random_state=42)
    pca.fit(X_sims[fit_idx])
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"Variance explained: {var_explained:.3f}")

    # ── Project both into PCA space ───────────────────────────────────────────
    print("\nProjecting sims and real patients into PCA space...")
    Z_sims = pca.transform(X_sims)   # (n_pool, n_components)
    Z_real = pca.transform(X_real)   # (802, n_components)

    # ── For each sim, find distance to nearest real patient ───────────────────
    print("Computing sim → nearest real patient distances...")
    nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(Z_real)
    dists, _ = nn.kneighbors(Z_sims)   # (n_pool, 1)
    dists = dists[:, 0]                # (n_pool,)

    # ── Select n_select sims with smallest distance ───────────────────────────
    print(f"\nSelecting {args.n_select} nearest sims...")
    selected_order = np.argsort(dists)[:args.n_select]
    selected_idx   = np.sort(selected_order)  # preserve original order

    print(f"Distance to nearest real patient — selected: "
          f"max={dists[selected_order].max():.4f}  "
          f"median={np.median(dists[selected_order]):.4f}")
    print(f"Distance — full pool: "
          f"max={dists.max():.4f}  "
          f"median={np.median(dists):.4f}")
    print(f"Fraction of pool selected: {args.n_select / args.n_pool:.1%}")

    # ── Save filtered manifest ────────────────────────────────────────────────
    filtered_index = [index[i] for i in selected_idx]
    filtered_manifest = {**manifest, "index": filtered_index}

    out_path = sim_root / args.out
    with open(out_path, "w") as f:
        json.dump(filtered_manifest, f)
    print(f"\nSaved filtered manifest ({len(filtered_index)} entries) → {out_path}")

    # ── Sanity check: parameter distribution of selected vs full ─────────────
    print("\nDone. Run a sanity-check plot with:")
    print(f"  python scripts/plot_pca_filter.py --manifest {out_path} \\")
    print(f"    --sim-data-root {sim_root} --real-data {args.real_data}")


if __name__ == "__main__":
    main()
