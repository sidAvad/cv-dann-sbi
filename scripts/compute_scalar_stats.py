"""
Compute per-summary scalar statistics across training sims and add to norm_stats.json.

For each sim extracts one summary value per scalar:
  map  = Pas waveform mean  (mmHg)
  sbp  = Pas waveform max   (mmHg)
  dbp  = Pas waveform min   (mmHg)
  sv   = Vlv_max - Vlv_min  (mL, raw)
  hr   = HR parameter       (bpm)

Then computes mean/std across all sims and writes to norm_stats.json["scalars"].

Usage:
  python scripts/compute_scalar_stats.py \
      --data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \
      --n-sims 300000

  # PCA-nearest subset:
  python scripts/compute_scalar_stats.py \
      --data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \
      --manifest /home/sa4604/cv-dann-sbi/manifest_train_v3.3.json \
      --stats-path norm_stats_v3c.json
"""

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--manifest", default=None,
                   help="Path to manifest JSON (default: <data-root>/manifest_train.json)")
    p.add_argument("--stats-path", default="norm_stats.json")
    p.add_argument("--n-sims", type=int, default=None,
                   help="Number of sims to use (default: all in manifest)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    manifest_path = Path(args.manifest) if args.manifest else data_root / "manifest_train.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    index = manifest["index"]
    if args.n_sims is not None:
        index = index[:args.n_sims]
    n = len(index)
    print(f"Computing scalar stats over {n} sims...")

    maps, sbps, dbps, svs, hrs = [], [], [], [], []
    handles = {}
    data_dir = data_root / "train"

    for i, entry in enumerate(index):
        path = str(data_dir / entry["file"])
        if path not in handles:
            handles[path] = h5py.File(path, "r")
        g = handles[path][entry["group"]]

        pas = g["waves/Pas"][:]
        vlv = g["waves/Vlv"][:]
        hr  = float(g["parameters/HR"][()])

        maps.append(float(pas.mean()))
        sbps.append(float(pas.max()))
        dbps.append(float(pas.min()))
        svs.append(float(vlv.max() - vlv.min()))
        hrs.append(hr)

        if (i + 1) % 10000 == 0:
            print(f"\r  {i+1}/{n}", end="", flush=True)

    for fh in handles.values():
        fh.close()
    print()

    scalar_stats = {
        "map": {"mean": float(np.mean(maps)), "std": float(np.std(maps))},
        "sbp": {"mean": float(np.mean(sbps)), "std": float(np.std(sbps))},
        "dbp": {"mean": float(np.mean(dbps)), "std": float(np.std(dbps))},
        "sv":  {"mean": float(np.mean(svs)),  "std": float(np.std(svs))},
        "hr":  {"mean": float(np.mean(hrs)),  "std": float(np.std(hrs))},
    }

    print("\nSim scalar stats:")
    for k, v in scalar_stats.items():
        print(f"  {k:6s}  mean={v['mean']:.3f}  std={v['std']:.3f}")

    with open(args.stats_path) as f:
        norm_stats = json.load(f)

    norm_stats["scalars"] = scalar_stats

    with open(args.stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nWritten to {args.stats_path}[\"scalars\"]")


if __name__ == "__main__":
    main()
