"""
Compute per-channel mean/std from real patient H5 files and save as real_norm_stats.json.

These stats are used to z-score real patient inputs independently of the sim distribution,
so both domains arrive at the encoder with ~zero mean and ~unit variance in their own space.

Scalars computed separately per summary type (map/sbp/dbp/sv/hr) so that zscore mode
can normalize each independently — matching the sim-side compute_scalar_stats.py approach.

Usage:
    python scripts/compute_real_stats.py \
        --real-data /home/sa4604/real_data/onebeat_300patients \
        --out real_norm_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import WAVE_KEYS_REDUCED

WAVE_KEYS_REAL = WAVE_KEYS_REDUCED  # Prv, Pra, Pvp, Pap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-data", required=True)
    p.add_argument("--out", default="real_norm_stats.json")
    args = p.parse_args()

    data_dir = Path(args.real_data)
    h5_files = sorted(data_dir.glob("*.h5"))
    assert h5_files, f"No H5 files found in {data_dir}"
    print(f"Found {len(h5_files)} patient files")

    wave_vals = {k: [] for k in WAVE_KEYS_REAL}
    map_vals, sbp_vals, dbp_vals, sv_vals, hr_vals = [], [], [], [], []

    for fpath in h5_files:
        with h5py.File(fpath, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("beat_"):
                    continue
                g = f[key]
                for ch in WAVE_KEYS_REAL:
                    wave_vals[ch].append(g[f"waves/{ch}"][:].astype(np.float32))
                map_vals.append(float(g["summaries/map"][()]))
                sbp_vals.append(float(g["summaries/sbp"][()]))
                dbp_vals.append(float(g["summaries/dbp"][()]))
                sv_vals.append(float(g["summaries/sv"][()]))
                hr_vals.append(float(g["parameters/HR"][()]))

    out = {"waves": {}, "scalars": {}}

    print("\nWaveform stats:")
    for ch in WAVE_KEYS_REAL:
        arr = np.concatenate(wave_vals[ch])
        out["waves"][ch] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"  {ch:<6}  mean={arr.mean():.4f}  std={arr.std():.4f}")

    scalar_data = {
        "map": np.array(map_vals),
        "sbp": np.array(sbp_vals),
        "dbp": np.array(dbp_vals),
        "sv":  np.array(sv_vals),
        "hr":  np.array(hr_vals),
    }

    print("\nScalar stats:")
    for key, arr in scalar_data.items():
        out["scalars"][key] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"  {key:<6}  mean={arr.mean():.4f}  std={arr.std():.4f}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
