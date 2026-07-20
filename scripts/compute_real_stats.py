"""
Compute per-channel mean/std from real patient H5 files and save as real_norm_stats.json.

These stats are used to z-score real patient inputs independently of the sim distribution,
so both domains arrive at the encoder with ~zero mean and ~unit variance in their own space.

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

    # Accumulators: waveforms (per channel), Pas scalars, SV, HR
    wave_vals  = {k: [] for k in WAVE_KEYS_REAL}
    pas_vals   = []   # map, sbp, dbp all drawn from the same Pas distribution
    sv_vals    = []
    hr_vals    = []

    for fpath in h5_files:
        with h5py.File(fpath, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("beat_"):
                    continue
                g = f[key]
                for ch in WAVE_KEYS_REAL:
                    wave_vals[ch].append(g[f"waves/{ch}"][:].astype(np.float32))
                pas_vals.extend([
                    float(g["summaries/map"][()]),
                    float(g["summaries/sbp"][()]),
                    float(g["summaries/dbp"][()]),
                ])
                sv_vals.append(float(g["summaries/sv"][()]))
                hr_vals.append(float(g["parameters/HR"][()]))

    out = {"waves": {}, "scalars": {}}

    for ch in WAVE_KEYS_REAL:
        arr = np.concatenate(wave_vals[ch])
        out["waves"][ch] = {"mean": float(arr.mean()), "std": float(arr.std())}
        print(f"  {ch:<6}  mean={arr.mean():.4f}  std={arr.std():.4f}")

    pas_arr = np.array(pas_vals)
    out["scalars"]["Pas"]  = {"mean": float(pas_arr.mean()), "std": float(pas_arr.std())}
    sv_arr  = np.array(sv_vals)
    out["scalars"]["sv"]   = {"mean": float(sv_arr.mean()),  "std": float(sv_arr.std())}
    hr_arr  = np.array(hr_vals)
    out["scalars"]["HR"]   = {"mean": float(hr_arr.mean()),  "std": float(hr_arr.std())}

    print(f"\n  Pas    mean={pas_arr.mean():.4f}  std={pas_arr.std():.4f}")
    print(f"  SV     mean={sv_arr.mean():.4f}   std={sv_arr.std():.4f}")
    print(f"  HR     mean={hr_arr.mean():.4f}   std={hr_arr.std():.4f}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
