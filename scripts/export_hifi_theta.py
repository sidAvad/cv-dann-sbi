"""Recover ground-truth theta for the hifi "real stand-in" beat files.

export_hifi_as_real_beats.py writes one sim_<global index>.h5 per surviving sim in the
real-patient layout, which carries only HR/PVR/SVR -- the three quantities a real patient
record has. The full 24-dim theta stays in the source target set's batch files. This walks
the stand-in directory, resolves each index through the target manifest, and writes a single
npz so training-time calibration scoring needs no h5 access.

Output: indices (N,), theta (N, 24) in PARAM_KEYS_INFER order, param_keys (24,).
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import PARAM_KEYS_INFER  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", required=True,
                    help="directory of sim_<idx>.h5 stand-in files")
    ap.add_argument("--target-dir", required=True,
                    help="source target set holding manifest.json + batch_*.h5")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    real_dir, target_dir = Path(a.real_dir), Path(a.target_dir)
    index = json.load(open(target_dir / "manifest.json"))["index"]

    # Row order must match load_real_beats, which iterates
    # sorted(data_dir.glob("*.h5")) -- a LEXICOGRAPHIC sort of the filenames, so
    # sim_1000.h5 precedes sim_200.h5. Sorting these indices numerically instead
    # silently misaligns theta against the beats it is meant to label.
    idx = [int(m.group(1)) for m in
           sorted((m for m in (re.fullmatch(r"sim_(\d+)\.h5", p.name)
                               for p in real_dir.glob("sim_*.h5")) if m),
                  key=lambda m: m.string)]
    if not idx:
        raise SystemExit(f"no sim_<idx>.h5 files under {real_dir}")
    print(f"{len(idx)} stand-in files; target manifest has {len(index)} entries")

    # group by batch file so each is opened once
    by_file = defaultdict(list)
    for gi in idx:
        e = index[gi]
        by_file[e["file"]].append((gi, e["group"]))

    theta = np.full((len(idx), len(PARAM_KEYS_INFER)), np.nan)
    pos = {gi: i for i, gi in enumerate(idx)}
    missing = []
    for fname in sorted(by_file):
        with h5py.File(target_dir / fname, "r") as f:
            for gi, grp in by_file[fname]:
                if grp not in f:
                    missing.append(gi)
                    continue
                p = f[grp]["parameters"]
                theta[pos[gi]] = [float(p[k][()]) for k in PARAM_KEYS_INFER]
        print(f"  {fname}: {len(by_file[fname])} sims", flush=True)

    if missing:
        raise SystemExit(f"{len(missing)} indices absent from their batch group, "
                         f"e.g. {missing[:5]}")
    bad = ~np.isfinite(theta).all(1)
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} rows have non-finite theta")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, indices=np.array(idx), theta=theta.astype(np.float64),
             param_keys=np.array(PARAM_KEYS_INFER))
    print(f"wrote {out}  theta {theta.shape}")
    print("  per-param min/max:")
    for j, k in enumerate(PARAM_KEYS_INFER):
        print(f"    {k:12s} {theta[:, j].min():12.5g} {theta[:, j].max():12.5g}")


if __name__ == "__main__":
    main()
