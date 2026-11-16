"""Phase-1 checks for the Route A base dataset.

Run with a small --n-sims; every check is representation-level and does not need the
full bank. Exits non-zero on the first failure.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from dataset import PARAM_KEYS_INFER, ReducedCVDataset, load_manifest, load_stats  # noqa
from train_gibbs_npe import WAVE_LEN                                               # noqa
import routeA_data as RD                                                           # noqa
import routeA_noise as RN                                                          # noqa

OK, FAIL = "  ok  ", " FAIL "
_fails = []


def check(name, cond, detail=""):
    print(f"[{OK if cond else FAIL}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-data-root", default="/home/sa4604/outputs/nested-sim/data/lofi_train_for_gibbs")
    p.add_argument("--stats-path", default="/home/sa4604/outputs/nested-sim/data/lofi_train/norm_stats.json")
    p.add_argument("--nested-sim-root", default="/home/sa4604/projects/nested-sim")
    p.add_argument("--n-sims", type=int, default=20_000)
    a = p.parse_args()

    d = RD.build_base(a.sim_data_root, a.stats_path, a.n_sims, 256,
                      a.nested_sim_root, log=lambda *x: None)
    theta_n, theta_u, x = d["theta_native"], d["theta_unit"], d["x"]
    print(f"\nloaded {a.n_sims} rows: theta {tuple(theta_n.shape)}  x {tuple(x.shape)}\n")

    # ---- T4: x is bit-for-bit the vector Gibbs NPE trains on ----------------
    check("x_dim == 809", x.shape[1] == RD.X_DIM, f"{x.shape[1]}")
    check("theta_dim == 24", theta_n.shape[1] == RD.THETA_DIM, f"{theta_n.shape[1]}")

    stats = load_stats(Path(a.stats_path))
    man = load_manifest(Path(a.sim_data_root) / "manifest_train.json")
    ds = ReducedCVDataset(str(Path(a.sim_data_root) / "train"), man["index"][:16], stats)
    ref_t = torch.stack([ds[i][0] for i in range(16)])
    ref_x = torch.stack([ds[i][1] for i in range(16)])
    ds.close()
    check("x identical to an independent ReducedCVDataset read (T4)",
          torch.equal(ref_x, x[:16]),
          f"max|diff|={float((ref_x - x[:16]).abs().max()):.3e}")
    check("theta_native identical to the same read",
          torch.equal(ref_t, theta_n[:16]))

    # ---- hr_z location ------------------------------------------------------
    check("WAVE_LEN == 804", WAVE_LEN == 804, f"{WAVE_LEN}")
    check("HR_Z_INDEX is the last x dim", RN.HR_Z_INDEX == RD.X_DIM - 1,
          f"{RN.HR_Z_INDEX} vs {RD.X_DIM - 1}")
    check("hr_z column is not constant (it is a real signal)",
          float(x[:, RN.HR_Z_INDEX].std()) > 1e-3,
          f"std={float(x[:, RN.HR_Z_INDEX].std()):.4f}")

    # ---- theta unit-cube coordinates ---------------------------------------
    check("theta_unit strictly inside [0,1]",
          bool((theta_u >= 0).all() and (theta_u <= 1).all()),
          f"min={float(theta_u.min()):.3e} max={float(theta_u.max()):.6f}")
    tf, log_dims = RD.box_to_unit(a.nested_sim_root)
    back, _ = tf.inverse(theta_u)
    rel = ((back - theta_n).abs() / theta_n.abs().clamp_min(1e-12)).max()
    check("unit-cube round-trip recovers native theta", float(rel) < 1e-4,
          f"max rel err={float(rel):.3e}")
    names = [PARAM_KEYS_INFER[i] for i in log_dims]
    check("log dims are exactly Emax_RV, Eap, Rap",
          sorted(names) == sorted(["Emax_RV", "Eap", "Rap"]), f"{names}")

    # ---- the prior really is uniform in these coordinates ------------------
    # This is what makes tempering the joint score equivalent to the intended power
    # posterior (REPRESENTATION.md section 2). If the bank were not log-uniform on the
    # log dims, those three would fail here.
    worst = (None, 0.0)
    for j in range(RD.THETA_DIM):
        u = np.sort(theta_u[:, j].numpy())
        ks = float(np.abs(np.arange(1, len(u) + 1) / len(u) - u).max())
        if ks > worst[1]:
            worst = (PARAM_KEYS_INFER[j], ks)
    crit = 1.63 / np.sqrt(a.n_sims)          # ~99% KS critical value
    check("every theta dim is uniform in unit-cube coords (KS)", worst[1] < crit,
          f"worst {worst[0]} D={worst[1]:.4f} vs crit {crit:.4f}")

    # ---- hr_z is held exact by both of our masked functions ----------------
    m = RN._clean_mask(RD.X_DIM, RN.CLEAN_X_DIMS, torch.device("cpu"), torch.float32)
    check("clean mask zeroes only hr_z",
          float(m.sum()) == RD.X_DIM - 1 and float(m[0, RN.HR_Z_INDEX]) == 0.0)

    class ZeroScore(torch.nn.Module):
        """Returns a nonzero score everywhere, so any unmasked dim must move."""
        def __init__(self, d):
            super().__init__()
            self.d = d
            self.p = torch.nn.Parameter(torch.zeros(1))
        def forward(self, th, xx, sig):
            return torch.ones(th.shape[0], self.d, device=th.device) * 0.5

    init = torch.cat([theta_u[:64], x[:64]], dim=1)
    out, diags = RN.Lgmcmc(torch.tensor([0.5, 1.0]), ZeroScore(RD.THETA_DIM + RD.X_DIM),
                           RD.THETA_DIM, RD.X_DIM, init, initial_step=2,
                           step_increase=0, device="cpu")
    hr_col = RD.THETA_DIM + RN.HR_Z_INDEX
    check("Lgmcmc leaves hr_z exactly unchanged",
          torch.equal(out[:, hr_col], init[:, hr_col]),
          f"max|diff|={float((out[:, hr_col] - init[:, hr_col]).abs().max()):.3e}")
    moved = (out[:, RD.THETA_DIM:-1] - init[:, RD.THETA_DIM:-1]).abs().max()
    check("Lgmcmc does move the other 808 x dims", float(moved) > 0, f"max|diff|={float(moved):.3e}")
    check("Lgmcmc returns per-noise-level diagnostics", len(diags) == 2 and
          "frac_theta_out_of_box" in diags[0])

    # ---- beta partition ----------------------------------------------------
    betas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]
    slices, per, unused = RD.beta_partition(a.n_sims, len(betas))
    allix = np.concatenate(slices)
    check("beta slices are disjoint", len(np.unique(allix)) == len(allix),
          f"{len(allix)} rows, {len(np.unique(allix))} unique")
    check("beta slices are equal sized and account for the bank",
          len(slices) == len(betas) and per * len(betas) + unused == a.n_sims,
          f"{len(betas)} x {per} + {unused} = {a.n_sims}")

    print()
    if _fails:
        print(f"FAILED: {len(_fails)} check(s): {_fails}")
        sys.exit(1)
    print("all phase-1 checks passed")


if __name__ == "__main__":
    main()
