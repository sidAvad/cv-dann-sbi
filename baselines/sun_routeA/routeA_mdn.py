"""Phase 6: Route A's inference network -- the conditional MDN p_beta(theta|x).

Thin driver around vendored `train_conditional_mdn`, which owns the loop, the z-scoring,
the ReduceLROnPlateau schedule and its own early stopping (patience 20). Nothing about the
method is reimplemented here; this module only feeds it the synthesized pairs, saves what
it returns, and measures the rejection rate its own inference path implies.

Two upstream call arguments are set away from their defaults, both for headless running
rather than for method reasons: `plot=False` (the default calls `plt.show()`) and
`verbose=True`.

Out-of-support theta is deliberately NOT filtered from the training set. Upstream's
synthesis applies no reflecting barrier and its inference path handles the consequence with
`sample_with_rejection` against explicit box bounds, so filtering here would depart from
their design and hide the cost. With ~74.5% of synthesized theta outside the unit cube (see
REPRESENTATION.md section 11), that cost is what `--rejection-probe` measures.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # vendor imports pyplot

from RouteB_utils import (train_conditional_mdn,                # noqa: E402 (vendored)
                          sample_with_rejection)


def rejection_probe(model, norm_theta, norm_x, x_obs, betas, n_samples, device, log=print):
    """Acceptance rate of upstream's own inference path, per beta.

    `sample_with_rejection` draws `num_samples*oversample` and keeps those inside the box,
    retrying up to `max_rounds`. Rather than infer the rate from its output (it returns a
    fixed count, padding from the last draw if it never fills), the raw MDN draw is measured
    directly: the fraction of a single unrejected draw that lands in [0,1]^24.
    """
    from RouteB_utils import mdn_sample
    lower = torch.zeros(model.theta_dim, device=device)
    upper = torch.ones(model.theta_dim, device=device)
    rows = []
    for beta in betas:
        accs = []
        for i in range(x_obs.shape[0]):
            th = mdn_sample(model, x_obs[i:i + 1], float(beta), norm_theta, norm_x,
                            num_samples=n_samples, device=device)
            inside = ((th >= lower) & (th <= upper)).all(dim=1).float().mean()
            accs.append(float(inside))
        # does upstream's wrapper actually fill its order at this beta?
        got = sample_with_rejection(model, norm_theta, norm_x, x_obs[0:1], float(beta),
                                    lower, upper, num_samples=n_samples, device=device)
        filled = float(((got >= lower) & (got <= upper)).all(dim=1).float().mean())
        rows.append({"beta": float(beta), "accept_mean": float(np.mean(accs)),
                     "accept_min": float(np.min(accs)), "accept_max": float(np.max(accs)),
                     "wrapper_frac_in_box": filled})
        log(f"  beta {beta:<4g} accept {np.mean(accs):.4f} "
            f"[{np.min(accs):.4f}, {np.max(accs):.4f}]  "
            f"wrapper output in-box {filled:.4f}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synth", required=True, help="dir holding pairs_all.pt")
    p.add_argument("--run", required=True)
    p.add_argument("--outputs-root", default="/home/sa4604/outputs/cv-dann-sbi")
    # upstream train_conditional_mdn defaults
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--n-components", type=int, default=8)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.1)
    # ours
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rejection-probe", type=int, default=64,
                   help="number of held-out x to measure acceptance on (0 to skip)")
    p.add_argument("--probe-samples", type=int, default=2000)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    run_dir = Path(a.outputs_root) / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    ckpt_dir = run_dir / "checkpoints" / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    d = torch.load(Path(a.synth) / "pairs_all.pt", weights_only=False)
    pairs, td, xd = d["pairs_all"], d["theta_dim"], d["x_dim"]
    theta = pairs[:, :td].clone()
    x = pairs[:, td:td + xd].clone()
    beta = pairs[:, td + xd].clone()
    del d, pairs
    log(f"pairs {theta.shape[0]}  theta {td}  x {xd}  betas {sorted(set(beta.tolist()))}")
    oob = float(((theta < 0) | (theta > 1)).any(dim=1).float().mean())
    log(f"theta outside unit cube: {oob:.4f} (kept; upstream rejects at inference)")

    with open(ckpt_dir / "config.json", "w") as f:
        json.dump({**vars(a), "theta_dim": td, "x_dim": xd,
                   "n_pairs": int(theta.shape[0]), "frac_theta_oob": oob}, f, indent=2)

    t0 = time.time()
    model, norm_theta, norm_x, best_val = train_conditional_mdn(
        theta, x, beta,
        hidden=a.hidden, n_components=a.n_components,
        n_hidden_layers=a.n_hidden_layers, batch_size=a.batch_size,
        epochs=a.epochs, lr=a.lr, patience=a.patience,
        val_fraction=a.val_fraction, device=device,
        verbose=True, plot=False)
    log(f"trained in {(time.time()-t0)/60:.1f} min; best val NLL {best_val:.6f}")

    torch.save({"model": model.state_dict(),
                "norm_theta": norm_theta, "norm_x": norm_x,
                "best_val": best_val,
                "theta_dim": td, "x_dim": xd,
                "arch": {"hidden": a.hidden, "n_components": a.n_components,
                         "n_hidden_layers": a.n_hidden_layers}},
               ckpt_dir / "mdn.pt")

    results = {"best_val_nll": best_val, "frac_theta_oob_train": oob}
    if a.rejection_probe:
        log(f"\nrejection probe on {a.rejection_probe} held-out x, "
            f"{a.probe_samples} draws each:")
        base = torch.load(Path(a.synth).parent / "routeA_base" / "base.pt",
                          weights_only=False)
        vp = base["val_pairs"]
        x_obs = vp[:a.rejection_probe, td:td + xd].to(device).float()
        del base, vp
        betas = sorted(set(beta.tolist()))
        results["rejection"] = rejection_probe(model, norm_theta, norm_x, x_obs, betas,
                                               a.probe_samples, device, log)
    with open(ckpt_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log(f"\ndone; artifacts in {ckpt_dir}")


if __name__ == "__main__":
    main()
