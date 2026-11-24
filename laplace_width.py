"""Laplace width of the Gibbs posterior: an unsupervised sharpness target.

The training loss targets pi_w(theta|x) ~ pi(theta) exp(-(w/2)||phi(x) - phi(f(theta))||^2_Sigma),
and near theta_true that target is approximately Gaussian with precision

    A(w) = w J^T Sigma^-1 J + PRIOR_PREC * I,   J = d phi(f(theta)) / d u

in unit-cube parameter coordinates u, so its central 95% width is 3.92 * sqrt(diag(A^-1)).
Comparing the flow's own width against that number separates a flow that has not reached
its target from a target that is itself mis-specified -- the two have identical signatures
in any score computed against the target.

Everything here uses the LOFI simulator (the one the Gibbs target is defined with) and
the validation sims' own theta, so it needs no labels and no high-fidelity model. J costs
2 * n_params simulator calls per case, paid once and cached, after which the per-epoch
cost is one flow sampling pass.

Build a cache before training:

    python laplace_width.py --metric-dir <ckpt> --nested-sim-root ~/projects/nested-sim \
        --data-root ~/outputs/nested-sim/data --out <cache>.npz

then pass it to train_gibbs_npe.py as --laplace-cache.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

PRIOR_PREC = 12.0        # 1 / Var(U[0,1]): the box prior, moment-matched to a Gaussian
Z95 = 3.92               # width of the central 95% interval of a unit normal
PRIOR_WIDTH95 = Z95 / PRIOR_PREC ** 0.5   # the width a wholly unconstrained parameter has

WAVES = ["Prv", "Pra", "Pvp", "Pap"]

# Worker-global state, set by _init in each forked process.
_SETTINGS = None
_KEYS = None
_W = {}


# --------------------------------------------------------------------------- the target

def sigma_inv_apply(M, correction):
    """Sigma_c^{-1} M for Sigma_c = I + U diag(a) U^T, columnwise in M."""
    if correction is None:
        return M
    U = np.asarray(correction["U"], float)
    a = np.asarray(correction["a"], float)
    return M - U @ ((a / (1.0 + a))[:, None] * (U.T @ M))


def laplace_width95(J, w, correction=None, prior_prec=PRIOR_PREC, per_param=False):
    """Mean central-95% width of the linearised Gibbs posterior, in unit-cube coords.

    J is (n_cases, d_phi, n_params). Averaged over cases, and over parameters unless
    per_param, so the result is directly comparable to a flow's mean width95.
    """
    J = np.asarray(J, float)
    eye = np.eye(J.shape[2])
    sds = np.empty((J.shape[0], J.shape[2]))
    for i, Ji in enumerate(J):
        A = w * (Ji.T @ sigma_inv_apply(Ji, correction)) + prior_prec * eye
        sds[i] = np.sqrt(np.clip(np.diag(np.linalg.inv(A)), 0.0, None))
    return Z95 * (sds.mean(0) if per_param else float(sds.mean()))


def load_cache(path):
    """Cache as written by build_cache: J, x (flow-input z-space), theta, meta."""
    d = np.load(path, allow_pickle=False)
    return {"J": d["J"], "x": d["x"], "theta": d["theta"],
            "meta": json.loads(str(d["meta"]))}


# ------------------------------------------------------------------- building the cache

def _init(ss_tol, Q, s_wave, s_scal, keys, nested_sim_root):
    global _SETTINGS, _KEYS
    sys.path.insert(0, str(nested_sim_root))
    from nested_sim import SolverSettings
    _SETTINGS = SolverSettings(ss_tol=ss_tol)
    _KEYS = keys
    _W.update(Q=Q, s_wave=s_wave, s_scal=s_scal)


def _phi_of_theta(task):
    """(theta row in binary-named order, HR) -> phi, or None if the solve fails."""
    from nested_sim import simulate
    from nested_sim.params import from_binary_names
    theta_row, hr = task
    th = from_binary_names(dict(zip(_KEYS, theta_row.tolist())))
    th["HR"] = hr
    r = simulate(th, None, _SETTINGS)
    if not r.success:
        return None
    wv = np.stack([r.waves201[c] for c in WAVES]).astype(np.float64)
    pas, vlv = r.waves201["Pas"], r.waves201["Vlv"]
    scal = np.array([pas.mean(), pas.max(), pas.min(),
                     np.log(max(vlv.max() - vlv.min(), 1e-6))])
    w = np.einsum("ct,tk->ck", wv, _W["Q"]) / _W["s_wave"][:, None]
    return np.concatenate([w.ravel(), scal / _W["s_scal"]])


def _keep_positions(root, load_manifest):
    """lofi_test positions excluding QC failures and the 1000 metric-fitting cases.

    The RandomState(0) draw reproduces the held-out split used to fit the metric, so the
    Jacobian cases are ones no metric estimate was tuned on.
    """
    man = load_manifest(root / "manifest.json")
    bad = set(json.load(open(root / "qc_report.json"))["excluded_groups"])
    keep = np.ones(10_000, bool)
    keep[np.random.RandomState(0).choice(10_000, 1000, replace=False)] = False
    for i, e in enumerate(man["index"]):
        if f"{e['file']}/{e['group']}" in bad:
            keep[i] = False
    return man, np.flatnonzero(keep)


def build_cache(a):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(a.nested_sim_root)))
    from dataset import ReducedCVDataset, load_manifest, load_stats
    from train_gibbs_npe import Metric, PARAM_KEYS_INFER
    from nested_sim.params import THETA_BOUNDS, LOG_SAMPLED, to_binary_names

    keys = list(PARAM_KEYS_INFER)
    data_root = Path(a.data_root)
    lo_root = data_root / a.split
    meta_j = json.load(open(Path(a.metric_dir) / "gibbs_metric.json"))
    metric = Metric.from_npz(Path(a.metric_dir) / "gibbs_metric.npz", meta_j["ell"])
    stats = load_stats(a.stats_path or (data_root / "lofi_train/norm_stats.json"))

    if a.indices:
        # Explicit manifest positions, for comparing against artifacts that already fixed
        # their own cases (the gold SMC campaign indexes the manifest directly). The QC /
        # metric-holdout filtering is bypassed: the caller has already chosen.
        man = load_manifest(lo_root / "manifest.json")
        pos = [int(v) for v in a.indices.split(",")]
        entries = [man["index"][i] for i in pos]
    else:
        man, kept = _keep_positions(lo_root, load_manifest)
        sel = np.random.RandomState(a.seed).choice(len(kept), a.n_cases, replace=False)
        pos = [int(kept[s]) for s in sel]
        entries = [man["index"][p] for p in pos]

    # The flow's own input, built by the training dataset class itself so the layout and
    # normalization cannot drift from what the flow was trained on.
    ds = ReducedCVDataset(str(lo_root), entries, stats)
    theta = np.stack([ds[i][0].numpy() for i in range(len(ds))])
    x = np.stack([ds[i][1].numpy() for i in range(len(ds))])
    ds.close()

    bb = to_binary_names(THETA_BOUNDS)
    lo = np.array([bb[k][0] for k in keys], float)
    hi = np.array([bb[k][1] for k in keys], float)
    log_idx = [keys.index(k) for k in to_binary_names({k: 0 for k in LOG_SAMPLED})]
    lo_z, hi_z = lo.copy(), hi.copy()
    lo_z[log_idx], hi_z[log_idx] = np.log(lo[log_idx]), np.log(hi[log_idx])
    step = a.h * (hi_z - lo_z)

    tasks, book = [], []
    for ci, e in enumerate(entries):
        with h5py.File(lo_root / e["file"], "r") as f:
            hr = float(f[e["group"]]["parameters/HR"][()])
        z = theta[ci].astype(float).copy()
        z[log_idx] = np.log(z[log_idx])
        for j in range(len(keys)):
            for sgn in (+1, -1):
                zp = z.copy()
                zp[j] += sgn * step[j]
                tp = zp.copy()
                tp[log_idx] = np.exp(zp[log_idx])
                tasks.append((tp, hr))
                book.append((ci, j if sgn > 0 else ~j))

    print(f"  {len(tasks)} simulator calls ({a.workers} workers) ...", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init,
                             initargs=(a.ss_tol, metric.Q, metric.s_wave, metric.s_scal,
                                       keys, a.nested_sim_root)) as ex:
        out = list(ex.map(_phi_of_theta, tasks, chunksize=4))
    print(f"  failed solves: {sum(o is None for o in out)} of {len(tasks)}")

    by = {}
    for (ci, j), o in zip(book, out):
        by.setdefault(ci, {})[j] = o
    nd, keep_ci, Js = len(keys), [], []
    for ci in range(len(entries)):
        rec = by[ci]
        if any(rec.get(j) is None or rec.get(~j) is None for j in range(nd)):
            continue
        # u is in unit-cube units and the FD step is a.h of the box, so the central
        # difference divides by 2 * a.h, not by the physical step.
        Js.append(np.stack([(rec[j] - rec[~j]) / (2 * a.h) for j in range(nd)], 1))
        keep_ci.append(ci)

    J = np.stack(Js)
    meta = {"metric_dir": str(a.metric_dir), "split": a.split, "ell": meta_j["ell"],
            "d_eff": meta_j["d_eff"], "h": a.h, "ss_tol": a.ss_tol, "seed": a.seed,
            "n_requested": a.n_cases, "param_keys": keys,
            "manifest_positions": [int(pos[c]) for c in keep_ci],
            "cases": [f"{entries[c]['file']}/{entries[c]['group']}" for c in keep_ci]}
    np.savez_compressed(a.out, J=J, x=x[keep_ci], theta=theta[keep_ci],
                        meta=json.dumps(meta))
    print(f"  usable cases: {len(Js)}  J {J.shape}  -> {a.out}")

    cor = np.load(a.correction) if a.correction else None
    print(f"\n{'w':>10}  {'width95':>9}")
    for w in [float(v) for v in a.w_grid.split(",")]:
        print(f"{w:10.4f}  {laplace_width95(J, w, cor):9.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric-dir", required=True,
                    help="checkpoint dir holding gibbs_metric.{json,npz}; J is "
                         "metric-dependent, so a cache is valid only for its own metric")
    ap.add_argument("--nested-sim-root", default=str(Path.home() / "projects/nested-sim"))
    ap.add_argument("--data-root", default=str(Path.home() / "outputs/nested-sim/data"))
    ap.add_argument("--stats-path", default="")
    ap.add_argument("--split", default="lofi_test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--correction", default="", help="Sigma_c npz, for the printed sweep")
    ap.add_argument("--n-cases", type=int, default=60)
    ap.add_argument("--indices", default="",
                    help="explicit comma-separated manifest positions instead of a random "
                         "draw; use when matching cases another artifact already fixed")
    ap.add_argument("--workers", type=int, default=192)
    ap.add_argument("--h", type=float, default=2e-3, help="FD step, fraction of the box")
    ap.add_argument("--ss-tol", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--w-grid", default="0.755,1.033,2.0,7.5539,20.0,75.5")
    build_cache(ap.parse_args())


if __name__ == "__main__":
    main()
