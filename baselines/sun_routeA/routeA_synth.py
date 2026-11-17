"""Phase 4b: synthesize Route A's (theta, x, beta) pairs with annealed Langevin.

Mirrors the synthesis block of `vendor/RouteA_generate_pairs.py`: for each beta, run
`Lgmcmc` with `t_power=beta` and emit `[theta | x | beta]` rows, then concatenate over
betas into one `pairs_all.pt` for the conditional MDN to train on.

Three things differ from upstream, all of them already recorded in REPRESENTATION.md:

* chains start from the lofi bank's disjoint per-beta partition rather than
  `prior`/`simulator` calls (section 5), so Route A's simulation budget equals Gibbs
  NPE's and no beta reuses another's rows;
* `hr_z` (x dim 808) is never moved (section 4);
* the 13-value un-shadowed beta grid is used, not the flattened `[0.1..1.0]` that
  upstream's line 107 overwrites line 106 with (section 1).

Each beta is written as its own file as soon as it finishes, so an interrupted run
resumes by re-invocation -- a completed beta is skipped unless `--overwrite`. The
per-noise-level diagnostics `Lgmcmc` returns are kept verbatim in `synth_meta.json`
alongside, per beta, the covariance-trace ratio of the sampled theta against the bank
slice it started from. That ratio is the real-data analogue of the quantity the phase-2
toy reported as `cov ratio theta`, where it came out at the untempered value for every
beta; it is logged here so the same question can be asked of the real run without any
extra compute.
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

import routeA_noise as RN                                      # noqa: E402
from NSEutils import ScoreNet_sigma                            # noqa: E402 (vendored)
from RouteA_utils import make_gamma_schedule                   # noqa: E402 (vendored)


def beta_tag(b):
    """Filename-safe, round-trip-stable tag for a beta value."""
    return f"{b:g}".replace(".", "p")


def load_score_net(ckpt_path, theta_dim, x_dim, device, log=print):
    """Build the net from the training run's own config.json, then load weights.

    Reading `emb_dim`/`hidden_dim`/`gamma_*`/`T` from the checkpoint directory rather than
    re-specifying them here means a mismatch between the trained and sampled architecture
    cannot happen silently.
    """
    ckpt_path = Path(ckpt_path)
    cfg_path = ckpt_path.parent / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json beside {ckpt_path}; cannot verify architecture")
    cfg = json.load(open(cfg_path))

    for k, v in (("theta_dim", theta_dim), ("x_dim", x_dim)):
        if cfg.get(k) != v:
            raise SystemExit(f"{k} mismatch: base.pt says {v}, score net was trained "
                             f"with {cfg.get(k)}")
    if list(cfg.get("clean_x_dims", [])) != list(RN.CLEAN_X_DIMS):
        raise SystemExit(f"clean_x_dims mismatch: score net trained with "
                         f"{cfg.get('clean_x_dims')}, sampler would use {RN.CLEAN_X_DIMS}")

    model = ScoreNet_sigma(theta_dim=theta_dim, x_dim=x_dim, amortize=True,
                           emb_dim=cfg["emb_dim"], sigma_emb_dim=cfg["emb_dim"],
                           hidden_dim=cfg["hidden_dim"],
                           sigma_min=cfg["gamma_min"], sigma_max=cfg["gamma_max"])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)
    log(f"score net {ckpt_path.name} from {ckpt_path.parent.name}: "
        f"emb {cfg['emb_dim']}, hidden {cfg['hidden_dim']}, "
        f"sigma [{cfg['gamma_min']}, {cfg['gamma_max']}], T {cfg['T']}")
    return model, cfg


def cov_trace_ratio(sampled, reference):
    """tr(cov(sampled)) / tr(cov(reference)), both (N, d) on CPU.

    Trace rather than per-dim so one number summarises all 24 theta dims; the toy
    reported the same quantity, which is what makes the two comparable.
    """
    a = sampled.double().var(dim=0, unbiased=True).sum()
    b = reference.double().var(dim=0, unbiased=True).sum()
    return float(a / b)


def synth_one_beta(model, gammas, base, beta, init_idx, theta_dim, x_dim,
                   initial_step, step_increase, device, cmin_x, cmax_x,
                   cmin_theta, cmax_theta, log=print):
    init = base["pairs"][torch.from_numpy(init_idx)]
    t0 = time.time()
    out, diags = RN.Lgmcmc(gammas=gammas, model=model,
                           theta_dim=theta_dim, x_dim=x_dim,
                           init_theta_and_x=init,
                           initial_step=initial_step, t_power=float(beta),
                           step_increase=step_increase, device=device,
                           cmin_x=cmin_x, cmax_x=cmax_x,
                           cmin_theta=cmin_theta, cmax_theta=cmax_theta,
                           collect_diagnostics=True)
    dt = time.time() - t0
    out = out.cpu()

    th_s, th_0 = out[:, :theta_dim], init[:, :theta_dim]
    x_s, x_0 = out[:, theta_dim:], init[:, theta_dim:]
    summary = {
        "beta": float(beta),
        "n_chains": int(out.shape[0]),
        "total_steps": sum(d["steps"] for d in diags),
        "secs": round(dt, 1),
        "cov_trace_ratio_theta": cov_trace_ratio(th_s, th_0),
        "cov_trace_ratio_x": cov_trace_ratio(x_s, x_0),
        "mean_abs_dtheta": float((th_s - th_0).abs().mean()),
        "mean_abs_dx": float((x_s - x_0).abs().mean()),
        "frac_theta_out_of_box": float(((th_s < 0) | (th_s > 1)).any(dim=1)
                                       .float().mean()),
        "max_abs_excursion": float((th_s - th_s.clamp(0, 1)).abs().max()),
        "hr_z_moved_max": float((x_s[:, RN.HR_Z_INDEX]
                                 - x_0[:, RN.HR_Z_INDEX]).abs().max()),
        "n_nonfinite": int((~torch.isfinite(out)).sum()),
        "per_sigma": diags,
    }
    log(f"  beta {beta:<4g} {dt/60:5.1f} min  covratio th {summary['cov_trace_ratio_theta']:.3f} "
        f"x {summary['cov_trace_ratio_x']:.3f}  |dth| {summary['mean_abs_dtheta']:.4f}  "
        f"|dx| {summary['mean_abs_dx']:.4f}  oob {summary['frac_theta_out_of_box']:.3f}  "
        f"clampbound th/x {np.mean([d['frac_steps_clamp_bound_theta'] for d in diags]):.2f}/"
        f"{np.mean([d['frac_steps_clamp_bound_x'] for d in diags]):.2f}")
    return out, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="dir holding base.pt from routeA_data.py")
    p.add_argument("--score-net", required=True,
                   help="path to score_net_best.pt (its sibling config.json is read too)")
    p.add_argument("--out", required=True, help="output dir for synth_beta_*.pt")
    p.add_argument("--betas", default=None,
                   help="comma-separated; default: the grid recorded in base.pt")
    p.add_argument("--only-beta", default=None,
                   help="synthesize just these betas (comma-separated), for a partial rerun")
    # upstream sampler settings
    p.add_argument("--initial-step", type=int, default=10)
    p.add_argument("--step-increase", type=int, default=10)
    # x step clamp. Upstream ships two different windows: Lgmcmc's signature says
    # [1e-2, 0.5] and is what its call site uses, while alpha_from_snr_blocks' own
    # defaults are [2e-5, 1e-4] (identical to theta's). At d_x=809 the former diverges;
    # see REPRESENTATION.md section 11 for the measured scan behind these values.
    p.add_argument("--cmin-x", type=float, default=1e-3)
    p.add_argument("--cmax-x", type=float, default=1e-2)
    # theta step clamp. Upstream's [2e-5, 1e-4] binds for ANY score in ANY dimension: with
    # an accurate score their own SNR rule returns 0.125*sigma^2/beta^2 (d_theta cancels),
    # which is 1250x above cmax_theta at beta=1. So beta never reaches theta's step. These
    # defaults are upstream's; raise cmax-theta to let their rule drive theta instead of
    # their guard rail. See REPRESENTATION.md section 12.
    p.add_argument("--cmin-theta", type=float, default=2e-5)
    p.add_argument("--cmax-theta", type=float, default=1e-4)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-assemble", action="store_true",
                   help="write per-beta files only; skip pairs_all.pt")
    p.add_argument("--smoke", type=int, default=0,
                   help="truncate each beta to this many chains and write nothing; for "
                        "wiring checks only. The step size is the batch median, so a "
                        "truncated run is NOT numerically the real one")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    base = torch.load(Path(a.base) / "base.pt", weights_only=False)
    theta_dim, x_dim = base["theta_dim"], base["x_dim"]
    betas = ([float(b) for b in a.betas.split(",")] if a.betas else list(base["betas"]))
    slices = base["init_slices"]
    if len(slices) != len(betas):
        raise SystemExit(f"base.pt holds {len(slices)} init slices but {len(betas)} betas "
                         f"were requested; the partition must match the grid")

    model, cfg = load_score_net(a.score_net, theta_dim, x_dim, device, log)
    gammas = make_gamma_schedule(T=cfg["T"], gamma_min=cfg["gamma_min"],
                                 gamma_max=cfg["gamma_max"]).to(device)
    n_steps = sum(a.initial_step + i * a.step_increase for i in range(cfg["T"]))
    log(f"betas {betas}")
    log(f"{len(slices[0])} chains/beta, {n_steps} Langevin steps/chain, "
        f"clean x dims {RN.CLEAN_X_DIMS}, device {device}")
    log(f"x step clamp [{a.cmin_x:g}, {a.cmax_x:g}]*sigma^2 "
        f"(upstream Lgmcmc call site uses [1e-2, 0.5], which diverges at this d_x)")
    log(f"theta step clamp [{a.cmin_theta:g}, {a.cmax_theta:g}]*sigma^2 "
        f"(upstream [2e-5, 1e-4]; their SNR rule asks for 0.125/beta^2)")

    wanted = (set(float(b) for b in a.only_beta.split(",")) if a.only_beta
              else set(betas))
    if a.smoke:
        slices = [s[:a.smoke] for s in slices]
        log(f"SMOKE: {a.smoke} chains/beta, nothing written, numbers not comparable")
    summaries = {}
    for i, beta in enumerate(betas):
        f = out / f"synth_beta_{beta_tag(beta)}.pt"
        if beta not in wanted:
            continue
        if a.smoke:
            _, summary = synth_one_beta(model, gammas, base, beta, slices[i],
                                        theta_dim, x_dim,
                                        a.initial_step, a.step_increase, device,
                                        a.cmin_x, a.cmax_x,
                                        a.cmin_theta, a.cmax_theta, log)
            summaries[beta_tag(beta)] = summary
            continue
        if f.exists() and not a.overwrite:
            d = torch.load(f, weights_only=False)
            summaries[beta_tag(beta)] = d["summary"]
            log(f"  beta {beta:<4g} already done -> {f.name} (use --overwrite to redo)")
            continue
        rows, summary = synth_one_beta(model, gammas, base, beta, slices[i],
                                       theta_dim, x_dim,
                                       a.initial_step, a.step_increase, device,
                                       a.cmin_x, a.cmax_x,
                                       a.cmin_theta, a.cmax_theta, log)
        tmp = f.with_suffix(".pt.tmp")
        torch.save({"rows": rows, "beta": float(beta),
                    "init_idx": slices[i], "summary": summary}, tmp)
        tmp.replace(f)
        summaries[beta_tag(beta)] = summary

    if a.smoke:
        log("\nSMOKE done; no files written")
        return

    meta = {"base": str(Path(a.base).resolve()),
            "score_net": str(Path(a.score_net).resolve()),
            "betas": betas, "n_per_beta": int(len(slices[0])),
            "langevin_steps_per_chain": n_steps,
            "initial_step": a.initial_step, "step_increase": a.step_increase,
            "cmin_x": a.cmin_x, "cmax_x": a.cmax_x,
            "cmin_theta": a.cmin_theta, "cmax_theta": a.cmax_theta,
            "clean_x_dims": list(RN.CLEAN_X_DIMS),
            "theta_coords": "unit_cube",
            "per_beta": summaries}
    with open(out / "synth_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    present = [b for b in betas if (out / f"synth_beta_{beta_tag(b)}.pt").exists()]
    if a.no_assemble or len(present) != len(betas):
        log(f"\n{len(present)}/{len(betas)} betas on disk; pairs_all.pt not written")
        return

    log("\nassembling pairs_all.pt ...")
    chunks = []
    for beta in betas:
        d = torch.load(out / f"synth_beta_{beta_tag(beta)}.pt", weights_only=False)
        r = d["rows"]
        chunks.append(torch.cat(
            [r, torch.full((r.shape[0], 1), float(beta), dtype=r.dtype)], dim=-1))
    pairs_all = torch.cat(chunks, dim=0)
    del chunks
    torch.save({"pairs_all": pairs_all, "theta_dim": theta_dim, "x_dim": x_dim,
                "betas": betas, "layout": "theta | x | beta"},
               out / "pairs_all.pt")
    log(f"wrote {out/'pairs_all.pt'}  {tuple(pairs_all.shape)}  "
        f"(theta {theta_dim} | x {x_dim} | beta 1)")

    log("\n beta  covratio th  covratio x  mean|dth|  mean|dx|   oob  clampbound th/x")
    for beta in betas:
        s = summaries[beta_tag(beta)]
        d = s["per_sigma"]
        log(f"{beta:5g}  {s['cov_trace_ratio_theta']:11.3f}  "
            f"{s['cov_trace_ratio_x']:10.3f}  {s['mean_abs_dtheta']:9.4f}  "
            f"{s['mean_abs_dx']:8.4f}  {s['frac_theta_out_of_box']:5.3f}  "
            f"{np.mean([x['frac_steps_clamp_bound_theta'] for x in d]):.2f}/"
            f"{np.mean([x['frac_steps_clamp_bound_x'] for x in d]):.2f}")


if __name__ == "__main__":
    main()
