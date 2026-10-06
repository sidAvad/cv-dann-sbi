"""
Gibbs-posterior NPE baseline: one w-conditioned q(theta | x, w), flow only, no critics.

Target: nu_w(theta | x) ∝ pi(theta) exp{-(w/2) ||x - f(theta)||^2_M}, learned by Gaussian
noise augmentation with covariance (wM)^{-1} and log w as an extra conditioning input.

Stages (every fresh invocation):
  1. Metric M: NN residuals of calibration patients against the sim bank give the temporal
     length scale ell, per-channel scales s_c / s_k, and a trace normalization. Frozen after.
  2. Plug-in w_hat = d_eff / mean_b ||x_b - f(theta_hat_b)||^2_M (calibration patients, NN
     residuals only; the simulator is not differentiable, so no gradient refinement).
  3. Training: log w ~ U[log(w_hat/10), log(10 w_hat)] independent of theta, fresh noise per
     batch. Conditioning = [E(x_noisy), WEmbed(standardized log w)].
     Phases: flow-only (encoder frozen), encoder-only (flow frozen), joint.

--resume-dir reuses a previous run's metric (M, w_hat) and continues training from its
epoch, so the noise model and conditioning stay identical across the extension.

Observation layout (dataset.py): x[:804] = 4 z-scored waveforms (Prv, Pra, Pvp, Pap; 201
steps each, channel-major), x[804:] = [MAP_z, SBP_z, DBP_z, SV_z, HR_z]. Noise and the
metric are defined in raw units (mmHg / log SV), so z-space perturbations are divided by
the normalization scale.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent))
from dataset import (
    PARAM_KEYS_INFER,
    ReducedCVDataset, load_stats, load_manifest,
    WAVE_KEYS_REDUCED, N_REDUCED_CHANNELS as NW, N_SCALARS, T,
)
from models import LipschitzReducedAutoencoderEncoder
from train_joint import build_flow_net, load_real_beats

WAVE_LEN = NW * T
MATERN_LAGS = 40
W_EMBED_DIM = 16


# ── Metric M ──────────────────────────────────────────────────────────────────

def matern32(ell: float, n: int = T) -> np.ndarray:
    d = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    a = np.sqrt(3.0) * d / ell
    return (1.0 + a) * np.exp(-a)


def eig_truncated(ell: float, thresh: float = 1e-4):
    K = matern32(ell) + 1e-6 * np.eye(T)
    lam, V = np.linalg.eigh(K)
    lam, V = lam[::-1], V[:, ::-1]
    keep = lam > thresh * lam[0]
    return lam[keep].copy(), V[:, keep].copy()


class Metric:
    """
    Block-diagonal M = blockdiag(s_c^{-2} K_t^{-1}, diag(s_k^{-2})) on the truncated temporal
    basis. phi(x) maps raw observations so that ||phi(x) - phi(y)||^2 = ||x - y||^2_M, and the
    matching noise is w^{-1/2} (s_c * V Lambda^{1/2} z_c, s_k z_k).
    """

    def __init__(self, ell, s_wave, s_scal, lam, V):
        self.ell = float(ell)
        self.s_wave = np.asarray(s_wave, dtype=np.float32)   # (NW,)
        self.s_scal = np.asarray(s_scal, dtype=np.float32)   # (4,) MAP, SBP, DBP mmHg; log SV
        self.lam = np.asarray(lam, dtype=np.float64)
        self.V = np.asarray(V, dtype=np.float64)
        self.k = len(self.lam)
        self.d_eff = NW * self.k + 4
        self.Q = (V / np.sqrt(lam)[None, :]).astype(np.float32)           # (T, k)
        self.Bn = (V * np.sqrt(lam)[None, :]).T.astype(np.float32)        # (k, T)

    def to_dict(self):
        return {"ell": self.ell, "k_t": self.k, "d_eff": self.d_eff,
                "s_wave": self.s_wave.tolist(), "s_scal": self.s_scal.tolist(),
                "eigvals": self.lam.tolist()}

    @classmethod
    def from_npz(cls, npz_path, ell):
        d = np.load(npz_path)
        return cls(ell, d["s_wave"], d["s_scal"], d["eigvals"], d["V"])


class Observables:
    """Dataset-layout z-scores -> raw units, using the shared sim-fit norm stats."""

    def __init__(self, stats):
        w = stats["waves"]
        self.wave_mean = np.array([w[k]["mean"] for k in WAVE_KEYS_REDUCED], np.float32)
        self.wave_std = np.array([w[k]["std"] for k in WAVE_KEYS_REDUCED], np.float32) + 1e-8
        self.pas_mean = np.float32(w["Pas"]["mean"])
        self.pas_std = np.float32(w["Pas"]["std"] + 1e-8)
        self.vlv_std = np.float32(w["Vlv"]["std"] + 1e-8)

    def raw(self, x: np.ndarray):
        n = len(x)
        waves = x[:, :WAVE_LEN].reshape(n, NW, T) * self.wave_std[None, :, None] + self.wave_mean[None, :, None]
        sc = x[:, WAVE_LEN:]
        map_ = sc[:, 0] * self.pas_std + self.pas_mean
        sbp = sc[:, 1] * self.pas_std + self.pas_mean
        dbp = sc[:, 2] * self.pas_std + self.pas_mean
        logsv = np.log(np.clip(sc[:, 3] * self.vlv_std, 1e-6, None))
        return waves.astype(np.float32), np.stack([map_, sbp, dbp, logsv], 1).astype(np.float32)


def phi_from_z(x: np.ndarray, obs: Observables, metric: Metric, chunk: int = 100_000) -> np.ndarray:
    out = np.empty((len(x), metric.d_eff), dtype=np.float32)
    for i in range(0, len(x), chunk):
        waves, scal = obs.raw(x[i:i + chunk])
        w = np.einsum("nct,tk->nck", waves, metric.Q) / metric.s_wave[None, :, None]
        s = scal / metric.s_scal[None, :]
        out[i:i + chunk] = np.concatenate([w.reshape(len(w), -1), s], 1)
    return out


def nearest_neighbours(bank_phi: np.ndarray, query_phi: np.ndarray, chunk: int = 64) -> np.ndarray:
    bank = torch.from_numpy(bank_phi)
    bank_sq = (bank ** 2).sum(1)
    idx = []
    for i in range(0, len(query_phi), chunk):
        q = torch.from_numpy(query_phi[i:i + chunk])
        d = (q ** 2).sum(1, keepdim=True) - 2 * q @ bank.T + bank_sq[None]
        idx.append(d.argmin(1))
    return torch.cat(idx).numpy()


def fit_ell(resid_waves: np.ndarray, lags: int = MATERN_LAGS):
    R = resid_waves.reshape(-1, T).astype(np.float64)
    denom = (R ** 2).mean()
    emp = np.array([(R[:, :T - h] * R[:, h:]).mean() / denom for h in range(lags + 1)])
    grid = np.logspace(0, np.log10(500), 300)
    lag_idx = np.arange(lags + 1)
    errs = [np.sum((np.exp(-np.sqrt(3.0) * lag_idx / g) * (1 + np.sqrt(3.0) * lag_idx / g) - emp) ** 2)
            for g in grid]
    return float(grid[int(np.argmin(errs))]), emp


def estimate_metric(sim_x, obs, cal_x, log=print, lam_thresh=1e-4):
    """Two-pass estimate of M from calibration residuals; returns (metric, w_hat, info)."""
    lam, V = eig_truncated(10.0, lam_thresh)
    m0 = Metric(10.0, np.ones(NW, np.float32), np.ones(4, np.float32), lam, V)
    bank0 = phi_from_z(sim_x, obs, m0)
    cal_w0, cal_s0 = obs.raw(cal_x)
    nn0 = nearest_neighbours(bank0, phi_from_z(cal_x, obs, m0))
    bank_w, bank_s = obs.raw(sim_x[nn0])
    r_w = cal_w0 - bank_w
    r_s = cal_s0 - bank_s

    if not (np.all(np.isfinite(r_w)) and np.all(np.isfinite(r_s)) and r_w.std() > 0):
        raise RuntimeError("calibration residuals are degenerate (zero or non-finite); "
                           "check that calibration beats are not in the sim bank")

    ell, emp = fit_ell(r_w)
    s_wave = np.sqrt((r_w ** 2).mean(axis=(0, 2))).astype(np.float32)
    s_scal = np.sqrt((r_s ** 2).mean(axis=0)).astype(np.float32)
    log(f"  metric pass 1: ell={ell:.2f} steps  s_wave={np.round(s_wave, 3)}  s_scal={np.round(s_scal, 3)}")

    lam, V = eig_truncated(ell, lam_thresh)
    k = len(lam)
    tr = sum(np.sum(1.0 / lam) / s ** 2 for s in s_wave) + np.sum(1.0 / s_scal.astype(np.float64) ** 2)
    d_eff = NW * k + 4
    c = d_eff / tr
    s_wave = (s_wave / np.sqrt(c)).astype(np.float32)
    s_scal = (s_scal / np.sqrt(c)).astype(np.float32)
    metric = Metric(ell, s_wave, s_scal, lam, V)

    bank1 = phi_from_z(sim_x, obs, metric)
    phi_cal = phi_from_z(cal_x, obs, metric)
    nn1 = nearest_neighbours(bank1, phi_cal)
    d2 = ((phi_cal - bank1[nn1]) ** 2).sum(1)
    w_hat = d_eff / d2.mean()
    log(f"  metric final: k_t={k} d_eff={d_eff} trace-scale c={c:.4g}  "
        f"w_hat={w_hat:.4g}  (NN residual, no refinement)  "
        f"mean ||r||^2_M={d2.mean():.3f} (expected {d_eff})")
    info = {"w_hat": float(w_hat), "d_eff": int(d_eff), "k_t": int(k), "ell_fit": ell,
            "trace_scale_c": float(c), "mean_resid_sq_M": float(d2.mean()),
            "empirical_acf_pass1": emp.tolist()}
    return metric, float(w_hat), info


def check_noise_consistency(sim_x, obs, metric, w_test, n_draws=256, tol=0.05, log=print):
    """Pre-launch unit check: noise from noisy_obs must have ||phi(noise)||^2 ~ d_eff / w."""
    m_t, o_t = metric_to_torch(metric, obs, "cpu")
    x0 = torch.from_numpy(sim_x[:n_draws])
    torch.manual_seed(0)
    lw = torch.full((n_draws, 1), float(np.log(w_test)))
    x_noisy = noisy_obs(x0, lw, m_t, o_t).numpy()
    delta = phi_from_z(x_noisy, obs, metric) - phi_from_z(sim_x[:n_draws], obs, metric)
    measured = (delta ** 2).sum(1).mean()
    expected = metric.d_eff / w_test
    rel = abs(measured - expected) / expected
    log(f"  noise check at w={w_test:.4g}: mean ||delta phi||^2 = {measured:.4g}, "
        f"expected d_eff/w = {expected:.4g}, rel err {rel:.3f}")
    if rel > tol:
        raise RuntimeError(f"noise/metric mismatch: rel err {rel:.3f} > {tol}")


def noisy_obs(x, log_w, metric_t, obs_t):
    """Apply w^{-1/2}-scaled metric-consistent noise to a batch in z-space. Fresh noise per call."""
    B = x.shape[0]
    inv_sqrt_w = torch.exp(-0.5 * log_w)
    waves = x[:, :WAVE_LEN].reshape(B, NW, T)
    z = torch.randn(B, NW, metric_t["k"], device=x.device)
    noise = z @ metric_t["Bn"]
    waves = waves + noise * (metric_t["s_wave"] / obs_t["wave_std"])[None, :, None] * inv_sqrt_w[:, :, None]
    sc = x[:, WAVE_LEN:].clone()
    e = torch.randn(B, 4, device=x.device)
    sc[:, :3] = sc[:, :3] + e[:, :3] * (metric_t["s_scal"][:3] / obs_t["pas_std"]) * inv_sqrt_w
    sc[:, 3:4] = sc[:, 3:4] * torch.exp(e[:, 3:4] * metric_t["s_scal"][3] * inv_sqrt_w)
    return torch.cat([waves.reshape(B, WAVE_LEN), sc], 1)


def load_sim_tensors(data_dir, index_entries, stats, log):
    """Preload a ReducedCVDataset split into (theta, x) tensors. The dataset reads HDF5 per
    item, which is too slow to iterate every epoch over 1M sims."""
    ds = ReducedCVDataset(str(data_dir), index_entries, stats)
    n = len(ds)
    theta = torch.empty(n, len(PARAM_KEYS_INFER), dtype=torch.float32)
    x = torch.empty(n, WAVE_LEN + N_SCALARS, dtype=torch.float32)
    for i in range(n):
        theta[i], x[i] = ds[i]
        if (i + 1) % 100_000 == 0:
            log(f"  sims {i + 1}/{n}")
    ds.close()
    return theta, x


def metric_to_torch(metric, obs, device):
    m = {"k": metric.k, "Bn": torch.from_numpy(metric.Bn).to(device),
         "s_wave": torch.from_numpy(metric.s_wave).to(device),
         "s_scal": torch.from_numpy(metric.s_scal).to(device)}
    o = {"wave_std": torch.from_numpy(obs.wave_std).to(device),
         "pas_std": float(obs.pas_std)}
    return m, o


class WEmbed(nn.Module):
    """log w -> W_EMBED_DIM features. Standardized to [-1, 1] over the training range first,
    so w is not a single unit-scale channel among the encoder's latent dims."""

    def __init__(self, center: float, half_width: float, dim: int = W_EMBED_DIM):
        super().__init__()
        self.center = float(center)
        self.half_width = float(half_width)
        self.net = nn.Sequential(nn.Linear(1, dim), nn.SiLU())

    def forward(self, log_w):
        return self.net((log_w - self.center) / self.half_width)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--version", default="1")
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--real-data", required=True)
    parser.add_argument("--n-sims", type=int, required=True)
    parser.add_argument("--calib-frac", type=float, default=0.5,
                        help="Fraction of real patients used for metric/w_hat estimation; the rest "
                             "are the test set (never used for estimation).")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--flow-warmup", type=int, default=2,
                        help="Epochs with the encoder frozen (flow-only)")
    parser.add_argument("--enc-warmup", type=int, default=10,
                        help="Epochs after flow-warmup with the flow frozen (encoder-only)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--stats-path", default="norm_stats.json")
    parser.add_argument("--outputs-root", default="/home/sa4604/outputs/cv-sbi-spin")
    parser.add_argument("--resume-dir", default=None,
                        help="Checkpoint dir (checkpoints/<ts>) of a previous run to continue. "
                             "Reuses its metric and w_hat; epochs continue from its last epoch.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.outputs_root) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = f"{datetime.now():%Y%m%d-%H%M%S}"
    ckpt_dir = run_dir / "checkpoints" / ts
    ckpt_dir.mkdir(parents=True)
    log_fh = open(run_dir / f"train_log_{ts}.txt", "w")

    def log(msg):
        print(msg, flush=True)
        log_fh.write(msg + "\n")
        log_fh.flush()

    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                           text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        git_hash = "unknown"
    run_info_path = run_dir / f"run_info_v{args.version}_{ts}.json"
    with open(run_info_path, "w") as f:
        json.dump({"run": args.run, "version": args.version, "status": "running", "started": ts,
                   "command": " ".join(["train_gibbs_npe.py"] + sys.argv[1:]),
                   "git_hash": git_hash}, f, indent=2)

    stats = load_stats(args.stats_path)
    obs = Observables(stats)
    sim_root = Path(args.sim_data_root)
    manifest = load_manifest(sim_root / "manifest_train.json")

    log(f"Loading {args.n_sims} sim observations...")
    sim_theta, sim_x_t = load_sim_tensors(sim_root / "train", manifest["index"][:args.n_sims], stats, log)
    sim_x = sim_x_t.numpy()

    real_x_t = load_real_beats(Path(args.real_data), stats, log)
    real_files = [fp.stem for fp in sorted(Path(args.real_data).glob("*.h5"))]
    assert len(real_files) == len(real_x_t)
    rng = np.random.RandomState(args.split_seed)
    perm = rng.permutation(len(real_x_t))
    n_cal = int(round(len(real_x_t) * args.calib_frac))
    cal_idx, test_idx = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
    cal_x = real_x_t.numpy()[cal_idx]
    log(f"Real patients: {len(cal_idx)} calibration / {len(test_idx)} test")

    start_epoch = 1
    if args.resume_dir:
        src = Path(args.resume_dir)
        meta = json.load(open(src / "gibbs_metric.json"))
        metric = Metric.from_npz(src / "gibbs_metric.npz", meta["ell"])
        w_hat = meta["w_hat"]
        start_epoch = meta["last_epoch"] + 1
        log(f"Resuming from {src}: reusing metric (ell={metric.ell:.2f}, k_t={metric.k}) and "
            f"w_hat={w_hat:.6g}; epochs continue from {start_epoch}")
    else:
        log("Stage 1-2: estimating metric M and w_hat on calibration patients...")
        metric, w_hat, cal_info = estimate_metric(sim_x, obs, cal_x, log=log)
        log(f"  w_hat = {w_hat:.6g}   training range [{w_hat/10:.4g}, {10*w_hat:.4g}]")
        check_noise_consistency(sim_x, obs, metric, w_hat, log=log)

    lo, hi = np.log(w_hat / 10.0), np.log(10.0 * w_hat)
    gibbs_meta = {**metric.to_dict(), "w_hat": w_hat, "w_log_range": [lo, hi],
                  "last_epoch": start_epoch - 1, "resumed_from": args.resume_dir}
    if not args.resume_dir:
        gibbs_meta["calib_info"] = cal_info
        gibbs_meta["calib_patients"] = [real_files[i] for i in cal_idx]
        gibbs_meta["test_patients"] = [real_files[i] for i in test_idx]
    np.savez(ckpt_dir / "gibbs_metric.npz", V=metric.V, eigvals=metric.lam,
             s_wave=metric.s_wave, s_scal=metric.s_scal)

    with open(ckpt_dir / "gibbs_metric.json", "w") as f:
        json.dump(gibbs_meta, f, indent=2)

    # ── Model: E + flow conditioned on [E(x), WEmbed(log w)] ─────────────────
    theta_all = sim_theta[:min(10_000, len(sim_theta))]
    if args.resume_dir:
        E = torch.load(src / "encoder.pt", weights_only=False).to(device)
        flow = torch.load(src / "flow_net.pt", weights_only=False).to(device)
        wemb = torch.load(src / "w_embed.pt", weights_only=False).to(device)
    else:
        E = LipschitzReducedAutoencoderEncoder(latent_dim=args.latent_dim, sn_ceiling=2.0,
                                               proj_hidden=None, n_scalars=N_SCALARS).to(device)
        wemb = WEmbed(center=(lo + hi) / 2, half_width=(hi - lo) / 2).to(device)
        flow = build_flow_net("maf", args.latent_dim + W_EMBED_DIM, theta_all,
                              hidden_features=128, num_transforms=5).to(device)
    flow_params = list(flow.parameters()) + list(wemb.parameters())
    opt = torch.optim.Adam(list(E.parameters()) + flow_params, lr=args.lr)
    if args.resume_dir:
        opt.load_state_dict(torch.load(src / "opt_state.pt", weights_only=False))

    sim_dl = DataLoader(TensorDataset(sim_theta, sim_x_t), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    m_t, o_t = metric_to_torch(metric, obs, device)

    flow_end = args.flow_warmup
    enc_end = flow_end + args.enc_warmup

    def phase(epoch):
        if epoch <= flow_end:
            return "flow"
        if epoch <= enc_end:
            return "enc"
        return "joint"

    csv_fh = open(run_dir / f"train_log_{ts}.csv", "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["epoch", "phase", "train_nll", "seconds"])

    gibbs_meta_path = ckpt_dir / "gibbs_metric.json"
    log(f"Training: epochs {start_epoch}-{args.max_epochs}, {len(sim_dl)} batches/epoch, lr={args.lr}, "
        f"phases flow<={flow_end}, enc<={enc_end}, joint after")
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        ph = phase(epoch)
        train_E = ph != "flow"
        train_flow = ph != "enc"
        for p in E.parameters():
            p.requires_grad_(train_E)
        for p in flow_params:
            p.requires_grad_(train_flow)
        trainable = (list(E.parameters()) if train_E else []) + (flow_params if train_flow else [])
        E.train(); flow.train(); wemb.train()

        total, n = 0.0, 0
        for theta, x in sim_dl:
            theta, x = theta.to(device, non_blocking=True), x.to(device, non_blocking=True)
            log_w = lo + (hi - lo) * torch.rand(x.shape[0], 1, device=device)
            x_noisy = noisy_obs(x, log_w, m_t, o_t)
            cond = torch.cat([E(x_noisy), wemb(log_w)], 1)
            loss = -flow.log_prob(theta, condition=cond).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += loss.item() * x.shape[0]; n += x.shape[0]
        train_nll = total / n
        dt = time.time() - t0
        writer.writerow([epoch, ph, f"{train_nll:.5f}", f"{dt:.1f}"])
        csv_fh.flush()
        log(f"ep {epoch:4d} [{ph:4s}]  train_nll {train_nll:.4f}  ({dt:.0f}s)")
        last_epoch = epoch

    torch.save(E, ckpt_dir / "encoder.pt")
    torch.save(flow, ckpt_dir / "flow_net.pt")
    torch.save(wemb, ckpt_dir / "w_embed.pt")
    torch.save(opt.state_dict(), ckpt_dir / "opt_state.pt")
    gibbs_meta["last_epoch"] = last_epoch
    with open(gibbs_meta_path, "w") as f:
        json.dump(gibbs_meta, f, indent=2)

    with open(run_info_path) as f:
        info = json.load(f)
    info.update({"status": "complete", "w_hat": w_hat, "d_eff": metric.d_eff,
                 "last_epoch": last_epoch, "n_sims": len(sim_theta), "checkpoint": ts})
    with open(run_info_path, "w") as f:
        json.dump(info, f, indent=2)
    log(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
