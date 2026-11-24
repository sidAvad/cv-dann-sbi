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
import shutil
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
from bounded_nsf import build_bounded_nsf
from mdn_flow import build_mdn, WPassthrough

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


def check_noise_consistency(sim_x, obs, metric, w_test, n_draws=256, tol=0.05, log=print,
                            correction=None):
    """Pre-launch unit check: noisy_obs must reproduce the model's own second moment.

    E||delta phi||^2 = ||mu_r||^2 + tr(Sigma_c)/w, which reduces to d_eff/w when there
    is no correction.
    """
    m_t, o_t = metric_to_torch(metric, obs, "cpu", correction)
    x0 = torch.from_numpy(sim_x[:n_draws])
    torch.manual_seed(0)
    lw = torch.full((n_draws, 1), float(np.log(w_test)))
    x_noisy = noisy_obs(x0, lw, m_t, o_t).numpy()
    delta = phi_from_z(x_noisy, obs, metric) - phi_from_z(sim_x[:n_draws], obs, metric)
    measured = (delta ** 2).sum(1).mean()
    expected = metric.d_eff / w_test
    if correction is not None:
        mu2 = np.atleast_2d(correction["mu_r"]) ** 2
        if "bin_edges" in correction:
            # each draw carries its own bin's offset, so average over the bins the
            # draws actually land in rather than summing every bin's
            b = bin_of(x0, m_t, o_t).numpy()
            mu2_mean = float(mu2.sum(1)[b].mean())
        else:
            mu2_mean = float(mu2.sum())
        expected = (mu2_mean
                    + (metric.d_eff + float(correction["a"].sum())) / w_test)
    rel = abs(measured - expected) / expected
    log(f"  noise check at w={w_test:.4g}: mean ||delta phi||^2 = {measured:.4g}, "
        f"expected = {expected:.4g}, rel err {rel:.3f}")
    if rel > tol:
        raise RuntimeError(f"noise/metric mismatch: rel err {rel:.3f} > {tol}")


def bin_of(x, metric_t, obs_t):
    """Bin index per row, from the raw peak of the binning channel.

    Read off the observation, not theta, so the same lookup applies unchanged to a real
    patient at inference.
    """
    c = metric_t["cor_ch"]
    ch = x[:, c * T:(c + 1) * T] * obs_t["wave_std"][c] + obs_t["wave_mean"][c]
    return torch.bucketize(ch.max(dim=1).values, metric_t["cor_edges"])


def noisy_obs(x, log_w, metric_t, obs_t, mu_scale=None):
    """Apply metric-consistent noise to a batch in z-space. Fresh noise per call.

    The draw is made in phi coordinates, where the metric says the noise is white: the
    per-wave block holds k_t Matern-mode coefficients and the tail holds the 4 scalars,
    in the same order phi_from_z concatenates them. Scaling by s_wave/s_scal and the
    Bn projection then carry it back to z-space, with the 4th scalar multiplicative
    because phi takes its log.

    With a correction present the draw is eps ~ N(mu_r, Sigma_c / w),
    Sigma_c = I + U diag(a) U^T. mu_r is NOT scaled by w^{-1/2}: it is a fixed
    displacement, so the loss (x - f(theta) - mu_r)^T Sigma_c^-1 (...) stays a fixed
    quadratic that w tempers, keeping the Gibbs correspondence and w-amortization exact.
    """
    B = x.shape[0]
    k = metric_t["k"]
    inv_sqrt_w = torch.exp(-0.5 * log_w)
    eps = torch.randn(B, NW * k + 4, device=x.device)
    U, a = metric_t.get("cor_U"), metric_t.get("cor_a")
    if U is not None:
        # Sigma_c^{1/2} eps = eps + U (sqrt(1+a) - 1) U^T eps
        eps = eps + (eps @ U) * (torch.sqrt(1.0 + a) - 1.0) @ U.T
    eps = eps * inv_sqrt_w
    mu = metric_t.get("cor_mu")
    if mu is not None:
        off = (mu[bin_of(x, metric_t, obs_t)]
               if "cor_edges" in metric_t else mu[0].expand(B, -1))
        if mu_scale is not None:
            off = off * mu_scale
        eps = eps + off
    z = eps[:, :NW * k].reshape(B, NW, k)
    e = eps[:, NW * k:]

    waves = x[:, :WAVE_LEN].reshape(B, NW, T)
    noise = z @ metric_t["Bn"]
    waves = waves + noise * (metric_t["s_wave"] / obs_t["wave_std"])[None, :, None]
    sc = x[:, WAVE_LEN:].clone()
    sc[:, :3] = sc[:, :3] + e[:, :3] * (metric_t["s_scal"][:3] / obs_t["pas_std"])
    sc[:, 3:4] = sc[:, 3:4] * torch.exp(e[:, 3:4] * metric_t["s_scal"][3])
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


def metric_to_torch(metric, obs, device, correction=None):
    m = {"k": metric.k, "Bn": torch.from_numpy(metric.Bn).to(device),
         "Q": torch.from_numpy(metric.Q).to(device),
         "s_wave": torch.from_numpy(metric.s_wave).to(device),
         "s_scal": torch.from_numpy(metric.s_scal).to(device)}
    o = {"wave_std": torch.from_numpy(obs.wave_std).to(device),
         "wave_mean": torch.from_numpy(obs.wave_mean).to(device),
         "pas_std": float(obs.pas_std), "pas_mean": float(obs.pas_mean),
         "vlv_std": float(obs.vlv_std)}
    if correction is not None:
        m["cor_U"] = torch.from_numpy(correction["U"].astype(np.float32)).to(device)
        m["cor_a"] = torch.from_numpy(correction["a"].astype(np.float32)).to(device)
        m["cor_mu"] = torch.from_numpy(
            np.atleast_2d(correction["mu_r"]).astype(np.float32)).to(device)
        if "bin_edges" in correction:
            # mu_r is (n_bins, d_eff) and the bin is read off the observation, so
            # Sigma_c stays global: |Sigma_c| does not depend on theta, the log-det is
            # constant and falls into the normaliser, and the target stays exactly Gibbs.
            m["cor_edges"] = torch.from_numpy(
                np.asarray(correction["bin_edges"], np.float32)).to(device)
            m["cor_ch"] = int(WAVE_KEYS_REDUCED.index(
                str(correction["bin_feature"]).split("_")[0]))
    return m, o


def phi_torch(x, m_t, o_t):
    """phi(x) for a z-space batch, on device: the whitened features the Gibbs likelihood
    is defined on. Mirrors Observables.raw + phi_from_z exactly, including the log on SV.
    """
    B = x.shape[0]
    waves = (x[:, :WAVE_LEN].reshape(B, NW, T) * o_t["wave_std"][None, :, None]
             + o_t["wave_mean"][None, :, None])
    wk = torch.einsum("nct,tk->nck", waves, m_t["Q"]) / m_t["s_wave"][None, :, None]
    sc = x[:, WAVE_LEN:]
    s = torch.stack([sc[:, 0] * o_t["pas_std"] + o_t["pas_mean"],
                     sc[:, 1] * o_t["pas_std"] + o_t["pas_mean"],
                     sc[:, 2] * o_t["pas_std"] + o_t["pas_mean"],
                     torch.log(torch.clamp(sc[:, 3] * o_t["vlv_std"], min=1e-6))], 1)
    return torch.cat([wk.reshape(B, -1), s / m_t["s_scal"][None, :]], 1)


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


class WFiLM(nn.Module):
    """log w modulates the conditioning features multiplicatively instead of being
    concatenated alongside them.

    Concatenation leaves w as 16 of 144 input dims against 128 encoder features that carry
    nearly all the variance, so the conditioner can reach most of its likelihood reduction
    without using w and the w pathway stays under-trained. A multiplicative gate cannot be
    routed around. Follows mdn_flow.ConditionalMDN's FiLM (gamma = 2 tanh, zero-centred
    shift) but standardises log w over the training range first, as WEmbed does -- the
    window is known, so there is no reason to feed an unstandardised scalar.

    forward returns the modulated features, so the caller appends nothing for w and
    cond_dim equals feature_dim.
    """

    def __init__(self, feature_dim: int, center: float, half_width: float,
                 hidden: int = 64):
        super().__init__()
        self.center = float(center)
        self.half_width = float(half_width)
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(),
                                 nn.Linear(hidden, feature_dim * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)   # starts as gamma=0 -> h unchanged by scale

    def forward(self, h, log_w):
        u = (log_w - self.center) / self.half_width
        gamma, shift = torch.chunk(self.net(u), 2, dim=-1)
        return torch.tanh(gamma) * 2.0 * h + shift


def prior_box(nested_sim_root, keys):
    """(lo, hi, log_mask) of the prior box, in PARAM_KEYS_INFER order."""
    sys.path.insert(0, nested_sim_root)
    from nested_sim.params import THETA_BOUNDS, LOG_SAMPLED, to_binary_names
    bb = to_binary_names(THETA_BOUNDS)
    lo = np.array([bb[k][0] for k in keys], dtype=np.float64)
    hi = np.array([bb[k][1] for k in keys], dtype=np.float64)
    mask = np.zeros(len(keys), dtype=bool)
    mask[[keys.index(k) for k in to_binary_names({k: 0 for k in LOG_SAMPLED})]] = True
    return lo, hi, mask


def to_unit_unclamped(phys, lo, hi, log_mask):
    """Prior box -> unit cube, logs on the log-uniform dims, without clamping to [0,1].

    Mirrors eval_pretrained_lofihifi.to_unit_unclamped so the calibration curve is on
    the same scale as the published scores. Non-positive draws on a log dimension are
    floored, landing far below 0, where the caller's winsorisation handles them.
    """
    x = np.where(log_mask, np.log(np.clip(phys, 1e-30, None)), phys)
    lo2 = np.where(log_mask, np.log(lo), lo)
    hi2 = np.where(log_mask, np.log(hi), hi)
    return (x - lo2) / (hi2 - lo2)


def load_calib_theta(path, real_files, cal_idx, keys):
    """Ground-truth theta for the calibration stand-ins, in real_x_t row order.

    The stand-in files carry only HR/PVR/SVR; scripts/export_hifi_theta.py recovers the
    full vector from the source target set and keys it by the global index in the file
    name.
    """
    d = np.load(path, allow_pickle=True)
    pk = [str(k) for k in d["param_keys"]]
    if pk != list(keys):
        raise SystemExit(f"{path} is in a different parameter order than "
                         f"PARAM_KEYS_INFER (first mismatch at "
                         f"{next(i for i, (a, b) in enumerate(zip(pk, keys)) if a != b)})")
    tbl = {int(i): r for i, r in zip(d["indices"], d["theta"])}
    out, missing = [], []
    for i in cal_idx:
        gi = int(str(real_files[i]).rsplit("_", 1)[1])
        (out.append(tbl[gi]) if gi in tbl else missing.append(gi))
    if missing:
        raise SystemExit(f"{len(missing)} calibration patients absent from {path}, "
                         f"e.g. {missing[:5]}")
    return np.stack(out)


def save_snapshot(dst, E, flow, wemb, ckpt_dir, gibbs_meta, epoch):
    """Mid-training snapshot, in the same format load_gibbs_models expects.

    Written INSIDE the run's own checkpoint dir, so resolve_gibbs_ckpt -- which globs
    one level under checkpoints/ and takes the last -- keeps resolving to the run
    itself and never to a snapshot.
    """
    dst.mkdir(parents=True, exist_ok=True)
    if E is not None:
        torch.save(E.state_dict(), dst / "encoder.pt")
    torch.save(flow, dst / "flow_net.pt")
    torch.save(wemb, dst / "w_embed.pt")
    shutil.copy2(ckpt_dir / "gibbs_metric.npz", dst / "gibbs_metric.npz")
    with open(dst / "gibbs_metric.json", "w") as f:
        json.dump({**gibbs_meta, "last_epoch": epoch}, f, indent=2)


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
    parser.add_argument("--flow-arch", choices=["maf", "bounded-nsf", "mdn"],
                        default="maf",
                        help="maf: the unconstrained Gaussian-base flow. bounded-nsf: "
                             "rational-quadratic splines on the prior box, so every "
                             "draw is in-prior by construction (see bounded_nsf.py). "
                             "mdn: the Route A baseline's mixture head, FiLM-conditioned "
                             "on log w and unbounded (see mdn_flow.py)")
    parser.add_argument("--mdn-hidden", type=int, default=128)
    parser.add_argument("--mdn-components", type=int, default=8)
    parser.add_argument("--mdn-layers", type=int, default=2)
    parser.add_argument("--head-lr", type=float, default=None,
                        help="learning rate for the density head and w embedding; "
                             "--lr applies to the encoder. Default: --lr for both")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--calib-every", type=int, default=0,
                        help="score the held-out calibration patients every N epochs "
                             "into calib_log_<ts>.csv; 0 disables")
    parser.add_argument("--calib-samples", type=int, default=500)
    parser.add_argument("--calib-theta", default=None,
                        help="npz from scripts/export_hifi_theta.py giving ground-truth "
                             "theta for the stand-in patients; required by --calib-every")
    parser.add_argument("--val-eps-draws", type=int, default=1,
                        help="average the validation NLL over this many independent "
                             "noise realisations")
    parser.add_argument("--laplace-cache", default="",
                        help="npz from laplace_width.py: the Jacobians and observations "
                             "for the width diagnostic. J is metric-dependent, so the "
                             "cache is rejected unless it was built with this run's own "
                             "metric. Unset disables the diagnostic.")
    parser.add_argument("--laplace-every", type=int, default=0,
                        help="log the flow/target width ratio every N epochs into "
                             "laplace_log_<ts>.csv; 0 disables")
    parser.add_argument("--laplace-w", type=float, default=0.0,
                        help="w at which the two widths are compared. 0 uses w_hat. The "
                             "ratio is w-dependent, so comparing runs needs a shared "
                             "value.")
    parser.add_argument("--laplace-samples", type=int, default=2000,
                        help="posterior draws per case for the flow's own width")
    parser.add_argument("--laplace-inf-frac", type=float, default=0.8,
                        help="a parameter counts as informative, and so enters "
                             "ratio_informative, when its target width95 is below this "
                             "fraction of the prior's")
    parser.add_argument("--nsf-bins", type=int, default=10)
    parser.add_argument("--nsf-transforms", type=int, default=8)
    parser.add_argument("--nsf-hidden", type=int, default=256)
    parser.add_argument("--nested-sim-root", default="/home/sa4604/projects/nested-sim",
                        help="source of the prior box (bounded-nsf only)")
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="snapshot encoder/flow/w_embed every N epochs into "
                             "<ckpt_dir>/epoch<NNNN>/, for tracking a metric against "
                             "training epoch. 0 disables.")
    parser.add_argument("--resume-dir", default=None,
                        help="Checkpoint dir (checkpoints/<ts>) of a previous run to continue. "
                             "Reuses its metric and w_hat; epochs continue from its last epoch.")
    parser.add_argument("--metric-dir", default=None,
                        help="Checkpoint dir (checkpoints/<ts>) to load a frozen metric M and "
                             "w_hat from, WITHOUT resuming weights, optimizer state, or the "
                             "epoch counter. Keeps M fixed across runs; training starts fresh.")
    parser.add_argument("--w-lo", type=float, default=None,
                        help="Lower end of the log-uniform w training range, in absolute "
                             "units. Must be given with --w-hi. Default: w_hat/10.")
    parser.add_argument("--w-hi", type=float, default=None,
                        help="Upper end of the log-uniform w training range, in absolute "
                             "units. Must be given with --w-lo. Default: 10*w_hat.")
    parser.add_argument("--w-cond", choices=["concat", "film"], default="concat",
                        help="How log w enters the conditioning. concat: WEmbed features "
                             "appended to the encoder output (default). film: log w gates "
                             "the encoder features multiplicatively, so the conditioner "
                             "cannot ignore it. Not available for --flow-arch mdn, whose "
                             "head already FiLMs on w internally.")
    parser.add_argument("--lr-cosine", action="store_true",
                        help="Cosine-decay the learning rate to zero over --max-epochs. "
                             "Lets a higher --lr be used early without carrying it late.")
    parser.add_argument("--mu-scale-range", type=float, nargs=2, default=None,
                        metavar=("LO", "HI"),
                        help="Amortize over the offset's MAGNITUDE: draw c ~ U[LO, HI] "
                             "per sample, use c*mu_r as the offset, and condition on c. "
                             "The eval then sweeps c, with c=0 the no-offset baseline "
                             "and c=1 the fitted value. mu_r's direction is far better "
                             "determined than its size, so one scalar covers the "
                             "uncertainty that matters. Needs --correction.")
    parser.add_argument("--correction", default=None,
                        help="npz with mu_r, U, a: the low-rank enrichment of the noise "
                             "covariance, Sigma_c = I + U diag(a) U^T, plus a mean "
                             "offset. Fitted off-line from theta-paired residuals; see "
                             "experiments.md. Omitted, the noise stays N(0, I/w) in phi.")
    parser.add_argument("--n-val-sims", type=int, default=4096,
                        help="Held-out sims for the validation NLL, taken from the "
                             "manifest entries AFTER the --n-sims training slice, so "
                             "they are never trained on. 0 disables validation.")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Compute the validation NLL every N epochs.")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop when the validation NLL has not improved on its best "
                             "for this many epochs. 0 disables it. Val NLL is noisy "
                             "relative to its own trend -- see experiments.md -- so a "
                             "short patience fires on noise; 50 is the smallest value "
                             "that tracked the trend on the runs measured so far.")
    parser.add_argument("--val-seed", type=int, default=1234,
                        help="Seed for the validation set's log w draw and its noise. The "
                             "noised validation batch is built ONCE and reused every epoch, "
                             "so val_nll is comparable across epochs and across runs; "
                             "resampling it per epoch would add the same per-epoch scatter "
                             "that makes train_nll noisy.")
    parser.add_argument("--val-batch", type=int, default=512)
    parser.add_argument("--sn-ceiling", type=float, default=2.0,
                        help="Per-layer spectral-norm ceiling on the encoder. The "
                             "parametrization scales a layer by min(1, ceiling/sigma_1), "
                             "so a very large value (e.g. inf) is an exact no-op. Ignored "
                             "on --resume-dir, which keeps the source run's architecture.")
    parser.add_argument("--cond-mode", default="encoder",
                        choices=["encoder", "phi", "concat"],
                        help="What the flow is conditioned on, alongside WEmbed(log w). "
                             "encoder: the learned latent E(x). phi: the metric's own "
                             "whitened feature vector phi(x), which the Gibbs likelihood "
                             "is defined on, so no encoder is built or trained. concat: "
                             "both, as [E(x), phi(x)] in that order.")
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

    val_idx = manifest["index"][args.n_sims:args.n_sims + args.n_val_sims]
    if args.n_val_sims and not val_idx:
        log(f"WARNING no held-out sims past --n-sims {args.n_sims} "
            f"({len(manifest['index'])} in manifest); validation disabled")
    val_theta, val_x_t = (load_sim_tensors(sim_root / "train", val_idx, stats, log)
                          if val_idx else (None, None))
    if val_idx:
        log(f"Validation: {len(val_idx)} held-out sims (manifest "
            f"{args.n_sims}..{args.n_sims + len(val_idx) - 1})")

    real_x_t = load_real_beats(Path(args.real_data), stats, log)
    real_files = [fp.stem for fp in sorted(Path(args.real_data).glob("*.h5"))]
    assert len(real_files) == len(real_x_t)
    rng = np.random.RandomState(args.split_seed)
    perm = rng.permutation(len(real_x_t))
    n_cal = int(round(len(real_x_t) * args.calib_frac))
    cal_idx, test_idx = np.sort(perm[:n_cal]), np.sort(perm[n_cal:])
    cal_x = real_x_t.numpy()[cal_idx]
    log(f"Real patients: {len(cal_idx)} calibration / {len(test_idx)} test")

    correction = None
    if args.correction:
        correction = dict(np.load(args.correction))
        mu2 = (np.atleast_2d(correction["mu_r"]) ** 2).sum(1)
        wv = np.atleast_1d(correction["w"])
        log(f"Noise correction from {args.correction}: rank {int(correction['rank'])}, "
            f"projected={bool(correction['projected'])}, "
            f"a in [{float(correction['a'].min()):.3g}, "
            f"{float(correction['a'].max()):.3g}]")
        if "bin_edges" in correction:
            ed = np.asarray(correction["bin_edges"])
            log(f"  per-bin offset on {str(correction['bin_feature'])}, "
                f"{len(mu2)} bins, edges {np.round(ed, 3)}")
            log(f"  ||mu_b||^2 {np.round(mu2, 3)}   self-consistent w_b "
                f"{np.round(wv, 4)}   s_b {np.round(wv / float(correction['w_global']), 3)}")
            log("  Sigma_c is global, so the bin shifts only the offset; the per-bin "
                "scale is applied at evaluation via --percase-w")
        else:
            log(f"  ||mu_r||^2={float(mu2.sum()):.4g}, "
                f"self-consistent w={float(wv[0]):.4g}")

    start_epoch = 1
    if args.resume_dir:
        src = Path(args.resume_dir)
        meta = src_meta = json.load(open(src / "gibbs_metric.json"))
        metric = Metric.from_npz(src / "gibbs_metric.npz", meta["ell"])
        w_hat = meta["w_hat"]
        start_epoch = meta["last_epoch"] + 1
        log(f"Resuming from {src}: reusing metric (ell={metric.ell:.2f}, k_t={metric.k}) and "
            f"w_hat={w_hat:.6g}; epochs continue from {start_epoch}")
    elif args.metric_dir:
        mdir = Path(args.metric_dir)
        mmeta = json.load(open(mdir / "gibbs_metric.json"))
        metric = Metric.from_npz(mdir / "gibbs_metric.npz", mmeta["ell"])
        w_hat = mmeta["w_hat"]
        log(f"Frozen metric from {mdir}: ell={metric.ell:.4f}, k_t={metric.k}, "
            f"d_eff={mmeta['d_eff']}, w_hat={w_hat:.6g}; weights train from scratch")
        # also verify here, not only on the estimate path: a correction fitted in a
        # different metric's phi (wrong d_eff, wrong U shape) is otherwise caught by
        # nothing until the noise is silently wrong
        check_noise_consistency(sim_x, obs, metric, w_hat, log=log, correction=correction)
    else:
        log("Stage 1-2: estimating metric M and w_hat on calibration patients...")
        metric, w_hat, cal_info = estimate_metric(sim_x, obs, cal_x, log=log)
        log(f"  w_hat = {w_hat:.6g}   training range [{w_hat/10:.4g}, {10*w_hat:.4g}]")
        check_noise_consistency(sim_x, obs, metric, w_hat, log=log, correction=correction)

    if (args.w_lo is None) != (args.w_hi is None):
        raise SystemExit("--w-lo and --w-hi must be given together")
    if args.w_lo is not None:
        if not 0 < args.w_lo < args.w_hi:
            raise SystemExit(f"need 0 < --w-lo < --w-hi, got {args.w_lo}, {args.w_hi}")
        lo, hi = np.log(args.w_lo), np.log(args.w_hi)
        log(f"w training range (explicit): [{args.w_lo:g}, {args.w_hi:g}]  "
            f"(w_hat={w_hat:.6g} is recorded but does not set the range)")
    else:
        lo, hi = np.log(w_hat / 10.0), np.log(10.0 * w_hat)
    gibbs_meta = {**metric.to_dict(), "w_hat": w_hat, "w_log_range": [lo, hi],
                  "last_epoch": start_epoch - 1, "resumed_from": args.resume_dir,
                  "metric_from": args.metric_dir,
                  "flow_arch": args.flow_arch, "cond_mode": args.cond_mode,
                  "sn_ceiling": (src_meta.get("sn_ceiling", 2.0) if args.resume_dir
                                 else args.sn_ceiling)}
    if not (args.resume_dir or args.metric_dir):
        gibbs_meta["calib_info"] = cal_info
    if not args.resume_dir:
        gibbs_meta["calib_patients"] = [real_files[i] for i in cal_idx]
        gibbs_meta["test_patients"] = [real_files[i] for i in test_idx]
    np.savez(ckpt_dir / "gibbs_metric.npz", V=metric.V, eigvals=metric.lam,
             s_wave=metric.s_wave, s_scal=metric.s_scal)
    if correction is not None:
        gibbs_meta["correction"] = {
            "path": args.correction, "rank": int(correction["rank"]),
            "projected": bool(correction["projected"]),
            "w_self_consistent": np.atleast_1d(correction["w"]).tolist(),
            "mu_r_sq": float((correction["mu_r"] ** 2).sum()),
            "a_sum": float(correction["a"].sum())}
        np.savez(ckpt_dir / "gibbs_correction.npz", **correction)

    with open(ckpt_dir / "gibbs_metric.json", "w") as f:
        json.dump(gibbs_meta, f, indent=2)

    # ── Model: flow conditioned on [E(x)?, phi(x)?, WEmbed(log w)] ──────────
    theta_all = sim_theta[:min(10_000, len(sim_theta))]
    # conditioning order is fixed as [E(x), phi(x), WEmbed(log w)]; a checkpoint is only
    # loadable under the cond_mode it was trained with, which gibbs_metric.json records
    use_enc = args.cond_mode in ("encoder", "concat")
    use_phi = args.cond_mode in ("phi", "concat")
    if args.w_cond == "film" and args.flow_arch == "mdn":
        raise SystemExit("--w-cond film is redundant for the mdn head, which FiLMs on w "
                         "internally; use --w-cond concat")
    # film modulates the features in place, so nothing is appended for w
    w_cond_dim = (0 if args.w_cond == "film"
                  else (1 if args.flow_arch == "mdn" else W_EMBED_DIM))
    feature_dim = ((args.latent_dim if use_enc else 0)
                   + (metric.d_eff if use_phi else 0))
    mu_lo, mu_hi = (args.mu_scale_range if args.mu_scale_range else (1.0, 1.0))
    mu_cond_dim = 1 if args.mu_scale_range else 0
    cond_dim = feature_dim + w_cond_dim + mu_cond_dim
    if args.resume_dir and src_meta.get("cond_mode", "encoder") != args.cond_mode:
        raise SystemExit(f"--resume-dir was trained with cond_mode "
                         f"{src_meta.get('cond_mode', 'encoder')!r}, not "
                         f"{args.cond_mode!r}; that is a different model")
    log(f"conditioning: {args.cond_mode}  cond_dim={cond_dim} "
        f"(E {args.latent_dim if use_enc else 0} + phi {metric.d_eff if use_phi else 0} "
        f"+ w {w_cond_dim})")
    if args.resume_dir:
        # a resume reproduces the source run's encoder exactly; --sn-ceiling does not
        # apply here, since a different ceiling is a different model
        E = None
        if use_enc:
            E = LipschitzReducedAutoencoderEncoder(latent_dim=args.latent_dim,
                                                   sn_ceiling=src_meta.get("sn_ceiling", 2.0),
                                                   proj_hidden=None, n_scalars=N_SCALARS).to(device)
            E.load_state_dict(torch.load(src / "encoder.pt", map_location=device))
        flow = torch.load(src / "flow_net.pt", weights_only=False).to(device)
        wemb = torch.load(src / "w_embed.pt", weights_only=False).to(device)
    else:
        E = LipschitzReducedAutoencoderEncoder(
            latent_dim=args.latent_dim, sn_ceiling=args.sn_ceiling,
            proj_hidden=None, n_scalars=N_SCALARS).to(device) if use_enc else None
        if args.w_cond == "film":
            wemb = WFiLM(feature_dim, center=(lo + hi) / 2,
                         half_width=(hi - lo) / 2).to(device)
        elif args.flow_arch == "mdn":
            wemb = WPassthrough().to(device)
        else:
            wemb = WEmbed(center=(lo + hi) / 2, half_width=(hi - lo) / 2).to(device)
        if args.flow_arch == "mdn":
            flow = build_mdn(theta_dim=theta_all.shape[1], feature_dim=feature_dim,
                             theta_all=theta_all, hidden=args.mdn_hidden,
                             n_components=args.mdn_components,
                             n_hidden_layers=args.mdn_layers).to(device)
            log(f"flow: conditional MDN, {args.mdn_components} components x "
                f"{args.mdn_hidden} hidden, {args.mdn_layers} layers, FiLM on log w; "
                f"unbounded support (draws are rejected downstream)")
        elif args.flow_arch == "bounded-nsf":
            sys.path.insert(0, args.nested_sim_root)
            from nested_sim.params import (THETA_BOUNDS, LOG_SAMPLED,  # noqa: E402
                                           to_binary_names)
            bb = to_binary_names(THETA_BOUNDS)
            theta_lo = [bb[k][0] for k in PARAM_KEYS_INFER]
            theta_hi = [bb[k][1] for k in PARAM_KEYS_INFER]
            log_dims = [PARAM_KEYS_INFER.index(k)
                        for k in to_binary_names({k: 0 for k in LOG_SAMPLED})]
            flow = build_bounded_nsf(cond_dim, theta_lo, theta_hi,
                                     log_dims, num_bins=args.nsf_bins,
                                     num_transforms=args.nsf_transforms,
                                     hidden_features=args.nsf_hidden).to(device)
            log(f"flow: bounded NSF, {args.nsf_transforms} transforms x "
                f"{args.nsf_bins} bins, {len(log_dims)} log dims")
        else:
            flow = build_flow_net("maf", cond_dim, theta_all,
                                  hidden_features=128, num_transforms=5).to(device)
    flow_params = list(flow.parameters()) + list(wemb.parameters())
    enc_params = list(E.parameters()) if E is not None else []
    head_lr = args.lr if args.head_lr is None else args.head_lr
    opt = torch.optim.Adam([{"params": enc_params, "lr": args.lr},
                            {"params": flow_params, "lr": head_lr}], lr=args.lr)
    if head_lr != args.lr:
        log(f"lr: encoder {args.lr:g}, density head {head_lr:g}")
    sched = None
    if args.lr_cosine:
        # per-epoch cosine to zero over the full schedule, so a higher --lr applies early
        # and decays out rather than being carried to the end
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_epochs)
        log(f"lr schedule: cosine {args.lr:g} -> 0 over {args.max_epochs} epochs")
    n_enc = sum(p.numel() for p in enc_params)
    n_head = sum(p.numel() for p in flow_params)
    log(f"parameters: encoder {n_enc:,} + head {n_head:,} = {n_enc + n_head:,}")
    if args.resume_dir:
        opt.load_state_dict(torch.load(src / "opt_state.pt", weights_only=False))

    sim_dl = DataLoader(TensorDataset(sim_theta, sim_x_t), batch_size=args.batch_size, shuffle=True,
                        num_workers=0, pin_memory=True, drop_last=True)
    m_t, o_t = metric_to_torch(metric, obs, device, correction)

    # One fixed (log w, noise) realisation, drawn in a forked RNG so the training stream
    # is untouched. log w uses the same U[log(w_hat/10), log(10 w_hat)] prior the training
    # loss averages over, so val_nll and train_nll are the same quantity on different data.
    val_pack = None
    if val_x_t is not None:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(args.val_seed)
            vx = val_x_t.to(device)
            vth = val_theta.to(device)
            val_pack = []
            for _ in range(max(1, args.val_eps_draws)):
                vlog_w = lo + (hi - lo) * torch.rand(vx.shape[0], 1, device=device)
                vmu = (torch.ones(vx.shape[0], 1) if args.mu_scale_range else None)
                val_pack.append((vth, noisy_obs(vx, vlog_w, m_t, o_t, vmu), vlog_w))
            del vx
        if len(val_pack) > 1:
            log(f"validation NLL averaged over {len(val_pack)} noise realisations")

    def _eval_mode():
        if E is not None:
            E.eval()
        flow.eval(); wemb.eval()

    def _cond(x_b, w_b, mu_c=1.0):
        """mu_c defaults to 1: validation and held-out scoring use the FITTED offset, so
        they stay comparable across epochs and against a run without amortization."""
        parts = []
        if use_enc:
            parts.append(E(x_b))
        if use_phi:
            parts.append(phi_torch(x_b, m_t, o_t))
        if args.w_cond == "film":
            c = wemb(torch.cat(parts, 1), w_b)
            if args.mu_scale_range:
                c = torch.cat([c, torch.full((x_b.shape[0], 1), float(mu_c),
                                             device=x_b.device)], 1)
            return c
        extra = [wemb(w_b)]
        if args.mu_scale_range:
            extra.append(torch.full((x_b.shape[0], 1), float(mu_c), device=x_b.device))
        return torch.cat(parts + extra, 1)

    def val_nll_now():
        """Mean NLL on the held-out slice, over the fixed noised batches."""
        _eval_mode()
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for th, xn, lw in val_pack:
                for b in range(0, th.shape[0], args.val_batch):
                    sl = slice(b, b + args.val_batch)
                    c = _cond(xn[sl], lw[sl])
                    tot += -flow.log_prob(th[sl], condition=c).sum().item()
                    cnt += th[sl].shape[0]
        return tot / cnt

    # Held-out scoring: the stand-in patients NOT used for metric estimation, with
    # ground-truth theta recovered from the source target set. Nothing in training sees
    # them -- the weights train on sims, the metric was fit on the other half, and the
    # published benchmark draws from hifi_v2_test instead -- so they are legitimate for
    # model selection. Unavailable on genuine patients, which have no theta, so this is a
    # development instrument; see experiments.md on what transfers and what does not.
    calib = None
    if args.calib_every:
        if not args.calib_theta:
            raise SystemExit("--calib-every needs --calib-theta")
        sys.path.insert(0, str(Path(args.nested_sim_root) / "scripts"))
        import probscores as ps                                      # noqa: E402
        keys = list(PARAM_KEYS_INFER)
        box_lo, box_hi, box_logmask = prior_box(args.nested_sim_root, keys)
        cal_theta = load_calib_theta(args.calib_theta, real_files, test_idx, keys)
        cal_x_t = torch.from_numpy(real_x_t.numpy()[test_idx]).float().to(device)
        u_true = to_unit_unclamped(cal_theta[:, None, :], box_lo, box_hi,
                                   box_logmask)[:, 0, :]
        pr_crps, pr_is = ps.prior_reference(u_true)
        calib_logw = float(np.log(w_hat))
        calib = True
        log(f"held-out scoring: {len(cal_theta)} patients, {args.calib_samples} draws, "
            f"every {args.calib_every} epochs at log w = {calib_logw:.4f} "
            f"(w_hat = {w_hat:.6g})")
        calib_fh = open(run_dir / f"calib_log_{ts}.csv", "w", newline="")
        calib_w = csv.writer(calib_fh)
        calib_w.writerow(["epoch", "crps", "crps_skill", "is95", "is95_skill", "cov95",
                          "width95", "accept_median", "post_std_mean", "n_degenerate"])

    def calib_now():
        """Scores on the held-out stand-ins, plus the theta-free surrogates.

        accept_median and post_std_mean need no ground truth, so they are the candidates
        for a stopping rule that would also work on genuine patients; the rest need theta
        and exist to tell us whether those surrogates track anything.
        """
        _eval_mode()
        chunks = []
        with torch.no_grad():
            for b in range(0, cal_x_t.shape[0], args.val_batch):
                x_b = cal_x_t[b:b + args.val_batch]
                lw = torch.full((x_b.shape[0], 1), calib_logw, device=device)
                d = flow.sample((args.calib_samples,), condition=_cond(x_b, lw))
                chunks.append(d.transpose(0, 1).cpu().numpy())        # (B, n, 24)
        samples = np.concatenate(chunks, 0)
        u = to_unit_unclamped(samples, box_lo, box_hi, box_logmask)
        in_box = ((u >= 0.0) & (u <= 1.0)).all(axis=2)
        sc = ps.score_set(np.clip(u, -5.0, 6.0), u_true, in_box)
        crps, is95 = float(np.nanmean(sc["crps_box"])), float(np.nanmean(sc["is_box"]))
        return {"crps": crps, "crps_skill": 1.0 - crps / float(np.nanmean(pr_crps)),
                "is95": is95, "is95_skill": 1.0 - is95 / float(np.nanmean(pr_is)),
                "cov95": float(np.nanmean(sc["cov_box"])),
                "width95": float(np.nanmean(sc["width_box"])),
                "accept_median": float(np.median(in_box.mean(1))),
                "post_std_mean": float(np.mean(u.std(axis=1))),
                "n_degenerate": int(sc["n_degenerate"])}

    # Laplace width diagnostic: the flow's own 95% width against the width the linearised
    # Gibbs target has at the same w, on lofi cases with no noise added -- the regime the
    # Laplace expansion is taken in. Unsupervised and target-internal: it separates a flow
    # that has not reached its target from a target that is itself wrong, which no score
    # computed against the target can do. See experiments.md.
    lap = None
    if args.laplace_cache:
        if not args.laplace_every:
            raise SystemExit("--laplace-cache needs --laplace-every")
        import laplace_width as lwmod                                # noqa: E402
        cache = lwmod.load_cache(args.laplace_cache)
        cmeta = cache["meta"]
        if (abs(cmeta["ell"] - metric.ell) > 1e-6
                or cmeta["d_eff"] != metric.d_eff):
            raise SystemExit(
                f"--laplace-cache was built on a different metric "
                f"(ell {cmeta['ell']:.6f} d_eff {cmeta['d_eff']} vs this run's "
                f"{metric.ell:.6f} / {metric.d_eff}); rebuild it with "
                f"laplace_width.py --metric-dir pointing at this run's metric")
        if list(cmeta["param_keys"]) != list(PARAM_KEYS_INFER):
            raise SystemExit("--laplace-cache has a different parameter order")
        lap_w = args.laplace_w or w_hat
        lap_x = torch.from_numpy(cache["x"]).float().to(device)
        lap_logw = float(np.log(lap_w))
        lap_lo, lap_hi, lap_logmask = prior_box(args.nested_sim_root,
                                                list(PARAM_KEYS_INFER))
        lap_tgt_pp = lwmod.laplace_width95(cache["J"], lap_w, correction,
                                           per_param=True)
        lap_target = float(lap_tgt_pp.mean())
        # Parameters the data barely constrains sit at the prior under both the target and
        # any sane flow, so including them pulls every ratio toward 1 and hides the signal.
        # The informative subset is the one the mean ratio is worth reading on.
        lap_inf = lap_tgt_pp < args.laplace_inf_frac * lwmod.PRIOR_WIDTH95
        log(f"Laplace width diagnostic: {lap_x.shape[0]} {cmeta['split']} cases, "
            f"{args.laplace_samples} draws, every {args.laplace_every} epochs at "
            f"w = {lap_w:.6g}; target width95 = {lap_target:.4f}, "
            f"{int(lap_inf.sum())}/{lap_inf.size} params informative "
            f"(< {args.laplace_inf_frac:g} x prior {lwmod.PRIOR_WIDTH95:.4f})"
            + (" (Sigma_c applied)" if correction is not None else ""))
        lap = True
        lap_fh = open(run_dir / f"laplace_log_{ts}.csv", "w", newline="")
        lap_wr = csv.writer(lap_fh)
        lap_wr.writerow(["epoch", "w", "flow_width95", "laplace_width95", "ratio",
                         "ratio_inf_median", "ratio_inf_mean", "in_box_frac"]
                        + [f"ratio_{k}" for k in PARAM_KEYS_INFER])

    def laplace_now():
        """Flow width95 on the cache's cases, in the same unit-cube coords as the target.

        The observations are the clean lofi phi(f(theta)), not noised: the Laplace is
        expanded at an observation whose mode sits at theta_true, so feeding a noised
        draw would compare against the wrong reference. Samples are left unclamped so an
        over-wide flow is not silently trimmed to the box.

        Sampling runs in a forked RNG at a fixed seed, so the diagnostic neither advances
        the training noise stream -- enabling it leaves the trained weights bit-identical
        -- nor contributes its own scatter to the epoch-to-epoch ratio.
        """
        _eval_mode()
        chunks = []
        with torch.no_grad(), torch.random.fork_rng(
                devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(args.val_seed)
            for b in range(0, lap_x.shape[0], args.val_batch):
                x_b = lap_x[b:b + args.val_batch]
                lw = torch.full((x_b.shape[0], 1), lap_logw, device=device)
                d = flow.sample((args.laplace_samples,), condition=_cond(x_b, lw))
                chunks.append(d.transpose(0, 1).cpu().numpy())        # (B, n, 24)
        u = to_unit_unclamped(np.concatenate(chunks, 0), lap_lo, lap_hi, lap_logmask)
        qs = np.percentile(u, [2.5, 97.5], axis=1)
        fw_pp = (qs[1] - qs[0]).mean(0)                               # (24,)
        ratio_pp = fw_pp / lap_tgt_pp
        # The median, not the mean, over the informative subset: per-parameter ratios are
        # bounded below by 0 and unbounded above, and a couple of parameters (AVD, Tmax_a)
        # disagree with the linearisation by 2-4x, which a mean lets dominate the summary.
        # See experiments.md -- a mean-of-ratios read "above target" where every robust
        # estimator reads "below".
        return {"flow": float(fw_pp.mean()),
                "ratio": float(fw_pp.mean() / lap_target),
                "ratio_inf": float(np.median(ratio_pp[lap_inf])) if lap_inf.any()
                else float("nan"),
                "ratio_inf_mean": float(ratio_pp[lap_inf].mean()) if lap_inf.any()
                else float("nan"),
                "in_box": float(np.mean(((u >= 0.0) & (u <= 1.0)).all(axis=2))),
                "ratio_pp": ratio_pp}

    flow_end = args.flow_warmup
    enc_end = flow_end + args.enc_warmup

    def phase(epoch):
        if not use_enc:
            return "phi"
        if epoch <= flow_end:
            return "flow"
        if epoch <= enc_end:
            return "enc"
        return "joint"

    csv_fh = open(run_dir / f"train_log_{ts}.csv", "w", newline="")
    writer = csv.writer(csv_fh)
    # since_best is logged whether or not --patience is set, so any patience threshold
    # can be read off a finished run without rerunning it: the rule with patience N would
    # have fired at the first epoch whose since_best reaches N.
    writer.writerow(["epoch", "phase", "train_nll", "val_nll", "seconds",
                     "best_val", "since_best"])

    gibbs_meta_path = ckpt_dir / "gibbs_metric.json"
    log(f"Training: epochs {start_epoch}-{args.max_epochs}, {len(sim_dl)} batches/epoch, lr={args.lr}, "
        f"phases flow<={flow_end}, enc<={enc_end}, joint after")
    last_epoch = start_epoch - 1
    best_val, best_epoch = float("inf"), start_epoch - 1
    for epoch in range(start_epoch, args.max_epochs + 1):
        if sched is not None and epoch > start_epoch:
            sched.step()
        t0 = time.time()
        ph = phase(epoch)
        train_E = E is not None and ph != "flow"
        train_flow = ph != "enc"
        for p in enc_params:
            p.requires_grad_(train_E)
        for p in flow_params:
            p.requires_grad_(train_flow)
        trainable = (enc_params if train_E else []) + (flow_params if train_flow else [])
        if E is not None:
            E.train()
        flow.train(); wemb.train()

        total, n = 0.0, 0
        for theta, x in sim_dl:
            theta, x = theta.to(device, non_blocking=True), x.to(device, non_blocking=True)
            log_w = lo + (hi - lo) * torch.rand(x.shape[0], 1, device=device)
            mu_c = (mu_lo + (mu_hi - mu_lo) * torch.rand(x.shape[0], 1, device=device)
                    if args.mu_scale_range else None)
            x_noisy = noisy_obs(x, log_w, m_t, o_t, mu_c)
            parts = []
            if use_enc:
                parts.append(E(x_noisy))
            if use_phi:
                parts.append(phi_torch(x_noisy, m_t, o_t))
            if args.w_cond == "film":
                cond = wemb(torch.cat(parts, 1), log_w)
                if mu_c is not None:
                    cond = torch.cat([cond, mu_c], 1)
            else:
                extra = [wemb(log_w)] + ([mu_c] if mu_c is not None else [])
                cond = torch.cat(parts + extra, 1)
            loss = -flow.log_prob(theta, condition=cond).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            opt.step()
            total += loss.item() * x.shape[0]; n += x.shape[0]
        train_nll = total / n
        v = (val_nll_now() if val_pack is not None and args.val_every
             and epoch % args.val_every == 0 else None)
        dt = time.time() - t0
        since_best = "" if v is None else epoch - (
            epoch if v < best_val else best_epoch)
        writer.writerow([epoch, ph, f"{train_nll:.5f}",
                         "" if v is None else f"{v:.5f}", f"{dt:.1f}",
                         "" if best_val == float("inf") and v is None
                         else f"{min(best_val, v if v is not None else best_val):.5f}",
                         since_best])
        csv_fh.flush()
        log(f"ep {epoch:4d} [{ph:4s}]  train_nll {train_nll:.4f}"
            + ("" if v is None else f"  val_nll {v:.4f}")
            + ("" if v is None else f"  since_best {since_best}")
            + f"  ({dt:.0f}s)")
        last_epoch = epoch

        if v is not None and v < best_val:
            best_val, best_epoch = v, epoch
            save_snapshot(ckpt_dir / "best_val", E, flow, wemb, ckpt_dir,
                          gibbs_meta, epoch)
        elif args.patience and v is not None and epoch - best_epoch >= args.patience:
            log(f"early stop: val_nll has not improved on {best_val:.4f} (ep "
                f"{best_epoch}) for {epoch - best_epoch} epochs")
            break

        if calib and epoch % args.calib_every == 0:
            c = calib_now()
            calib_w.writerow([epoch] + [f"{c[k]:.6g}" for k in
                             ("crps", "crps_skill", "is95", "is95_skill", "cov95",
                              "width95", "accept_median", "post_std_mean")]
                             + [c["n_degenerate"]])
            calib_fh.flush()
            log(f"       calib  CRPS {c['crps']:.4f} (skill {c['crps_skill']:+.3f})  "
                f"IS95 {c['is95']:.4f} (skill {c['is95_skill']:+.3f})  "
                f"cov95 {c['cov95']:.3f}  width95 {c['width95']:.4f}  "
                f"accept {c['accept_median']:.3f}")

        if lap and epoch % args.laplace_every == 0:
            c = laplace_now()
            lap_wr.writerow([epoch, f"{lap_w:.6g}", f"{c['flow']:.6f}",
                             f"{lap_target:.6f}", f"{c['ratio']:.6f}",
                             f"{c['ratio_inf']:.6f}", f"{c['ratio_inf_mean']:.6f}",
                             f"{c['in_box']:.4f}"]
                            + [f"{r:.6f}" for r in c["ratio_pp"]])
            lap_fh.flush()
            log(f"       laplace  flow width95 {c['flow']:.4f}  "
                f"target {lap_target:.4f}  ratio {c['ratio']:.3f}  "
                f"ratio_inf(med) {c['ratio_inf']:.3f}  "
                f"in-box {c['in_box']:.3f}")

        if args.ckpt_every and epoch % args.ckpt_every == 0:
            save_snapshot(ckpt_dir / f"epoch{epoch:04d}", E, flow, wemb,
                          ckpt_dir, gibbs_meta, epoch)
            log(f"       snapshot -> epoch{epoch:04d}")

    if E is not None:
        torch.save(E.state_dict(), ckpt_dir / "encoder.pt")
    torch.save(flow, ckpt_dir / "flow_net.pt")
    torch.save(wemb, ckpt_dir / "w_embed.pt")
    torch.save(opt.state_dict(), ckpt_dir / "opt_state.pt")
    gibbs_meta["last_epoch"] = last_epoch
    with open(gibbs_meta_path, "w") as f:
        json.dump(gibbs_meta, f, indent=2)

    with open(run_info_path) as f:
        info = json.load(f)
    info.update({"status": "complete", "w_hat": w_hat, "d_eff": metric.d_eff,
                 "last_epoch": last_epoch, "n_sims": len(sim_theta), "checkpoint": ts,
                 "n_val_sims": (0 if val_pack is None else int(val_pack[0][0].shape[0])),
                 "val_seed": args.val_seed})
    with open(run_info_path, "w") as f:
        json.dump(info, f, indent=2)
    log(f"Saved checkpoint to {ckpt_dir}")


if __name__ == "__main__":
    main()
