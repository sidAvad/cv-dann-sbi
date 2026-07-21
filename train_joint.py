"""
Joint training: LipschitzReducedAutoencoderEncoder + WDGRL + task head.

Three-phase training schedule:

  Phase 0  flow-warmup    encoder frozen, flow trains        (sim only)
  Phase 1  enc-warmup     flow frozen, encoder trains        (sim only, no domain pressure)
  Phase 2  joint          encoder + flow + WDGRL, λ ramps    (sim + real)

WDGRL: Wasserstein critic (MLP, no sigmoid) trained with WGAN-GP.
  - n_critic inner critic updates per encoder step, encoder detached during critic updates
  - Gradient penalty at random interpolates enforces 1-Lipschitz on critic
  - Encoder step minimizes W1 estimate (critic(z_sim).mean() - critic(z_real).mean())
  - Real data has no theta labels; it only contributes domain gradient to the encoder

Separate Adam optimizers for encoder, flow, and critic.
Critic Adam uses betas=(0.5, 0.9) per WGAN-GP convention.
λ ramps linearly or sigmoid from 0 → --lambda-target over --lambda-warmup epochs in phase 2.
Early stopping on total loss, patience counted from the start of phase 2.

Run names: exp-v{N}_encoder-lipschitz_dann_{flow-maf5|flow-nsf8|reconstruction}
run_info:  outputs/{run}/run_info_v{version}.json

Usage:
    python train_joint.py \\
        --run exp-v2_encoder-lipschitz_dann_flow-maf5 \\
        --version 2 \\
        --objective flow-maf5 \\
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \\
        --real-data ~/real_data/onebeat_300patients
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

from sbi.neural_nets import posterior_nn

from dataset import (
    ReducedCVDataset, PARAM_KEYS_INFER,
    N_REDUCED_CHANNELS, N_SCALARS, T,
    load_stats, load_manifest,
)
from models import (
    LipschitzReducedAutoencoderEncoder,
    VAEReducedAutoencoderEncoder,
    WaveformDecoder,
    WassersteinCritic,
)


# ─── Constants ────────────────────────────────────────────────────────────────

STATS_PATH     = Path("norm_stats.json")
N_SIMS_FULL    = 100_000
N_SIMS_DRY     = 512
BATCH_SIZE     = 512
WAVE_KEYS_REAL = ["Prv", "Pra", "Pvp", "Pap"]
N_PARAMS_INFER = len(PARAM_KEYS_INFER)   # 24 (HR is in x, not θ)
WAVE_LEN       = N_REDUCED_CHANNELS * T  # 804

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Helpers ─────────────────────────────────────────────────────────────────

def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tee:
    def __init__(self, fh):
        self._fh     = fh
        self._stdout = sys.stdout

    def write(self, msg):
        self._fh.write(msg)
        self._stdout.write(msg)

    def flush(self):
        self._fh.flush()
        self._stdout.flush()


def parse_run(name: str):
    if name.startswith("dry-"):
        return "dry", Path("dry-runs") / name
    elif name.startswith("exp-"):
        return "exp", Path("outputs") / name
    else:
        raise ValueError("--run must start with 'exp-' or 'dry-'")


def mixup_real(real_beats: torch.Tensor, n: int, alpha: float, device: str) -> torch.Tensor:
    """On-the-fly Mixup of real beats. Returns n interpolated samples."""
    idx_i = torch.randint(0, len(real_beats), (n,), device=device)
    idx_j = torch.randint(0, len(real_beats), (n,), device=device)
    lam   = torch.distributions.Beta(alpha, alpha).sample((n,)).to(device).unsqueeze(1)
    return lam * real_beats[idx_i] + (1 - lam) * real_beats[idx_j]


def gradient_penalty(critic: nn.Module, z_sim: torch.Tensor,
                     z_real: torch.Tensor, device: str) -> torch.Tensor:
    """1-Lipschitz gradient penalty for WGAN-GP at random interpolates.

    create_graph=True is critical — without it the GP has no gradient through
    the critic and the Lipschitz constraint is not enforced.
    """
    eps   = torch.rand(z_sim.shape[0], 1, device=device)
    z_hat = (eps * z_sim.detach() + (1 - eps) * z_real.detach()).requires_grad_(True)
    f_hat = critic(z_hat)
    grads = torch.autograd.grad(f_hat.sum(), z_hat, create_graph=True)[0]
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sim_data(data_dir: Path, manifest: dict, stats: dict, n: int, log,
                  include_sv: bool = True, scalar_norm: str = "legacy"):
    index   = manifest["index"][:n]
    dataset = ReducedCVDataset(str(data_dir), index, stats,
                               include_sv=include_sv, scalar_norm=scalar_norm)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    thetas, xs, loaded = [], [], 0
    for theta_b, x_b in loader:
        thetas.append(theta_b)
        xs.append(x_b)
        loaded += len(theta_b)
        print(f"\r  loaded {loaded}/{n}", end="", flush=True)
    print()
    dataset.close()
    theta_all = torch.cat(thetas)
    x_all     = torch.cat(xs)
    log(f"Sim: {loaded} sims  θ={tuple(theta_all.shape[1:])}  x={tuple(x_all.shape[1:])}")
    return theta_all, x_all


def load_real_beats(data_dir: Path, stats: dict, log,
                    real_stats: dict | None = None,
                    include_sv: bool = True,
                    scalar_norm: str = "legacy") -> torch.Tensor:
    """Load all patient beats into a flat tensor (N_beats, OBS_DIM).

    scalar_norm="legacy"  — waveform-level Pas stats for MAP/SBP/DBP; sv/vlv_std for SV
    scalar_norm="zscore"  — per-summary distribution stats from real_stats["scalars"]
                            (real_stats must be provided)
    """
    w = stats["waves"]
    p = stats["parameters"]

    # Waveform normalization: use real stats if provided, else sim stats
    if real_stats is not None:
        rw = real_stats["waves"]
        wave_mean = torch.tensor([rw[k]["mean"] for k in WAVE_KEYS_REAL], dtype=torch.float32).unsqueeze(1)
        wave_std  = torch.tensor([rw[k]["std"]  for k in WAVE_KEYS_REAL], dtype=torch.float32).unsqueeze(1)
        log("Real beats: using real_norm_stats for waveform normalisation")
    else:
        wave_mean = torch.tensor([w[k]["mean"] for k in WAVE_KEYS_REAL], dtype=torch.float32).unsqueeze(1)
        wave_std  = torch.tensor([w[k]["std"]  for k in WAVE_KEYS_REAL], dtype=torch.float32).unsqueeze(1)
        log("Real beats: using sim norm_stats for normalisation (legacy)")

    # Scalar normalization
    if scalar_norm == "zscore":
        rs = real_stats["scalars"]
        map_mean, map_std = rs["map"]["mean"], rs["map"]["std"] + 1e-8
        sbp_mean, sbp_std = rs["sbp"]["mean"], rs["sbp"]["std"] + 1e-8
        dbp_mean, dbp_std = rs["dbp"]["mean"], rs["dbp"]["std"] + 1e-8
        sv_mean,  sv_std  = rs["sv"]["mean"],  rs["sv"]["std"]  + 1e-8
        hr_mean,  hr_std  = rs["hr"]["mean"],  rs["hr"]["std"]  + 1e-8
        log("Real beats: using zscore scalar normalisation from real_norm_stats")
    else:
        map_mean = sbp_mean = dbp_mean = w["Pas"]["mean"]
        map_std  = sbp_std  = dbp_std  = w["Pas"]["std"] + 1e-8
        sv_mean,  sv_std  = 0.0, w["Vlv"]["std"] + 1e-8
        hr_mean,  hr_std  = p["HR"]["mean"], p["HR"]["std"] + 1e-8

    beats = []
    for fpath in sorted(data_dir.glob("*.h5")):
        with h5py.File(fpath, "r") as f:
            for key in sorted(f.keys()):
                if not key.startswith("beat_"):
                    continue
                g     = f[key]
                waves = np.stack([g[f"waves/{k}"][:].astype(np.float32) for k in WAVE_KEYS_REAL])
                wt    = (torch.from_numpy(waves) - wave_mean) / (wave_std + 1e-8)
                sbp   = float(g["summaries/sbp"][()])
                dbp   = float(g["summaries/dbp"][()])
                map_  = float(g["summaries/map"][()])
                sv    = float(g["summaries/sv"][()])
                hr    = float(g["parameters/HR"][()])
                sc_vals = [
                    (map_ - map_mean) / map_std,
                    (sbp  - sbp_mean) / sbp_std,
                    (dbp  - dbp_mean) / dbp_std,
                ]
                if include_sv:
                    sc_vals.append((sv - sv_mean) / sv_std)
                sc_vals.append((hr - hr_mean) / hr_std)
                sc = torch.tensor(sc_vals, dtype=torch.float32)
                beats.append(torch.cat([wt.reshape(-1), sc]))

    real = torch.stack(beats)
    log(f"Real: {len(beats)} beats from {data_dir.name}")
    return real


# ─── Flow construction ────────────────────────────────────────────────────────

def build_flow_net(objective: str, latent_dim: int, theta_stats: torch.Tensor,
                   hidden_features: int, num_transforms: int) -> nn.Module:
    """Build MAF or NSF via sbi's posterior_nn with identity embedding."""
    model    = "maf" if "maf" in objective else "nsf"
    build_fn = posterior_nn(
        model=model,
        embedding_net=nn.Identity(),
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_theta="independent",
        z_score_x="none",
    )
    z_dummy = torch.zeros(len(theta_stats), latent_dim)
    return build_fn(theta_stats.cpu(), z_dummy)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Identity
    parser.add_argument("--run",     required=True,
                        help="e.g. exp-v2_encoder-lipschitz_dann_flow-maf5")
    parser.add_argument("--version", required=True,
                        help="Version string for run_info filename, e.g. '2'")
    parser.add_argument("--objective",
                        choices=["flow-maf5", "flow-nsf8", "reconstruction"], required=True)
    # Data
    parser.add_argument("--sim-data-root",   required=True)
    parser.add_argument("--real-data",       required=True)
    parser.add_argument("--stats-path", default="norm_stats.json",
                        help="Path to sim norm_stats.json (default: norm_stats.json)")
    parser.add_argument("--real-norm-stats", default=None,
                        help="Path to real_norm_stats.json; if provided, real patient waveforms "
                             "are z-scored with real-data statistics instead of sim statistics")
    parser.add_argument("--scalar-norm", choices=["legacy", "zscore"], default="legacy",
                        help="Scalar normalization mode: legacy=waveform-derived stats, "
                             "zscore=per-summary distribution stats (requires norm_stats[scalars] "
                             "for sims and real_norm_stats[scalars] for reals)")
    parser.add_argument("--no-sv", action="store_true",
                        help="Drop SV scalar from observation (808-dim instead of 809-dim)")
    parser.add_argument("--manifest-train",  default="manifest_train.json",
                        help="Manifest filename under sim-data-root (default: manifest_train.json)")
    parser.add_argument("--n-sims",          type=int, default=None)
    # Encoder
    parser.add_argument("--encoder-type",  choices=["lipschitz", "vae"], default="lipschitz")
    parser.add_argument("--latent-dim",    type=int,   default=128)
    parser.add_argument("--sn-ceiling",    type=float, default=2.0,
                        help="Spectral norm ceiling (lipschitz encoder only)")
    parser.add_argument("--proj-hidden",   type=int,   default=None)
    parser.add_argument("--kl-weight",     type=float, default=1e-4,
                        help="Beta weight on KL term (VAE encoder only)")
    # Flow (ignored for reconstruction)
    parser.add_argument("--hidden-features", type=int, default=None,
                        help="Default: 128 for MAF, 256 for NSF")
    parser.add_argument("--num-transforms",  type=int, default=None,
                        help="Default: 5 for MAF, 8 for NSF")
    # Phase schedule (cumulative epoch counts)
    parser.add_argument("--flow-warmup",   type=int, default=2,
                        help="Phase 0: epochs training flow only (encoder frozen)")
    parser.add_argument("--enc-warmup",    type=int, default=10,
                        help="Phase 1: epochs training encoder only + WDGRL (flow frozen)")
    parser.add_argument("--lambda-warmup", type=int, default=20,
                        help="Phase 2: epochs over which λ ramps 0→lambda-target")
    parser.add_argument("--max-epochs",    type=int, default=200)
    # WDGRL
    parser.add_argument("--lambda-target",     type=float, default=0.1,
                        help="WDGRL weight at full ramp")
    parser.add_argument("--lambda-schedule",   choices=["linear", "sigmoid"], default="linear",
                        help="λ ramp schedule in phase 2")
    parser.add_argument("--lambda-gamma",      type=float, default=10.0,
                        help="Sigmoid schedule steepness")
    parser.add_argument("--use-mixup",          action="store_true",
                        help="Augment real beats with on-the-fly Mixup (Beta interpolation)")
    parser.add_argument("--mixup-alpha",        type=float, default=0.4,
                        help="Beta distribution concentration for Mixup")
    parser.add_argument("--n-critic",          type=int,   default=5,
                        help="Critic updates per encoder step")
    parser.add_argument("--gp-weight",         type=float, default=10.0,
                        help="Gradient penalty coefficient")
    parser.add_argument("--critic-hidden",     type=int,   default=128,
                        help="Hidden dim for WassersteinCritic MLP")
    # Optimisation
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--patience",   type=int,   default=30,
                        help="Early stopping patience from start of phase 2")
    parser.add_argument("--log-every",  type=int,   default=5)
    args = parser.parse_args()

    is_flow    = args.objective.startswith("flow")
    run_type, run_dir = parse_run(args.run)
    is_dry     = run_type == "dry"
    n_sims     = N_SIMS_DRY if is_dry else (args.n_sims or N_SIMS_FULL)

    # Phase boundary epochs (cumulative)
    flow_end = args.flow_warmup
    enc_end  = flow_end + args.enc_warmup

    def get_phase(epoch: int) -> int:
        if epoch <= flow_end: return 0
        if epoch <= enc_end:  return 1
        return 2

    def get_lambda(epoch: int) -> float:
        if get_phase(epoch) != 2:
            return 0.0
        t = min(1.0, (epoch - enc_end) / max(1, args.lambda_warmup))
        if args.lambda_schedule == "linear":
            return args.lambda_target * t
        # sigmoid: 2/(1+e^{-γt})-1, reaches ~lambda_target at t=1 with γ=10
        return args.lambda_target * (2 / (1 + np.exp(-args.lambda_gamma * t)) - 1)

    if args.hidden_features is None:
        args.hidden_features = 256 if "nsf" in args.objective else 128
    if args.num_transforms is None:
        args.num_transforms  = 8   if "nsf" in args.objective else 5

    run_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"train_{args.run}_{date_str}.log"
    log_fh   = open(log_path, "w")
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(log_fh)
    sys.stderr = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Run: {args.run}  v={args.version}  ({'dry' if is_dry else 'full'})")
    log(f"Objective: {args.objective}  Device: {DEVICE}")
    log(f"Phase boundaries — flow_end={flow_end}  enc_end={enc_end}  max={args.max_epochs}")
    log(f"λ: target={args.lambda_target}  schedule={args.lambda_schedule}  "
        f"warmup_epochs={args.lambda_warmup}")
    log(f"WDGRL: n_critic={args.n_critic}  gp_weight={args.gp_weight}  "
        f"critic_hidden={args.critic_hidden}")
    log(f"Mixup: {'on' if args.use_mixup else 'off'}  alpha={args.mixup_alpha}")
    log(f"Encoder type: {args.encoder_type}"
        + (f"  kl_weight={args.kl_weight}" if args.encoder_type == "vae" else ""))

    # ── Load data ──────────────────────────────────────────────────────────────
    stats    = load_stats(Path(args.stats_path))
    manifest = load_manifest(Path(args.sim_data_root) / args.manifest_train)

    include_sv  = not args.no_sv
    scalar_norm = args.scalar_norm
    log(f"Loading {n_sims} sim observations...")
    theta_all, x_all = load_sim_data(
        Path(args.sim_data_root) / "train", manifest, stats, n_sims, log,
        include_sv=include_sv, scalar_norm=scalar_norm,
    )

    real_stats = None
    if args.real_norm_stats:
        with open(args.real_norm_stats) as f:
            real_stats = json.load(f)

    log("Loading real patient beats...")
    real_beats = load_real_beats(Path(args.real_data), stats, log, real_stats=real_stats,
                                 include_sv=include_sv, scalar_norm=scalar_norm)

    # ── Normalization sanity check ─────────────────────────────────────────────
    log(f"x_all      mean={x_all.mean():.4f}  std={x_all.std():.4f}  "
        f"shape={tuple(x_all.shape)}")
    log(f"real_beats mean={real_beats.mean():.4f}  std={real_beats.std():.4f}  "
        f"shape={tuple(real_beats.shape)}")

    real_beats = real_beats.to(DEVICE)

    # ── Build models ───────────────────────────────────────────────────────────
    is_vae = args.encoder_type == "vae"
    if is_vae:
        encoder = VAEReducedAutoencoderEncoder(
            latent_dim=args.latent_dim,
            proj_hidden=args.proj_hidden,
        ).to(DEVICE)
    else:
        encoder = LipschitzReducedAutoencoderEncoder(
            latent_dim=args.latent_dim,
            sn_ceiling=args.sn_ceiling,
            proj_hidden=args.proj_hidden,
            n_scalars=4 if args.no_sv else N_SCALARS,
        ).to(DEVICE)
    log(f"Encoder: {encoder.describe()}")

    if is_flow:
        log("Building flow network...")
        n_stats   = min(10_000, len(theta_all))
        task_head = build_flow_net(
            args.objective, args.latent_dim,
            theta_all[:n_stats], args.hidden_features, args.num_transforms,
        ).to(DEVICE)
        log(f"Flow: {args.objective}  transforms={args.num_transforms}  hidden={args.hidden_features}")
    else:
        task_head = WaveformDecoder(
            latent_dim=args.latent_dim, hidden=512, n_layers=3,
            out_channels=N_REDUCED_CHANNELS,
        ).to(DEVICE)
        log(f"Decoder: {task_head.describe()}")

    critic = WassersteinCritic(latent_dim=args.latent_dim, hidden=args.critic_hidden).to(DEVICE)
    log(f"WassersteinCritic: {critic.describe()}")

    # ── Separate optimizers ────────────────────────────────────────────────────
    enc_opt    = torch.optim.Adam(encoder.parameters(),   lr=args.lr)
    flow_opt   = torch.optim.Adam(task_head.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(),    lr=args.lr, betas=(0.5, 0.9))

    enc_flow_params = list(encoder.parameters()) + list(task_head.parameters())
    log(f"Adam lr={args.lr}  enc+flow params={sum(p.numel() for p in enc_flow_params):,}  "
        f"critic params={sum(p.numel() for p in critic.parameters()):,}")

    # ── Write run_info ─────────────────────────────────────────────────────────
    run_info = dict(
        run=args.run,
        version=args.version,
        type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        git_hash=git_hash(),
        device=DEVICE,
        encoder_type=args.encoder_type,
        encoder=encoder.describe(),
        task_head=dict(
            type=args.objective,
            hidden_features=args.hidden_features if is_flow else None,
            num_transforms=args.num_transforms   if is_flow else None,
        ),
        wdgrl_critic=critic.describe(),
        data=dict(
            n_sims=n_sims,
            n_real_beats=len(real_beats),
            sim_data_root=args.sim_data_root,
            real_data=args.real_data,
            real_norm_stats=args.real_norm_stats,
            scalar_norm=args.scalar_norm,
            no_sv=args.no_sv,
        ),
        schedule=dict(
            flow_end=flow_end, enc_end=enc_end,
            lambda_warmup=args.lambda_warmup, max_epochs=args.max_epochs,
            lambda_target=args.lambda_target,
            lambda_schedule=args.lambda_schedule,
        ),
        wdgrl=dict(
            n_critic=args.n_critic,
            gp_weight=args.gp_weight,
            critic_hidden=args.critic_hidden,
            use_mixup=args.use_mixup,
            mixup_alpha=args.mixup_alpha,
        ),
        training=dict(lr=args.lr, batch_size=args.batch_size, patience=args.patience,
                      kl_weight=args.kl_weight if is_vae else None),
    )
    info_path = run_dir / f"run_info_v{args.version}.json"
    info_path.write_text(json.dumps(run_info, indent=2))
    log(f"run_info_v{args.version}.json written  git={run_info['git_hash']}")

    # ── Training loop ──────────────────────────────────────────────────────────
    n_total      = len(theta_all)
    best_loss    = float("inf")
    wait         = 0
    best_enc_sd  = {k: v.clone() for k, v in encoder.state_dict().items()}
    best_task_sd = {k: v.clone() for k, v in task_head.state_dict().items()}

    csv_path   = run_dir / f"train_log_{date_str}.csv"
    csv_fh     = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["epoch", "task", "kl", "w1_est", "gp", "total", "lambda", "phase"])

    log("Training...")
    for epoch in range(1, args.max_epochs + 1):
        phase     = get_phase(epoch)
        lambda_e  = get_lambda(epoch)
        use_wdgrl = (phase == 2)

        encoder.train(); task_head.train(); critic.train()

        perm = torch.randperm(n_total)
        ep_task = ep_kl = ep_w1 = ep_gp = ep_tot = 0.0
        n_batches = 0

        for start in range(0, n_total, args.batch_size):
            idx     = perm[start:start + args.batch_size]
            x_sim_b = x_all[idx].to(DEVICE)
            theta_b = theta_all[idx].to(DEVICE)

            def sample_real(n):
                if args.use_mixup:
                    return mixup_real(real_beats, n, args.mixup_alpha, DEVICE)
                idx = torch.randint(0, len(real_beats), (n,), device=DEVICE)
                return real_beats[idx]

            def _encode_z(x):
                """Return just z, discarding mu/log_var for VAE."""
                out = encoder(x)
                return out[0] if is_vae else out

            # ── Critic inner loop (phase 2 only) ──────────────────────────────
            batch_gp = 0.0
            if use_wdgrl:
                for _ in range(args.n_critic):
                    z_sim_d  = _encode_z(x_sim_b).detach()
                    z_real_d = _encode_z(sample_real(len(idx))).detach()
                    gp       = gradient_penalty(critic, z_sim_d, z_real_d, DEVICE)
                    w_diff   = critic(z_sim_d).mean() - critic(z_real_d).mean()
                    c_loss   = -w_diff + args.gp_weight * gp
                    critic_opt.zero_grad()
                    c_loss.backward()
                    critic_opt.step()
                    batch_gp += gp.item()
                batch_gp /= args.n_critic

            # ── Encoder + flow step ───────────────────────────────────────────
            if is_vae:
                z_sim, mu_sim, log_var_sim = encoder(x_sim_b)
                kl_loss = -0.5 * (1 + log_var_sim - mu_sim**2 - log_var_sim.exp()).sum(-1).mean()
            else:
                z_sim   = encoder(x_sim_b)
                kl_loss = x_sim_b.new_zeros(1).squeeze()

            if is_flow:
                task_loss = -task_head.log_prob(theta_b, condition=z_sim).mean()
            else:
                task_loss = F.mse_loss(task_head(z_sim), x_sim_b[:, :WAVE_LEN])

            if use_wdgrl:
                z_real = _encode_z(sample_real(len(idx)))
                w1_est      = critic(z_sim).mean() - critic(z_real).mean()
                domain_loss = w1_est
            else:
                domain_loss = x_sim_b.new_zeros(1).squeeze()
                w1_est      = domain_loss

            total = task_loss + lambda_e * domain_loss + args.kl_weight * kl_loss

            enc_opt.zero_grad(); flow_opt.zero_grad()
            total.backward()

            if phase == 0:
                torch.nn.utils.clip_grad_norm_(task_head.parameters(), 1.0)
                flow_opt.step()
            elif phase == 1:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                enc_opt.step()
            else:  # phase 2: joint
                torch.nn.utils.clip_grad_norm_(enc_flow_params, 1.0)
                enc_opt.step(); flow_opt.step()

            ep_task += task_loss.item()
            ep_kl   += kl_loss.item()
            ep_w1   += w1_est.item()
            ep_gp   += batch_gp
            ep_tot  += total.item()
            n_batches += 1

        avg_task = ep_task / n_batches
        avg_kl   = ep_kl   / n_batches
        avg_w1   = ep_w1   / n_batches
        avg_gp   = ep_gp   / n_batches
        avg_tot  = ep_tot  / n_batches

        csv_writer.writerow([epoch,
                             f"{avg_task:.5f}", f"{avg_kl:.5f}", f"{avg_w1:.5f}",
                             f"{avg_gp:.5f}", f"{avg_tot:.5f}", f"{lambda_e:.4f}", phase])
        csv_fh.flush()

        if phase == 2 and avg_tot < best_loss:
            best_loss    = avg_tot
            best_enc_sd  = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_task_sd = {k: v.clone() for k, v in task_head.state_dict().items()}
            wait = 0
        elif phase == 2:
            wait += 1

        if epoch % args.log_every == 0 or epoch == 1:
            phase_name = ["flow-warmup", "enc-warmup", "joint"][phase]
            kl_str = f"  kl={avg_kl:.4f}" if is_vae else ""
            log(f"  ep {epoch:4d}/{args.max_epochs}  [{phase_name}]"
                f"  task={avg_task:.4f}{kl_str}  w1={avg_w1:.4f}  gp={avg_gp:.4f}"
                f"  total={avg_tot:.4f}  λ={lambda_e:.3f}  wait={wait}")

        if phase == 2 and wait >= args.patience:
            log(f"Early stop at epoch {epoch}  best_total={best_loss:.4f}")
            break

    csv_fh.close()

    # ── Save best checkpoints ──────────────────────────────────────────────────
    encoder.load_state_dict(best_enc_sd)
    task_head.load_state_dict(best_task_sd)

    torch.save(encoder.state_dict(), run_dir / "encoder.pt")
    if is_flow:
        torch.save(task_head, run_dir / "flow_net.pt")
    else:
        torch.save(task_head.state_dict(), run_dir / "decoder.pt")
    torch.save(critic.state_dict(), run_dir / "wdgrl_critic.pt")
    log(f"Saved encoder.pt  {'flow_net.pt' if is_flow else 'decoder.pt'}  wdgrl_critic.pt")

    log_fh.close()
    sys.stdout = _stdout
    sys.stderr = _stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
