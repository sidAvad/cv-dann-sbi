"""
Joint training: LipschitzReducedAutoencoderEncoder + DANN + task head.

Objective (per batch):
  L = L_task  +  λ · L_domain(z_sim, z_real)

  L_task = -E[log p_flow(θ|z)]         --objective flow-maf5 / flow-nsf8
         = E[||decoder(z) - x_waves||²] --objective reconstruction

  L_domain uses GradientReversalLayer so the encoder is trained adversarially.

Phase schedule:
  Phase 1 (epoch 1 .. --phase1-epochs):           λ = 0           (task only, sim data)
  Phase 2 (.. --phase1-epochs + --phase2-epochs):  λ = lambda-p2  (light domain pressure)
  Phase 3 (.. --max-epochs):                       λ = lambda      (full joint objective)

Run names: exp-v{N}_encoder-lipschitz_{flow-maf5|flow-nsf8|reconstruction}
run_info written to: outputs/{run}/run_info_v{version}.json

Usage:
    python train_joint.py \\
        --run exp-v1_encoder-lipschitz_flow-maf5 \\
        --version 1 \\
        --objective flow-maf5 \\
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \\
        --real-data ~/real_data/multibeat
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
    N_REDUCED_CHANNELS, T,
    load_stats, load_manifest,
)
from models import (
    LipschitzReducedAutoencoderEncoder,
    WaveformDecoder,
    GradientReversalLayer,
    DomainClassifier,
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


# ─── Phase schedule ───────────────────────────────────────────────────────────

def get_lambda(epoch: int, phase1_end: int, phase2_end: int,
               lambda_p2: float, lambda_target: float) -> float:
    if epoch <= phase1_end:
        return 0.0
    elif epoch <= phase2_end:
        return lambda_p2
    else:
        return lambda_target


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_sim_data(data_dir: Path, manifest: dict, stats: dict, n: int, log):
    index   = manifest["index"][:n]
    dataset = ReducedCVDataset(str(data_dir), index, stats)
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


def load_real_beats(data_dir: Path, stats: dict, log) -> torch.Tensor:
    """Load all patient beats into a flat tensor (N_beats, OBS_DIM)."""
    w = stats["waves"]
    p = stats["parameters"]
    wave_mean = torch.tensor([w[k]["mean"] for k in WAVE_KEYS_REAL], dtype=torch.float32).unsqueeze(1)
    wave_std  = torch.tensor([w[k]["std"]  for k in WAVE_KEYS_REAL], dtype=torch.float32).unsqueeze(1)
    pas_mean, pas_std = w["Pas"]["mean"], w["Pas"]["std"] + 1e-8
    vlv_std           = w["Vlv"]["std"] + 1e-8
    hr_mean,  hr_std  = p["HR"]["mean"],  p["HR"]["std"]  + 1e-8

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
                sc    = torch.tensor([
                    (map_ - pas_mean) / pas_std,
                    (sbp  - pas_mean) / pas_std,
                    (dbp  - pas_mean) / pas_std,
                    sv    / vlv_std,
                    (hr   - hr_mean)  / hr_std,
                ], dtype=torch.float32)
                beats.append(torch.cat([wt.reshape(-1), sc]))

    real = torch.stack(beats)
    log(f"Real: {len(beats)} beats from {data_dir.name}")
    return real


# ─── Flow construction ────────────────────────────────────────────────────────

def build_flow_net(objective: str, latent_dim: int, theta_stats: torch.Tensor,
                   hidden_features: int, num_transforms: int) -> nn.Module:
    """Build MAF or NSF via sbi's posterior_nn with identity embedding.

    The returned module accepts log_prob(theta, context=z) where z is our
    encoded latent. z_score stats for theta are fitted to theta_stats.
    """
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
    parser.add_argument("--run",           required=True,
                        help="e.g. exp-v1_encoder-lipschitz_flow-maf5")
    parser.add_argument("--version",       required=True,
                        help="Version string for run_info filename, e.g. '1' or '1.1'")
    parser.add_argument("--objective",
                        choices=["flow-maf5", "flow-nsf8", "reconstruction"], required=True)
    # Data
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--real-data",     required=True)
    parser.add_argument("--n-sims",        type=int, default=None)
    # Encoder
    parser.add_argument("--latent-dim",    type=int,   default=64)
    parser.add_argument("--sn-ceiling",    type=float, default=2.0)
    parser.add_argument("--proj-hidden",   type=int,   default=None)
    # Flow (ignored for reconstruction)
    parser.add_argument("--hidden-features", type=int, default=None,
                        help="Default: 128 for MAF, 256 for NSF")
    parser.add_argument("--num-transforms",  type=int, default=None,
                        help="Default: 5 for MAF, 8 for NSF")
    # Phase schedule
    parser.add_argument("--phase1-epochs",  type=int,   default=20,
                        help="Epochs with λ=0 — task only on sim data")
    parser.add_argument("--phase2-epochs",  type=int,   default=30,
                        help="Warmup epochs with λ=lambda-p2")
    parser.add_argument("--max-epochs",     type=int,   default=200)
    parser.add_argument("--lambda-target",  type=float, default=0.1,
                        help="DANN weight at target (phase 3)")
    parser.add_argument("--lambda-p2",      type=float, default=0.01,
                        help="DANN weight during warmup (phase 2)")
    # Optimisation
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch-size",     type=int,   default=BATCH_SIZE)
    parser.add_argument("--patience",       type=int,   default=30,
                        help="Early stopping patience, counted from end of phase 1")
    parser.add_argument("--log-every",      type=int,   default=5)
    args = parser.parse_args()

    is_flow    = args.objective.startswith("flow")
    run_type, run_dir = parse_run(args.run)
    is_dry     = run_type == "dry"
    n_sims     = N_SIMS_DRY if is_dry else (args.n_sims or N_SIMS_FULL)
    phase1_end = args.phase1_epochs
    phase2_end = args.phase1_epochs + args.phase2_epochs

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
    log(f"Phase schedule — p1_end={phase1_end}  p2_end={phase2_end}  max={args.max_epochs}")
    log(f"λ_p2={args.lambda_p2}  λ_target={args.lambda_target}")

    # ── Load data ──────────────────────────────────────────────────────────────
    stats    = load_stats(STATS_PATH)
    manifest = load_manifest(Path(args.sim_data_root) / "manifest_train.json")

    log(f"Loading {n_sims} sim observations...")
    theta_all, x_all = load_sim_data(
        Path(args.sim_data_root) / "train", manifest, stats, n_sims, log
    )

    log("Loading real patient beats...")
    real_beats = load_real_beats(Path(args.real_data), stats, log)

    # ── Build models ───────────────────────────────────────────────────────────
    encoder = LipschitzReducedAutoencoderEncoder(
        latent_dim=args.latent_dim,
        sn_ceiling=args.sn_ceiling,
        proj_hidden=args.proj_hidden,
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

    grl        = GradientReversalLayer(alpha=1.0)
    domain_clf = DomainClassifier(latent_dim=args.latent_dim, hidden=256).to(DEVICE)
    log(f"Domain classifier: {domain_clf.describe()}")

    # ── Optimizer ──────────────────────────────────────────────────────────────
    all_params = (list(encoder.parameters())
                  + list(task_head.parameters())
                  + list(domain_clf.parameters()))
    opt = torch.optim.Adam(all_params, lr=args.lr)
    log(f"Adam  lr={args.lr}  total_params={sum(p.numel() for p in all_params):,}")

    # ── Write run_info ─────────────────────────────────────────────────────────
    run_info = dict(
        run=args.run,
        version=args.version,
        type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv),
        git_hash=git_hash(),
        device=DEVICE,
        encoder=encoder.describe(),
        task_head=dict(
            type=args.objective,
            hidden_features=args.hidden_features if is_flow else None,
            num_transforms=args.num_transforms   if is_flow else None,
        ),
        domain_clf=domain_clf.describe(),
        data=dict(
            n_sims=n_sims,
            n_real_beats=len(real_beats),
            sim_data_root=args.sim_data_root,
            real_data=args.real_data,
        ),
        schedule=dict(
            phase1_end=phase1_end,
            phase2_end=phase2_end,
            max_epochs=args.max_epochs,
            lambda_p2=args.lambda_p2,
            lambda_target=args.lambda_target,
        ),
        training=dict(lr=args.lr, batch_size=args.batch_size, patience=args.patience),
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
    csv_writer.writerow(["epoch", "task", "domain", "total", "lambda", "phase"])

    log("Training...")
    for epoch in range(1, args.max_epochs + 1):
        lambda_e   = get_lambda(epoch, phase1_end, phase2_end, args.lambda_p2, args.lambda_target)
        use_domain = lambda_e > 0.0
        phase      = 1 if epoch <= phase1_end else (2 if epoch <= phase2_end else 3)

        encoder.train(); task_head.train(); domain_clf.train()

        perm = torch.randperm(n_total)
        ep_task = ep_dom = ep_tot = 0.0
        n_batches = 0

        for start in range(0, n_total, args.batch_size):
            idx     = perm[start:start + args.batch_size]
            x_sim_b = x_all[idx].to(DEVICE)
            theta_b = theta_all[idx].to(DEVICE)

            z_sim = encoder(x_sim_b)

            # ── Task loss ─────────────────────────────────────────────────────
            if is_flow:
                task_loss = -task_head.log_prob(theta_b, context=z_sim).mean()
            else:
                task_loss = F.mse_loss(task_head(z_sim), x_sim_b[:, :WAVE_LEN])

            # ── Domain loss (λ > 0) ───────────────────────────────────────────
            if use_domain:
                real_idx = torch.randint(0, len(real_beats), (len(idx),), device=DEVICE)
                x_real_b = real_beats[real_idx].to(DEVICE)
                z_real   = encoder(x_real_b)
                logits   = torch.cat([domain_clf(grl(z_sim)), domain_clf(grl(z_real))])
                labels   = torch.cat([
                    torch.zeros(len(z_sim),  device=DEVICE),
                    torch.ones(len(z_real), device=DEVICE),
                ])
                domain_loss = F.binary_cross_entropy_with_logits(logits, labels)
            else:
                domain_loss = x_sim_b.new_zeros(1).squeeze()

            total = task_loss + lambda_e * domain_loss

            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step()

            ep_task += task_loss.item()
            ep_dom  += domain_loss.item()
            ep_tot  += total.item()
            n_batches += 1

        avg_task = ep_task / n_batches
        avg_dom  = ep_dom  / n_batches
        avg_tot  = ep_tot  / n_batches

        csv_writer.writerow([epoch,
                             f"{avg_task:.5f}", f"{avg_dom:.5f}", f"{avg_tot:.5f}",
                             f"{lambda_e:.4f}", phase])
        csv_fh.flush()

        if avg_tot < best_loss:
            best_loss    = avg_tot
            best_enc_sd  = {k: v.clone() for k, v in encoder.state_dict().items()}
            best_task_sd = {k: v.clone() for k, v in task_head.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % args.log_every == 0 or epoch == 1:
            log(f"  ep {epoch:4d}/{args.max_epochs}  ph={phase}"
                f"  task={avg_task:.4f}  dom={avg_dom:.4f}"
                f"  total={avg_tot:.4f}  λ={lambda_e:.3f}  wait={wait}")

        # Only apply patience after phase 1 — total isn't comparable across phases
        if epoch > phase1_end and wait >= args.patience:
            log(f"Early stop at epoch {epoch}  best_total={best_loss:.4f}")
            break

    csv_fh.close()

    # ── Save best checkpoints ──────────────────────────────────────────────────
    encoder.load_state_dict(best_enc_sd)
    task_head.load_state_dict(best_task_sd)

    torch.save(encoder.state_dict(), run_dir / "encoder.pt")
    if is_flow:
        torch.save(task_head, run_dir / "flow_net.pt")   # full module: z-score stats live inside
    else:
        torch.save(task_head.state_dict(), run_dir / "decoder.pt")
    torch.save(domain_clf.state_dict(), run_dir / "domain_clf.pt")
    log(f"Saved encoder.pt  {'flow_net.pt' if is_flow else 'decoder.pt'}  domain_clf.pt")

    log_fh.close()
    sys.stdout = _stdout
    sys.stderr = _stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
