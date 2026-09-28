"""
NPE + self-consistency baseline (Schmitt et al., "Leveraging Self-Consistency for
Data-Efficient Amortized Bayesian Inference", ICML 2024, arXiv:2310.04395, Case II /
NPLE variant): jointly trains the posterior flow q_phi(theta|h(x)) with a second,
likelihood flow q_eta(h(x)|theta) over the SAME encoder embedding h(x) the posterior
conditions on -- not raw x. This is the theoretically correct pairing given q_phi only
ever sees h(x), never raw x (see design discussion, session notes).

L_SC(x) = Var_k[ log p(theta_k) + log q_eta(h(x)|theta_k) - log q_phi(theta_k|h(x)) ]
          for theta_k ~ q_phi(.|h(x)), k=1..K

L_total = L_NPE + L_NLE + lambda(epoch) * L_SC

Sim-only training, matching train_joint.py's vanilla NPE baseline protocol exactly
(same encoder/posterior-flow architecture and hyperparameters) -- no WDGRL critic (no
domain adaptation of any kind), no real data touched during training at all. Real
patients are only used at evaluation time, via the existing eval scripts (encoder.pt/
flow_net.pt are saved in the same format vanilla NPE uses, so eval_calibration.py,
eval_zscore_shrinkage.py, eval_real_patient_inference.py, eval_recon_sim.py all work
unchanged -- q_eta is training-only and never touches inference).

Usage:
    python train_npe_selfconsistency.py \\
        --run exp-npe-selfconsistency-baseline_cvdannsbi --version 1 \\
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \\
        --n-sims 300000 --max-epochs 400
"""

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')

from sbi.neural_nets import posterior_nn

from dataset import ReducedCVDataset, PARAM_KEYS_INFER, N_SCALARS, load_stats, load_manifest
from models import LipschitzReducedAutoencoderEncoder

BATCH_SIZE  = 512
N_SIMS_FULL = 100_000
N_SIMS_DRY  = 512
N_PARAMS_INFER = len(PARAM_KEYS_INFER)  # 24

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


class Tee:
    def __init__(self, fh):
        self._fh = fh
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
    log(f"Sim: {loaded} sims  theta={tuple(theta_all.shape[1:])}  x={tuple(x_all.shape[1:])}")
    return theta_all, x_all


def build_flow(target_dim: int, condition_dim: int, hidden_features: int,
               num_transforms: int, z_score_target: str, z_score_condition: str,
               target_batch: torch.Tensor, condition_batch: torch.Tensor) -> nn.Module:
    """
    Generic MAF builder via sbi's posterior_nn -- fully symmetric in target/condition,
    so the same call builds either q_phi(theta|h) or q_eta(h|theta), just by swapping
    which tensor plays "theta" (target, gets the autoregressive treatment) vs "x"
    (condition, fed as context to every MADE block, not autoregressively masked).
    """
    build_fn = posterior_nn(
        model="maf",
        embedding_net=nn.Identity(),
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        z_score_theta=z_score_target,
        z_score_x=z_score_condition,
    )
    return build_fn(target_batch.cpu(), condition_batch.cpu())


def prior_log_prob(theta: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor,
                    floor: float = -1e6) -> torch.Tensor:
    """Log-density of a uniform box prior, with a finite floor for out-of-box samples
    (posterior-sampled theta_k aren't prior-clipped, so some land outside)."""
    in_box   = ((theta >= lo) & (theta <= hi)).all(dim=-1)
    log_vol  = torch.log(hi - lo).sum()
    return torch.where(in_box, -log_vol.expand_as(in_box.float()), torch.full_like(in_box, floor, dtype=theta.dtype))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--stats-path", default="norm_stats.json")
    parser.add_argument("--manifest-train", default="manifest_train.json")
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--no-sv", action="store_true")
    # Encoder
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--sn-ceiling", type=float, default=2.0)
    parser.add_argument("--proj-hidden", type=int, default=None)
    # Posterior flow q_phi(theta|h) -- matches vanilla NPE baseline exactly
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--num-transforms", type=int, default=5)
    # Likelihood flow q_eta(h|theta) -- wider/deeper, target dim is 128 not 24
    parser.add_argument("--lik-hidden-features", type=int, default=512)
    parser.add_argument("--lik-num-transforms", type=int, default=8)
    # Self-consistency
    parser.add_argument("--sc-k", type=int, default=10, help="theta_k samples per x for L_SC")
    parser.add_argument("--lam-sc-max", type=float, default=1.0)
    parser.add_argument("--sc-warmup", type=int, default=50, help="epochs into joint phase before L_SC ramp starts")
    parser.add_argument("--sc-ramp", type=int, default=50, help="epochs to ramp lambda_sc 0 -> max")
    # Phase schedule
    parser.add_argument("--flow-warmup", type=int, default=2)
    parser.add_argument("--enc-warmup", type=int, default=10)
    parser.add_argument("--max-epochs", type=int, default=400)
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    run_type, run_dir = parse_run(args.run)
    is_dry  = run_type == "dry"
    n_sims  = N_SIMS_DRY if is_dry else (args.n_sims or N_SIMS_FULL)
    include_sv = not args.no_sv

    flow_end = args.flow_warmup
    enc_end  = flow_end + args.enc_warmup

    def get_phase(epoch: int) -> int:
        if epoch <= flow_end: return 0
        if epoch <= enc_end:  return 1
        return 2

    def lambda_sc(epoch: int) -> float:
        joint_epoch = epoch - enc_end
        if joint_epoch <= args.sc_warmup:
            return 0.0
        t = (joint_epoch - args.sc_warmup) / max(args.sc_ramp, 1)
        return args.lam_sc_max * min(t, 1.0)

    run_dir.mkdir(parents=True, exist_ok=True)
    start_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"train_{args.run}_{start_ts}.log"
    log_fh   = open(log_path, "w")
    sys.stdout = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

    ghash = git_hash()
    log(f"Run: {args.run}  v={args.version}  ({'dry' if is_dry else 'full'})  git={ghash}")
    log(f"Phase boundaries — flow_end={flow_end}  enc_end={enc_end}  max={args.max_epochs}")
    log(f"Self-consistency: K={args.sc_k}  lam_sc_max={args.lam_sc_max}  "
        f"sc_warmup={args.sc_warmup}  sc_ramp={args.sc_ramp}")
    log(f"Posterior flow: hidden={args.hidden_features}  transforms={args.num_transforms}")
    log(f"Likelihood flow: hidden={args.lik_hidden_features}  transforms={args.lik_num_transforms}")

    # ── Data ──────────────────────────────────────────────────────────────────
    stats    = load_stats(Path(args.stats_path))
    manifest = load_manifest(Path(args.sim_data_root) / args.manifest_train)
    log(f"Loading {n_sims} sim observations...")
    theta_all, x_all = load_sim_data(
        Path(args.sim_data_root) / "train", manifest, stats, n_sims, log,
        include_sv=include_sv, scalar_norm="legacy",
    )

    lo_t = torch.tensor([manifest["config"]["pvar_low"][k]  for k in PARAM_KEYS_INFER], dtype=torch.float32).to(DEVICE)
    hi_t = torch.tensor([manifest["config"]["pvar_high"][k] for k in PARAM_KEYS_INFER], dtype=torch.float32).to(DEVICE)

    # ── Models ────────────────────────────────────────────────────────────────
    encoder = LipschitzReducedAutoencoderEncoder(
        latent_dim=args.latent_dim, sn_ceiling=args.sn_ceiling,
        proj_hidden=args.proj_hidden, n_scalars=(4 if args.no_sv else N_SCALARS),
    ).to(DEVICE)
    log(f"Encoder: {encoder.describe()}")

    with torch.no_grad():
        seed_idx = torch.randperm(len(x_all))[:2048]
        h_seed   = encoder(x_all[seed_idx].to(DEVICE)).cpu()
    theta_seed = theta_all[seed_idx]

    flow_post = build_flow(
        target_dim=N_PARAMS_INFER, condition_dim=args.latent_dim,
        hidden_features=args.hidden_features, num_transforms=args.num_transforms,
        z_score_target="independent", z_score_condition="none",
        target_batch=theta_seed, condition_batch=h_seed,
    ).to(DEVICE)
    log(f"Posterior flow params: {sum(p.numel() for p in flow_post.parameters()):,}")

    flow_lik = build_flow(
        target_dim=args.latent_dim, condition_dim=N_PARAMS_INFER,
        hidden_features=args.lik_hidden_features, num_transforms=args.lik_num_transforms,
        z_score_target="none", z_score_condition="independent",
        target_batch=h_seed, condition_batch=theta_seed,
    ).to(DEVICE)
    log(f"Likelihood flow params: {sum(p.numel() for p in flow_lik.parameters()):,}")

    all_params = list(encoder.parameters()) + list(flow_post.parameters()) + list(flow_lik.parameters())
    opt = torch.optim.Adam(all_params, lr=args.lr)
    log(f"Adam lr={args.lr}  total params={sum(p.numel() for p in all_params):,}")

    # ── run_info ──────────────────────────────────────────────────────────────
    run_info = dict(
        run=args.run, version=args.version, type=run_type, timestamp=datetime.now().isoformat(),
        command=" ".join(sys.argv), git_hash=ghash, device=DEVICE,
        encoder=dict(type="LipschitzReducedAutoencoderEncoder", latent_dim=args.latent_dim,
                     sn_ceiling=args.sn_ceiling, proj_hidden=args.proj_hidden,
                     n_params=sum(p.numel() for p in encoder.parameters())),
        posterior_flow=dict(hidden_features=args.hidden_features, num_transforms=args.num_transforms,
                             n_params=sum(p.numel() for p in flow_post.parameters())),
        likelihood_flow=dict(hidden_features=args.lik_hidden_features, num_transforms=args.lik_num_transforms,
                              n_params=sum(p.numel() for p in flow_lik.parameters())),
        self_consistency=dict(k=args.sc_k, lam_sc_max=args.lam_sc_max,
                               sc_warmup=args.sc_warmup, sc_ramp=args.sc_ramp),
        data=dict(n_sims=n_sims, sim_data_root=args.sim_data_root),
        schedule=dict(flow_end=flow_end, enc_end=enc_end, max_epochs=args.max_epochs),
        training=dict(lr=args.lr, batch_size=args.batch_size, grad_clip=args.grad_clip),
    )
    info_path = run_dir / f"run_info_v{args.version}.json"
    info_path.write_text(json.dumps(run_info, indent=2))
    log(f"run_info_v{args.version}.json written")

    csv_path = run_dir / f"train_log_{start_ts}.csv"
    csv_fh   = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["epoch", "phase", "l_npe", "l_nle", "l_sc", "lambda_sc", "total"])

    n = len(theta_all)
    for epoch in range(1, args.max_epochs + 1):
        phase = get_phase(epoch)
        lam_sc = lambda_sc(epoch) if phase == 2 else 0.0

        encoder.train(); flow_post.train(); flow_lik.train()
        perm = torch.randperm(n)
        ep_npe = ep_nle = ep_sc = ep_total = 0.0
        n_batches = 0

        for bstart in range(0, n, args.batch_size):
            idx = perm[bstart:bstart + args.batch_size]
            theta_b = theta_all[idx].to(DEVICE)
            x_b     = x_all[idx].to(DEVICE)
            B       = len(idx)

            if phase == 0:
                with torch.no_grad():
                    h = encoder(x_b)
            else:
                h = encoder(x_b)

            l_npe = -flow_post.log_prob(theta_b, condition=h).mean()
            l_nle = -flow_lik.log_prob(h, condition=theta_b).mean()

            if lam_sc > 0:
                theta_k = flow_post.sample((args.sc_k,), condition=h)          # (K, B, 24)
                theta_k_flat = theta_k.reshape(args.sc_k * B, N_PARAMS_INFER)
                h_rep = h.unsqueeze(0).expand(args.sc_k, B, args.latent_dim).reshape(args.sc_k * B, args.latent_dim)

                log_qphi = flow_post.log_prob(theta_k_flat, condition=h_rep)
                log_qeta = flow_lik.log_prob(h_rep, condition=theta_k_flat)
                log_prior = prior_log_prob(theta_k_flat, lo_t, hi_t)

                log_phat = (log_prior + log_qeta - log_qphi).reshape(args.sc_k, B)
                l_sc = log_phat.var(dim=0, unbiased=False).mean()
            else:
                l_sc = torch.zeros((), device=DEVICE)

            if phase == 0:
                loss = l_npe + l_nle
            else:
                loss = l_npe + l_nle + lam_sc * l_sc

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            opt.step()

            ep_npe += l_npe.item(); ep_nle += l_nle.item()
            ep_sc  += l_sc.item();  ep_total += loss.item()
            n_batches += 1

        ep_npe /= n_batches; ep_nle /= n_batches; ep_sc /= n_batches; ep_total /= n_batches
        phase_name = ["flow-warmup", "enc-warmup", "joint"][phase]
        csv_writer.writerow([epoch, phase_name, ep_npe, ep_nle, ep_sc, lam_sc, ep_total])
        csv_fh.flush()

        if epoch % args.log_every == 0 or epoch == args.max_epochs:
            log(f"  ep {epoch:4d}/{args.max_epochs}  [{phase_name}]  "
                f"l_npe={ep_npe:.4f}  l_nle={ep_nle:.4f}  l_sc={ep_sc:.4f}  "
                f"lambda_sc={lam_sc:.3f}  total={ep_total:.4f}")

    csv_fh.close()

    torch.save(encoder.state_dict(), run_dir / "encoder.pt")
    torch.save(flow_post, run_dir / "flow_net.pt")
    torch.save(flow_lik, run_dir / "flow_lik.pt")
    log("Saved encoder.pt  flow_net.pt  flow_lik.pt")


if __name__ == "__main__":
    main()
