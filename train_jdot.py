"""
v4: JDOT-based domain adaptation for cardiovascular SBI.

Two independent encoders, two independent flows — no shared flow, no shared encoder,
no WDGRL critic. Alignment is not adversarial; it's an explicit, differentiable
optimal-transport coupling between z_sim and z_real, solved via Sinkhorn every step,
jointly with model training (not a separate "align, then freeze and pseudo-label" pipeline).

  E_sim  + flow_sim  : standard NPE, trained directly on labeled sim data
                       (theta_sim, unconditional hard anchor — unchanged from v3).
  E_real + flow_real : trained on entropy-regularized OT (JDOT)-transported soft
                       pseudo-labels from sim thetas. This is E_real's ONLY gradient
                       source.

Cost for matching sim i to real j combines feature proximity and label plausibility
(the "joint" part of Joint Distribution OT):

  C_ij = lam_feat * ||z_sim_i - z_real_j||^2
       + lam_label * ||theta_sim_i - real_guess_j||^2

real_guess_j is a cheap proxy for "what does flow_real currently believe about real
patient j" — the mean of a few samples drawn from flow_real's own conditional at z_real_j,
NOT a full log_prob evaluation for every (sim, real) pair (that would be ~B*N_real flow
evaluations per step — computationally prohibitive). real_guess is computed under
torch.no_grad(); it's a fixed reference point for deciding which matches are cheap, not
a differentiable path.

Gradient routing (this is the load-bearing part — see cv-spin-latent's session notes for
the full design discussion of why each detach placement matters):
  E_sim  : L_sim (always, full strength) + L_ot's feature term (z_real is DETACHED when
           building the cost matrix for L_ot, so L_ot's gradient reaches E_sim only —
           this is "the JDOT objective finds a better map by shaping E_sim").
  E_real : L_real only. z_real is ATTACHED here (a second, separate use of E_real's
           output from the one used to build the cost matrix). The soft weights used to
           combine L_real's per-candidate losses are DETACHED before use, so L_real's
           gradient can't leak back through the coupling into E_sim — "the flow objective
           shapes E_real, and only E_real."

L_real itself only evaluates flow_real.log_prob against each real patient's top-K
highest-weight sim candidates (not all B), for the same computational reason as
real_guess above — with reasonably peaked Sinkhorn couplings this is a lossless-in-
practice truncation, not an approximation that changes what's being optimized.

No warmup-then-freeze phases beyond a brief sim-only warmup (E_sim/flow_sim train alone
on true theta for --sim-warmup epochs) before the OT/real-side machinery turns on — early
matching is close to noise before E_sim has had any time to organize itself, so this
avoids feeding early-noise-derived labels into flow_real from epoch 1. lam_real is then
additionally ramped in over --real-ramp epochs once the joint phase starts, so E_real
doesn't fully commit to trusting the matching before it's had a chance to improve.

Usage:
    python train_jdot.py \\
        --run exp-v4_jdot \\
        --version 4 \\
        --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \\
        --real-data ~/data/real_data/onebeat_300patients
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import torch

from dataset import PARAM_KEYS_INFER, load_stats, load_manifest
from models import LipschitzReducedAutoencoderEncoder
from train_joint import (
    git_hash, Tee, parse_run, load_sim_data, load_real_beats, build_flow_net,
)

N_PARAMS_INFER = len(PARAM_KEYS_INFER)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── Sinkhorn / JDOT ────────────────────────────────────────────────────────────

def sinkhorn(C: torch.Tensor, epsilon: float, n_iters: int = 50) -> torch.Tensor:
    """
    Entropy-regularized OT coupling for cost matrix C (n_sim, n_real), uniform marginals.
    Log-domain stabilized (avoids overflow/underflow from raw exp(-C/epsilon) at small
    epsilon). Fully differentiable — every op here is a standard torch op, gradient
    flows straight back through to whatever C depended on.
    """
    n_sim, n_real = C.shape
    log_mu = -math.log(n_sim)
    log_nu = -math.log(n_real)

    f = torch.zeros(n_sim, device=C.device, dtype=C.dtype)
    g = torch.zeros(n_real, device=C.device, dtype=C.dtype)

    for _ in range(n_iters):
        f = epsilon * (log_mu - torch.logsumexp((-C + g[None, :]) / epsilon, dim=1)) + f
        g = epsilon * (log_nu - torch.logsumexp((-C + f[:, None]) / epsilon, dim=0)) + g

    log_gamma = (-C + f[:, None] + g[None, :]) / epsilon
    return log_gamma.exp()


def jdot_step(E_sim, E_real, flow_sim, flow_real, x_sim_b, theta_sim_b, real_beats_all,
             lam_feat, lam_label, lam_ot, lam_real,
             sinkhorn_epsilon, sinkhorn_iters, ot_label_samples, ot_topk):
    """
    One combined JDOT step. Returns (loss, info_dict). See module docstring for the
    gradient-routing rationale behind each detach.
    """
    z_sim = E_sim(x_sim_b)                              # attached
    L_sim = -flow_sim.log_prob(theta_sim_b, condition=z_sim).mean()

    z_real_full     = E_real(real_beats_all)             # attached — used later for L_real
    z_real_detached = z_real_full.detach()                # detached — used for the OT cost only

    # Cheap proxy for "what does flow_real currently believe at z_real_j" — amortizes to
    # one batched sample call over all real points, not one flow evaluation per (sim,real)
    # pair. Detached: this is a fixed reference point, not a differentiable path.
    with torch.no_grad():
        samples    = flow_real.sample((ot_label_samples,), condition=z_real_detached)  # (S, N_real, theta_dim)
        real_guess = samples.mean(dim=0)                                                # (N_real, theta_dim)

    feat_dist  = torch.cdist(z_sim, z_real_detached, p=2) ** 2        # (B, N_real)
    label_dist = torch.cdist(theta_sim_b, real_guess, p=2) ** 2       # (B, N_real)
    C = lam_feat * feat_dist + lam_label * label_dist

    gamma = sinkhorn(C, sinkhorn_epsilon, sinkhorn_iters)              # (B, N_real)
    L_ot  = (gamma * C).sum()                                          # gradient -> z_sim only (z_real detached above)

    # Soft weights per real patient, detached before use in L_real so that loss can't
    # leak gradient back through the coupling into E_sim — L_real shapes E_real only.
    w = (gamma / (gamma.sum(dim=0, keepdim=True) + 1e-12)).detach()    # (B, N_real)

    # Truncate to each real patient's top-K highest-weight sim candidates — with a
    # reasonably peaked coupling this drops only near-zero-weight pairs, and turns an
    # O(B * N_real) flow-evaluation cost into O(K * N_real).
    k_eff = min(ot_topk, w.shape[0])
    topk_w, topk_idx = torch.topk(w, k=k_eff, dim=0)                   # (K, N_real) each
    topk_w = (topk_w / (topk_w.sum(dim=0, keepdim=True) + 1e-12))

    n_real = real_beats_all.shape[0]
    theta_topk  = theta_sim_b[topk_idx]                                            # (K, N_real, theta_dim)
    theta_flat  = theta_topk.reshape(k_eff * n_real, -1)
    z_real_flat = z_real_full.unsqueeze(0).expand(k_eff, -1, -1).reshape(k_eff * n_real, -1)  # attached

    logp = flow_real.log_prob(theta_flat, condition=z_real_flat).reshape(k_eff, n_real)
    L_real = (topk_w * (-logp)).sum(dim=0).mean()

    loss = L_sim + lam_ot * L_ot + lam_real * L_real
    return loss, {
        "L_sim": L_sim.item(), "L_ot": L_ot.item(), "L_real": L_real.item(),
        "gamma_max": gamma.max().item(), "gamma_entropy": (-(gamma + 1e-12) * (gamma + 1e-12).log()).sum().item(),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    # Identity
    parser.add_argument("--run",     required=True, help="e.g. exp-v4_jdot")
    parser.add_argument("--version", required=True)
    # Data
    parser.add_argument("--sim-data-root", required=True)
    parser.add_argument("--real-data",     required=True)
    parser.add_argument("--stats-path",    default="norm_stats.json")
    parser.add_argument("--manifest-train", default="manifest_train.json")
    parser.add_argument("--n-sims",        type=int, default=None)
    # Encoder
    parser.add_argument("--latent-dim", type=int,   default=128)
    parser.add_argument("--sn-ceiling", type=float, default=2.0)
    # Flow (both flow_sim and flow_real share this config)
    parser.add_argument("--hidden-features", type=int, default=128)
    parser.add_argument("--num-transforms",  type=int, default=5)
    # Schedule
    parser.add_argument("--sim-warmup", type=int, default=10,
                        help="Epochs of E_sim/flow_sim-only training before JDOT/E_real turn on")
    parser.add_argument("--real-ramp",  type=int, default=30,
                        help="Epochs (after sim-warmup) over which lam_real ramps 0 -> target")
    parser.add_argument("--max-epochs", type=int, default=400)
    # JDOT
    parser.add_argument("--lam-feat",  type=float, default=1.0, help="Weight on feature-distance cost term")
    parser.add_argument("--lam-label", type=float, default=1.0, help="Weight on label-consistency cost term")
    parser.add_argument("--lam-ot",    type=float, default=0.1, help="Weight on L_ot in E_sim's total loss")
    parser.add_argument("--lam-real",  type=float, default=1.0, help="Target weight on L_real (ramped)")
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters",   type=int,   default=50)
    parser.add_argument("--ot-label-samples", type=int,   default=8,
                        help="Samples drawn per real patient for the cheap real_guess proxy")
    parser.add_argument("--ot-topk", type=int, default=8,
                        help="Top-K sim candidates per real patient used to compute L_real")
    # Optimization
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--log-every",  type=int,   default=5)
    args = parser.parse_args()

    run_type, run_dir = parse_run(args.run)
    n_sims = args.n_sims or 100_000
    run_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = run_dir / f"train_{args.run}_{date_str}.log"
    log_fh   = open(log_path, "w")
    _stdout, _stderr = sys.stdout, sys.stderr
    sys.stdout = Tee(log_fh)
    sys.stderr = Tee(log_fh)

    def log(msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Run: {args.run}  v={args.version}  ({run_type})")
    log(f"Device: {DEVICE}")
    log(f"Schedule: sim_warmup={args.sim_warmup}  real_ramp={args.real_ramp}  max_epochs={args.max_epochs}")
    log(f"JDOT: lam_feat={args.lam_feat}  lam_label={args.lam_label}  lam_ot={args.lam_ot}  "
        f"lam_real={args.lam_real}  epsilon={args.sinkhorn_epsilon}  iters={args.sinkhorn_iters}  "
        f"label_samples={args.ot_label_samples}  topk={args.ot_topk}")

    ghash = git_hash()

    # ── Data ──────────────────────────────────────────────────────────────────
    stats    = load_stats(Path(args.stats_path))
    manifest = load_manifest(Path(args.sim_data_root) / args.manifest_train)

    log(f"Loading {n_sims} sim observations...")
    theta_all, x_all = load_sim_data(
        Path(args.sim_data_root) / "train", manifest, stats, n_sims, log,
    )

    log("Loading real patient beats...")
    real_beats = load_real_beats(Path(args.real_data), stats, log)
    real_beats = real_beats.to(DEVICE)
    log(f"x_all      mean={x_all.mean():.4f}  std={x_all.std():.4f}  shape={tuple(x_all.shape)}")
    log(f"real_beats mean={real_beats.mean():.4f}  std={real_beats.std():.4f}  shape={tuple(real_beats.shape)}")

    # ── Models ────────────────────────────────────────────────────────────────
    E_sim  = LipschitzReducedAutoencoderEncoder(latent_dim=args.latent_dim, sn_ceiling=args.sn_ceiling).to(DEVICE)
    E_real = LipschitzReducedAutoencoderEncoder(latent_dim=args.latent_dim, sn_ceiling=args.sn_ceiling).to(DEVICE)

    n_stats = min(10_000, len(theta_all))
    flow_sim  = build_flow_net("flow-maf5", args.latent_dim, theta_all[:n_stats],
                               args.hidden_features, args.num_transforms).to(DEVICE)
    flow_real = build_flow_net("flow-maf5", args.latent_dim, theta_all[:n_stats],
                               args.hidden_features, args.num_transforms).to(DEVICE)

    log(f"E_sim/E_real params: {sum(p.numel() for p in E_sim.parameters()):,} each")
    log(f"flow_sim/flow_real params: {sum(p.numel() for p in flow_sim.parameters()):,} each")

    opt_sim  = torch.optim.Adam(list(E_sim.parameters()) + list(flow_sim.parameters()), lr=args.lr)
    opt_real = torch.optim.Adam(list(E_real.parameters()) + list(flow_real.parameters()), lr=args.lr)

    # ── run_info ──────────────────────────────────────────────────────────────
    run_info = dict(
        run=args.run, version=args.version, type=run_type,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        command=" ".join(sys.argv), git_hash=ghash, device=DEVICE,
        schedule=dict(sim_warmup=args.sim_warmup, real_ramp=args.real_ramp, max_epochs=args.max_epochs),
        jdot=dict(lam_feat=args.lam_feat, lam_label=args.lam_label, lam_ot=args.lam_ot,
                  lam_real=args.lam_real, sinkhorn_epsilon=args.sinkhorn_epsilon,
                  sinkhorn_iters=args.sinkhorn_iters, ot_label_samples=args.ot_label_samples,
                  ot_topk=args.ot_topk),
        data=dict(n_sims=n_sims, n_real_beats=len(real_beats),
                  sim_data_root=args.sim_data_root, real_data=args.real_data),
        training=dict(lr=args.lr, batch_size=args.batch_size),
    )
    info_path = run_dir / f"run_info_v{args.version}.json"
    info_path.write_text(json.dumps(run_info, indent=2))
    log(f"run_info_v{args.version}.json written  git={ghash}")

    # ── Training loop ─────────────────────────────────────────────────────────
    n_total = len(theta_all)
    csv_path   = run_dir / f"train_log_{date_str}.csv"
    csv_fh     = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(["epoch", "phase", "L_sim", "L_ot", "L_real", "total",
                        "lam_real", "gamma_max", "gamma_entropy"])

    log("Training...")
    for epoch in range(1, args.max_epochs + 1):
        joint = epoch > args.sim_warmup
        lam_real_ep = 0.0
        if joint:
            t = min(1.0, (epoch - args.sim_warmup) / max(1, args.real_ramp))
            lam_real_ep = args.lam_real * t

        E_sim.train(); flow_sim.train(); E_real.train(); flow_real.train()

        perm = torch.randperm(n_total)
        sums = dict.fromkeys(["L_sim", "L_ot", "L_real", "total", "gamma_max", "gamma_entropy"], 0.0)
        n_batches = 0

        for start in range(0, n_total, args.batch_size):
            idx     = perm[start:start + args.batch_size]
            x_sim_b = x_all[idx].to(DEVICE)
            theta_b = theta_all[idx].to(DEVICE)

            opt_sim.zero_grad()
            if not joint:
                z_sim = E_sim(x_sim_b)
                L_sim = -flow_sim.log_prob(theta_b, condition=z_sim).mean()
                L_sim.backward()
                torch.nn.utils.clip_grad_norm_(list(E_sim.parameters()) + list(flow_sim.parameters()), 1.0)
                opt_sim.step()
                info = {"L_sim": L_sim.item(), "L_ot": 0.0, "L_real": 0.0,
                       "gamma_max": 0.0, "gamma_entropy": 0.0}
            else:
                opt_real.zero_grad()
                loss, info = jdot_step(
                    E_sim, E_real, flow_sim, flow_real, x_sim_b, theta_b, real_beats,
                    args.lam_feat, args.lam_label, args.lam_ot, lam_real_ep,
                    args.sinkhorn_epsilon, args.sinkhorn_iters, args.ot_label_samples, args.ot_topk,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(E_sim.parameters()) + list(flow_sim.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(list(E_real.parameters()) + list(flow_real.parameters()), 1.0)
                opt_sim.step(); opt_real.step()

            total = info["L_sim"] + args.lam_ot * info["L_ot"] + lam_real_ep * info["L_real"]
            sums["L_sim"] += info["L_sim"]; sums["L_ot"] += info["L_ot"]; sums["L_real"] += info["L_real"]
            sums["total"] += total
            sums["gamma_max"] += info["gamma_max"]; sums["gamma_entropy"] += info["gamma_entropy"]
            n_batches += 1

        avg = {k: v / n_batches for k, v in sums.items()}
        phase_name = "sim-warmup" if not joint else "joint"

        csv_writer.writerow([epoch, phase_name, f"{avg['L_sim']:.5f}", f"{avg['L_ot']:.5f}",
                            f"{avg['L_real']:.5f}", f"{avg['total']:.5f}", f"{lam_real_ep:.4f}",
                            f"{avg['gamma_max']:.5f}", f"{avg['gamma_entropy']:.5f}"])
        csv_fh.flush()

        if epoch % args.log_every == 0 or epoch == 1:
            log(f"  ep {epoch:4d}/{args.max_epochs}  [{phase_name}]"
                f"  L_sim={avg['L_sim']:.4f}  L_ot={avg['L_ot']:.4f}  L_real={avg['L_real']:.4f}"
                f"  lam_real={lam_real_ep:.3f}  gamma_max={avg['gamma_max']:.4f}")

    csv_fh.close()

    # ── Save checkpoints ──────────────────────────────────────────────────────
    torch.save(E_sim.state_dict(),  run_dir / "encoder_sim.pt")
    torch.save(E_real.state_dict(), run_dir / "encoder_real.pt")
    torch.save(flow_sim,  run_dir / "flow_sim.pt")
    torch.save(flow_real, run_dir / "flow_real.pt")
    log("Saved encoder_sim.pt  encoder_real.pt  flow_sim.pt  flow_real.pt")

    log_fh.close()
    sys.stdout = _stdout
    sys.stderr = _stderr
    print(f"\nLog written to {log_path}")


if __name__ == "__main__":
    main()
