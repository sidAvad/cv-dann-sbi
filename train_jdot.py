"""
v4: JDOT-based domain adaptation for cardiovascular SBI.

Two independent encoders, two independent flows — no shared flow, no shared encoder,
no WDGRL critic. Alignment is not adversarial; it's a hard nearest-neighbor match between
z_sim and z_real, recomputed every step, jointly with model training (not a separate
"align, then freeze and pseudo-label" pipeline). No entropic/Sinkhorn regularization and
no LP-based bipartite assignment — each real point in the batch just takes whichever sim
point is currently cheapest under the cost below (vanilla JDOT, matching the original
paper's mechanism rather than a differentiable-soft-coupling variant of it).

  E_sim  + flow_sim  : standard NPE, trained directly on labeled sim data
                       (theta_sim, unconditional hard anchor — unchanged from v3).
  E_real + flow_real : trained on hard-OT-transported pseudo-labels from sim thetas.
                       This is E_real's ONLY gradient source.

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

Sim and real batches are both --batch-size (real side via a fresh Mixup draw each step,
--mixup-n defaulting to --batch-size) so the cost matrix is symmetric (B, B) and every
real point gets exactly one nearest-sim match every step — no coverage gaps, unlike a
rectangular assignment. A given real patient's match still varies step to step (both the
sim permutation and the mixup draw are independently randomized), which is the mechanism
relied on to avoid fixating on one bad match rather than any explicit regularization.

Gradient routing (this is the load-bearing part — see cv-spin-latent's session notes for
the full design discussion of why each detach placement matters):
  E_sim  : L_sim (always, full strength) + L_ot's feature term (z_real is DETACHED when
           building the cost matrix for L_ot, so L_ot's gradient reaches E_sim only —
           this is "the JDOT objective finds a better map by shaping E_sim").
  E_real : L_real only. z_real is ATTACHED here (a second, separate use of E_real's
           output from the one used to build the cost matrix) — matching itself
           (argmin over a detached cost) contributes no gradient, so L_real's gradient
           can't leak back into E_sim through the match — "the flow objective shapes
           E_real, and only E_real."

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
import sys
from datetime import datetime
from pathlib import Path

import torch

from dataset import PARAM_KEYS_INFER, load_stats, load_manifest
from models import LipschitzReducedAutoencoderEncoder
from train_joint import (
    git_hash, Tee, load_sim_data, load_real_beats, build_flow_net, mixup_real,
)

N_PARAMS_INFER = len(PARAM_KEYS_INFER)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─── JDOT (vanilla: hard nearest-neighbor matching, no entropic regularization) ────────

def jdot_step(E_sim, E_real, flow_sim, flow_real, x_sim_b, theta_sim_b, real_beats_b,
             lam_feat, lam_label, lam_ot, lam_real, ot_label_samples):
    """
    One combined JDOT step. Returns (loss, info_dict). See module docstring for the
    gradient-routing rationale behind each detach.

    Matching is plain hard nearest-neighbor: each real point in the batch independently
    takes the closest sim point (by cost) as its pseudo-label -- no Sinkhorn/entropic
    regularization, no LP-based bipartite assignment. sim and real batches are both
    --batch-size, so this is symmetric (B, B) each step. Batches are reshuffled every
    step (real side via fresh Mixup draws, sim side via the usual permutation), so a
    given real patient sees different nearest-sim matches across steps/epochs -- that
    variability is the mechanism that keeps this from just memorizing one fixed match.
    """
    z_sim = E_sim(x_sim_b)                              # attached
    L_sim = -flow_sim.log_prob(theta_sim_b, condition=z_sim).mean()

    z_real_full     = E_real(real_beats_b)                # attached — used later for L_real
    z_real_detached = z_real_full.detach()                # detached — used for the OT cost only

    # Cheap proxy for "what does flow_real currently believe at z_real_j" — amortizes to
    # one batched sample call over all real points, not one flow evaluation per (sim,real)
    # pair. Detached: this is a fixed reference point, not a differentiable path.
    with torch.no_grad():
        samples    = flow_real.sample((ot_label_samples,), condition=z_real_detached)  # (S, B_real, theta_dim)
        real_guess = samples.mean(dim=0)                                                # (B_real, theta_dim)

    feat_dist  = torch.cdist(z_sim, z_real_detached, p=2) ** 2        # (B_sim, B_real)
    label_dist = torch.cdist(theta_sim_b, real_guess, p=2) ** 2       # (B_sim, B_real)
    C = lam_feat * feat_dist + lam_label * label_dist

    # C's raw magnitude is arbitrary (depends on ||z||^2/||theta||^2 scale, which itself
    # drifts as E_sim trains) and was blowing out L_ot to 10^4-10^5 against L_sim's ~50,
    # swamping L_sim's gradient direction on E_sim even after grad-norm clipping (clipping
    # only rescales magnitude, not direction). Rescale so lam_ot's effective weight stays
    # stable regardless of that drift -- same fix the earlier Sinkhorn version needed, for
    # the same reason; unrelated to Sinkhorn itself. argmin's result is scale-invariant so
    # this doesn't change which pair gets matched, only L_ot's reported/used magnitude.
    C_scale = C.mean().detach().clamp(min=1e-6)
    C_scaled = C / C_scale

    nn_idx = C_scaled.argmin(dim=0)                                          # (B_real,) -- nearest sim per real
    n_real = real_beats_b.shape[0]
    L_ot   = C_scaled[nn_idx, torch.arange(n_real, device=C.device)].mean()  # gradient -> z_sim only (z_real detached above)

    theta_matched = theta_sim_b[nn_idx]                                 # (B_real, theta_dim)
    logp   = flow_real.log_prob(theta_matched, condition=z_real_full)   # z_real_full attached -> gradient -> E_real only
    L_real = (-logp).mean()

    loss = L_sim + lam_ot * L_ot + lam_real * L_real
    return loss, {"L_sim": L_sim.item(), "L_ot": L_ot.item(), "L_real": L_real.item()}


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
    parser.add_argument("--outputs-root", default="/home/sa4604/outputs/cv-dann-sbi",
                        help="Absolute root for run_dir -- train_joint.py's parse_run() uses a "
                             "CWD-relative Path('outputs')/run, which silently writes into "
                             "~/projects/cv-dann-sbi/outputs/ instead of ~/outputs/cv-dann-sbi/ "
                             "when launched from the code checkout (same bug already found and "
                             "fixed in cv-sbi-spin/cv-spin-latent; not yet fixed in train_joint.py "
                             "itself). This script doesn't use parse_run for that reason.")
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
    parser.add_argument("--ot-label-samples", type=int,   default=8,
                        help="Samples drawn per real patient for the cheap real_guess proxy")
    # Mixup (real side) -- each step draws a fresh batch of Mixup-interpolated real points
    # (size = --mixup-n, default matches --batch-size so sim/real batches are symmetric),
    # rather than reusing the same static 802-patient set verbatim every step. mixup-n=0
    # falls back to that original static-802-every-step behavior.
    parser.add_argument("--mixup-n",     type=int,   default=None,
                        help="Real points drawn via mixup per step (default: --batch-size, "
                             "matching sim batch size 1:1). 0 disables mixup entirely.")
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="Beta(alpha, alpha) mixing coefficient -- small alpha biases interpolates near one endpoint")
    # Optimization
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int,   default=512)
    parser.add_argument("--log-every",  type=int,   default=5)
    args = parser.parse_args()
    if args.mixup_n is None:
        args.mixup_n = args.batch_size

    if not (args.run.startswith("exp-") or args.run.startswith("dry-")):
        raise ValueError("--run must start with 'exp-' or 'dry-'")
    run_type = "dry" if args.run.startswith("dry-") else "exp"
    run_dir  = Path(args.outputs_root) / args.run
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
        f"lam_real={args.lam_real}  label_samples={args.ot_label_samples}  (hard nearest-neighbor matching)")
    log(f"Mixup: n={args.mixup_n}  alpha={args.mixup_alpha}")

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
                  lam_real=args.lam_real, ot_label_samples=args.ot_label_samples,
                  matching="hard-nearest-neighbor", mixup_n=args.mixup_n, mixup_alpha=args.mixup_alpha),
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
    csv_writer.writerow(["epoch", "phase", "L_sim", "L_ot", "L_real", "total", "lam_real"])

    log("Training...")
    for epoch in range(1, args.max_epochs + 1):
        joint = epoch > args.sim_warmup
        lam_real_ep = 0.0
        if joint:
            t = min(1.0, (epoch - args.sim_warmup) / max(1, args.real_ramp))
            lam_real_ep = args.lam_real * t

        E_sim.train(); flow_sim.train(); E_real.train(); flow_real.train()

        perm = torch.randperm(n_total)
        sums = dict.fromkeys(["L_sim", "L_ot", "L_real", "total"], 0.0)
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
                info = {"L_sim": L_sim.item(), "L_ot": 0.0, "L_real": 0.0}
            else:
                opt_real.zero_grad()
                # Fresh Mixup-drawn real batch every step, sized to match the sim batch
                # (args.mixup_n defaults to args.batch_size) -- symmetric (B, B) cost matrix,
                # and a real patient's nearest-sim match varies step to step since both the
                # sim permutation and the mixup draw are randomized independently.
                if args.mixup_n > 0:
                    real_batch = mixup_real(real_beats, args.mixup_n, args.mixup_alpha, DEVICE)
                else:
                    real_batch = real_beats
                loss, info = jdot_step(
                    E_sim, E_real, flow_sim, flow_real, x_sim_b, theta_b, real_batch,
                    args.lam_feat, args.lam_label, args.lam_ot, lam_real_ep, args.ot_label_samples,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(E_sim.parameters()) + list(flow_sim.parameters()), 1.0)
                torch.nn.utils.clip_grad_norm_(list(E_real.parameters()) + list(flow_real.parameters()), 1.0)
                opt_sim.step(); opt_real.step()

            total = info["L_sim"] + args.lam_ot * info["L_ot"] + lam_real_ep * info["L_real"]
            sums["L_sim"] += info["L_sim"]; sums["L_ot"] += info["L_ot"]; sums["L_real"] += info["L_real"]
            sums["total"] += total
            n_batches += 1

        avg = {k: v / n_batches for k, v in sums.items()}
        phase_name = "sim-warmup" if not joint else "joint"

        csv_writer.writerow([epoch, phase_name, f"{avg['L_sim']:.5f}", f"{avg['L_ot']:.5f}",
                            f"{avg['L_real']:.5f}", f"{avg['total']:.5f}", f"{lam_real_ep:.4f}"])
        csv_fh.flush()

        if epoch % args.log_every == 0 or epoch == 1:
            log(f"  ep {epoch:4d}/{args.max_epochs}  [{phase_name}]"
                f"  L_sim={avg['L_sim']:.4f}  L_ot={avg['L_ot']:.4f}  L_real={avg['L_real']:.4f}"
                f"  lam_real={lam_real_ep:.3f}")

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
