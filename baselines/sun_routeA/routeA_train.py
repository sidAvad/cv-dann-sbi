"""Score-net training for Route A, with checkpointing and resume.

Upstream (`vendor/RouteA_generate_pairs.py`) trains the score net inline with no resume:
it writes `model.state_dict()` only on validation improvement and only **after epoch
500**, keeps the early stopper's best weights in RAM, and saves no optimizer, scheduler,
epoch counter or RNG state. `load_model = True` is reload-for-inference, not resume. That
is fine at their 10k-pair scale (minutes); at 980k pairs this is a multi-hour run whose
first ~third would write nothing.

This module owns the loop so it can snapshot, and changes nothing about the method:

* model is vendored `ScoreNet_sigma`, loss is `routeA_noise.score_matching_loss`
  (upstream's, with the hr_z mask -- see REPRESENTATION.md section 4)
* AdamW(lr=1e-3, weight_decay=1e-5), StepLR(step_size=10, gamma=0.95), grad-clip 1.0,
  batch 1024, 90/10 split, T=10 sigma schedule over [0.01, 1.0]
* `EarlyStopping(patience=100, min_delta=1e-5, restore_best_weights=True)` and the
  best-model write stay **armed only after `--es-warmup` (default 500)**, exactly as
  upstream. Periodic snapshots are orthogonal to model selection: they exist so a crash
  does not cost hours, and they never change which model is chosen.

Validation follows upstream and redraws `t_idx` every epoch, so the signal is noisy.
`--val-seed` fixes the realisation instead (what our Gibbs trainer does), which makes
selection cleaner but is a deviation; it is off by default.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))

import routeA_noise as RN                                      # noqa: E402
from NSEutils import ScoreNet_sigma, EarlyStopping             # noqa: E402 (vendored)
from RouteA_utils import make_gamma_schedule                   # noqa: E402 (vendored)


def rng_state():
    return {"torch": torch.get_rng_state(),
            "torch_cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
            "numpy": np.random.get_state(),
            "python": random.getstate()}


def set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["torch_cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def atomic_save(obj, path):
    """Write to a sibling tmp file then rename, so a crash mid-write cannot corrupt the
    only snapshot we have."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def split_indices(n, val_fraction, seed):
    """Deterministic 90/10 split. Upstream uses random_split off the global RNG; keying it
    to an explicit seed is what lets a resume reproduce the same split."""
    g = np.random.RandomState(seed)
    perm = g.permutation(n)
    n_val = int(round(n * val_fraction))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def run_epoch(model, data, idx, gammas, theta_dim, clean_x_dims, batch, opt, device,
              train=True, fixed_t=None):
    model.train() if train else model.eval()
    T = len(gammas)
    tot, nb = 0.0, 0
    order = (torch.randperm(len(idx), device=device) if train
             else torch.arange(len(idx), device=device))
    with torch.set_grad_enabled(train):
        for i in range(0, len(idx) - (batch - 1 if train else 0), batch):
            sel = idx[order[i:i + batch]]
            z = data[sel]
            if fixed_t is None:
                t_idx = torch.randint(0, T, (z.shape[0],), device=device)
            else:
                t_idx = fixed_t[i:i + z.shape[0]]
            loss = RN.score_matching_loss(model, z[:, :theta_dim], z[:, theta_dim:],
                                          t_idx, gammas, clean_x_dims=clean_x_dims)
            if train:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += float(loss.detach())
            nb += 1
    return tot / max(nb, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="dir holding base.pt from routeA_data.py")
    p.add_argument("--run", required=True)
    p.add_argument("--outputs-root", default="/home/sa4604/outputs/cv-dann-sbi")
    p.add_argument("--resume-dir", default=None,
                   help="checkpoints/<ts> of an interrupted run; restores model, optimizer, "
                        "scheduler, epoch, early-stopper and RNG state")
    # upstream values
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--sched-step", type=int, default=10)
    p.add_argument("--sched-gamma", type=float, default=0.95)
    p.add_argument("--val-fraction", type=float, default=0.10)
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--gamma-min", type=float, default=1e-2)
    p.add_argument("--gamma-max", type=float, default=1.0)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--es-patience", type=int, default=100)
    p.add_argument("--es-min-delta", type=float, default=1e-5)
    p.add_argument("--es-warmup", type=int, default=500,
                   help="upstream arms early stopping and best-model saving only after "
                        "this epoch; snapshots are unaffected")
    # ours
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-seed", type=int, default=None,
                   help="fix the validation noise-level draw (deviation; upstream redraws)")
    p.add_argument("--data-device", default=None, help="default: same as compute device")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)

    run_dir = Path(a.outputs_root) / a.run
    run_dir.mkdir(parents=True, exist_ok=True)
    ts = (Path(a.resume_dir).name if a.resume_dir
          else time.strftime("%Y%m%d-%H%M%S"))
    ckpt_dir = run_dir / "checkpoints" / ts
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"train_log_{ts}.csv"

    def log(msg):
        print(msg, flush=True)

    base = torch.load(Path(a.base) / "base.pt", weights_only=False)
    theta_dim, x_dim = base["theta_dim"], base["x_dim"]
    data = base["pairs"].to(torch.device(a.data_device) if a.data_device else device)
    n = data.shape[0]
    gammas = make_gamma_schedule(T=a.T, gamma_min=a.gamma_min,
                                 gamma_max=a.gamma_max).to(device)
    tr_idx_np, va_idx_np = split_indices(n, a.val_fraction, a.split_seed)
    tr_idx = torch.from_numpy(tr_idx_np).to(data.device)
    va_idx = torch.from_numpy(va_idx_np).to(data.device)
    log(f"pairs {n} ({len(tr_idx)} train / {len(va_idx)} val)  theta {theta_dim}  x {x_dim}")
    log(f"sigma schedule {[round(float(s), 5) for s in gammas]}")
    log(f"clean x dims (never noised): {RN.CLEAN_X_DIMS}")

    model = ScoreNet_sigma(theta_dim=theta_dim, x_dim=x_dim, amortize=True,
                           emb_dim=a.emb_dim, sigma_emb_dim=a.emb_dim,
                           hidden_dim=a.hidden_dim,
                           sigma_min=a.gamma_min, sigma_max=a.gamma_max).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=a.sched_step, gamma=a.sched_gamma)
    stopper = EarlyStopping(patience=a.es_patience, min_delta=a.es_min_delta, mode="min",
                            restore_best_weights=True, verbose=True)

    fixed_t = None
    if a.val_seed is not None:
        g = torch.Generator(device="cpu").manual_seed(a.val_seed)
        fixed_t = torch.randint(0, a.T, (len(va_idx),), generator=g).to(device)

    start_epoch, best_val = 1, float("inf")
    snap_path = ckpt_dir / "snapshot.pt"
    if a.resume_dir:
        s = torch.load(Path(a.resume_dir) / "snapshot.pt", weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        best_val = s["best_val"]; start_epoch = s["epoch"] + 1
        stopper.best_score = s["es"]["best_score"]
        stopper.counter = s["es"]["counter"]
        stopper.should_stop = s["es"]["should_stop"]
        stopper.best_state = s["es"]["best_state"]
        set_rng_state(s["rng"])
        log(f"resumed from {a.resume_dir} at epoch {start_epoch} (best_val {best_val:.6f})")
    else:
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,val_loss,lr,secs\n")
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump({**vars(a), "n_pairs": n, "theta_dim": theta_dim, "x_dim": x_dim,
                       "clean_x_dims": list(RN.CLEAN_X_DIMS),
                       "sigma_schedule": [float(s) for s in gammas]}, f, indent=2)

    log(f"training epochs {start_epoch}-{a.epochs}, "
        f"{len(tr_idx)//a.batch_size} batches/epoch, ckpt every {a.ckpt_every}")
    for epoch in range(start_epoch, a.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, data, tr_idx, gammas, theta_dim, RN.CLEAN_X_DIMS,
                       a.batch_size, opt, device, train=True)
        sched.step()
        va = run_epoch(model, data, va_idx, gammas, theta_dim, RN.CLEAN_X_DIMS,
                       a.batch_size, opt, device, train=False, fixed_t=fixed_t)
        dt = time.time() - t0
        with open(log_path, "a") as f:
            f.write(f"{epoch},{tr:.6f},{va:.6f},{opt.param_groups[0]['lr']:.3e},{dt:.1f}\n")
        if epoch % max(a.epochs // 50, 1) == 0 or epoch == start_epoch:
            log(f"ep {epoch:>5d}  train {tr:.6f}  val {va:.6f}  ({dt:.0f}s)")

        stop = False
        if epoch > a.es_warmup:                      # upstream gating, unchanged
            if va < best_val - 1e-8:
                best_val = float(va)
                atomic_save(model.state_dict(), ckpt_dir / "score_net_best.pt")
            if stopper.step(float(va), model):
                log(f"[EarlyStopping] stop at epoch {epoch}; best val {stopper.best_score:.6f}")
                stop = True

        if epoch % a.ckpt_every == 0 or stop or epoch == a.epochs:
            atomic_save({"epoch": epoch, "model": model.state_dict(),
                         "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "best_val": best_val,
                         "es": {"best_score": stopper.best_score,
                                "counter": stopper.counter,
                                "should_stop": stopper.should_stop,
                                "best_state": stopper.best_state},
                         "rng": rng_state()}, snap_path)
        if stop:
            break

    # upstream's EarlyStopping restores best weights into `model` on stop; if it never
    # armed (short run), score_net_best.pt may not exist, so always leave a final artifact.
    atomic_save(model.state_dict(), ckpt_dir / "score_net_final.pt")
    log(f"done; artifacts in {ckpt_dir}")


if __name__ == "__main__":
    main()
