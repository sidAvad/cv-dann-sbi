"""Phase-2 gate: does the DSM recover a known closed-form score?

A linear-Gaussian toy where every quantity is analytic:

    theta ~ N(mu, S_th),    x | theta ~ N(A theta + b, S_x)

so (theta, x) is jointly Gaussian with mean m and covariance S, and the sigma-noised
joint used by denoising score matching is exactly

    p_sigma(z) = N(m, S + sigma^2 D),    D = diag(1_theta, mask_x)

giving the closed-form score  grad log p_sigma(z) = -(S + sigma^2 D)^-1 (z - m).

D carries our hr_z mask (0 on dims never noised), so the masked loss in routeA_noise is
testable against an analytic target too, not just the unmasked one.

Checks A/B are the gate: our code must reproduce the analytic score. Checks C/D then feed
the *analytic* score to the sampler, which isolates whether upstream's annealed Langevin
reaches the tempered target p^beta -- for a Gaussian, N(m, S/beta) -- independently of any
score-network error.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))

import routeA_noise as RN                                        # noqa: E402
from NSEutils import ScoreNet_sigma                              # noqa: E402 (vendored)
from RouteA_utils import make_gamma_schedule                     # noqa: E402 (vendored)

_fails = []


def check(name, cond, detail=""):
    print(f"[{'  ok  ' if cond else ' FAIL '}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


class Toy:
    """Linear-Gaussian joint with analytic noised score."""

    def __init__(self, theta_dim=2, x_dim=5, clean_x_dims=(), seed=0, device="cpu"):
        g = torch.Generator().manual_seed(seed)
        self.theta_dim, self.x_dim = theta_dim, x_dim
        self.device = device

        mu = torch.randn(theta_dim, generator=g)
        Lth = torch.randn(theta_dim, theta_dim, generator=g) / np.sqrt(theta_dim)
        S_th = Lth @ Lth.T + 0.5 * torch.eye(theta_dim)
        A = torch.randn(x_dim, theta_dim, generator=g) / np.sqrt(theta_dim)
        b = torch.randn(x_dim, generator=g)
        Lx = torch.randn(x_dim, x_dim, generator=g) / np.sqrt(x_dim)
        S_x = Lx @ Lx.T + 0.5 * torch.eye(x_dim)

        self.m = torch.cat([mu, A @ mu + b]).to(device)
        top = torch.cat([S_th, S_th @ A.T], dim=1)
        bot = torch.cat([A @ S_th, A @ S_th @ A.T + S_x], dim=1)
        self.S = torch.cat([top, bot], dim=0).to(device)
        self.L = torch.linalg.cholesky(self.S)

        d = torch.ones(theta_dim + x_dim, device=device)
        for j in clean_x_dims:
            d[theta_dim + j] = 0.0
        self.D = torch.diag(d)
        self.clean_x_dims = tuple(clean_x_dims)

    def sample(self, n, seed=1):
        g = torch.Generator(device="cpu").manual_seed(seed)
        e = torch.randn(n, self.m.numel(), generator=g).to(self.device)
        return self.m + e @ self.L.T

    def score(self, z, sigma):
        P = torch.linalg.inv(self.S + float(sigma) ** 2 * self.D)
        return -(z - self.m) @ P.T

    def tempered_cov(self, beta, sigma=0.0):
        return (self.S + float(sigma) ** 2 * self.D) / beta


class AnalyticScore(torch.nn.Module):
    """Exact score with ScoreNet_sigma's call signature, for the sampler checks."""

    def __init__(self, toy):
        super().__init__()
        self.toy = toy
        self.p = torch.nn.Parameter(torch.zeros(1))

    def forward(self, theta, x, sigma):
        z = torch.cat([theta, x], dim=-1)
        return self.toy.score(z, float(sigma.reshape(-1)[0]))


def train_score_net(toy, gammas, n, epochs, batch, lr, device, clean_x_dims, log):
    net = ScoreNet_sigma(theta_dim=toy.theta_dim, x_dim=toy.x_dim, amortize=True,
                         emb_dim=128, sigma_emb_dim=128, hidden_dim=128,
                         sigma_min=float(gammas.min()), sigma_max=float(gammas.max())).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.95)  # upstream

    data = toy.sample(n, seed=1).to(device)
    val = toy.sample(n // 10, seed=2).to(device)
    T = len(gammas)
    for ep in range(epochs):
        perm = torch.randperm(data.shape[0], device=device)
        tot = 0.0
        for i in range(0, data.shape[0] - batch + 1, batch):
            z = data[perm[i:i + batch]]
            t_idx = torch.randint(0, T, (z.shape[0],), device=device)
            loss = RN.score_matching_loss(net, z[:, :toy.theta_dim], z[:, toy.theta_dim:],
                                          t_idx, gammas, clean_x_dims=clean_x_dims)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss)
        sched.step()
        if (ep + 1) % max(epochs // 5, 1) == 0:
            with torch.no_grad():
                t_idx = torch.randint(0, T, (val.shape[0],), device=device)
                vl = float(RN.score_matching_loss(net, val[:, :toy.theta_dim],
                                                  val[:, toy.theta_dim:], t_idx, gammas,
                                                  clean_x_dims=clean_x_dims))
            log(f"    ep {ep+1:>4d}  train {tot/max(1,(data.shape[0]//batch)):.5f}  val {vl:.5f}")
    return net


def score_errors(net, toy, gammas, device, n=4096):
    """Relative L2 error of the learned score vs analytic, per sigma, split by block."""
    z = toy.sample(n, seed=3).to(device)
    rows = []
    with torch.no_grad():
        for sigma in gammas:
            eps = torch.randn_like(z) * torch.diag(toy.D)
            zt = z + float(sigma) * eps
            sig = torch.full((n, 1), float(sigma), device=device)
            got = net(zt[:, :toy.theta_dim], zt[:, toy.theta_dim:], sig)
            want = toy.score(zt, sigma)
            td = toy.theta_dim
            noisy_x = [j for j in range(toy.x_dim) if j not in toy.clean_x_dims]
            def rel(gi, wi):
                return float((gi - wi).norm() / wi.norm().clamp_min(1e-12))
            rows.append({
                "sigma": float(sigma),
                "theta": rel(got[:, :td], want[:, :td]),
                "x_noisy": rel(got[:, td:][:, noisy_x], want[:, td:][:, noisy_x]),
                "x_clean": (None if not toy.clean_x_dims else
                            (float(got[:, td:][:, list(toy.clean_x_dims)].abs().mean()),
                             float(want[:, td:][:, list(toy.clean_x_dims)].abs().mean()))),
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--theta-dim", type=int, default=2)
    p.add_argument("--x-dim", type=int, default=5)
    p.add_argument("--n", type=int, default=200_000)
    # Upstream trains the score net for 1500 epochs with EarlyStopping(patience=100)
    # armed only after epoch 500. That budget is for a real simulator; this toy's score is
    # exactly linear in z and converges in far fewer, so the epoch count is the single
    # knob dialled down for a fast gate. Everything else matches upstream.
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=1024)   # upstream
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tol", type=float, default=0.10,
                   help="max relative L2 score error allowed at any sigma")
    p.add_argument("--n-chains", type=int, default=20_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()

    dev = torch.device(a.device)
    gammas = make_gamma_schedule(T=10, gamma_min=1e-2, gamma_max=1.0).to(dev)
    print(f"device {dev}   sigma schedule {[round(float(s),4) for s in gammas]}\n")

    # ---------- A: unmasked DSM reproduces the analytic score -----------------
    print("A. unmasked DSM (identical to upstream's loss: mask is all ones)")
    toy = Toy(a.theta_dim, a.x_dim, clean_x_dims=(), device=dev)
    net = train_score_net(toy, gammas, a.n, a.epochs, a.batch, a.lr, dev, (), print)
    rows = score_errors(net, toy, gammas, dev)
    print(f"    {'sigma':>8s} {'rel err theta':>14s} {'rel err x':>12s}")
    for r in rows:
        print(f"    {r['sigma']:>8.4f} {r['theta']:>14.4f} {r['x_noisy']:>12.4f}")
    worst = max(max(r["theta"], r["x_noisy"]) for r in rows)
    check("unmasked DSM recovers the closed-form score at every sigma",
          worst < a.tol, f"worst rel L2 = {worst:.4f} (tol {a.tol})")

    # ---------- B: masked DSM ------------------------------------------------
    print("\nB. masked DSM (hr_z analogue: last x dim never noised)")
    clean = (a.x_dim - 1,)
    toy_m = Toy(a.theta_dim, a.x_dim, clean_x_dims=clean, device=dev)
    net_m = train_score_net(toy_m, gammas, a.n, a.epochs, a.batch, a.lr, dev, clean, print)
    rows_m = score_errors(net_m, toy_m, gammas, dev)
    print(f"    {'sigma':>8s} {'rel err theta':>14s} {'rel err x_noisy':>16s} "
          f"{'|s| clean (got/analytic)':>26s}")
    for r in rows_m:
        g_, w_ = r["x_clean"]
        print(f"    {r['sigma']:>8.4f} {r['theta']:>14.4f} {r['x_noisy']:>16.4f} "
              f"{g_:>12.4f} /{w_:>11.4f}")
    worst_m = max(max(r["theta"], r["x_noisy"]) for r in rows_m)
    check("masked DSM recovers the closed-form score on the noised dims",
          worst_m < a.tol, f"worst rel L2 = {worst_m:.4f} (tol {a.tol})")
    # The clean dim's target is 0 by construction, while the true marginal score there is
    # nonzero because S couples it to the noised dims. Learning 0 is correct for our use:
    # the sampler holds that coordinate fixed, so its drift is never read.
    got_c = np.mean([r["x_clean"][0] for r in rows_m])
    want_c = np.mean([r["x_clean"][1] for r in rows_m])
    check("clean dim's learned score is driven to ~0, as the masked target implies",
          got_c < 0.1 * want_c,
          f"mean |s_learned| {got_c:.4f} vs |s_analytic| {want_c:.4f}")

    # ---------- C/D: does upstream's sampler reach p^beta? -------------------
    print("\nC/D. upstream annealed Langevin fed the ANALYTIC score (no net error)")
    toy_s = Toy(a.theta_dim, a.x_dim, clean_x_dims=(), device=dev)
    exact = AnalyticScore(toy_s).to(dev)
    init = toy_s.sample(a.n_chains, seed=7).to(dev)
    sig_min = float(gammas.min())
    print(f"    {'beta':>5s} {'cov ratio theta':>16s} {'cov ratio x':>12s} "
          f"{'mean |dtheta|':>14s} {'mean |dx|':>10s} {'clampbound th/x':>16s}")
    for beta in (0.5, 1.0, 2.0):
        out, diags = RN.Lgmcmc(gammas, exact, toy_s.theta_dim, toy_s.x_dim, init,
                               t_power=beta, device=dev, clean_x_dims=())
        tgt = toy_s.tempered_cov(beta, sig_min)
        emp = torch.cov(out.T)
        td = toy_s.theta_dim
        rt = float(emp[:td, :td].diagonal().mean() / tgt[:td, :td].diagonal().mean())
        rx = float(emp[td:, td:].diagonal().mean() / tgt[td:, td:].diagonal().mean())
        dth = float((out[:, :td] - init[:, :td]).norm(dim=1).mean())
        dx = float((out[:, td:] - init[:, td:]).norm(dim=1).mean())
        cb_t = np.mean([d["frac_steps_clamp_bound_theta"] for d in diags])
        cb_x = np.mean([d["frac_steps_clamp_bound_x"] for d in diags])
        print(f"    {beta:>5.1f} {rt:>16.3f} {rx:>12.3f} {dth:>14.4f} {dx:>10.4f} "
              f"{cb_t:>7.2f}/{cb_x:>7.2f}")
        if beta == 1.0:
            check("sampler preserves the beta=1 joint it was initialised from",
                  0.8 < rt < 1.25 and 0.8 < rx < 1.25,
                  f"theta cov ratio {rt:.3f}, x cov ratio {rx:.3f}")

    print()
    if _fails:
        print(f"PHASE 2 GATE FAILED: {_fails}")
        sys.exit(1)
    print("phase-2 gate passed (A/B); see C/D table for sampler behaviour")


if __name__ == "__main__":
    main()
