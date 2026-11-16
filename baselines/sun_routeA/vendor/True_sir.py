from pathlib import Path
import sbibm
from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import plot_posteriors_compare,to_float


import math, torch
from typing import Optional, Union

Tensor = torch.Tensor

# ----------------------------
# SIR RHS
# ----------------------------
def _sir_rhs(y: Tensor, beta: Tensor, gamma: Tensor, N: float) -> Tensor:
    S, I, R = y[:, 0], y[:, 1], y[:, 2]
    dS = -beta * S * I / N
    dI =  beta * S * I / N - gamma * I
    dR =  gamma * I
    return torch.stack([dS, dI, dR], dim=1)

# ----------------------------
# RK4 (records t=0 as well)
# ----------------------------
@torch.no_grad()
def rk4_I_hit_targets(beta: torch.Tensor, gamma: torch.Tensor, N: float,
                      t_eval: torch.Tensor, dt_max: float, dtype: torch.dtype) -> torch.Tensor:
    """
    beta,gamma: (B,)
    t_eval: (T,) 递增，可为任意实数（0..160，含 0）
    dt_max: 最大步长；每个子区间最后一步会缩到正好命中 t_eval
    return: I(t_eval)  shape (B,T)
    """
    dev = beta.device
    B, T = beta.shape[0], t_eval.shape[0]
    y = torch.stack([
        torch.full((B,), N-1.0, dtype=dtype, device=dev),
        torch.full((B,), 1.0,   dtype=dtype, device=dev),
        torch.full((B,), 0.0,   dtype=dtype, device=dev),
    ], dim=1)

    I_out = torch.empty((B, T), dtype=dtype, device=dev)
    t_cur = torch.tensor(0.0, dtype=dtype, device=dev)

    # 若第一个观测就是 t=0，先记录
    if torch.isclose(t_eval[0], torch.tensor(0.0, dtype=dtype, device=dev)):
        I_out[:, 0] = y[:, 1]
        start_idx = 1
    else:
        start_idx = 0

    for j in range(start_idx, T):
        t_target = t_eval[j]
        while t_cur < t_target - 1e-15:
            h = torch.clamp(t_target - t_cur, max=dt_max).item()  # 最后一步自动缩短
            k1 = _sir_rhs(y, beta, gamma, N)
            k2 = _sir_rhs(y + 0.5*h*k1, beta, gamma, N)
            k3 = _sir_rhs(y + 0.5*h*k2, beta, gamma, N)
            k4 = _sir_rhs(y +     h*k3, beta, gamma, N)
            y = y + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t_cur = t_cur + h
        I_out[:, j] = y[:, 1]
    return I_out

# ----------------------------
# Binomial log-likelihood (no constants, faster)
# x_counts must be counts in [0, n]; times are 0..160 (10 points)
# ----------------------------
def sir_loglik_binomial_counts_batch(theta, x_counts, *, N=1_000_000.0, n=1000,
                                     t_eval=None, dt=0.1, eps=1e-12, dtype=torch.float64):
    if theta.ndim == 1: theta = theta.view(1,2)
    dev = theta.device
    if t_eval is None:
        t_eval = torch.linspace(0.0, 160.0, 10, dtype=dtype, device=dev)  # ★ 含 0
    else:
        t_eval = t_eval.to(dev, dtype)
    x = torch.as_tensor(x_counts, dtype=dtype, device=dev).flatten()

    beta, gamma = theta[:,0].to(dtype), theta[:,1].to(dtype)
    # ★ 命中目标时刻的积分（dt 作为最大步长）
    I = rk4_I_hit_targets(beta, gamma, N, t_eval, dt_max=dt, dtype=dtype)   # (B,T)
    p = (I / N).clamp(eps, 1-eps)
    # 非常数版对数似然（对 MCMC 足够）
    return (x.view(1,-1)*p.log() + (n-x.view(1,-1))*(1-p).log()).sum(dim=1)

@torch.no_grad()
def pcn_power_posterior_samples(
    x_counts: Tensor,
    t: float,
    mu: Tensor, sigma: Tensor,                  # prior N(mu, diag(sigma^2)) in u-space
    n_samples: int,
    *, burnin: int = 1000, thin: int = 1,
    step_beta: float = 0.3,                     # 0<beta<1
    n_chains: int = 64,                         # parallel chains
    device: str = "cuda", dtype: torch.dtype = torch.float64,
    dt_ode: float = 0.1,
    seed: Optional[int] = None,
    progress: bool = True,
) -> Tensor:
    assert t >= 0.0
    dev = torch.device(device)
    mu    = mu.to(dev, dtype)
    sigma = sigma.to(dev, dtype)
    x_counts = torch.as_tensor(x_counts, dtype=dtype, device=dev).flatten()

    # RNG
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)

    # 初始：u = mu + sigma ⊙ z, z~N(0,I)
    z = torch.randn(n_chains, 2, generator=g, device=dev, dtype=dtype)
    u = mu.unsqueeze(0) + sigma.unsqueeze(0) * z
    theta = torch.exp(u)
    ll = sir_loglik_binomial_counts_batch(theta, x_counts, dt=dt_ode, dtype=dtype)  # (C,)

    def one_step(u: Tensor, z: Tensor, ll: Tensor):
        # pCN: z' = sqrt(1-b^2) z + b * eta
        eta    = torch.randn_like(z)
        z_prop = math.sqrt(1.0 - step_beta**2) * z + step_beta * eta
        u_prop = mu.unsqueeze(0) + sigma.unsqueeze(0) * z_prop
        th_prop = torch.exp(u_prop)
        ll_prop = sir_loglik_binomial_counts_batch(th_prop, x_counts, dt=dt_ode, dtype=dtype)
        # 先验被 pCN 保持不变 → 接受率只看似然
        loga   = t * (ll_prop - ll)
        accept = (torch.log(torch.rand_like(loga)) < loga)
        # 返回新状态（不做原地写）
        z_new  = torch.where(accept.unsqueeze(1), z_prop, z)
        u_new  = torch.where(accept.unsqueeze(1), u_prop, u)
        ll_new = torch.where(accept, ll_prop, ll)
        return u_new, z_new, ll_new, int(accept.sum().item())

    # burn-in
    accepts = 0
    total_steps = 0
    for _ in range(burnin):
        u, z, ll, nacc = one_step(u, z, ll)
        accepts += nacc; total_steps += n_chains

    # 采样（按样本数计数；每步收集 n_chains 个）
    samples = torch.empty(n_samples, 2, device=dev, dtype=dtype)
    pos = 0
    while pos < n_samples:
        for _ in range(thin):
            u, z, ll, nacc = one_step(u, z, ll)
            accepts += nacc; total_steps += n_chains
        batch = torch.exp(u)  # (n_chains, 2)
        k = min(n_chains, n_samples - pos)
        samples[pos:pos+k] = batch[:k]
        pos += k
        if progress and pos % (n_chains*10) == 0:
            print(f"  collected {pos}/{n_samples}")

    acc_rate = accepts / total_steps
    if progress:
        print(f"[pCN] chains={n_chains}, beta={step_beta}, accept rate={acc_rate:.3f}")
    return samples

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    task = sbibm.get_task("sir")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observation = task.get_observation(num_observation=1).to(device)
    reference_samples = task.get_reference_posterior_samples(num_observation=1).to(device)
    save_dir = Path("sir")

    # t_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
    t_list = [1.0]
    for t in t_list:
        mu = torch.tensor([math.log(0.4), math.log(1 / 8)], dtype=torch.float64, device=device)
        sigma = torch.tensor([0.5, 0.2], dtype=torch.float64, device=device)

        samples = pcn_power_posterior_samples(
            x_counts=observation, t=t,
            mu=mu, sigma=sigma,
            n_samples=10000, burnin=500, thin=2,
            step_beta=0.3, n_chains=128,
            device=device, dtype=torch.float64, dt_ode=0.1,
        )


        samples = samples.to(device)
        c2st_accuracy = to_float(c2st(reference_samples, samples))
        mmd_value = to_float(mmd(reference_samples, samples))
        print(f"c2st:{c2st_accuracy:.4f}")
        print(f"MMD: {mmd_value:.4g}")
        print(samples.shape)
        print(reference_samples.shape)
        plot_posteriors_compare(reference_samples, posterior_samples=samples,
                                title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={t}")


