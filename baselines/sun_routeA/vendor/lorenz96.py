import numpy as np
from dataclasses import dataclass
from typing import List, Optional,Dict,Any
import torch
import math
from pathlib import Path
import matplotlib.pyplot as plt

Array = np.ndarray

def _l96_core_torch(x: torch.Tensor) -> torch.Tensor:
    x_plus_1 = torch.roll(x, shifts=-1, dims=-1)
    x_minus_2 = torch.roll(x, shifts=2, dims=-1)
    x_minus_1 = torch.roll(x, shifts=1, dims=-1)

    # (x_{k+1} - x_{k-2}) * x_{k-1} - x_k + 10.0
    return (x_plus_1 - x_minus_2) * x_minus_1 - x + 10.0


@dataclass
class Theta:
    b0: float
    b1: float
    sigma_e: float

def _step_mean(x: torch.Tensor, th: Theta, dt: float) -> torch.Tensor:
    return x + dt * (_l96_core_torch(x) - (th.b0 + th.b1 * x))

def simulate_path(
        th: Theta, K: int = 8, T: int = 20, dt: float = 3.0 / 40.0,
        x0: Optional[torch.Tensor] = None,
        device = "cuda",
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:

    if x0 is None:
        x = torch.ones(K, device=device, dtype=dtype)
    else:
        x = torch.tensor(x0, device=device, dtype=dtype).clone()

    outputs = [x]

    for _ in range(T):
        mean = _step_mean(x, th, dt)
        noise = torch.randn(K, device=x.device, dtype=x.dtype)
        x = mean - dt * th.sigma_e * noise
        outputs.append(x)

    return torch.stack(outputs, dim=0)


def simulate_dataset(
        th: Theta, n_series: int = 10, K: int = 8, T: int = 20, dt: float = 3.0 / 40.0,
        x0: Optional[torch.Tensor] = None, seed: int = 0,
        device = "cuda",
        dtype: torch.dtype = torch.float32
) -> torch.Tensor:

    if seed is not None:
        torch.manual_seed(seed)

    paths = [
        simulate_path(th, K, T, dt, x0, device, dtype)
        for _ in range(n_series)
    ]

    return torch.stack(paths, dim=0)
# -----------
#   Prior
# -----------
@dataclass
class PriorBox:
    b0_low: float = 1.4
    b0_high: float = 2.2
    b1_low: float = 0.1
    b1_high: float = 1.0
    se_low: float = 1.5
    se_high: float = 2.5

    def sample(
            self,
            M: int,
            generator: Optional[torch.Generator] = None,
            device: Optional[torch.device] = None,
            dtype: torch.dtype = torch.float32
    ) -> torch.Tensor:

        rand_samples = torch.rand(
            M, 3,
            generator=generator,
            device=device,
            dtype=dtype
        )

        lows = torch.tensor(
            [self.b0_low, self.b1_low, self.se_low],
            device=device, dtype=dtype
        )
        highs = torch.tensor(
            [self.b0_high, self.b1_high, self.se_high],
            device=device, dtype=dtype
        )

        return rand_samples * (highs - lows) + lows

    def inside(self, thetas: torch.Tensor) -> torch.Tensor:
        b0, b1, se = thetas[:, 0], thetas[:, 1], thetas[:, 2]

        return (
                (self.b0_low <= b0) & (b0 <= self.b0_high) &
                (self.b1_low <= b1) & (b1 <= self.b1_high) &
                (self.se_low <= se) & (se <= self.se_high)
        )

    def logpdf(self, thetas: torch.Tensor) -> torch.Tensor:
        vol = (self.b0_high - self.b0_low) * \
              (self.b1_high - self.b1_low) * \
              (self.se_high - self.se_low)

        out = torch.full(
            (thetas.shape[0],),
            fill_value=float('-inf'),
            device=thetas.device,
            dtype=thetas.dtype
        )

        m = self.inside(thetas)

        if m.any():
            out[m] = -math.log(vol)

        return out

# ----------------
#  Likelihood
# ----------------
def loglik_dataset(
        dataset: torch.Tensor,
        th: Theta,
        dt = 3.0/40.0
) -> torch.Tensor:
    B, T_plus_1, K = dataset.shape
    T = T_plus_1 - 1

    sig2 = (dt * th.sigma_e) ** 2

    log_2pi_sig2 = math.log(2 * math.pi * sig2)
    const = -0.5 * B * K * T * log_2pi_sig2

    x_t = dataset[:, :-1, :]

    x_t_plus_1 = dataset[:, 1:, :]

    means = _step_mean(x_t, th, dt)

    resid = x_t_plus_1 - means

    quad = -0.5 * torch.sum(resid ** 2) / sig2
    return const + quad


def precompute_l96_stats(dataset: torch.Tensor, dt: float, device=None, dtype=torch.float32):
    """
    dataset: (B, T+1, K)
    返回充要统计量，用于 O(1)/粒子的似然计算。
    全部标量以 dtype 计算，放在 device 上。
    """
    if dataset.dim() == 2:
        dataset = dataset.unsqueeze(0)
    B, Tplus1, K = dataset.shape
    T = Tplus1 - 1
    if device is None:
        device = dataset.device
    x_t  = dataset[:, :-1, :].to(device=device, dtype=dtype)   # (B,T,K)
    x_tp1= dataset[:, 1:, :].to(device=device, dtype=dtype)    # (B,T,K)

    # core(x_t)
    x_plus_1 = torch.roll(x_t, shifts=-1, dims=-1)
    x_minus_2 = torch.roll(x_t, shifts=2, dims=-1)
    x_minus_1 = torch.roll(x_t, shifts=1, dims=-1)
    core = (x_plus_1 - x_minus_2) * x_minus_1 - x_t + 10.0

    U = x_tp1 - (x_t + dt * core)   # (B,T,K)
    V = x_t                         # (B,T,K)

    # 用更高精度累加，最后再转回 dtype（数值更稳）
    acc_dtype = torch.float64
    U64 = U.to(acc_dtype); V64 = V.to(acc_dtype)

    S0   = torch.tensor(B * T * K, dtype=acc_dtype, device=device)
    SU   = U64.sum()
    SV   = V64.sum()
    SU2  = (U64 * U64).sum()
    SV2  = (V64 * V64).sum()
    SUV  = (U64 * V64).sum()

    stats = {
        "S0":  S0.to(dtype),
        "SU":  SU.to(dtype),
        "SV":  SV.to(dtype),
        "SU2": SU2.to(dtype),
        "SV2": SV2.to(dtype),
        "SUV": SUV.to(dtype),
        "N":   torch.tensor(B * T * K, dtype=dtype, device=device)  # 同 S0
    }
    # 释放中间量（对 GPU 友好）
    del x_t, x_tp1, x_plus_1, x_minus_2, x_minus_1, core, U, V, U64, V64
    if device == "cuda":
        torch.cuda.empty_cache()
    return stats


@torch.no_grad()
def batch_loglik_from_stats(th_arr: torch.Tensor, dt: float, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    th_arr: (M,3) -> (b0, b1, sigma_e)
    返回 (M,) 的 log-likelihood。
    完全 O(1)/粒子，不做任何 (M,B,T,K) 广播。
    """
    device = th_arr.device
    dtype  = th_arr.dtype

    b0 = th_arr[:, 0]
    b1 = th_arr[:, 1]
    se = th_arr[:, 2]

    a   = dt * b0          # (M,)
    b   = dt * b1          # (M,)
    sig2= (dt * se) ** 2   # (M,)

    S0  = stats["S0"].to(device=device, dtype=dtype)   # 标量
    SU  = stats["SU"].to(device=device, dtype=dtype)
    SV  = stats["SV"].to(device=device, dtype=dtype)
    SU2 = stats["SU2"].to(device=device, dtype=dtype)
    SV2 = stats["SV2"].to(device=device, dtype=dtype)
    SUV = stats["SUV"].to(device=device, dtype=dtype)

    # sum resid^2 for each theta
    resid2 = (
        SU2
        + (a * a) * S0
        + (b * b) * SV2
        + 2.0 * a * SU
        + 2.0 * b * SUV
        + 2.0 * a * b * SV
    )  # (M,)

    const = -0.5 * S0 * torch.log(2.0 * math.pi * sig2)   # (M,)
    quad  = -0.5 * resid2 / sig2                          # (M,)
    return const + quad


def make_prior_fn(prior_box: PriorBox, device="cpu", dtype=torch.float32):
    """
    返回一个函数 prior(M, seed=None) -> thetas(M,3)
    """
    device = torch.device(device)
    def prior(M: int, seed: Optional[int] = None) -> torch.Tensor:
        gen = torch.Generator(device=device)
        if seed is not None:
            gen = gen.manual_seed(seed)
        thetas = prior_box.sample(M, generator=gen, device=device, dtype=dtype)  # (M,3)
        return thetas
    return prior

from typing import Optional
import torch

def make_simulator_fn(
    K: int, T: int, dt: float,R,
    R_default: Optional[int] = None,         # 可选：默认每个 theta 生成的轨迹条数
    x0: Optional[torch.Tensor] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    out_device: str = "cpu",
    chunk_size: int = 2048                    # 按 θ 维分块，避免一次吃满显存
):
    """
    返回: simulator(thetas, R=None, seed=None, chunk_size=None) -> X
      - thetas: (M,3)  float32/float64 tensor
      - R: 每个 theta 生成的轨迹条数；若 None 则用 R_default 或 1
      - seed: 可复现种子（分块会影响具体序列，但同配置下复现）
      - chunk_size: 覆盖默认分块大小
    输出:
      - X: (M, R*(T+1)*K)  展平特征
    实现要点:
      * 对 (θ, R) 同时并行，时间 T 用轻量 for 循环
      * 对 θ 维分块，显存可控
      * 计算在 `device` 上完成，分块拷回 `out_device`
    """
    dev = torch.device(device)
    out_dev = torch.device(out_device)

    # 预处理 x0 -> (1,1,K) 以便广播到 (m, R, K)
    if x0 is None:
        x0_base = torch.ones(K, device=dev, dtype=dtype)
    else:
        x0_base = x0 if isinstance(x0, torch.Tensor) else torch.tensor(x0)
        x0_base = x0_base.to(device=dev, dtype=dtype)
        assert x0_base.shape == (K,), f"x0 需为形状 (K,), 实际 {x0_base.shape}"

    @torch.no_grad()
    def simulator(thetas: torch.Tensor, seed: Optional[int] = None,
                  chunk_size_: Optional[int] = None) -> torch.Tensor:
        assert thetas.ndim == 2 and thetas.shape[1] == 3, f"thetas 需为 (M,3)，得到 {thetas.shape}"
        M = thetas.shape[0]
        chunk = chunk_size_ or chunk_size

        feat_dim = R * (T + 1) * K
        # 直接在输出设备上分配，后续分块写入，节省 GPU 峰值内存
        X_out = torch.empty((M, feat_dim), device=out_dev, dtype=dtype)

        # 分块遍历 θ
        for start in range(0, M, chunk):
            end = min(M, start + chunk)
            m = end - start

            th_chunk = thetas[start:end].to(device=dev, dtype=dtype)  # (m,3)
            b0 = th_chunk[:, 0].view(m, 1, 1)                          # (m,1,1)
            b1 = th_chunk[:, 1].view(m, 1, 1)
            se = th_chunk[:, 2].view(m, 1, 1)

            # 初始化状态 (m, R, K)
            x = x0_base.view(1, 1, K).expand(m, R, K).clone()

            # 存储当前分块的完整轨迹 (m, R, T+1, K)
            y = torch.empty((m, R, T + 1, K), device=dev, dtype=dtype)
            y[:, :, 0, :] = x

            # 为了复现性：每个分块用独立 generator（注意：改变 chunk_size 会改变具体序列）
            gen = None
            if seed is not None:
                gen = torch.Generator(device=dev).manual_seed(int(seed) + start)

            # 演化 T 步；_l96_core_torch 对最后一维 roll，天然支持 (m,R,K)
            for t in range(1, T + 1):
                core = _l96_core_torch(x)                                 # (m,R,K)
                mean = x + dt * (core - (b0 + b1 * x))                    # 广播到 (m,R,K)
                noise = torch.randn((m, R, K), device=dev, dtype=dtype, generator=gen)
                x = mean - dt * se * noise                                # 与你当前生成模型一致
                y[:, :, t, :] = x

            # 展平到 (m, R*(T+1)*K)，再写入输出
            X_chunk = y.reshape(m, -1).to(out_dev)
            X_out[start:end] = X_chunk

            # 释放中间，降低峰值
            del th_chunk, b0, b1, se, x, y, X_chunk, core, mean, noise
            if dev.type == "cuda":
                torch.cuda.empty_cache()

        return X_out

    return simulator




# -----------------------------
#  Power Posterior via SMC
# -----------------------------
@dataclass
class SMCConfig:
    M: int = 800
    ess_target_frac: float = 0.8
    resample_frac: float = 0.6
    mh_steps: int = 3
    mh_scales: tuple = (0.03, 0.03, 0.03)
    seed: int = 123


def _reflect_box_torch(
        props: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor
) -> torch.Tensor:
    """在先验盒子内“反射”粒子 (PyTorch 版本)"""
    y = props.clone()
    for d in range(y.shape[1]):
        L, H = low[d], high[d]
        while True:
            below = y[:, d] < L
            above = y[:, d] > H
            if not (below.any() or above.any()):
                break
            y[below, d] = 2 * L - y[below, d]
            y[above, d] = 2 * H - y[above, d]
    return y


# ---------- 2) Extract exact power posterior at ANY t using saved intermediates ----------
def get_exact_power_posterior_from_saved(
        intermediates: List[Dict[str, torch.Tensor]],
        target_t: float,
        num_sample: int = 1000,
        gen: torch.Generator = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-300
) -> Dict[str, Any]:
    if gen is None:
        gen = torch.Generator(device=device).manual_seed(0)

    # input checks
    if not isinstance(intermediates, list) or len(intermediates) == 0:
        raise ValueError("intermediates must be a non-empty list from smc_power_posterior_save.")

    # collect betas
    betas = torch.tensor([float(x["beta"]) for x in intermediates], dtype=dtype, device=device)
    target = float(target_t)

    # find index of closest saved beta
    diffs = torch.abs(betas - target)
    idx = int(torch.argmin(diffs).item())
    entry = intermediates[idx]
    beta_saved = float(entry["beta"])

    # ensure tensors on desired device/dtype
    thetas = entry["thetas"].to(device=device, dtype=dtype)   # (M,3)
    logL = entry["logL"].to(device=device, dtype=dtype)       # (M,)
    weights_saved = entry["weights"].to(device=device, dtype=dtype)  # normalized at beta_saved

    # compute log of saved weights (avoid log(0) via eps)
    w_safe = torch.clamp(weights_saved, min=eps)
    log_old_w = torch.log(w_safe)

    delta = float(target - beta_saved)
    # new log-weights: log_old_w + delta * logL
    log_new_raw = log_old_w + delta * logL

    # normalize in log-space
    m = torch.max(log_new_raw)
    w_new = torch.exp(log_new_raw - m)
    w_new = w_new / (w_new.sum() + 1e-300)

    # sample num_sample indices according to w_new
    idxs = torch.multinomial(w_new, num_samples=num_sample, replacement=True, generator=gen)
    samples = thetas[idxs, :].clone()

    return {
        "target": target,
        "used_beta": beta_saved,
        "probs": w_new,          # tensor (M,)
        "samples": samples       # tensor (num_sample,3)
    }

def _to_numpy(arr):
    if torch is not None and isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)

def _kde_pdf(values, grid, bw=None):
    values = np.asarray(values, dtype=float)
    n = values.size
    if n == 0:
        return np.zeros_like(grid)
    if bw is None:
        std = values.std(ddof=1) if n > 1 else 0.0
        if std <= 0 or not np.isfinite(std):
            std = 1.0
        bw = 1.06 * std * n ** (-1/5)
        if bw <= 0 or not np.isfinite(bw):
            bw = 1e-3
    diffs = (grid[:, None] - values[None, :]) / bw
    kernel = np.exp(-0.5 * (diffs ** 2))
    pdf = kernel.sum(axis=1) / (n * bw * np.sqrt(2 * np.pi))
    return pdf

def plot_parameter_pdfs_subplot(samples, param_names=None, power = None,num_grid=512, margin=3.0,
                                figsize=(12,4), show=True):
    arr = _to_numpy(samples)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"samples must be shape (num_samples, 3). Got {arr.shape}")
    n, d = arr.shape
    if param_names is None:
        param_names = ['b0', 'b1', 'sigma_e']

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for i in range(3):
        vals = arr[:, i]
        # compute grid range: mean ± margin*std, fallback to 1-99 percentiles if std~=0
        if np.isfinite(vals).all():
            mean = np.mean(vals)
            std = np.std(vals, ddof=1) if n > 1 else 0.0
        else:
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                mean = 0.0; std = 1.0
            else:
                mean = np.mean(vals); std = np.std(vals, ddof=1) if vals.size>1 else 0.0

        if std <= 0 or not np.isfinite(std):
            lo, hi = np.percentile(vals, [1, 99]) if vals.size>0 else (-1.0, 1.0)
            pad = max(0.1, 0.1 * (hi - lo))
            grid_min = lo - pad
            grid_max = hi + pad
        else:
            grid_min = mean - margin * std
            grid_max = mean + margin * std

        # ensure data is inside grid
        data_min, data_max = np.nanmin(vals), np.nanmax(vals)
        grid_min = min(grid_min, data_min - 0.05 * abs(data_min) - 1e-8)
        grid_max = max(grid_max, data_max + 0.05 * abs(data_max) + 1e-8)

        grid = np.linspace(grid_min, grid_max, num_grid)
        pdf = _kde_pdf(vals, grid)

        ax = axes[i]
        ax.plot(grid, pdf)
        ax.set_xlabel(param_names[i])
        ax.set_ylabel("pdf")
        ax.set_title(f"{param_names[i]} (n={n},t={power})")
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    if show:
        plt.show()
    return fig


@dataclass
class SimpleMHCfg:
    target_acc: float = 0.30     # 目标验收率
    tune_steps: int = 2000       # 自适应调参步数
    n_samples: int = 10000       # 正式采样条数
    map_random_search: int = 4000  # 先验里随机搜高似然初始点的样本数
    seed: int = 42
    base_scale_frac: float = 1/6  # 提案各维基础尺度 = 先验宽度 * 该系数
    rm_eta0: float = 0.05
    rm_kappa: float = 0.5        # γ_t = (t+1)^(-kappa), 0.5~1

def _choose_init_via_prior_search(prior: PriorBox, dt: float, stats, N: int,
                                  device="cpu", dtype=torch.float32, seed: int = 0):
    gen = torch.Generator(device=device).manual_seed(int(seed))
    thetas = prior.sample(N, generator=gen, device=device, dtype=dtype)         # (N,3)
    logL   = batch_loglik_from_stats(thetas, dt, stats)                          # (N,)
    idx    = int(torch.argmax(logL).item())
    return thetas[idx].clone(), logL[idx].clone()

def mh_sample_l96(batch_loglik_from_stats,
                  dataset: torch.Tensor,
                  dt: float,
                  prior: PriorBox,
                  cfg: SimpleMHCfg = SimpleMHCfg(),
                  power=1.0,
                  device: str = "cpu",
                  dtype: torch.dtype = torch.float32):
    """
    返回:
      dict(chain=(N,3), accept_rate=float, step_size=float, theta0=(3,), base_scale_vec=(3,))
    """
    # 预计算似然统计量（O(1)/粒子）
    stats = precompute_l96_stats(dataset, dt, device=device, dtype=dtype)

    # 初始点：先验随机搜 MAP（高似然）
    theta, cur_ll = _choose_init_via_prior_search(
        prior, dt, stats, cfg.map_random_search, device=device, dtype=dtype, seed=cfg.seed
    )
    theta = theta.to(device=device, dtype=dtype)
    cur_ll = cur_ll.to(device=device, dtype=dtype)

    # 提案基础尺度（各维）：用先验盒子宽度的一个分数
    widths = torch.tensor([prior.b0_high - prior.b0_low,
                           prior.b1_high - prior.b1_low,
                           prior.se_high - prior.se_low], device=device, dtype=dtype)
    base_scale_vec = cfg.base_scale_frac * widths

    # 盒子边界（用于反射）
    lows  = torch.tensor([prior.b0_low, prior.b1_low, prior.se_low], device=device, dtype=dtype)
    highs = torch.tensor([prior.b0_high, prior.b1_high, prior.se_high], device=device, dtype=dtype)

    # 随机源
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))

    # 自适应阶段：只调一个全局 step（对称 RW-Normal + 反射），接受率只看 loglik 差
    log_step = torch.tensor(0.0, device=device, dtype=dtype)  # 初始 step=1
    for t in range(cfg.tune_steps):
        step = torch.exp(log_step)
        prop = theta + step * base_scale_vec * torch.randn(3, generator=gen, device=device, dtype=dtype)
        prop = _reflect_box_torch(prop.view(1,3), lows, highs)[0]
        prop_ll = batch_loglik_from_stats(prop.view(1,3), dt, stats)[0]

        log_alpha = power*(prop_ll - cur_ll)  # uniform prior 内部抵消
        u = torch.log(torch.rand((), generator=gen, device=device, dtype=dtype))
        accept = (u < log_alpha).item()
        if accept:
            theta = prop
            cur_ll = prop_ll

        # Robbins–Monro 调整 log_step
        gamma_t = (t + 1)**(-cfg.rm_kappa)
        log_step = log_step + cfg.rm_eta0 * gamma_t * ((1.0 if accept else 0.0) - cfg.target_acc)

    tuned_step = float(torch.exp(log_step).item())

    # 正式采样（固定步长）
    N = int(cfg.n_samples)
    chain = torch.empty((N, 3), device=device, dtype=dtype)
    accept_cnt = 0
    step_tensor = torch.tensor(tuned_step, device=device, dtype=dtype)

    for n in range(N):
        prop = theta + step_tensor * base_scale_vec * torch.randn(3, generator=gen, device=device, dtype=dtype)
        prop = _reflect_box_torch(prop.view(1,3), lows, highs)[0]
        prop_ll = batch_loglik_from_stats(prop.view(1,3), dt, stats)[0]

        log_alpha = power*(prop_ll - cur_ll)
        u = torch.log(torch.rand((), generator=gen, device=device, dtype=dtype))
        if u < log_alpha:
            theta = prop
            cur_ll = prop_ll
            accept_cnt += 1

        chain[n] = theta

    return {
        "chain": chain.detach().cpu(),
        "accept_rate": accept_cnt / N,
        "step_size": tuned_step,
        "theta0": theta.detach().cpu(),
        "base_scale_vec": base_scale_vec.detach().cpu()
    }



if __name__ == "__main__":
    torch.set_grad_enabled(False)
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K, dt, T = 8, 3.0/40.0, 20
    theta_true = Theta(2.0, 0.8, 1.7)
    save_dir = Path("lorenz96")
    dataset = simulate_dataset(theta_true, n_series=5, K=K, T=T, dt=dt, x0=torch.ones(K), seed=42)

    prior = PriorBox(1.4, 2.2, 0.1, 1.0, 1.5, 2.5)

    targets = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    cfg = SimpleMHCfg(target_acc=0.30, tune_steps=6000, n_samples=10000, seed=42, map_random_search=1500)
    for beta in targets:
        # res = mh_sample_l96(batch_loglik_from_stats,dataset, dt, prior, cfg, power= beta,device=device, dtype=dtype)
        # plot_parameter_pdfs_subplot(res['chain'], power=beta,param_names=["b0", "b1", "sigma_e"])

        # print(res["accept_rate"],res["chain"].shape)
        ckpt_path = save_dir / f"True_Power_posterior_with_t_power={beta}_10000.pt"
        # torch.save(res['chain'], ckpt_path)
        posterior = torch.load(ckpt_path)
        plot_parameter_pdfs_subplot(posterior, power=beta,param_names=["b0", "b1", "sigma_e"])

