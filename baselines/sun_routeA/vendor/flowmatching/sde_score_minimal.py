import math
import argparse
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# --------------------------- Utils: Standardization ---------------------------

class Standardizer:
    """Zero-mean, unit-variance per dimension, with inverse transform."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: torch.Tensor):
        # 统计量用 float32 存在 CPU，上下文设备无关
        Xc = X.detach().to(torch.float32).cpu()
        self.mean_ = Xc.mean(0, keepdim=True)
        self.scale_ = Xc.std(0, keepdim=True).clamp_min(1e-12)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        # 按输入 X 的 device/dtype 对齐统计量
        m = self.mean_.to(device=X.device, dtype=X.dtype)
        s = self.scale_.to(device=X.device, dtype=X.dtype)
        return (X - m) / s

    def inverse_transform(self, Z: torch.Tensor) -> torch.Tensor:
        # 按输入 Z 的 device/dtype 对齐统计量
        m = self.mean_.to(device=Z.device, dtype=Z.dtype)
        s = self.scale_.to(device=Z.device, dtype=Z.dtype)
        return Z * s + m

# --------------------------- Sigma schedule (Karras) --------------------------

def make_sigma_schedule(n, sigma_min=2e-3, sigma_max=80.0, rho=7.0, device="cpu"):
    ramp = torch.linspace(0, 1, n, device=device)
    sigmas = (sigma_max**(1/rho) + ramp*(sigma_min**(1/rho) - sigma_max**(1/rho)))**rho
    sigmas, _ = torch.sort(sigmas)  # 升序
    return sigmas

# --------------------------- Score Network (tiny MLP) -------------------------

class SinusoidalEmbedding(nn.Module):
    """对 log_sigma 做正弦位置嵌入（扩展条件维度）"""
    def __init__(self, dim=64, min_freq=1.0, max_freq=1000.0):
        super().__init__()
        self.dim = dim
        freqs = torch.exp(torch.linspace(math.log(min_freq), math.log(max_freq), dim // 2))
        self.register_buffer("freqs", freqs)

    def forward(self, log_sigma):  # log_sigma: (B,1)
        x = log_sigma * self.freqs[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B,dim)
        return emb

class ResBlock(nn.Module):
    """残差 MLP 块 + FiLM 条件化（用 σ 的嵌入调制每层）"""
    def __init__(self, hidden):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.fc1   = nn.Linear(hidden, hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.fc2   = nn.Linear(hidden, hidden)
        self.film  = nn.Linear(hidden, hidden * 2)  # 由 cond 产生 (gamma, beta)

    def forward(self, h, cond_vec):  # h: (B,H), cond_vec: (B,H)
        gamma, beta = self.film(cond_vec).chunk(2, dim=-1)
        x = self.norm1(h)
        x = x * (1 + gamma) + beta          # FiLM 调制
        x = F.silu(self.fc1(x))
        x = self.norm2(x)
        x = F.silu(self.fc2(x))
        return h + x                         # 残差

class ResidualScoreNet(nn.Module):
    """Input: x∈R^D, σ -> Output: score s(x,σ)∈R^D（更强但依然轻量）"""
    def __init__(self, dim: int, hidden=256, blocks=4, emb_dim=64):
        super().__init__()
        self.inp = nn.Linear(dim, hidden)

        self.sigma_emb = SinusoidalEmbedding(emb_dim)
        self.cond_mlp  = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.out   = nn.Linear(hidden, dim)
        # “零初始化”最后一层，稳定早期训练（常见小技巧）
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        if sigma.ndim == 0: sigma = sigma.expand(x.shape[0])
        sigma = sigma.view(-1, 1)
        log_sigma = torch.log(sigma.clamp_min(1e-12))

        cond = self.cond_mlp(self.sigma_emb(log_sigma))  # (B,H)
        h = F.silu(self.inp(x))
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(F.silu(h))

# --------------------------- Training: DSM objective --------------------------

@dataclass
class TrainConfig:
    # 训练循环
    steps: int = 10000
    batch: int = 1024
    lr: float = 1e-3
    clip_grad: float = 5.0
    log_every: int = 500

    # EDM 训练目标
    objective: str = "edm"    # 固定为 EDM
    sigma_data: float = 0.5   # 论文默认
    p_mean: float = -1.2      # 论文/官方实现默认
    p_std: float = 1.2

    # 采样用的 Karras σ 调度（论文默认）
    sigma_min: float = 2e-3
    sigma_max: float = 80.0
    rho: float = 7.0
    n_sigmas: int = 35        # 与论文“~35 NFE”一致

    # 早停/验证（你之前就有）
    early_stop: bool = True
    val_ratio: float = 0.10
    patience: int = 2000
    min_delta: float = 1e-4
    eval_every: int = 200

    # 采样时可选的 churn（论文里是可选，默认关）
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0


def edm_precond(sigma: torch.Tensor, sigma_data: float = 0.5):
    s2 = sigma**2
    sd2 = sigma_data**2
    c_skip = sd2 / (s2 + sd2)
    c_out  = sigma * sigma_data / torch.sqrt(s2 + sd2)
    c_in   = 1.0 / torch.sqrt(s2 + sd2)
    return c_skip, c_out, c_in

def denoise_x0(model: nn.Module, x: torch.Tensor, sigma: torch.Tensor, sigma_data: float = 0.5):
    """
    \hat{x}_0 = c_skip*x + c_out*F_theta(c_in*x, sigma)
    兼容 sigma 为标量 / (1,) / (B,) / (B,1) 的各种情况。
    """
    B, dev, dtype = x.shape[0], x.device, x.dtype

    # --- 统一 sigma 形状到 (B,) 并对齐设备/类型 ---
    if not torch.is_tensor(sigma):
        sigma = torch.tensor(sigma, device=dev, dtype=dtype)
    else:
        sigma = sigma.to(device=dev, dtype=dtype)

    if sigma.ndim == 0:
        sigma = sigma.expand(B)                # 标量 → (B,)
    elif sigma.ndim == 1 and sigma.shape[0] == 1:
        sigma = sigma.expand(B)                # (1,) → (B,)
    elif sigma.ndim == 2 and sigma.shape[1] == 1 and sigma.shape[0] == 1:
        sigma = sigma.view(-1).expand(B)       # (1,1) → (B,)

    # --- 预条件化并广播 ---
    c_skip, c_out, c_in = edm_precond(sigma, sigma_data)
    shp = (B,) + (1,)*(x.ndim-1)               # (B,1[,1...])
    x_in = c_in.view(*shp) * x
    F = model(x_in, sigma)                     # 模型内部会对 sigma 做 log 处理

    return c_skip.view(*shp) * x + c_out.view(*shp) * F


def edm_loss(model: nn.Module, x: torch.Tensor, cfg: TrainConfig):
    B, dev = x.shape[0], x.device
    log_sigma = torch.randn(B, device=dev) * cfg.p_std + cfg.p_mean
    sigma = log_sigma.exp()
    shp = (B,) + (1,)*(x.ndim-1)

    noise = torch.randn_like(x) * sigma.view(*shp)
    y = x + noise

    # 预条件化
    s2, sd2 = sigma**2, cfg.sigma_data**2
    c_skip = sd2 / (s2 + sd2)
    c_out  = sigma * cfg.sigma_data / torch.sqrt(s2 + sd2)
    c_in   = 1.0 / torch.sqrt(s2 + sd2)

    x_in = c_in.view(*shp) * y
    F = model(x_in, sigma)               # 预测残差
    D = c_skip.view(*shp)*y + c_out.view(*shp)*F

    weight = (((sigma**2 + cfg.sigma_data**2) / (sigma*cfg.sigma_data))**2).view(*shp)
    return (weight * (D - x)**2).mean()



def dsm_loss(model: nn.Module, x: torch.Tensor, sigmas: torch.Tensor) -> torch.Tensor:
    """Continuous denoising score matching (VE formulation).
       Sample σ from discrete schedule uniformly; target = -ε/σ; optional σ^2 weight.
    """
    B, D = x.shape
    device = x.device
    j = torch.randint(0, sigmas.shape[0], (B,), device=device)
    sigma = sigmas[j]                               # (B,)
    eps = torch.randn_like(x)
    y = x + sigma.view(-1,1) * eps                  # perturbed
    s_hat = model(y, sigma)                         # predicted score
    target = - eps / sigma.view(-1,1)
    # Weighting by σ^2 stabilizes training across noise levels (common choice).
    w = (sigma**2).view(-1,1)
    loss = (w * (s_hat - target)**2).mean()
    return loss

def train(model: nn.Module, X: torch.Tensor, cfg: TrainConfig, device="cpu"):
    """
    训练：连续DSM。
    新增：
      - 验证集拆分（按 cfg.val_ratio；若无该字段则默认 0.1）
      - 早停（按最小验证损失；若 cfg.early_stop 不存在则默认开启）
      - 回滚至最佳权重
    日志：每 cfg.log_every 打印一次训练loss；每 cfg.eval_every 评估一次val loss。
    """
    import copy

    # ------- 读 cfg 参数（若无则给默认值） -------
    val_ratio  = getattr(cfg, "val_ratio", 0.10)     # 验证集比例
    early_stop = getattr(cfg, "early_stop", True)    # 是否早停
    patience   = getattr(cfg, "patience", 2000)      # 耐心步数
    min_delta  = getattr(cfg, "min_delta", 1e-4)     # 认为“有提升”的最小幅度
    eval_every = getattr(cfg, "eval_every", 200)     # 多久评估一次验证损失

    # ------- 训练/验证集划分 -------
    X = X.detach().cpu()
    N = X.shape[0]
    if val_ratio > 0.0:
        perm = torch.randperm(N)
        n_val = max(1, int(N * val_ratio))
        idx_val = perm[:n_val]
        idx_tr  = perm[n_val:]
        X_tr, X_val = X[idx_tr], X[idx_val]
    else:
        X_tr, X_val = X, None

    # ------- 仅用训练集拟合标准化器 -------
    std = Standardizer().fit(X_tr)
    Xs_tr = std.transform(X_tr).to(device)
    Xs_val = std.transform(X_val).to(device) if X_val is not None else None

    # ------- 优化器与噪声表 -------
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sigmas = make_sigma_schedule(cfg.n_sigmas, cfg.sigma_min, cfg.sigma_max, cfg.rho, device)
    Ntr = Xs_tr.shape[0]

    # ------- 监控/早停状态 -------
    best_score = float("inf")
    best_state = None
    last_improve_step = 0

    def eval_loss_on(Xs):
        with torch.no_grad():
            B = min(cfg.batch, Xs.shape[0])
            idx = torch.randint(0, Xs.shape[0], (B,), device=device)
            xb = Xs[idx]
            if getattr(cfg, "objective", "edm") == "edm":
                return edm_loss(model, xb, cfg).item()
            else:
                return dsm_loss(model, xb, sigmas).item()

    # ------- 训练循环 -------
    for t in range(1, cfg.steps + 1):
        idx = torch.randint(0, Ntr, (min(cfg.batch, Ntr),), device=device)
        xb = Xs_tr[idx]

        if getattr(cfg, "objective", "edm") == "edm":
            loss = edm_loss(model, xb, cfg)
        else:
            loss = dsm_loss(model, xb, sigmas)

        opt.zero_grad(); loss.backward()
        if getattr(cfg, "clip_grad", None) is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        opt.step()

        if t % getattr(cfg, "log_every", 500) == 0:
            print(f"[{t}/{cfg.steps}] train_loss={loss.item():.4f}")

        # ---- 验证评估 & 早停 ----
        if early_stop and (t % eval_every == 0 or t == 1):
            metric = eval_loss_on(Xs_val) if Xs_val is not None else eval_loss_on(Xs_tr)
            tag = "val" if Xs_val is not None else "train"
            improved = (best_score - metric) > min_delta
            if improved:
                best_score = metric
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                last_improve_step = t
                print(f"  ↳ {tag}_loss={metric:.4f} (best so far)")
            else:
                if (t - last_improve_step) >= patience:
                    print(f"[Early Stop] {tag}_loss no improv ≥{patience} steps; stop @ {t}. best={best_score:.4f}")
                    break

    # 回滚到最佳权重（若已找到）
    if early_stop and best_state is not None:
        model.load_state_dict(best_state)

    return model, std, sigmas

@torch.no_grad()
def edm_sampler(model: nn.Module, sigmas: torch.Tensor, n: int, std,
                sigma_data: float = 0.5,
                s_churn: float = 0.0, s_tmin: float = 0.0, s_tmax: float = float("inf"), s_noise: float = 1.0,
                device="cpu", seed=0):
    torch.manual_seed(seed)
    D = std.mean_.shape[1]
    model = model.to(device).eval()

    # 降序 σ
    sigs = torch.flip(sigmas, [0]) if sigmas[0] < sigmas[-1] else sigmas.clone()
    # 以最大 σ 初始化
    x = torch.randn(n, D, device=device) * sigs[0]

    def denoise_x0(x, sigma):
        # 兼容标量/向量 σ
        B = x.shape[0]
        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=x.device, dtype=x.dtype)
        sigma = sigma.to(x.device, x.dtype)
        if sigma.ndim == 0: sigma = sigma.expand(B)
        elif sigma.ndim == 1 and sigma.shape[0] == 1: sigma = sigma.expand(B)
        s2, sd2 = sigma**2, sigma_data**2
        c_skip = sd2 / (s2 + sd2)
        c_out  = sigma * sigma_data / torch.sqrt(s2 + sd2)
        c_in   = 1.0 / torch.sqrt(s2 + sd2)
        shp = (B,) + (1,)*(x.ndim-1)
        F = model((c_in.view(*shp) * x), sigma)
        return c_skip.view(*shp) * x + c_out.view(*shp) * F

    for i in range(len(sigs)-1):
        si, sj = sigs[i], sigs[i+1]          # si > sj
        sigma_hat = si

        # （可选）churn：论文/实现里可开可关，这里默认关
        if s_churn > 0.0 and (si >= s_tmin) and (si <= s_tmax):
            gamma = min(s_churn/len(sigs), math.sqrt(2.0) - 1.0)
            sigma_hat = si * (1.0 + gamma)
            x = x + torch.sqrt(torch.clamp(sigma_hat**2 - si**2, min=0.0)).to(x.dtype) \
                    * s_noise * torch.randn_like(x)

        h = (sj - sigma_hat).to(x.dtype)     # 负数
        # Heun 预测
        x0  = denoise_x0(x,  sigma_hat)
        s1  = (x0 - x) / (sigma_hat**2)      # score(x,σ) = (x0 - x)/σ^2
        d1  = - sigma_hat * s1               # f = -σ * score
        x_p = x + h * d1

        # Heun 修正
        x0_2 = denoise_x0(x_p, sj)
        s2   = (x0_2 - x_p) / (sj**2)
        d2   = - sj * s2
        x    = x + h * 0.5 * (d1 + d2)

    # 终点再做一次“到 σ≈0 的去噪”以提高洁净度（论文/实现常见做法）
    x = denoise_x0(x, sigs[-1])

    return std.inverse_transform(x).cpu()




# --------- VE-SDE Reverse SDE PC sampler & Probability-flow ODE sampler -------

def pc_sampler(model: nn.Module, sigmas: torch.Tensor, n: int, std: Standardizer,
               corrector_steps: int = 1, corrector_eta: float = 0.1,
               device="cpu", seed=0):
    """
    Predictor–Corrector sampling for VE-SDE.
      Predictor: Reverse SDE Euler–Maruyama in u = σ^2
      Corrector: Annealed Langevin at each σ (optional)
    """
    torch.manual_seed(seed)
    D = std.mean_.shape[1]
    model = model.to(device).eval()

    # Ensure descending sigmas (big -> small)
    sigs = torch.flip(sigmas, dims=[0]) if sigmas[0] < sigmas[-1] else sigmas.clone()

    # init at largest sigma  # CHANGED
    x = torch.randn(n, D, device=device) * sigs[0]

    with torch.no_grad():
        for i in range(len(sigs)-1):
            sigma_i = sigs[i]
            sigma_j = sigs[i+1]

            # ---- Corrector: Langevin (optional) ----
            if corrector_steps > 0:
                for _ in range(corrector_steps):
                    grad = model(x, sigma_i)
                    step_size = (corrector_eta * sigma_i)**2
                    noise = torch.randn_like(x)
                    x = x + step_size * grad + torch.sqrt(2 * step_size) * noise

            # ---- Predictor: reverse VE-SDE (Euler–Maruyama) ----
            delta = (sigma_j.square() - sigma_i.square()).to(x.dtype)  # negative
            grad = model(x, sigma_i)
            noise = torch.randn_like(x)
            x = x - delta * grad + torch.sqrt(torch.abs(delta)) * noise  # CHANGED

    return std.inverse_transform(x).cpu()


def ode_sampler(model: nn.Module, sigmas: torch.Tensor, n: int, std: Standardizer,
                solver="heun", device="cpu", seed=0):
    """
    Probability-flow ODE sampler for VE-SDE (deterministic).
    ODE: dx = -0.5 * g(t)^2 * score(x, t) dt  → in σ^2 coords: Δx = -0.5 * Δ(σ^2) * score
    We use fixed-step Heun (2nd order) over the σ grid.
    """
    torch.manual_seed(seed)
    D = std.mean_.shape[1]
    model = model.to(device).eval()

    # descending sigmas
    sigs = torch.flip(sigmas, dims=[0]) if sigmas[0] < sigmas[-1] else sigmas.clone()

    # init at the largest sigma (now sigs[0] is the largest)
    x = torch.randn(n, D, device=device) * sigs[0]

    with torch.no_grad():
        for i in range(len(sigs)-1):
            si, sj = sigs[i], sigs[i+1]
            delta = (sj.square() - si.square()).to(x.dtype)  # scalar tensor (negative)

            if solver == "euler":
                f = -0.5 * model(x, si)
                x = x + delta * f
            else:  # heun (default)
                f1 = -0.5 * model(x, si)
                x_p = x + delta * f1
                f2 = -0.5 * model(x_p, sj)
                x  = x + delta * 0.5 * (f1 + f2)

    return std.inverse_transform(x).cpu()
