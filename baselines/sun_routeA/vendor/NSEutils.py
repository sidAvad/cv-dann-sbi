import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
from torch import nn
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import csv, uuid
from datetime import datetime


class SigmaEmbedding(nn.Module):
    """
     sigma -> embedding
    """
    def __init__(self, emb_dim: int = 128, sigma_min: float = 1e-2, sigma_max: float = 1.0, normalize: bool = True):
        super().__init__()
        self.normalize = normalize

        log_min, log_max = math.log(sigma_min), math.log(sigma_max)
        center = 0.5 * (log_min + log_max)
        scale  = 0.5 * (log_max - log_min) if normalize else 1.0
        self.register_buffer("log_center", torch.tensor(center, dtype=torch.float32))
        self.register_buffer("log_scale",  torch.tensor(scale,  dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(1, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim), nn.SiLU()
        )

    def forward(self, sigma: torch.Tensor):
        s = sigma.view(-1, 1).clamp_min(1e-12)
        s = torch.log(s)
        if self.normalize:
            s = (s - self.log_center) / (self.log_scale + 1e-12)
        return self.net(s)   # [B, emb_dim]

# ------------------------
# Residual MLP block
# ------------------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm2(x)
        return x + residual

# ------------------------
# MLP with residual connections
# ------------------------
class ResidualMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.input_fc = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualMLPBlock(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.out_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.input_fc(x))
        for block in self.blocks:
            x = block(x)
        return self.out_fc(x)


class ScoreNet_sigma(nn.Module):
    def __init__(self, theta_dim, x_dim, amortize = False,emb_dim=128, sigma_emb_dim=128, hidden_dim=128,sigma_min = 1e-2, sigma_max = 1):
        super().__init__()
        # θ embedding
        self.theta_net = ResidualMLP(theta_dim, emb_dim, hidden_dim)
        # x embedding
        self.x_net = ResidualMLP(x_dim, emb_dim, hidden_dim)
        # t embedding
        self.sigma_emb = SigmaEmbedding(sigma_emb_dim,sigma_min = sigma_min,sigma_max = sigma_max)
        # final MLP
        if amortize:
            self.final_net = ResidualMLP(emb_dim*2 + sigma_emb_dim, theta_dim + x_dim, hidden_dim)
        else:
            self.final_net = ResidualMLP(emb_dim*2 + sigma_emb_dim, theta_dim, hidden_dim)

    def forward(self, theta, x, sigma):
        theta_emb = self.theta_net(theta)
        x_emb = self.x_net(x)
        sigma_emb = self.sigma_emb(sigma)
        h = torch.cat([theta_emb, x_emb, sigma_emb], dim=-1)
        score = self.final_net(h)
        return score


class PairDataset(Dataset):
    def __init__(self, prior, simulator,n,device = "cuda" if torch.cuda.is_available() else "cpu"):
        self.simulator = simulator
        self.prior = prior
        self.n = n
        self.theta = self.prior(self.n)
        self.x = self.simulator(self.theta).to(device)
        self.theta = self.theta.to(device)
        self.theta = torch.as_tensor(self.theta, dtype=torch.float32)
        self.x = torch.as_tensor(self.x, dtype=torch.float32)
    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.cat([self.theta[idx], self.x[idx]], dim=-1)


class EarlyStopping:
    def __init__(
        self,
        patience: int = 20,
        min_delta: float = 0.0,
        mode: str = "min",
        restore_best_weights: bool = True,
        verbose: bool = True,
    ):
        assert mode in ("min", "max")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.best_score = None
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:  # "max"
            return current > (best + self.min_delta)

    def step(self, current: float, model: torch.nn.Module) -> bool:
        """
        在每个 epoch 验证后调用。
        参数:
          current: 本轮监控指标（如 val_loss）
          model:   当前模型（用于保存/恢复权重）
        返回:
          True  -> 触发 early stop
          False -> 继续训练
        """
        # 首次
        if self.best_score is None:
            self.best_score = float(current)
            if self.restore_best_weights:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Init best={self.best_score:.6f}")
            return False

        # 是否变好
        if self._is_improvement(float(current), float(self.best_score)):
            self.best_score = float(current)
            if self.restore_best_weights:
                self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improved best -> {self.best_score:.6f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                pass
                # print(f"[EarlyStopping] No improv ({self.counter}/{self.patience}) | current={current:.6f} best={self.best_score:.6f}")
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best_weights and self.best_state is not None:
                    model.load_state_dict(self.best_state)
                    if self.verbose:
                        print("[EarlyStopping] Restored best weights.")
                return True
            return False


def _to_xy(samples):
    """
    支持两种输入：
    - Tensor 形状 [N,2]
    - list/tuple，里面每个元素是形状 [2] 的 tensor
    返回：x,y 的 numpy 数组
    """
    if isinstance(samples, (list, tuple)):
        t = torch.stack(samples, dim=0)
    else:
        t = samples
    t = t.detach().cpu()
    assert t.shape[-1] == 2, f"Expect last dim=2, got {t.shape}"
    x = t[..., 0].numpy()
    y = t[..., 1].numpy()
    return x, y

def plot_posteriors_compare(reference_samples, posterior_samples, title="Posterior comparison",
                            save_path: Path = None, show: bool = True):
    x_ref, y_ref = _to_xy(reference_samples[:, :2])
    x_post, y_post = _to_xy(posterior_samples[:, :2])


    all_x = np.concatenate([x_ref, x_post])
    all_y = np.concatenate([y_ref, y_post])
    x_min, x_max = np.percentile(all_x, [1, 99])
    y_min, y_max = np.percentile(all_y, [1, 99])
    pad_x = 0.05 * (x_max - x_min + 1e-12)
    pad_y = 0.05 * (y_max - y_min + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)


    axes[0].scatter(x_ref, y_ref, s=8, alpha=0.6, linewidths=0, rasterized=True)
    axes[0].set_title("Reference posterior")
    axes[0].set_xlabel("θ₁")
    axes[0].set_ylabel("θ₂")


    label = f"approximate posterior"
    axes[1].scatter(x_post, y_post, s=8, alpha=0.6, linewidths=0, rasterized=True)
    axes[1].set_title(label)
    axes[1].set_xlabel("θ₁")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(title, y=0.98, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)



import math





def to_float(x):

    if isinstance(x, torch.Tensor):

        return x.detach().cpu().item()
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return float(x)


import os, random


SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _style_ax(ax):
    ax.tick_params(labelsize=10)
    for s in ("top","right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, ls="--", alpha=0.3)

def _compute_ylims(arr_list, pad=0.05):
    # arr_list: [y1, y2, ...]  每个是 1D numpy 数组
    vals = np.concatenate([np.asarray(a, dtype=float) for a in arr_list if len(a) > 0])
    if vals.size == 0:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        if lo == 0.0:
            return (-0.05, 0.05)
        span = abs(lo) if lo != 0 else 1.0
        return (lo - 0.05*span, hi + 0.05*span)
    span = hi - lo
    return (lo - pad*span, hi + pad*span)

def plot_gm_1x4_using_collector(collect_fn, out_base):
    """
    使用你提供的 collect_metrics_for_budget(budget) 生成 Gaussian Mixture 的 1×4 图。
    参数：
      - collect_fn: 你的收集函数（就是 collect_metrics_for_budget）
      - out_base: 输出文件前缀（Path 或 str），会生成 out_base.pdf / out_base.png
    返回：(pdf_path, png_path)
    """
    # 1) 取两个预算的结果
    res10  = collect_fn(10_000)
    res100 = collect_fn(100_000)

    # 2) 解包 + 兼容可能缺失某些 beta 的情况（长度以指标数组为准）
    def _unpack(res, metric):
        yA = np.asarray(res[metric]["A"], dtype=float)
        yB = np.asarray(res[metric]["B"], dtype=float)
        betas_full = np.asarray(res["betas"], dtype=float)
        L = min(len(yA), len(yB), len(betas_full))
        betas = betas_full[:L]
        return betas, yA[:L], yB[:L]

    betas1, c2stA_10k, c2stB_10k = _unpack(res10,  "c2st")
    _,      mmdA_10k,  mmdB_10k  = _unpack(res10,  "mmd")
    betas2, c2stA_100k,c2stB_100k= _unpack(res100, "c2st")
    _,      mmdA_100k, mmdB_100k = _unpack(res100, "mmd")

    # 3) 为 C2ST 和 MMD 分别统一 y 轴范围（方便比较）
    ylim_c2st = _compute_ylims([c2stA_10k, c2stB_10k, c2stA_100k, c2stB_100k])
    ylim_mmd  = _compute_ylims([mmdA_10k,  mmdB_10k,  mmdA_100k,  mmdB_100k])

    # 4) 画 1×4 子图
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharex=False)
    panels = [
        ("simulation budget 10,000\nC2ST", betas1, c2stA_10k,  c2stB_10k,  "C2ST ", ylim_c2st),
        ("simulation budget 10,000\nMMD",  betas1, mmdA_10k,   mmdB_10k,   "MMD ",  ylim_mmd),
        ("simulation budget 100,000\nC2ST",betas2, c2stA_100k, c2stB_100k, "C2ST ", ylim_c2st),
        ("simulation budget 100,000\nMMD", betas2, mmdA_100k,  mmdB_100k,  "MMD ",  ylim_mmd),
    ]

    for i, (ttl, bx, yA, yB, ylabel, ylim) in enumerate(panels):
        ax = axes[i]
        ax.plot(bx, yA, marker="o", lw=2, ms=5, label="Route A")
        ax.plot(bx, yB, marker="s", lw=2, ms=5, label="Route B")
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("beta", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim is not None:
            ax.set_ylim(ylim)
        # 可读的 beta 刻度
        ax.set_xticks(bx)
        ax.set_xticklabels([f"{b:g}" for b in bx], rotation=0, fontsize=9)
        _style_ax(ax)
        if i == 0:
            ax.legend(frameon=False, fontsize=9, loc="best")

    fig.suptitle("Two Moons: Route A/B vs. beta", y=1.02, fontsize=12)
    fig.tight_layout()

    out_base = Path(out_base)
    pdf_path = out_base.with_suffix(".pdf")
    png_path = out_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(pdf_path), str(png_path)



class ExperimentLogger:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.header = [
            "run_id", "timestamp",
            "method", "n", "beta",
            "c2st", "mmd",
            # 可选元数据（按需填充）
            "seed", "device", "ckpt_path", "notes"
        ]
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log(self, *, method, n, beta, c2st, mmd, seed=None, device=None, ckpt_path=None, notes=None):
        row = [
            str(uuid.uuid4()),
            datetime.now().isoformat(timespec="seconds"),
            str(method),
            int(n),
            float(beta),
            float(c2st),
            float(mmd),
            seed if seed is not None else "",
            str(device) if device is not None else "",
            str(ckpt_path) if ckpt_path is not None else "",
            notes if notes is not None else "",
        ]
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)