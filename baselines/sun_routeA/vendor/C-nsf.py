import math, os
from dataclasses import dataclass
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import numpy as np


def _ema(y, k=15):
    y = np.asarray(y, dtype=float)
    a = 2.0 / (k + 1.0)
    s = np.empty_like(y)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = (1 - a) * s[i-1] + a * y[i]
    return s

@dataclass
class BoxSpec:
    low: torch.Tensor   # (D,)
    high: torch.Tensor  # (D,)

class BoxSquash:
    def __init__(self, box: BoxSpec, eps: float = 1e-6):
        self.low = box.low.detach().float()
        self.high = box.high.detach().float()
        self.eps = eps

    def theta_to_z(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = theta.float()

        low = self.low.to(theta.device)
        high = self.high.to(theta.device)
        u = (theta - low) / (high - low)
        u = torch.clamp(u, self.eps, 1.0 - self.eps)

        z = torch.log(u) - torch.log(1.0 - u)
        logdet = -(torch.log(high - low) + torch.log(u) + torch.log(1.0 - u)).sum(dim=-1)
        return z, logdet

    def z_to_theta(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = z.float()

        low = self.low.to(z.device)
        high = self.high.to(z.device)
        u = torch.sigmoid(z)
        u = torch.clamp(u, self.eps, 1.0 - self.eps)

        theta = low + u * (high - low)
        logdet_inv = (torch.log(high - low) + torch.log(u) + torch.log(1.0 - u)).sum(dim=-1)
        return theta, logdet_inv


def _searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor):
    # 返回每个 inputs 落在哪个 bin（0..K-1）
    return torch.sum(inputs[..., None] >= bin_locations[..., :-1], dim=-1) - 1

def rational_quadratic_spline_vec(
    inputs: torch.Tensor,
    unnormalized_widths: torch.Tensor,
    unnormalized_heights: torch.Tensor,
    unnormalized_derivatives: torch.Tensor,
    left: float = -3.0,
    right: float = 3.0,
    bottom: float = -3.0,
    top: float = 3.0,
    inverse: bool = False,
    min_bin_width: float = 1e-3,
    min_bin_height: float = 1e-3,
    min_derivative: float = 1e-3,
):
    K = unnormalized_widths.size(-1)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = widths * (right - left - K * min_bin_width) + min_bin_width
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, (1, 0), value=0.0) + left  # (B,M,K+1)
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]       # (B,M,K)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = heights * (top - bottom - K * min_bin_height) + min_bin_height
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, (1, 0), value=0.0) + bottom
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = F.softplus(unnormalized_derivatives) + min_derivative  # (B,M,K+1)

    if not inverse:
        inside = (inputs >= left) & (inputs <= right)
        x = inputs.clamp(left, right)

        # bin 索引 (B,M)
        bin_idx = torch.sum(x.unsqueeze(-1) >= cumwidths[..., :-1], dim=-1) - 1
        bin_idx = bin_idx.clamp_(0, K-1)
        idx = bin_idx.unsqueeze(-1)

        x0 = torch.gather(cumwidths[..., :-1], -1, idx).squeeze(-1)
        y0 = torch.gather(cumheights[..., :-1], -1, idx).squeeze(-1)
        w_k = torch.gather(widths, -1, idx).squeeze(-1)
        h_k = torch.gather(heights, -1, idx).squeeze(-1)

        s0 = torch.gather(derivatives[..., :-1], -1, idx).squeeze(-1)
        s1 = torch.gather(derivatives[..., 1:],  -1, idx).squeeze(-1)

        t = ((x - x0) / w_k).clamp(0.0, 1.0)
        delta = h_k / w_k
        a = (s0 + s1 - 2.0 * delta)

        numerator = delta * t * t + s0 * t * (1.0 - t)
        denom = (delta + a * t * (1.0 - t)).clamp_min(1e-8)
        y = y0 + h_k * (numerator / denom)

        derivative = (delta* delta * (s1 * t * t + 2.0 * delta * t * (1.0 - t) + s0 * (1.0 - t) * (1.0 - t))) / (denom * denom)
        derivative = derivative.clamp_min(1e-12)

        outputs = inputs.clone()
        outputs[inside] = y[inside]
        logabsdet = torch.where(inside, derivative.log(), torch.zeros_like(derivative)).sum(dim=-1)
        return outputs, logabsdet

    else:
        inside = (inputs >= bottom) & (inputs <= top)
        y = inputs.clamp(bottom, top)

        bin_idx = torch.sum(y.unsqueeze(-1) >= cumheights[..., :-1], dim=-1) - 1
        bin_idx = bin_idx.clamp_(0, K-1)
        idx = bin_idx.unsqueeze(-1)

        x0 = torch.gather(cumwidths[..., :-1], -1, idx).squeeze(-1)
        y0 = torch.gather(cumheights[..., :-1], -1, idx).squeeze(-1)
        w_k = torch.gather(widths, -1, idx).squeeze(-1)
        h_k = torch.gather(heights, -1, idx).squeeze(-1)

        s0 = torch.gather(derivatives[..., :-1], -1, idx).squeeze(-1)
        s1 = torch.gather(derivatives[..., 1:],  -1, idx).squeeze(-1)

        delta = h_k / w_k
        i = (s0 + s1 - 2.0 * delta)

        a = h_k * (delta - s0) + (y - y0) * i
        b = h_k * s0 - (y-y0)*i
        c = -delta * (y-y0)

        t =  (2*c) / (-b - torch.sqrt( b*b - 4*a*c))

        x_rec = x0 + w_k * t

        outputs = inputs.clone()
        outputs[inside] = x_rec[inside]

        return outputs,0


class Conditioner(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, n_hidden: int = 2, activation=nn.ReLU):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), activation()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden, hidden), activation()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z, x,beta):

        if torch.is_tensor(beta):
            if beta.dim() == 0:
                beta = beta.view(1, 1).expand(z.size(0), 1)
            elif beta.dim() == 1:
                beta = beta.unsqueeze(-1)
            # 若给的是 (1,1) 且 batch>1，就广播
            if beta.size(0) == 1 and z.size(0) > 1:
                beta = beta.expand(z.size(0), 1)
            # 安全检查
            if beta.size(0) != z.size(0):
                raise ValueError(f"beta batch {beta.size(0)} != z batch {z.size(0)}")
        else:
            # 传进来不是张量（比如标量），转为 (B,1)
            beta = torch.tensor(beta, dtype=z.dtype, device=z.device).view(1, 1).expand(z.size(0), 1)

        inp = torch.cat([z, x,beta], dim=-1)
        return self.net(inp)

class CouplingRQS(nn.Module):

    def __init__(self, dim: int, x_dim: int, mask: torch.Tensor, K: int = 16, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.x_dim = x_dim
        self.register_buffer("mask", mask.float().view(1, -1))
        self.K = K
        self.num_trans = int((1.0 - mask).sum().item())

        out_dim = self.num_trans * (3 * K + 1)
        in_dim = int(mask.sum().item()) + x_dim +1
        self.cond = Conditioner(in_dim=in_dim, out_dim=out_dim, hidden=hidden, n_hidden=2)

        # RQS 的工作区间（unbounded 上设线性尾部）
        bound = 7.0
        self.left = -bound
        self.right = bound
        self.bottom = -bound
        self.top = bound

    def _split_params(self, params: torch.Tensor):
        # params: (B, num_trans*(3K+1)) -> (B, num_trans, K), (B, num_trans, K), (B, num_trans, K+1)
        B = params.size(0)
        t = params.view(B, self.num_trans, 3 * self.K + 1)
        w = t[..., :self.K]
        h = t[..., self.K:2*self.K]
        d = t[..., 2*self.K:]
        return w, h, d

    def forward(self, z: torch.Tensor, x: torch.Tensor, beta: torch.Tensor,inverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        m = self.mask  # (1,D)


        idx_B = (1.0 - m).bool()[0]
        idx_A = m.bool()[0]
        zA = z[:, idx_A]                         # (B, D_A)
        zB = z[:, idx_B]                         # (B, D_B=num_trans)


        params = self.cond(zA, x,beta)                # (B, num_trans*(3K+1))
        w, h, d = self._split_params(params)     # 各 (B, D_B, ...)


        zBout, logdet = rational_quadratic_spline_vec(
            inputs=zB,
            unnormalized_widths=w,
            unnormalized_heights=h,
            unnormalized_derivatives=d,
            left=self.left, right=self.right, bottom=self.bottom, top=self.top,
            inverse=inverse,
        )
        z_ = z.clone()
        z_[:, idx_B] = zBout
        return z_, logdet

# -----------------------------
# 条件样条耦合流（堆多层 + 通道置换）
# -----------------------------

def _alternating_mask(D: int, even: bool):
    m = torch.zeros(D, dtype=torch.float32)
    m[::2] = 1.0 if even else 0.0
    m[1::2] = 0.0 if even else 1.0
    return m

def _cyclic_perm(D: int, shift: int):

    return torch.roll(torch.arange(D, dtype=torch.long), shifts=shift)

class ConditionalSplineFlow(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        n_layers: int = 6,
        K: int = 16,
        hidden: int = 128,
    ):
        super().__init__()
        self.D = theta_dim
        self.x_dim = x_dim

        self.layers = nn.ModuleList()
        self.perms = nn.ParameterList()

        for i in range(n_layers):
            mask = _alternating_mask(self.D, even=(i % 2 == 0))
            layer = CouplingRQS(dim=self.D, x_dim=x_dim, mask=mask, K=K, hidden=hidden)
            self.layers.append(layer)

            shift = (i + 1) % max(1, self.D)
            perm = _cyclic_perm(self.D, shift=shift)
            self.perms.append(nn.Parameter(perm, requires_grad=False))

    def _permute(self, z: torch.Tensor, perm: torch.Tensor):
        return z[:, perm.long()]

    def _inv_permute(self, z: torch.Tensor, perm: torch.Tensor):
        inv = torch.empty_like(perm)
        inv[perm.long()] = torch.arange(perm.numel(), device=perm.device)
        return z[:, inv.long()]

    @torch.no_grad()  # 如果你希望前向也参与梯度，就移除此装饰器
    def _check_shapes(self, z: torch.Tensor, x: torch.Tensor):
        assert z.dim() == 2 and z.size(1) == self.D, f"z shape {z.shape} != (B,{self.D})"
        assert x.dim() == 2 and x.size(1) == self.x_dim, f"x shape {x.shape} != (B,{self.x_dim})"

    def forward_logprob(self, z: torch.Tensor, x: torch.Tensor,beta:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向：z -> y（base），返回 (y, log|det J|)
        """
        logdet_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        y = z
        for layer, perm in zip(self.layers, self.perms):

            y = self._permute(y, perm)
            y, logdet = layer(y, x,beta, inverse=False)
            logdet_total = logdet_total + logdet
        return y, logdet_total

    def inverse_transform(self, y: torch.Tensor, x: torch.Tensor,beta:torch.Tensor) :
        """
        反向：y（base） -> z，返回 (z, log|det J_inv|)
        注意：layer(inverse=True) 应该返回 log|dx/dy|（即前向 logdet 的相反数），这里直接累加即可。
        """
        z = y
        for layer, perm in reversed(list(zip(self.layers, self.perms))):
            # 先 inverse-coupling -> 再逆置换
            z, _ = layer(z, x, beta,inverse=True)
            z = self._inv_permute(z, perm)
        return z


# -----------------------------
# 总模型：box-squash + 条件样条耦合流
# -----------------------------
class RouteB_Flow(nn.Module):
    def __init__(self, theta_dim: int, x_dim: int, box: BoxSpec, n_layers=6, K=16, hidden=128, base='standard_normal'):
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.box = BoxSquash(box)
        self.flow = ConditionalSplineFlow(theta_dim, x_dim, n_layers=n_layers, K=K, hidden=hidden)
        self.base = base  # 'standard_normal' 或 'logistic'

    def _log_base(self, y: torch.Tensor):
        if self.base == 'standard_normal':
            # -0.5*(y^2 + D*log(2π))
            const = -0.5 * self.theta_dim * math.log(2 * math.pi)
            return const - 0.5 * (y * y).sum(dim=-1)
        elif self.base == 'logistic':
            # logistic pdf: log σ'(y) = -y - 2*softplus(-y)
            log_pdf_1d = -y - 2.0 * F.softplus(-y)
            return log_pdf_1d.sum(dim=-1)
        else:
            raise ValueError("unknown base")

    def log_prob(self, theta: torch.Tensor, x: torch.Tensor,beta: torch.Tensor) -> torch.Tensor:
        # box-squash: theta->z（加 logdet_box）
        z, logdet_box = self.box.theta_to_z(theta)
        # 条件流：z->y（加 logdet_flow）
        y, logdet_flow = self.flow.forward_logprob(z, x,beta)
        # 基分布
        log_base = self._log_base(y)
        return log_base + logdet_flow + logdet_box

    @torch.no_grad()
    def sample(self, num_samples: int, x: torch.Tensor, beta,device=None) -> torch.Tensor:
        device = device or x.device
        x = x.float().to(device)
        if x.dim() == 1:  # ← 新增
            x = x.unsqueeze(0)  # 变成 (1, Dx)
        if x.dim() == 2 and x.size(0) == 1:
            x = x.expand(num_samples, -1)
        elif x.size(0) != num_samples:
            x = x[torch.randint(0, x.size(0), (num_samples,), device=device)]

        if not torch.is_tensor(beta):
            beta = torch.tensor(beta, dtype=torch.float32, device=device)
        else:
            beta = beta.to(device).float()

        if beta.dim() == 0:
            beta = beta.view(1, 1).expand(num_samples, 1)
        elif beta.dim() == 1:
            if beta.numel() == 1:
                beta = beta.view(1, 1).expand(num_samples, 1)
            elif beta.numel() == num_samples:
                beta = beta.view(num_samples, 1)
            else:
                idx = torch.randint(0, beta.numel(), (num_samples,), device=device)
                beta = beta[idx].view(num_samples, 1)
        elif beta.dim() == 2:
            if beta.size(0) == 1 and beta.size(1) == 1:
                beta = beta.expand(num_samples, 1)
            elif not (beta.size(0) == num_samples and beta.size(1) == 1):
                raise ValueError("beta 必须是标量、(N,) 或 (N,1)")
        else:
            raise ValueError("beta 必须是标量、(N,) 或 (N,1)")


        if self.base == 'standard_normal':
            y = torch.randn(num_samples, self.theta_dim, device=device)
        else:
            # Logistic: inverse-CDF 采样
            u = torch.rand(num_samples, self.theta_dim, device=device).clamp(1e-6, 1-1e-6)
            y = torch.log(u) - torch.log(1.0 - u)

        z = self.flow.inverse_transform(y, x,beta)
        theta, _ = self.box.z_to_theta(z)
        return theta



# -----------------------------
# 数据集与训练/验证
# ----------------------------~
def make_loaders(theta, x, weights, beta, batch_size=1024, val_fraction=0.2):
    theta = theta.cpu(); x = x.cpu();beta = beta.cpu()
    if weights is None:
        weights = torch.ones(theta.size(0), dtype=torch.float32)
    else:
        weights = weights.detach().float().cpu()

    beta = beta.detach().float().cpu().view(-1, 1)  # <-- 关键：保证二维

    N = theta.size(0)
    N_val = max(1, int(N * val_fraction))
    perm = torch.randperm(N)  # CPU
    id_val, id_tr = perm[:N_val], perm[N_val:]

    ds_tr  = TensorDataset(theta[id_tr], x[id_tr], weights[id_tr],beta[id_tr])
    ds_val = TensorDataset(theta[id_val], x[id_val], weights[id_val],beta[id_val])

    dl_tr = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  drop_last=True,
                       num_workers=4, pin_memory=True, persistent_workers=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False,
                       num_workers=2, pin_memory=True, persistent_workers=True)
    return dl_tr, dl_val


def train_epoch(model: RouteB_Flow,
                loader: DataLoader,
                opt: torch.optim.Optimizer,
                grad_clip: float = 5.0, device=None):
    model.train()
    tot, n = 0.0, 0
    for th, x, w ,beta in loader:
        th = th.to(device)
        x = x.to(device)
        w = w.to(device)
        beta = beta.to(device)


        lp = model.log_prob(th, x,beta)  # (B,)
        loss = -(w * lp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        tot += loss.item() * th.size(0); n += th.size(0)
    return tot / max(1, n)

@torch.no_grad()
def eval_epoch(model: RouteB_Flow, loader: DataLoader, device=None):
    model.eval()
    tot, n = 0.0, 0
    for th, x, w,beta in loader:
        th = th.to(device)
        x = x.to(device)
        w = w.to(device)
        beta = beta.to(device)


        lp = model.log_prob(th, x,beta)
        loss = -(w * lp).mean()
        tot += loss.item() * th.size(0)
        n += th.size(0)
    return tot / n

# -----------------------------
# Route-B：给定 β 的单网训练
# -----------------------------
@dataclass
class FlowConfig:
    n_layers: int = 6
    K: int = 16
    hidden: int = 128
    base: str = 'standard_normal'

def train_single_beta_flow(
    theta_train: torch.Tensor,  # (N, Dθ) 原空间（在盒内）
    x_train: torch.Tensor,      # (N, Dx)
    weights: Optional[torch.Tensor],  # (N,) 对应 β 的样本权重（均值≈1 更好）
    beta:torch.Tensor,
    box: BoxSpec,
    *,
    cfg: FlowConfig = FlowConfig(),
    batch_size: int = 1024,
    epochs: int = 200,
    patience: int = 30,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    plot = True,
):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    theta_train = theta_train.detach().float().to(device)
    x_train = x_train.detach().float().to(device)
    w = None if weights is None else weights.detach().float().to(device)

    train_hist = []
    val_hist = []

    model = RouteB_Flow(
        theta_dim=theta_train.size(1),
        x_dim=x_train.size(1),
        box=box,
        n_layers=cfg.n_layers,
        K=cfg.K,
        hidden=cfg.hidden,
        base=cfg.base,
    ).to(device)

    # 切分与优化器
    dl_tr, dl_val = make_loaders(theta_train, x_train, w, beta,batch_size=batch_size, val_fraction=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val, best_sd, bad = float("inf"), None, 0
    for ep in tqdm(range(1, epochs+1)):
        train_loss = train_epoch(model, dl_tr, opt, grad_clip=5.0, device=device)
        val_loss = eval_epoch(model, dl_val, device=device)
        train_hist.append(train_loss)
        val_hist.append(val_loss)
        if val_loss < best_val - 1e-4:
            best_val, bad = val_loss, 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"[early stop] epoch={ep} best_val={best_val:.4f}")
                break
        if ep % 30 == 0 or ep == 1:
            print(f"[ep {ep:03d}] train {train_loss:.4f} | val {val_loss:.4f}")
    if best_sd is not None:
        model.load_state_dict(best_sd)

    if plot:
        epochs_ran = np.arange(1, len(train_hist) + 1)
        tr_s = _ema(train_hist, k=15)
        va_s = _ema(val_hist, k=15)

        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)

        ax.plot(epochs_ran, tr_s, linewidth=2, label="train NLL", antialiased=True)
        ax.plot(epochs_ran, va_s, linewidth=2, label="val NLL", antialiased=True)

        step = max(1, len(epochs_ran) // 60)
        ax.scatter(epochs_ran[::step], np.array(train_hist)[::step], s=12, alpha=0.35)
        ax.scatter(epochs_ran[::step], np.array(val_hist)[::step], s=12, alpha=0.35)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("NLL")

        plt.title(f"H:{cfg.hidden},L:{cfg.n_layers},K:{cfg.K},BS:{batch_size}")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()

    return model

# -----------------------------
# 保存 / 加载
# -----------------------------
def save_flow(path: str, model: RouteB_Flow, meta: Optional[dict] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    ckpt = {
        "state_dict": model.state_dict(),
        "theta_dim": model.theta_dim,
        "x_dim": model.x_dim,
        "box_low": model.box.low.cpu(),
        "box_high": model.box.high.cpu(),
        "cfg": {
            "base": model.base,
            "n_layers": len(model.flow.layers),
            "K": model.flow.layers[0].K,
            "hidden": model.flow.layers[0].cond.net[0].out_features if isinstance(model.flow.layers[0].cond.net[0], nn.Linear) else 128
        },
        "meta": meta or {}
    }
    torch.save(ckpt, path)

def load_flow(path: str, device: Optional[torch.device] = None) -> Tuple[RouteB_Flow, dict]:
    ckpt = torch.load(path, map_location="cpu")
    box = BoxSpec(low=ckpt["box_low"], high=ckpt["box_high"])
    cfg = FlowConfig(
        n_layers=ckpt["cfg"]["n_layers"],
        K=ckpt["cfg"]["K"],
        hidden=ckpt["cfg"]["hidden"],
        base=ckpt["cfg"]["base"],
    )
    model = RouteB_Flow(theta_dim=ckpt["theta_dim"], x_dim=ckpt["x_dim"], box=box,
                        n_layers=cfg.n_layers, K=cfg.K, hidden=cfg.hidden, base=cfg.base)
    model.load_state_dict(ckpt["state_dict"])
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()
    return model, ckpt.get("meta", {})

from sbi.inference import SNLE
from sbi.utils import BoxUniform

from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import plot_posteriors_compare,to_float,ExperimentLogger
from pathlib import Path


from sbibm import get_task

from RouteB_utils import (
    log_r_hat,
    get_routeB_samples,
)

if __name__ == "__main__":

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # name = ["gaussian_mixture", "two_moons", "slcp", "sir"]
    # number = 0
    # n = 10_000
    #
    # task = get_task(name[number])
    # prior = task.get_prior()
    # simulator = task.get_simulator()
    # theta_dim, x_dim = task.dim_parameters, task.dim_data
    # save_dir = Path(name[number])

    # nle_ckpt = save_dir / f"NLE_ckpt_{n}.pt"
    # metrics_csv = save_dir / "metrics.csv"
    # logger = ExperimentLogger(metrics_csv)
    # method = "RouteB_NLE"
    #
    # with torch.no_grad():
    #     theta_train = prior(n)
    #     x_train = simulator(theta_train).to(device)
    #     theta_train = theta_train.to(device)
    #
    # inference_L = SNLE(prior=None, density_estimator="mdn")
    # load_model_NRE = False

    # if number == 0:
    #     low = torch.tensor([-10.0, -10.0], device=device)
    #     high = torch.tensor([10.0, 10.0], device=device)
    # elif number == 1:
    #     low = torch.tensor([-1.0, -1.0], device=device)
    #     high = torch.tensor([1.0, 1.0], device=device)
    # elif number == 2:
    #     low = torch.tensor([-3.0, -3.0, -3.0, -3.0, -3.0], device=device)
    #     high = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0], device=device)

    # prior = BoxUniform(low=low, high=high)
    # box = BoxSpec(low=low, high=high)
    #
    # if not load_model_NRE:
    #     inference_L = inference_L.append_simulations(theta_train.cpu(), x_train.cpu())
    #
    #     density_estimator = inference_L.train(training_batch_size=512,
    #                                           validation_fraction=0.1,
    #                                           max_num_epochs=3000,
    #                                           show_train_summary=False,
    #                                           stop_after_epochs=50)
    #
    #     posterior_r = inference_L.build_posterior(prior=prior, density_estimator=density_estimator)
    #
    #     torch.save(density_estimator.state_dict(), nle_ckpt)
    # else:
    #     with torch.no_grad():
    #         dummy_theta = torch.randn(256, theta_dim) * 0.1
    #         dummy_x = torch.randn(256, x_dim) * 0.1
    #     inference_L.append_simulations(dummy_theta, dummy_x)
    #     density_estimator = inference_L.train(stop_after_epochs=0)
    #
    #     density_estimator.load_state_dict(torch.load(nle_ckpt, map_location=device))
    #     density_estimator.to("cpu")
    #     posterior_r = inference_L.build_posterior(prior=prior, density_estimator=density_estimator)
    #
    lamda = 0.9999
    change_proposal = False
    #
    # flow_ckpt = save_dir / f"routeB_NRE_weighted_npe_{n}.pt"
    #
    # betas_sets = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
    betas_sets = [1.0]

    # load_model_NPE = False
    #
    # with torch.no_grad():
    #     log_r_all = log_r_hat(theta_train, x_train, prior=prior, posterior_r=posterior_r)
    #
    # theta_train = theta_train.to(device).float()
    # x_train = x_train.to(device).float()
    # log_r_all = log_r_all.to(device).float()
    #
    # theta_joint, x_joint, beta_joint, w_joint = get_routeB_samples(theta_train, x_train, log_r_all,
    #                                                                betas_sets, change_proposal=change_proposal,
    #                                                                lamda=lamda)
    # print("Joint dataset:", theta_joint.shape, x_joint.shape, beta_joint.shape, w_joint.shape)


    import torch
    from lorenz96 import PriorBox, make_prior_fn, make_simulator_fn, Theta

    prior_box = PriorBox(1.4, 2.2, 0.1, 1.0, 1.5, 2.5)

    prior = make_prior_fn(prior_box, device=device, dtype=torch.float32)
    K, T, dt = 8, 20, 3.0 / 40.0
    x0 = torch.ones(K, device=device)
    simulator = make_simulator_fn(K, T, dt, R=5, x0=x0, device=device, dtype=torch.float32, out_device="cpu")
    save_dir = Path("lorenz96")
    theta_dim, x_dim = 3, 840

    low = torch.tensor([1.4, 0.1, 1.5], device=device)
    high = torch.tensor([2.2, 1.0, 2.5], device=device)

    theta_true = torch.tensor([[2.0, 0.8, 1.7]])
    x_obs = simulator(theta_true, seed=42).to(device)
    n = 10_000

    theta_train = prior(n)
    x_train = simulator(theta_train)
    box = BoxSpec(low=low, high=high)
    log_r_all = torch.ones(n,1)

    theta_train = theta_train.to(device).float()
    x_train = x_train.to(device).float()
    log_r_all = log_r_all.to(device).float()

    theta_joint, x_joint, beta_joint, w_joint = get_routeB_samples(theta_train, x_train, log_r_all,
                                                                   betas_sets, change_proposal=change_proposal,
                                                                   lamda=lamda)

    print("Joint dataset:", theta_joint.shape, x_joint.shape, beta_joint.shape, w_joint.shape)
    # === 训练单张 flow ===
    cfg = FlowConfig(n_layers=8, K=24, hidden=128, base='logistic')
    model = train_single_beta_flow(
        theta_train=theta_joint, x_train=x_joint, weights=None,beta=beta_joint, box=box,
        cfg=cfg, batch_size=256, epochs=500, patience=30, lr=1e-3, device=device
    )

    for beta_scalar in betas_sets:
        with torch.no_grad():
            # x_obs = task.get_observation(num_observation=1)
            posterior_samples =model.sample(
                                 num_samples=10_000,x = x_obs,beta=beta_scalar,
                                 device=device)

            ckpt_path = save_dir / f"True_Power_posterior_with_t_power={beta_scalar}_10000.pt"
            reference_samples = torch.load(ckpt_path, map_location=device)

            c2st_accuracy = to_float(c2st(reference_samples, posterior_samples))
            mmd_value = to_float(mmd(reference_samples, posterior_samples))
            print(f"c2st:{c2st_accuracy:.4f}")
            print(f"MMD: {mmd_value:.4g}")
            print(posterior_samples.shape)
            print(reference_samples.shape)

            plot_posteriors_compare(reference_samples, posterior_samples=posterior_samples,
                                    title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={beta_scalar}")

