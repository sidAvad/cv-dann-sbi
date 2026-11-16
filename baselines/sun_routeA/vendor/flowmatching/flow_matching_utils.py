from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, depth: int,
                 activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden),
            activation,
        )

        n_res = max(0, depth - 2)
        self.res_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                activation,
                nn.Linear(hidden, hidden)
            ) for _ in range(n_res)
        ])

        for blk in self.res_layers:
            nn.init.zeros_(blk[-1].weight)
            nn.init.zeros_(blk[-1].bias)

        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for blk in self.res_layers:
            h = h +  blk(h)
        y = self.out(h)
        return y

class ConditionalVectorField(nn.Module):
    def __init__(self, theta_dim: int, x_dim: int,
                 hidden: int = 256, depth: int = 4,
                 x_feat_dim: int = 256, x_enc_hidden: int = 128,
                 x_enc_depth: int = 2):
        super().__init__()
        self.x_feat_dim = x_feat_dim
        self.x_encoder = MLP(in_dim=x_dim, out_dim=x_feat_dim,
                             hidden=x_enc_hidden, depth=x_enc_depth)
        if self.x_feat_dim > 0:
            in_dim = theta_dim + x_feat_dim + 1
        else:
            in_dim = theta_dim + x_dim + 1

        self.backbone = MLP(in_dim=in_dim, out_dim=theta_dim,
                            hidden=hidden, depth=depth)

    def forward(self, theta_t: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.x_feat_dim > 0 :
            x = self.x_encoder(x)

        h = torch.cat([theta_t, t, x], dim=-1)
        return self.backbone(h)


# ======================= Loss (paper style CFM) =======================
def _approx_ot_pair(z0: torch.Tensor, z1: torch.Tensor, n_proj: int = 8):
    """
    小批 sliced-OT 近似：多投影聚合名次后排序，返回
      z0_perm, z1_perm, idx1
    其中 idx1 是 z1 的置换索引，用于同步重排 x。
    """
    B, D = z0.shape
    device = z0.device
    rank0 = torch.zeros(B, device=device)
    rank1 = torch.zeros(B, device=device)
    for _ in range(max(1, n_proj)):
        w = torch.randn(D, device=device)
        r0 = (z0 @ w).argsort()
        r1 = (z1 @ w).argsort()
        rank0[r0] += torch.arange(B, device=device, dtype=torch.float32)
        rank1[r1] += torch.arange(B, device=device, dtype=torch.float32)
    s0 = rank0.argsort()
    s1 = rank1.argsort()
    return z0[s0], z1[s1], s1  # <- 返回 s1，别丢！



def cfm_loss_paper(v_model: nn.Module,
                   theta: torch.Tensor,
                   x: torch.Tensor,
                   ref_sampler: Callable[[int], torch.Tensor],
                   pairing: str = "independent",
                   sigma_min : float = 0.005,
                   ot_projections: int = 8) -> torch.Tensor:
    """
    论文风格 CFM（线性路径 + 均匀 t）。开启 OT 时要同步重排 x！
    """
    B = theta.shape[0]
    device = theta.device
    t = torch.rand(B, 1, device=device)          # U(0,1)

    if pairing == "ot":
        sigma = t * (sigma_min - 1)  + 1
        eps = torch.randn_like(theta)

        z_t = t*theta + eps * sigma
        target_v = (theta - (1 - sigma_min)*z_t) /(1- (1 - sigma_min)*t + 1e-8)
        pred_v = v_model(z_t,t,x)
        return F.mse_loss(pred_v, target_v)

    z0 = ref_sampler(B).to(device)
    z_t = (1.0 - t) * z0 + t * theta
    target_v = theta - z0
    pred_v = v_model(z_t, t, x)
    return F.mse_loss(pred_v, target_v)

# ======================= Config/Trainer =======================
@dataclass
class FMPEConfig:
    theta_dim: int
    x_dim: int
    hidden: int = 256
    depth: int = 4
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    pairing: str = "independent"   # or "ot"
    device: str = None
    x_feat_dim: int  = None
    x_enc_hidden: int = 256
    x_enc_depth: int = 2

class FMPE:
    """离线训练：先生成 (theta, x) 数据，再训练/验证（不打印、不保存，history 留好）。"""
    def __init__(self, cfg: FMPEConfig,
                 ref_sampler: Optional[Callable[[int], torch.Tensor]] = None
                ):
        self.cfg = cfg
        self.device = cfg.device
        self.ref_sampler = ref_sampler or (lambda B: torch.randn(B, cfg.theta_dim))
        self.model = ConditionalVectorField(cfg.theta_dim, cfg.x_dim, cfg.hidden, cfg.depth,
                                            cfg.x_feat_dim,cfg.x_enc_hidden,cfg.x_enc_depth).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.history = []
        self.step = 0

    def evaluate(self, loader: Optional[DataLoader]) -> float:
        if loader is None: return float('nan')
        self.model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for th, xx in loader:
                th = th.to(self.device); xx = xx.to(self.device)
                loss = cfm_loss_paper(self.model, th, xx,
                                      lambda B: self.ref_sampler(B).to(self.device),
                                      pairing=self.cfg.pairing)
                bs = th.shape[0]; tot += loss.item() * bs; n += bs
        self.model.train()
        return tot / max(1, n)

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 50, early_stop_patience: Optional[int] = None,
             ckpt_path: Optional[str] = None):
        best_val, wait = float('inf'), 0
        for ep in range(1, epochs + 1):
            run, n = 0.0, 0
            for th, xx in train_loader:
                self.step += 1
                th = th.to(self.device); xx = xx.to(self.device)
                loss = cfm_loss_paper(self.model, th, xx,
                                      lambda B: self.ref_sampler(B).to(self.device),
                                      pairing=self.cfg.pairing)
                self.opt.zero_grad(set_to_none=True); loss.backward()
                if self.cfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                self.opt.step()
                bs = th.shape[0]
                run += loss.item()* bs; n += bs

            train_loss = run / n
            val_loss = self.evaluate(val_loader)
            self.history.append({"epoch": ep, "train_loss": float(train_loss), "val_loss": float(val_loss)})

            if ep > 50:
                if val_loader is not None:
                    improved = val_loss < best_val - 1e-8
                    if improved:
                        best_val = val_loss; wait = 0
                        if ckpt_path is not None:
                            torch.save({"model": self.model.state_dict(),
                                        "opt": self.opt.state_dict(),
                                        "cfg": self.cfg,
                                        "history": self.history}, ckpt_path)
                    else:
                        wait += 1
                        if early_stop_patience is not None and wait >= early_stop_patience:
                            break

    @torch.no_grad()
    def sample_posterior(self, x_obs: torch.Tensor, n_samples=1000, ode_steps=400, ref: Optional[torch.Tensor]=None):
        self.model.eval()
        if x_obs.dim() == 1: x_obs = x_obs.unsqueeze(0)
        x_obs = x_obs.to(self.device)
        z0 = ref.to(self.device) if ref is not None else self.ref_sampler(n_samples).to(self.device)
        dt = 1.0 / ode_steps
        z = z0
        for i in range(ode_steps):
            N = z.shape[0]
            t  = torch.full((N,1), i      / ode_steps, device=self.device)
            tm = torch.full((N,1), (i+0.5)/ ode_steps, device=self.device)
            te = torch.full((N,1), (i+1.0)/ ode_steps, device=self.device)
            k1 = self.model(z,             t,  x_obs.expand(N, -1))
            k2 = self.model(z + 0.5*dt*k1, tm, x_obs.expand(N, -1))
            k3 = self.model(z + 0.5*dt*k2, tm, x_obs.expand(N, -1))
            k4 = self.model(z +     dt*k3, te, x_obs.expand(N, -1))
            z = z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        theta = z
        return theta

# ======================= Dataset helper =======================
class PairDataset(Dataset):
    def __init__(self, prior: Callable[[int], torch.Tensor], simulator: Callable[[torch.Tensor], torch.Tensor],
                 n: int, device):
        th = prior(n); x = simulator(th)
        self.theta = torch.as_tensor(th, dtype=torch.float32, device=device)
        self.x     = torch.as_tensor(x,   dtype=torch.float32, device=device)
    def __len__(self): return self.theta.shape[0]
    def __getitem__(self, i): return self.theta[i], self.x[i]

# ======================= Plot helper (可选) =======================
def plot_losses(history, save_path: Optional[str] = None, show: bool = True, title="FMPE training"):
    import matplotlib.pyplot as plt, math
    ep = [h["epoch"] for h in history]
    tr = [h["train_loss"] for h in history]
    va = [h.get("val_loss", float("nan")) for h in history]
    plt.figure(); plt.plot(ep, tr, label="train")
    if not all(isinstance(v, float) and math.isnan(v) for v in va): plt.plot(ep, va, label="val")
    plt.xlabel("epoch"); plt.ylabel("CFM loss"); plt.title(title); plt.legend(); plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=150)
    if show: plt.show(); plt.close()


