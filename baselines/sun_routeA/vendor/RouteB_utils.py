import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# -------------------------
# Standardizer
# -------------------------
@dataclass
class ZScore:
    mu: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, x: torch.Tensor, eps: float = 1e-8):
        x = x.detach().float()
        mu = x.mean(0)
        std = x.std(0) + eps
        return cls(mu, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, std = self.mu.to(x.device), self.std.to(x.device)
        return (x - mu) / std

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        mu, std = self.mu.to(z.device), self.std.to(z.device)
        return z * std + mu


def _ema(y, k=15):
    # 指数滑动平均：k 越大越平滑
    y = np.asarray(y, dtype=float)
    a = 2.0 / (k + 1.0)
    s = np.empty_like(y)
    s[0] = y[0]
    for i in range(1, len(y)):
        s[i] = (1 - a) * s[i-1] + a * y[i]
    return s



class ResBlock(nn.Module):
    def __init__(self, hidden: int, activation=nn.ReLU):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = activation()
        self.fc2 = nn.Linear(hidden, hidden)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


    def forward(self, x):
        h = self.ln(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return x + h


class ConditionalMDN(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        beta_dim: int = 1,
        hidden: int = 128,
        n_components: int = 4,
        n_hidden_layers: int = 3,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.theta_dim = theta_dim
        self.x_dim = x_dim
        self.beta_dim = beta_dim
        self.K = n_components
        self.hidden = hidden

        self.in_proj = nn.Sequential(
            nn.Linear(x_dim, hidden), activation()
        )

        n_blocks = max(n_hidden_layers - 1, 0)
        self.blocks = nn.ModuleList([ResBlock(hidden, activation) for _ in range(n_blocks)])

        self.film = nn.Sequential(
            nn.Linear(beta_dim, hidden),
            activation(),
            nn.Linear(hidden, hidden * 2)
        )

        self.pi_head = nn.Linear(hidden, self.K)
        self.mu_head = nn.Linear(hidden, self.K * theta_dim)
        self.log_sigma_head = nn.Linear(hidden, self.K * theta_dim)

        self.logsigma_min = -7.0
        self.logsigma_max = 10.0

    def _backbone(self, x: torch.Tensor):

        h = self.in_proj(x)
        for blk in self.blocks:
            h = blk(h)
        return h

    def _params(self, x, beta):
        h = self._backbone(x.float())
        if beta.dim() == 1:
            beta = beta.unsqueeze(-1)

        gamma_shift = self.film(beta)
        gamma, shift = torch.chunk(gamma_shift, 2, dim=-1)
        gamma = torch.tanh(gamma) * 2.0

        h_mod = gamma * h + shift

        logit_pi = self.pi_head(h_mod)
        mu = self.mu_head(h_mod).view(-1, self.K, self.theta_dim)
        log_sigma = self.log_sigma_head(h_mod).view(-1, self.K, self.theta_dim)
        log_sigma = torch.clamp(log_sigma, self.logsigma_min, self.logsigma_max)
        return logit_pi, mu, log_sigma

    def log_prob(self, theta, context):
        theta = theta.float()
        x, beta = context
        x = x.float()
        B, D = theta.shape
        logit_pi, mu, log_sigma = self._params(x, beta)

        inv_var = torch.exp(-2.0 * log_sigma)
        diff2 = (theta.unsqueeze(1) - mu) ** 2
        quad = -0.5 * (diff2 * inv_var).sum(dim=-1)
        log_det = log_sigma.sum(dim=-1)
        const = -0.5 * D * math.log(2 * math.pi)
        log_norm = const - log_det + quad
        log_pi = F.log_softmax(logit_pi, dim=-1)
        return torch.logsumexp(log_pi + log_norm, dim=-1)

    @torch.no_grad()
    def sample(self, num_samples, context):
        x, beta = context
        x = x.float()
        beta = beta.float()
        if x.dim() == 2 and x.size(0) == 1:
            x = x.expand(num_samples, -1)
            beta = beta.expand(num_samples, -1)
        logit_pi, mu, log_sigma = self._params(x, beta)
        comp = torch.distributions.Categorical(logits=logit_pi).sample()
        idx = comp.view(-1, 1, 1).expand(-1, 1, self.theta_dim)
        chosen_mu = mu.gather(1, idx).squeeze(1)
        chosen_log_sigma = log_sigma.gather(1, idx).squeeze(1)
        eps = torch.randn_like(chosen_mu)
        return chosen_mu + eps * torch.exp(chosen_log_sigma)


# -------------------------
# Build per-sample weights from log r and beta
# -------------------------

@torch.no_grad()
def log_r_hat(theta: torch.Tensor, x: torch.Tensor,prior,posterior_r) -> torch.Tensor:
    """ return (N,)：log r(x_i|theta_i) = log p(theta_i|x_i) - log pi(theta_i) """
    dev = theta.device
    out_list = []
    theta = theta.detach().cpu()
    x = x.detach().cpu()
    lp_list = []
    for j in range(theta.shape[0]):
        lp_ij = posterior_r.log_prob(theta[j:j+1],
                                     x=x[j:j+1])
        lp_list.append(lp_ij)
    log_post = torch.cat(lp_list, dim=0) # (B,)
    log_prior = prior.log_prob(theta) # (B,)

    out_list.append(log_post - log_prior) # (B,)
    out = torch.cat(out_list, dim=0) # (N,)
    return out.to(dev)


@torch.no_grad()
def build_weights_from_logr(
    log_r_all: torch.Tensor,     # (N,)
    beta: torch.Tensor,          # (N,) or (1,) or (N,1)/(1,1)
    change_proposal = False,
    lamda = 0.1,) -> torch.Tensor:


    log_r_all = log_r_all.view(-1).float()
    if beta.dim() == 2 and beta.size(1) == 1:
        beta = beta.view(-1)
    if beta.numel() == 1:
        beta = beta.expand_as(log_r_all)
    assert beta.shape == log_r_all.shape, "beta must align with log_r_all elementwise"

    if change_proposal:
        lw = beta * log_r_all - torch.log((1-lamda)*torch.exp(log_r_all) + lamda)
    else:
        lw = (beta - 1.0) * log_r_all

    lw = lw - lw.max()  # stabilize
    w = torch.exp(lw)

    w = w / w.mean()
    print(w.size(0)/(w ** 2).sum())
    return w

def get_routeB_samples(theta_train,x_train,log_r_all,betas_set,change_proposal,lamda):
    dev = theta_train.device
    thetas, xs, betas, weights = [], [], [], []
    for b in betas_set:
        thetas.append(theta_train)
        xs.append(x_train)
        beta_vec = torch.full((theta_train.size(0),), float(b), device=dev)
        betas.append(beta_vec)
        w_b = build_weights_from_logr(log_r_all, beta_vec,change_proposal=change_proposal,lamda=lamda)
        weights.append(w_b)

    theta_joint = torch.cat(thetas, dim=0)           # (N*|B|, Dθ)
    x_joint     = torch.cat(xs, dim=0)               # (N*|B|, Dx)
    beta_joint  = torch.cat(betas, dim=0)            # (N*|B|,)
    w_joint     = torch.cat(weights, dim=0)          # (N*|B|,)

    return theta_joint,x_joint,beta_joint,w_joint

# -------------------------
# Train function (weighted conditional MLE on FIXED dataset)

def train_conditional_mdn(
    theta: torch.Tensor,           # (N, Dθ)
    x: torch.Tensor,               # (N, Dx)
    beta: torch.Tensor,            # (N,) or (N,1)
    weights: Optional[torch.Tensor] = None,  # (N,)
    *,
    hidden: int = 128,
    n_components: int = 8,
    n_hidden_layers: int = 2,
    batch_size: int = 512,
    epochs: int = 200,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    grad_clip: float = 10.0,
    device: Optional[torch.device] = None,
    verbose: bool = False,
    patience: int = 20,
    fixed_split: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    plot = True,
) -> Tuple[torch.nn.Module, "ZScore", "ZScore", float]:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta = theta.detach().float()
    x = x.detach().float()
    beta = beta.view(-1, 1).detach().float()   # ensure shape (N,1)
    N, Dθ = theta.shape
    Dx = x.shape[1]

    # Standardize
    norm_theta = ZScore.fit(theta)
    norm_x = ZScore.fit(x)
    theta_z = norm_theta.forward(theta)
    x_z = norm_x.forward(x)

    train_hist = []
    val_hist = []

    if weights is None:
        weights = torch.ones(N, dtype=torch.float32)
    else:
        weights = weights.detach().float()

    if fixed_split is None:
        N_val = max(1, int(N * val_fraction))
        perm = torch.randperm(N)
        id_val = perm[:N_val]
        id_tr = perm[N_val:]
    else:
        id_tr, id_val = fixed_split

    th_tr, x_tr, b_tr, w_tr = theta_z[id_tr], x_z[id_tr], beta[id_tr], weights[id_tr]
    th_val, x_val, b_val, w_val = theta_z[id_val], x_z[id_val], beta[id_val], weights[id_val]

    dl_tr = DataLoader(TensorDataset(th_tr, x_tr, b_tr, w_tr),
                       batch_size=batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(TensorDataset(th_val, x_val, b_val, w_val),
                        batch_size=batch_size, shuffle=False, drop_last=False)

    model = ConditionalMDN(
        theta_dim=Dθ,
        x_dim=Dx,
        beta_dim=1,
        hidden=hidden,
        n_components=n_components,
        n_hidden_layers=n_hidden_layers,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.9, patience=5, min_lr=lr * 5e-3
    )

    best_val = float("inf")
    best_sd: Dict[str, torch.Tensor] = {}
    bad = 0

    for ep in tqdm(range(1, epochs + 1)):
        model.train()
        tr_losses = []
        for th_b, x_b, b_b, w_b in dl_tr:
            th_b = th_b.to(device)
            x_b = x_b.to(device)
            b_b = b_b.to(device)
            w_b = w_b.to(device)

            logq = model.log_prob(th_b, (x_b, b_b))      # (B,)
            loss = -(w_b * logq ).mean()
            opt.zero_grad()
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            tr_losses.append(loss.item())

        # ---- val ----
        model.eval()
        with torch.no_grad():
            val_losses = []
            for th_b, x_b, b_b, w_b in dl_val:
                th_b = th_b.to(device)
                x_b = x_b.to(device)
                b_b = b_b.to(device)
                w_b = w_b.to(device)

                logq = model.log_prob(th_b, (x_b, b_b))
                loss = -(w_b * logq).mean()
                val_losses.append(loss.item())
            val_nll = float(np.mean(val_losses))
            train_nll = float(np.mean(tr_losses))

        train_hist.append(train_nll)
        val_hist.append(val_nll)
        scheduler.step(val_nll)

        if verbose and (ep % 20 == 0 or ep == 1):
            print(f"[{ep:03d}] train NLL={train_nll:.6f} | val NLL={val_nll:.6f}")

        if val_nll < best_val - 1e-4:
            best_val = val_nll
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"Early stop at epoch {ep}, best val NLL={best_val:.6f}")
                break

    if best_sd:
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

        plt.title(f"{hidden},K:{n_components},L:{n_hidden_layers},BS:{batch_size}")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        ax.legend(frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.show()


    return model, norm_theta, norm_x, best_val


# -------------------------
# Inference sampling (with normalizers)
# -------------------------
@torch.no_grad()
def mdn_sample(
    model: ConditionalMDN,
    x_obs: torch.Tensor,
    beta: float,
    norm_theta: "ZScore",
    norm_x: "ZScore",
    num_samples: int = 5000,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    if device is None:
        device = next(model.parameters()).device

    x_obs = x_obs.detach().float().reshape(1, -1).to(device)
    beta_t = torch.tensor([[beta]], device=device, dtype=torch.float32)

    x_z = norm_x.forward(x_obs)
    z_samples = model.sample(num_samples, (x_z, beta_t))
    theta_samples = norm_theta.inverse(z_samples)
    return theta_samples


@torch.no_grad()
def sample_with_rejection(
    model: ConditionalMDN,
    norm_theta: "ZScore",
    norm_x: "ZScore",
    x_obs: torch.Tensor,
    beta: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    num_samples: int = 10_000,
    oversample: float = 1.5,
    max_rounds: int = 5,
    device=None,
):

    if device is None:
        device = next(model.parameters()).device

    keep = []
    need = num_samples
    last_drawn = None
    for _ in range(max_rounds):
        n_try = int(max(need * oversample, need))
        th = mdn_sample(model, x_obs, beta, norm_theta, norm_x, num_samples=n_try, device=device)  # (n_try, Dθ)
        last_drawn = th
        mask = (th >= lower) & (th <= upper)
        mask = mask.all(dim=1)
        if mask.any():
            keep.append(th[mask])
            need = num_samples - sum(k.size(0) for k in keep)
            if need <= 0:
                break
    if len(keep) == 0:
        return last_drawn[:num_samples]
    out = torch.cat(keep, dim=0)
    return out[:num_samples]




# -------------------------
# Save / Load
# -------------------------


def save_checkpoint(
    path: str,
    model: ConditionalMDN,
    norm_theta: ZScore,
    norm_x: ZScore,
    meta: Optional[dict] = None,
):
    """
    Save model + normalizers + metadata.
    Stores an `arch` dict containing x_dim and beta_dim so loading can reconstruct the model.
    """
    # prefer storing CPU copies for portability
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "theta_mu": norm_theta.mu.cpu(),
        "theta_std": norm_theta.std.cpu(),
        "x_mu": norm_x.mu.cpu(),
        "x_std": norm_x.std.cpu(),
        "meta": meta or {},
        "arch": {
            "theta_dim": int(model.theta_dim),
            "x_dim": int(model.x_dim),
            "beta_dim": int(getattr(model, "beta_dim", 1)),
            "n_components": int(model.K),
            # optional: preserve hidden / n_hidden_layers if present on model
            "hidden": int(getattr(model, "hidden", -1)),
            "n_hidden_layers": int(getattr(model, "n_hidden_layers", -1)),
        }
    }
    torch.save(ckpt, path)


def load_checkpoint(path: str, hidden: Optional[int] = None, n_hidden_layers: Optional[int] = None, device=None):
    """
    Load a checkpoint saved by save_checkpoint.

    If `hidden` or `n_hidden_layers` are None, the function will use the values stored in the ckpt.arch when available.
    Returns: model (on device, eval), norm_theta, norm_x, meta.
    """
    ckpt = torch.load(path, map_location="cpu")

    arch = ckpt.get("arch", {})
    theta_dim = int(arch.get("theta_dim", arch.get("context_dim", None)))
    x_dim = int(arch.get("x_dim", arch.get("context_dim", None)))  # fallback to context_dim if old ckpt
    beta_dim = int(arch.get("beta_dim", 1))
    n_components = int(arch.get("n_components", arch.get("n_components", 4)))

    # prefer saved hidden/n_hidden_layers unless caller overrides
    saved_hidden = arch.get("hidden", None)
    saved_n_hidden_layers = arch.get("n_hidden_layers", None)

    if hidden is None:
        if saved_hidden is None or saved_hidden < 0:
            hidden = 128
        else:
            hidden = int(saved_hidden)
    if n_hidden_layers is None:
        if saved_n_hidden_layers is None or saved_n_hidden_layers < 0:
            n_hidden_layers = 2
        else:
            n_hidden_layers = int(saved_n_hidden_layers)

    # instantiate model with recovered architecture
    model = ConditionalMDN(
        theta_dim=theta_dim,
        x_dim=x_dim,
        beta_dim=beta_dim,
        hidden=hidden,
        n_components=n_components,
        n_hidden_layers=n_hidden_layers,
    )

    # load weights (state dict keys must match)
    model.load_state_dict(ckpt["state_dict"])

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # reconstruct normalizers and move to device
    theta_mu = ckpt["theta_mu"].to(device)
    theta_std = ckpt["theta_std"].to(device)
    x_mu = ckpt["x_mu"].to(device)
    x_std = ckpt["x_std"].to(device)

    norm_theta = ZScore(theta_mu, theta_std)
    norm_x = ZScore(x_mu, x_std)

    return model, norm_theta, norm_x, ckpt.get("meta", {})




  # import optuna
    # from optuna.samplers import TPESampler

    # g = torch.Generator().manual_seed(SEED)
    # N_joint = theta_joint.size(0)
    # perm = torch.randperm(N_joint, generator=g)
    # N_val = max(1, int(0.1 * N_joint))
    # id_val = perm[:N_val]
    # id_tr = perm[N_val:]
    # fixed_split = (id_tr, id_val)


    # def objective(trial: optuna.Trial):
    #
    #     hidden = trial.suggest_categorical("hidden", [256])
    #     n_hidden_layers = trial.suggest_categorical("n_hidden_layers", [3])
    #     n_components = trial.suggest_categorical("n_components", [8,10,12])
    #     batch_size = trial.suggest_categorical("batch_size", [1024,2048])
    #     # lr = trial.suggest_float("lr", 5e-4, 1e-3, log=True)
    #     lr = 1e-3
    #     model_i, norm_theta_i, norm_x_i, val_i = train_conditional_mdn(
    #         theta=theta_joint, x=x_joint, beta=beta_joint, weights=w_joint,
    #         hidden=hidden, n_components=n_components, n_hidden_layers=n_hidden_layers,
    #         batch_size=batch_size, epochs=500, lr=lr,
    #         val_fraction=0.1, device=device, patience=20, verbose=False,
    #         fixed_split=fixed_split,
    #     )
    #     betas_sets = [0.1]
    #     posterior_samples, reference_samples, c2st_accuracy, mmd_value = draw_samples_and_references(
    #         betas_sets=betas_sets, model=model_i, x_obs=x_obs, norm_theta=norm_theta_i, norm_x=norm_x_i,
    #         num_samples=10_000, lower=low, upper=high, device=device, save_dir=save_dir,title = f"{hidden,n_hidden_layers,n_components}"
    #     )
    #
    #     # 保存 ckpt，方便最后加载最优
    #     ckpt_i = save_dir / (f"optuna_val{val_i:.6f}"
    #                          f"_h{hidden}_L{n_hidden_layers}_K{n_components}"
    #                          f"_bs{batch_size}_lr{lr}.pt")
    #     meta = dict(hidden=hidden, n_hidden_layers=n_hidden_layers,
    #                 n_components=n_components, batch_size=batch_size, lr=lr,
    #                 best_val=float(val_i))
    #     save_checkpoint(str(ckpt_i), model_i, norm_theta_i, norm_x_i, meta=meta)
    #
    #
    #     trial.set_user_attr("ckpt_path", str(ckpt_i))
    #     trial.set_user_attr("meta", meta)
    #
    #     return float(val_i)
    #
    #
    # study = optuna.create_study(direction="minimize",
    #                             sampler=TPESampler(seed=SEED))
    #
    # study.optimize(objective, n_trials=5, n_jobs=1, show_progress_bar=True)
    #
    # best = study.best_trial
    # print("\n>>> OPTUNA BEST <<<")
    # print("val_loss:", best.value)
    # print("params:", best.params)
    # print("ckpt:", best.user_attrs["ckpt_path"])
    #
    # best_hp = best.params
    # best_ckpt = best.user_attrs["ckpt_path"]
    # model, norm_theta, norm_x, meta = load_checkpoint(
    #     best_ckpt, hidden=best_hp["hidden"],
    #     n_hidden_layers=best_hp["n_hidden_layers"], device=device
    # )