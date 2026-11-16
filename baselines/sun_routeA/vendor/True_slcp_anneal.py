import numpy as np
import sbibm
import torch
from pathlib import Path

from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd

from NSEutils import seed_everything,plot_posteriors_compare,to_float
import math



torch.set_default_dtype(torch.float32)
LOG2PI = math.log(2*math.pi)


device = "cuda" if torch.cuda.is_available() else "cpu"

def make_gamma_schedule(
    T=10,
    gamma_min=0.01,
    gamma_max=1.0,
    direction="increasing",   # "decreasing" 或 "increasing"
):
    """
    生成几何/指数间隔的 gamma_t 序列（长度 T）。
    - direction="decreasing": t=1..T 单调递减：gamma_max → gamma_min（和你原注释一致）
    - direction="increasing": t=1..T 单调递增：gamma_min → gamma_max
    """
    assert T >= 1, "T 必须 >= 1"
    assert gamma_min > 0 and gamma_max > 0, "gamma 必须为正"
    # 在 log 空间线性插值，相当于几何级数
    log_min, log_max = math.log(gamma_min), math.log(gamma_max)
    gammas = torch.exp(torch.linspace(log_min, log_max, T))  # 递增：min → max

    if direction == "decreasing":
        gammas = torch.flip(gammas, dims=[0])  # 递减：max → min
    elif direction != "increasing":
        raise ValueError("direction 必须是 'decreasing' 或 'increasing'")

    return gammas.float()  # shape: [T]


def sample_prior(n, device=device):
    return (-3.0 + 6.0*torch.rand(n, 5, device=device))

def log_prior_box(theta, low=-3.0, high=3.0):
    # theta: (..., D)
    in_box = (theta >= low) & (theta <= high)
    ok = in_box.all(dim=-1)                     # (...,)
    out = theta.new_full(ok.shape, float('-inf'))
    out[ok] = 0.0                               # 盒内常数随便取，0 就行
    return out.to(device)

def _cov_inv_and_logdet(theta):
    s1  = torch.clamp(theta[..., 2]**2, min=1e-12)
    s2  = torch.clamp(theta[..., 3]**2, min=1e-12)
    rho = torch.tanh(theta[..., 4])

    a = s1**2
    c = s2**2
    b = rho*s1*s2
    det = torch.clamp(a*c - b*b, min=1e-24)     # (...,)
    logdet = torch.log(det)
    inv = torch.stack([
        torch.stack([ c/det, -b/det], dim=-1),
        torch.stack([-b/det,  a/det], dim=-1),
    ], dim=-2)                                   # (...,2,2)
    return inv, logdet

def log_likelihood(theta, x):
    """
    x: (..., 8) 或 (...,4,2) 或 (8,)
    theta: (..., 5)
    返回: (...,)
    """
    device = theta.device
    x = x.to(device)

    # 统一成 (...,4,2)
    if x.ndim == 1:                  # (8,)
        x = x.view(1, 4, 2)
    elif x.shape[-1] == 8:           # (...,8)
        x = x.view(*x.shape[:-1], 4, 2)
    else:
        assert x.shape[-2:] == (4, 2), "x must end with 8 or (4,2)"

    # 将 x 在前面维度上对齐并广播到 theta 的 batch 形状
    while x.dim() < theta.dim():     # 给 x 补齐缺少的批量维
        x = x.unsqueeze(0)
    # 现在做显式 expand（不会复制内存）
    x = x.expand(*theta.shape[:-1], 4, 2)

    mu = theta[..., :2]
    inv, logdet = _cov_inv_and_logdet(theta)

    diff = x - mu.unsqueeze(-2)      # (...,4,2)
    quad_each = torch.einsum('...ni,...ij,...nj->...n', diff, inv, diff)
    quad_sum  = quad_each.sum(dim=-1)

    const = -4.0 * LOG2PI
    return const - 0.5 * (4 * logdet + quad_sum)

@torch.no_grad()
def parallel_tempering_rw(
    x_obs,
    log_likelihood,
    sample_prior,
    D,                                         # 维度 D
    betas=None,                                # β 阶梯（张量，形如 [1.0,...,低温]）
    n_iters=30_000, burn_in=10_000,
    local_steps_per_iter=5,                    # 每轮交换前每条链做多少步RW
    step_init=0.2,                             # β=1 的步长标量或 (D,)；低温自动放大 1/sqrt(β)
    swap_interval=10,                          # 每多少轮尝试一次交换
    # —— 自适应步长（只在 burn-in 期间按 Robbins–Monro 调，目标接受率≈0.25）
    adapt=True, target_acc=0.25, adapt_interval=25, adapt_lr=0.05,
    seed=42, init_thetas=None                   # init_thetas: (R,D) 或 None（自动用先验挑）
):
    x_obs = x_obs.to(device).reshape(-1)
    gen = torch.Generator(device=device).manual_seed(int(seed))

    betas = betas.to(device)
    R = betas.numel()


    step_init = torch.as_tensor(step_init, device=device, dtype=x_obs.dtype)
    if step_init.ndim == 0: step_init = step_init.expand(D)
    step_all = step_init.unsqueeze(0) / torch.sqrt(betas).unsqueeze(1)   # (R,D)


    if init_thetas is None:
        cand = sample_prior(2048).to(device)

        ll   = log_likelihood(cand, x_obs)
        idx  = torch.topk(ll, k=R).indices
        theta= cand[idx].clone()                                         # (R,D)
    else:
        theta= init_thetas.to(device).clone()                            # (R,D)

    cur_lp = log_prior_box(theta)  # (R,)                                     # (R,)
    cur_ll  = log_likelihood(theta, x_obs).to(device)
    cur_lpo = betas*cur_ll + cur_lp                                       # (R,)

    keep = []
    mh_acc = torch.zeros(R, device=device); mh_trials = torch.zeros(R, device=device)
    sw_acc = torch.zeros(R-1, device=device); sw_trials = torch.zeros(R-1, device=device)

    for t in range(n_iters):

        for _ in range(local_steps_per_iter):

            prop = theta + torch.randn_like(theta) * step_all
            prop_lp = log_prior_box(prop).to(device)
            prop_ll  = log_likelihood(prop, x_obs).to(device)
            prop_lpo = betas*prop_ll + prop_lp

            loga = prop_lpo - cur_lpo

            U = torch.log(torch.rand(R, device=device, generator=gen))
            acc = U < loga
            mh_acc += acc.float(); mh_trials += 1

            # 接受/拒绝
            theta   = torch.where(acc.unsqueeze(-1), prop, theta)
            cur_ll  = torch.where(acc, prop_ll, cur_ll)
            cur_lpo = torch.where(acc, prop_lpo, cur_lpo)

        # ---- 交换相邻副本（奇偶轮换）
        if (t+1) % swap_interval == 0:
            offset = 0 if ((t//swap_interval) % 2 == 0) else 1
            for i in range(offset, R-1, 2):
                j = i+1
                # 交换比：exp( (β_i-β_j) * (ll_j - ll_i) )
                delta = (betas[i]-betas[j]) * (cur_ll[j]-cur_ll[i])
                U = torch.log(torch.rand(1, device=device, generator=gen)).item()
                sw_trials[i] += 1
                if U < delta.item():
                    sw_acc[i] += 1
                    theta[i], theta[j] = theta[j].clone(), theta[i].clone()
                    cur_ll[i], cur_ll[j] = cur_ll[j].clone(), cur_ll[i].clone()
                    cur_lpo[i],cur_lpo[j]=cur_lpo[j].clone(),cur_lpo[i].clone()

        if (t < burn_in) and adapt and ((t+1) % adapt_interval == 0):
            recent_acc = (mh_acc / torch.clamp_min(mh_trials, 1)).unsqueeze(-1)  # (R,1)
            adj = torch.exp(adapt_lr * (recent_acc - target_acc))
            step_all = step_all * adj
            mh_acc.zero_(); mh_trials.zero_()

        # ---- 记录 β=1 样本（链 0）
        if t >= burn_in:
            keep.append(theta[0].unsqueeze(0).cpu())

    samples = torch.cat(keep, dim=0)                                     # (N, D)
    stats = {
        "betas": betas.cpu().tolist(),
        "final_step_beta1": step_all[0].cpu().tolist(),
        "swap_accept_rate_between_betas": (sw_acc/torch.clamp_min(sw_trials,1)).cpu().tolist(),
    }
    return samples, stats




if __name__ == "__main__":
    rng = np.random.default_rng(42)
    seed_everything(42)
    task = sbibm.get_task("slcp")
    observation = task.get_observation(num_observation=1).to(device)  # 10 per task
    reference_samples = task.get_reference_posterior_samples(num_observation=1).to(device)
    save_dir = Path("slcp")
    theta_dim, x_dim = task.dim_parameters, task.dim_data
    t_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
    # t_list = [1.0]

    for t in t_list:
        betas = make_gamma_schedule(T=200, gamma_min=0.01, gamma_max=t, direction="decreasing")
        posterior_samples,acc = parallel_tempering_rw(
        x_obs=observation,
        log_likelihood=log_likelihood,
        sample_prior=sample_prior,
        D=5,
        betas=betas,
        n_iters=15000, burn_in=5000,
        local_steps_per_iter=5,
        step_init=0.2,
        swap_interval=2,
        seed=42
    )
        posterior_samples = posterior_samples.to(device)
        print(acc,posterior_samples.shape)
        ckpt_path = save_dir / f"True_Power_posterior_with_t_power={t}_10000.pt"
        torch.save(posterior_samples, ckpt_path)
        if t == 1.0:
            c2st_accuracy = to_float(c2st(reference_samples, posterior_samples))
            mmd_value = to_float(mmd(reference_samples, posterior_samples))
            print(f"c2st:{c2st_accuracy:.4f}")
            print(f"MMD: {mmd_value:.4g}")
            print(posterior_samples.shape)
            print(reference_samples.shape)
            plot_posteriors_compare(reference_samples[:, :2], posterior_samples=posterior_samples[:, :2],
                                    title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={t}")