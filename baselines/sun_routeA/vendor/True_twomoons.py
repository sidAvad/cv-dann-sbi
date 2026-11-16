import math
import torch
import sbibm
from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from pathlib import Path
from NSEutils import plot_posteriors_compare,to_float

def A_theta(theta):
    # theta: (..., 2)
    t1, t2 = theta[..., 0], theta[..., 1]
    comp1 = -torch.abs(t1 + t2) / math.sqrt(2.0)
    comp2 = (-t1 + t2) / math.sqrt(2.0)
    return torch.stack([comp1, comp2], dim=-1)  # (..., 2)

def loglik_two_moons(theta, x, sigma=0.01, eps=1e-12):
    # theta: (..., 2), x: (2,)
    v = x - A_theta(theta) - torch.tensor([0.25, 0.0], device=theta.device, dtype=theta.dtype)
    r = torch.linalg.norm(v, dim=-1).clamp_min(eps)
    # ✔ 用 vx>0 代替 atan2 的区间判断，数值更稳
    in_sector = (v[..., 0] > 0)

    logN = -0.5 * ((r - 0.1) / sigma) ** 2 - math.log(math.sqrt(2 * math.pi) * sigma)
    logdens = logN - math.log(math.pi) - torch.log(r)
    logdens = torch.where(in_sector, logdens, torch.full_like(logdens, -float("inf")))
    return logdens


def in_prior_box(theta):
    return ((theta >= -1.0) & (theta <= 1.0)).all(dim=-1)
# ---------- MH sampler ----------
@torch.no_grad()
def sample_posterior_two_moons(x_obs, num_samples=10_000, burnin=5_000,
                               step_init=0.3, adapt=True, chains=2, thin=1,
                               t_power = 1.0,log_fn = loglik_two_moons,
                               dtype=torch.float32, device="cpu", seed=42):
    torch.manual_seed(seed)
    x = torch.as_tensor(x_obs, dtype=dtype, device=device)

    inits = torch.tensor([[ 0.8,  0.8],
                          [-0.8, -0.8]], dtype=dtype, device=device)
    if chains == 1: inits = inits[:1]
    thetas = inits

    step = torch.full((chains, 2), step_init, dtype=dtype, device=device)
    cur_loglik = log_fn(thetas, x)

    cur_ok = in_prior_box(thetas)
    cur_logpost = torch.where(cur_ok, cur_loglik, torch.full_like(cur_loglik, -float("inf")))

    keep = []
    acc_count = torch.zeros(chains, dtype=dtype, device=device)
    total_count = torch.zeros(chains, dtype=dtype, device=device)

    total_iters = burnin + num_samples*thin
    for it in range(total_iters):
        prop = thetas + torch.randn_like(thetas) * step

        prop_loglik = log_fn(prop, x)
        prop_ok = in_prior_box(prop)
        prop_logpost = torch.where(prop_ok, prop_loglik, torch.full_like(prop_loglik, -float("inf")))

        log_alpha = (prop_logpost - cur_logpost) * t_power
        u = torch.log(torch.rand(chains, dtype=dtype, device=device))

        accept = (u < log_alpha)
        thetas = torch.where(accept[:, None], prop, thetas)
        cur_logpost = torch.where(accept, prop_logpost, cur_logpost)

        acc_count += accept.to(dtype)
        total_count += 1

        if adapt and it < burnin:
            target = 0.3
            eta = 1.0 / math.sqrt(it + 1.0)
            step = step * torch.exp(eta * (acc_count/total_count - target)[:, None])
        elif (it - burnin) % thin == 0 and it >= burnin:
            keep.append(thetas.clone())


    samples = torch.vstack(keep)
    return samples, {"acc_rate": (acc_count/total_count).cpu().numpy(),
                     "step": step.cpu().numpy()}


if __name__ == "__main__":
    task = sbibm.get_task("two_moons")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observation = task.get_observation(num_observation=1).to(device)

    reference_samples = task.get_reference_posterior_samples(num_observation=1).to(device)
    observation = observation.squeeze(0)
    save_dir = Path("two_moons")

    # t_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
    t_list = [1.0]
    for t in t_list:
        samples, info = sample_posterior_two_moons(observation, num_samples=5_000, burnin=10_000, chains=2, thin=2,
                                                   t_power=t,step_init = 0.2
                                                   )
        # ckpt_path = save_dir / f"True_Power_posterior_with_t_power={t}_10000.pt"
        # torch.save(samples, ckpt_path)
        print("acceptance:", info["acc_rate"])

        reference_samples = task.get_reference_posterior_samples(num_observation=1)
        c2st_accuracy = to_float(c2st(reference_samples, samples))
        mmd_value = to_float(mmd(reference_samples, samples))
        print(f"c2st:{c2st_accuracy:.4f}")
        print(f"MMD: {mmd_value:.4g}")
        print(samples.shape)
        print(reference_samples.shape)
        plot_posteriors_compare(reference_samples, posterior_samples=samples,
                                title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={t}")

