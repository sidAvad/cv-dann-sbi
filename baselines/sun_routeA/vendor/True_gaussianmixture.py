import sbibm
from pathlib import Path
from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import plot_posteriors_compare,to_float

import math, torch

def _in_box(theta):
    return ((theta >= -10.0) & (theta <= 10.0)).all(dim=-1)

def _log_normal_iso(theta, mean, s):  # cov = s * I
    diff = theta - mean
    return -0.5*(diff*diff).sum(-1)/s - 0.5*2*math.log(2*math.pi*s)

def _K_iso(s, beta):  # Σ = s I
    return (2*math.pi)**(-0.5*2*(beta-1.0)) * (s**(-0.5*2*(beta-1.0))) * (beta**(0.5*2))

@torch.no_grad()
def mh_sample(logpost, theta0, x_obs,n_samples=10_000, beta = 1.0,
              prop_scale=0.5, seed=0, device='cpu', dtype=torch.float32):
    torch.manual_seed(seed)
    theta = theta0.clone().to(device=device, dtype=dtype)
    d = theta.numel()
    theta = theta.reshape(1, d)
    cur_lp = logpost(theta,x_obs)
    out = []
    acc = 0
    for _ in range(n_samples*3):  # 留裕度，box 拒绝会浪费步
        prop = theta + prop_scale*torch.randn_like(theta)
        if not (((prop >= -10.0) & (prop <= 10.0)).all()):
            continue
        prop_lp = logpost(prop,x_obs)
        if torch.isfinite(prop_lp) and torch.log(torch.rand(())) < (prop_lp - cur_lp) * beta:
            theta, cur_lp = prop, prop_lp
            acc += 1
        out.append(theta.clone())
        if len(out) >= n_samples:
            break
    return torch.vstack(out).contiguous(), acc / max(1, len(out))


@torch.no_grad()
def sample_power_posterior_gm_anybeta(x_obs, beta, num_samples=10_000,
                                      device="cpu", dtype=torch.float64, seed=44):

    torch.manual_seed(seed)
    x = torch.as_tensor(x_obs, dtype=dtype, device=device).reshape(2)

    s1, s2 = 1.0, 0.01

    # 包络的“公共权重系数” w_beta
    w_beta = (0.5**beta) if (beta <= 1.0) else 0.5

    # 各分量的常数与提议分布（协方差放大为 Σ/β）
    w1 = w_beta * _K_iso(s1, beta)
    w2 = w_beta * _K_iso(s2, beta)
    W  = w1 + w2
    p1, p2 = w1/W, w2/W

    mvn1 = torch.distributions.MultivariateNormal(x, covariance_matrix=(s1/beta)*torch.eye(2, dtype=dtype, device=device))
    mvn2 = torch.distributions.MultivariateNormal(x, covariance_matrix=(s2/beta)*torch.eye(2, dtype=dtype, device=device))
    cat  = torch.distributions.Categorical(probs=torch.tensor([p1, p2], dtype=dtype, device=device))

    out = []
    batch = 4096
    while len(out) < num_samples:
        ks = cat.sample((batch,))
        n1 = (ks==0).sum().item(); n2 = batch - n1
        parts = []
        if n1>0: parts.append(mvn1.sample((n1,)))
        if n2>0: parts.append(mvn2.sample((n2,)))
        thetas = torch.vstack(parts)

        # 先验盒外直接丢
        mask = _in_box(thetas)
        thetas = thetas[mask]
        if thetas.numel() == 0:
            continue

        # q(θ) = 0.5 N(.; s1) + 0.5 N(.; s2)
        logN1 = _log_normal_iso(thetas, x, s1)
        logN2 = _log_normal_iso(thetas, x, s2)
        m = torch.maximum(logN1, logN2)
        log_q = m + torch.log(0.5*torch.exp(logN1-m) + 0.5*torch.exp(logN2-m))

        # Gβ(θ) = w1 N(.; s1/β) + w2 N(.; s2/β)
        logG1 = math.log(w1) + _log_normal_iso(thetas, x, s1/beta)
        logG2 = math.log(w2) + _log_normal_iso(thetas, x, s2/beta)
        mg = torch.maximum(logG1, logG2)
        log_G = mg + torch.log(torch.exp(logG1-mg) + torch.exp(logG2-mg))

        # 接受概率：q(θ)^β / Gβ(θ)
        log_acc = beta*log_q - log_G
        accept = torch.log(torch.rand_like(log_acc)) < log_acc
        if accept.any():
            out.append(thetas[accept])

    return torch.vstack(out)[:num_samples].contiguous()


if __name__ == "__main__":
    task = sbibm.get_task("gaussian_mixture")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    observation = task.get_observation(num_observation=1).to(device)
    observation = observation.squeeze(0)
    save_dir = Path("gaussian_mixture")
    # t_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
    t_list = [1.0]

    for t in t_list:
        samples = sample_power_posterior_gm_anybeta(observation, beta=t, num_samples=10_000, device=device,)

        ckpt_path = save_dir / f"True_Power_posterior_with_t_power={t}_10000.pt"
        torch.save(samples, ckpt_path)
        print(samples.shape)
        reference_samples = task.get_reference_posterior_samples(num_observation=1).to(device)
        c2st_accuracy = to_float(c2st(reference_samples, samples))
        mmd_value = to_float(mmd(reference_samples, samples))
        print(f"c2st:{c2st_accuracy:.4f}")
        print(f"MMD: {mmd_value:.4g}")
        print(samples.shape)
        print(reference_samples.shape)
        plot_posteriors_compare(reference_samples, posterior_samples=samples,
                                title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={t}")


