from NSEutils import *
from tqdm import tqdm


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
    assert gamma_min > 0 and gamma_max > 0, "gamma must be positive"
    # 在 log 空间线性插值，相当于几何级数
    log_min, log_max = math.log(gamma_min), math.log(gamma_max)
    gammas = torch.exp(torch.linspace(log_min, log_max, T))  # 递增：min → max

    if direction == "decreasing":
        gammas = torch.flip(gammas, dims=[0])  # 递减：max → min
    elif direction != "increasing":
        raise ValueError("direction must be 'decreasing' or 'increasing'")

    return gammas.float()  # shape: [T]


def score_matching_loss(score_net, theta, x,t_idx, gammas):
    """
    theta: [B, theta_dim]
    x:     [B, x_dim]
    t_idx: [B]  ∈ {0..T-1}
    gammas: [T] 噪声方差 γ_t
    """
    p = next(score_net.parameters())
    device = p.device
    dtype = p.dtype

    theta = theta.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    t_idx = t_idx.to(device=device, dtype=torch.long)

    gammas = torch.as_tensor(gammas, device=device, dtype=dtype)
    sigma = gammas[t_idx].unsqueeze(-1)

    eps_theta = torch.randn_like(theta, dtype=dtype)
    noise_theta = theta + sigma * eps_theta

    eps_x = torch.randn_like(x, dtype=dtype)
    noise_x = x + sigma * eps_x
    eps = torch.cat([eps_theta, eps_x], dim=-1)
    s_hat = score_net(noise_theta,noise_x, sigma) * sigma

    target = - eps
    loss = F.mse_loss(s_hat, target)

    return loss


def train(
    model,
    dataset,
    gammas,
    opt,
    theta_dim,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)

    model.train()
    T = len(gammas)
    total = 0.0
    for theta_and_x in dataset:
        theta = theta_and_x[:, :theta_dim].to(device)  # [B, d_theta]
        x = theta_and_x[:, theta_dim:].to(device)  # [B, d_x]
        B = theta.size(0)

        t_idx = torch.randint(0, T, (B,), device=device)  # [B]
        loss_k = score_matching_loss(model, theta, x, t_idx, gammas)

        opt.zero_grad()
        loss_k.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += (loss_k / B).detach()

    total = total/len(dataset)

    return total

@torch.no_grad()
def validate(
    model,
    dataset,
    gammas,
    theta_dim,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model = model.to(device)

    model.eval()
    T = len(gammas)
    total = 0.0
    with torch.inference_mode():
        for theta_and_x in dataset:
            theta = theta_and_x[:, :theta_dim].to(device)  # [B, d_theta]
            x = theta_and_x[:, theta_dim:].to(device)  # [B, d_x]
            B = theta.size(0)

            t_idx = torch.randint(0, T, (B,), device=device)  # [B]
            loss_k = score_matching_loss(model, theta, x, t_idx, gammas)

            total += (loss_k / B).detach()

    total = total/len(dataset)

    return total


@torch.no_grad()
def alpha_from_snr_blocks(
    score: torch.Tensor,
    theta_dim: int,
    x_dim: int,
    sigma: float,
    t_power: float,
    tau_theta: float = 0.5,
    tau_x: float = 0.5,
    cmin_theta: float = 2e-5,
    cmax_theta: float = 1e-4,
    cmin_x: float = 2e-5,
    cmax_x: float = 1e-4,
    per_sample: bool = False,   # True: 每个样本单独步长；False: 用batch中位数得到“批量共享”步长
):
    """
    输入:
      score: (N, theta_dim + x_dim)
      sigma: 当前温度/噪声标度（标量）
    输出:
      a_theta:  (N,1) 或 (1,1) 形状，方便广播到 θ 的每个维度
      a_x:      (N,1) 或 (1,1) 形状，方便广播到 x 的每个维度
    """
    device, dtype = score.device, score.dtype
    score_theta = score[:, :theta_dim]
    score_x     = score[:, theta_dim:]

    # 每个样本的二范数平方
    g2_theta = score_theta.pow(2).sum(dim=1).clamp_min(1e-12)  # (N,)
    g2_x     = score_x.pow(2).sum(dim=1).clamp_min(1e-12)      # (N,)

    if per_sample:
        # 每个样本一个步长
        a_theta = (0.5 * theta_dim * (tau_theta**2)) / (g2_theta * (t_power**2))  # (N,)
        a_x     = (0.5 * x_dim     * (tau_x**2))     / (g2_x     * (t_power**2))  # (N,)
    else:
        # 批量共享：用中位数得到一个标量步长（数值更稳）
        g2_theta_med = g2_theta.median()
        g2_x_med     = g2_x.median()
        a_theta = (0.5 * theta_dim * (tau_theta**2)) / (g2_theta_med * (t_power**2))
        a_x     = (0.5 * x_dim     * (tau_x**2))     / (g2_x_med     * (t_power**2))

    sig2 = torch.as_tensor(sigma, device=device, dtype=dtype)**2
    # 独立裁剪区间（可按块调不同范围）
    a_theta = a_theta.clamp(cmin_theta * sig2, cmax_theta * sig2)
    a_x     = a_x.clamp(    cmin_x     * sig2, cmax_x     * sig2)

    # 变成 (N,1) 或 (1,1)，便于广播
    if per_sample:
        a_theta = a_theta.view(-1, 1)
        a_x     = a_x.view(-1, 1)
    else:
        a_theta = a_theta.view(1, 1)
        a_x     = a_x.view(1, 1)
    return a_theta, a_x


@torch.no_grad()
def Lgmcmc(
    gammas,
    prior,
    simulator,
    model,
    theta_dim,
    x_dim,
    n_samples=10,
    initial_step=10,
    t_power=1.0,
    step_increase=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    # 可选：分块步长的超参
    tau_theta=0.5, tau_x=0.5,
    cmin_theta=2e-5, cmax_theta=1e-4,
    cmin_x=1e-2, cmax_x=0.5,
    per_sample_stepsize=False,   # True: 每样本独立步长
):
    model.eval()
    model.to(device)

    T = len(gammas)
    theta = prior(n_samples).to(device)     # (N, theta_dim)
    x     = simulator(theta).to(device)     # (N, x_dim)  这里假设你的 simulator 已经给出展平后的 x_dim
    theta_and_x = torch.cat([theta, x], dim=-1)  # (N, theta_dim + x_dim)

    step = initial_step
    samples = []

    for t in reversed(range(T)):
        sigma = gammas[t]

        for _ in range(step):
            # 拆块视图（不拷贝）
            theta_view = theta_and_x[:, :theta_dim]
            x_view     = theta_and_x[:, theta_dim:]

            # 联合score（假定 model 返回与输入同形状的 score 向量）
            sig_batch = torch.full((n_samples, 1), float(sigma), device=device, dtype=theta_and_x.dtype)
            score = model(theta_view, x_view, sig_batch)  # (N, theta_dim + x_dim)

            # 分块步长
            a_theta, a_x = alpha_from_snr_blocks(
                score, theta_dim, x_dim, sigma, t_power,
                tau_theta=tau_theta, tau_x=tau_x,
                cmin_theta=cmin_theta, cmax_theta=cmax_theta,
                cmin_x=cmin_x, cmax_x=cmax_x,
                per_sample=per_sample_stepsize
            )

            # 拆分 score
            score_theta = score[:, :theta_dim]
            score_x     = score[:, theta_dim:]

            # 独立噪声
            eps_theta = torch.randn_like(score_theta)
            eps_x     = torch.randn_like(score_x)

            # ULA 形式：x_{k+1} = x_k + t_power * a * score + sqrt(2a) * eps
            theta_new = theta_view + t_power * a_theta * score_theta + torch.sqrt(2.0 * a_theta) * eps_theta
            x_new     = x_view     + t_power * a_x     * score_x     + torch.sqrt(2.0 * a_x)     * eps_x

            # 回写
            theta_and_x = torch.cat([theta_new, x_new], dim=-1)

        step += step_increase

    samples.append(theta_and_x.detach().clone())
    chain_tilde = torch.stack(samples, dim=1)  # (N, 1, theta_dim + x_dim)
    samples_theta_and_x = chain_tilde.reshape(-1, theta_dim + x_dim)
    return samples_theta_and_x