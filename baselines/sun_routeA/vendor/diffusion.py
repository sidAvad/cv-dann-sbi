import numpy as np
import torch
import sbibm

from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import plot_posteriors_compare,to_float
from pathlib import Path

from RouteB_utils import (
    train_conditional_mdn,
    sample_with_rejection,
)

torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
name = ["gaussian_mixture","two_moons","gaussian_linear","bernoulli_glm"]
number = 1
n = 10_000

task = sbibm.get_task(name[number])
prior = task.get_prior()
simulator = task.get_simulator()
theta_dim, x_dim = task.dim_parameters,task.dim_data

save_dir = Path(name[number])
theta = prior(n)

x = simulator(theta).to(device)
theta = theta.to(device)

theta_and_x = torch.cat([theta,x],dim=-1)
posterior_samples = torch.cat([theta,x],dim=-1)
# model = ResidualScoreNet(dim=theta_dim + x_dim, hidden=256, blocks=5)
# cfg = TrainConfig(objective="edm",
#                   sigma_data=0.5, p_mean=-1.2, p_std=1.2,
#                   sigma_min=2e-3, sigma_max=80.0, rho=7.0, n_sigmas=35)
#
# model, std, sigmas = train(model, theta_and_x, cfg, device="cuda")   # 你的train里已支持EDM loss
# posterior_samples = edm_sampler(model, sigmas, n=10000, std=std, device="cuda").to(device)

low = torch.tensor([-1.0, -1.0],device = device)
high=torch.tensor([1.0, 1.0],device = device)

beta_col = torch.full((posterior_samples.size(0), 1), float(1.0), device=device, dtype=x.dtype)

theta_x_beta = torch.cat([posterior_samples, beta_col], dim=-1)

theta_train,x_train,beta_train = theta_x_beta[:,:theta_dim],theta_x_beta[:,theta_dim:theta_dim+x_dim],theta_x_beta[:,theta_dim + x_dim:]


theta_train = theta_train.to(device).float()
x_train     = x_train.to(device).float()
beta_train = beta_train.to(device).float()


load_model_NPE = False

if not load_model_NPE:
    model, norm_theta, norm_x, best_val = train_conditional_mdn(
        theta=theta_train,
        x=x_train,
        beta=beta_train,
        hidden=128,
        n_components=10,
        n_hidden_layers=3,
        batch_size=512,
        epochs=200,
        lr=1e-3,
        val_fraction=0.1,
        device=device,
        verbose=True,
        patience=30
    )
else:
    pass

# betas_sets = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
betas_sets = [1.0]


for beta_scalar in betas_sets:
    with torch.no_grad():
        x_obs = task.get_observation(num_observation=1)
        posterior_samples = sample_with_rejection(model, x_obs=x_obs, beta=beta_scalar,
                             norm_theta=norm_theta, norm_x=norm_x,
                             num_samples=10_000, lower=low,upper=high,
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


        print("samples:", posterior_samples.shape)
