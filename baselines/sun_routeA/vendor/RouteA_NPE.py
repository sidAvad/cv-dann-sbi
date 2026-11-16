from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import *

import sbibm

from RouteB_utils import (
    train_conditional_mdn,
    sample_with_rejection,
    save_checkpoint,
    load_checkpoint,
)

SEED = 42
seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# name = ["gaussian_mixture","two_moons","slcp","sir"]
# number = 2
# n = 100_000
#
# task = sbibm.get_task(name[number])
# task_prior = task.get_prior()
# simulator = task.get_simulator()
# theta_dim, x_dim = task.dim_parameters,task.dim_data
#
#
#
#
# if number ==0 :
#     low = torch.tensor([-10.0, -10.0],device = device)
#     high=torch.tensor([10.0, 10.0],device = device)
# elif number == 1:
#     low = torch.tensor([-1.0, -1.0], device=device)
#     high = torch.tensor([1.0, 1.0], device=device)
# elif number == 2:
#     low = torch.tensor([-3.0, -3.0, -3.0, -3.0, -3.0], device=device)
#     high = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0], device=device)
#
#
# save_dir = Path(name[number])


import torch
from lorenz96 import PriorBox, make_prior_fn,make_simulator_fn,Theta

prior_box = PriorBox(1.4, 2.2, 0.1, 1.0, 1.5, 2.5)

prior = make_prior_fn(prior_box, device=device, dtype=torch.float32)
K, T, dt = 8, 20, 3.0/40.0
x0 = torch.ones(K, device=device)
simulator = make_simulator_fn(K, T, dt, R = 5,x0=x0, device=device, dtype=torch.float32, out_device="cpu")
save_dir = Path("lorenz96")
theta_dim,x_dim = 3, 840

low = torch.tensor([1.4,0.1,1.5], device=device)
high = torch.tensor([2.2,1.0,2.5], device=device)

theta_true = torch.tensor([[2.0, 0.8, 1.7]])
x_obs = simulator(theta_true, seed=42).to(device)


metrics_csv = save_dir / "metrics.csv"
logger = ExperimentLogger(metrics_csv)
method = "RouteA_NPE"

n = 10_000

ckpt = save_dir / f"pairs_all_{n}.pt"

theta_x_beta = torch.load(ckpt)

theta_train,x_train,beta_train = theta_x_beta[:,:theta_dim],theta_x_beta[:,theta_dim:theta_dim+x_dim],theta_x_beta[:,theta_dim + x_dim:]


theta_train = theta_train.to(device).float()
x_train     = x_train.to(device).float()
beta_train = beta_train.to(device).float()

print("Joint dataset:", theta_train.shape, x_train.shape, beta_train.shape)
npe_ckpt = save_dir / f"routeA_weighted_npe_{n}.pt"

load_model_NPE = True

if not load_model_NPE:
    theta_train = prior(n)
    x_train = simulator(theta_train)
    beta_train = tensor_ones = torch.ones(n,1)
    theta_train = theta_train.to(device).float()
    x_train = x_train.to(device).float()
    beta_train = beta_train.to(device).float()
    print("Joint dataset:", theta_train.shape, x_train.shape, beta_train.shape)
    model, norm_theta, norm_x, best_val = train_conditional_mdn(
        theta=theta_train,
        x=x_train,
        beta=beta_train,
        hidden=128,
        n_components=9,
        n_hidden_layers=3,
        batch_size=128,
        epochs=500,
        lr=1e-3,
        val_fraction=0.1,
        device=device,
        patience=20
    )

    meta = {
        "hidden": 128,
        "n_hidden_layers": 3,
        "n_components": 9,
        "batch_size": 1024
    }
    save_checkpoint(str(npe_ckpt), model, norm_theta, norm_x, meta=meta)

else:
    model, norm_theta, norm_x, meta = load_checkpoint(
        str(npe_ckpt), hidden=128, n_hidden_layers=3, device=device
    )

# x_obs = task.get_observation(num_observation=1)

betas_sets = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
betas_sets = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
betas_sets = [1.0]
for beta_scalar in betas_sets:
    with torch.no_grad():
        posterior_samples = sample_with_rejection(model=model, x_obs=x_obs, beta=beta_scalar,
                                                  norm_theta=norm_theta, norm_x=norm_x,
                                                  num_samples=10_000, lower=low, upper=high,
                                                  device=device)
        ckpt_path = save_dir / f"True_Power_posterior_with_t_power={beta_scalar}_10000.pt"
        reference_samples = torch.load(ckpt_path, map_location=device)
        c2st_accuracy = to_float(c2st(reference_samples, posterior_samples))
        mmd_value = to_float(mmd(reference_samples, posterior_samples))
        print(f"c2st:{c2st_accuracy:.4f}")
        print(f"MMD: {mmd_value:.4g}")
        print(posterior_samples.shape)
        print(reference_samples.shape)
        plot_posteriors_compare(reference_samples[:, :2], posterior_samples=posterior_samples[:, :2],
                                title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={beta_scalar}")

        logger.log(
            method=method,
            n=n,
            beta=float(beta_scalar),
            c2st=float(c2st_accuracy),
            mmd=float(mmd_value),
            seed=SEED,
            device=device,
            ckpt_path=ckpt_path,
            notes=None
        )




