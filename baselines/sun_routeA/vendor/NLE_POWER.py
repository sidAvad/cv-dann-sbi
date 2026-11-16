import torch
import numpy as np

from sbi.inference import SNLE
from sbi.utils import BoxUniform

from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import plot_posteriors_compare,to_float,ExperimentLogger
from pathlib import Path
from True_twomoons import sample_posterior_two_moons
from True_gaussianmixture import mh_sample
from True_slcp_anneal import make_gamma_schedule,parallel_tempering_rw,sample_prior

from sbibm import get_task

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# name = ["gaussian_mixture","two_moons","slcp","sir"]
# number = 1
# n = 100_000
#
# task = get_task(name[number])
# prior = task.get_prior()
# theta = prior(1)
# simulator = task.get_simulator()
# theta_dim, x_dim = task.dim_parameters,task.dim_data
# save_dir = Path(name[number])

import torch
from lorenz96 import PriorBox, make_prior_fn,make_simulator_fn,Theta

prior_box = PriorBox(1.4, 2.2, 0.1, 1.0, 1.5, 2.5)

prior = make_prior_fn(prior_box, device=device, dtype=torch.float32)
K, T, dt = 8, 20, 3.0/40.0
x0 = torch.ones(K, device=device)
simulator = make_simulator_fn(K, T, dt,R=5, x0=x0, device=device, dtype=torch.float32, out_device="cpu")
save_dir = Path("lorenz96")
theta_dim,x_dim = 3, 840
low = torch.tensor([1.4,0.1,1.5], device=device)
high = torch.tensor([2.2,1.0,2.5], device=device)
theta_true = torch.tensor([[2.0, 0.8, 1.7]])
x_obs = simulator(theta_true, seed=42).to(device)
n = 10_000



nle_ckpt = save_dir / f"NLE_ckpt_{n}.pt"

metrics_csv = save_dir / "metrics.csv"
logger = ExperimentLogger(metrics_csv)
method = "NlE"

with torch.no_grad():
    theta_train = prior(n)
    x_train = simulator(theta_train).to(device)
    theta_train = theta_train.to(device)


inference_L = SNLE(prior=None, density_estimator="mdn")

# if number ==0 :
#     low = torch.tensor([-10.0, -10.0],device = device)
#     high=torch.tensor([10.0, 10.0],device = device)
# elif number == 1:
#     low = torch.tensor([-1.0, -1.0], device=device)
#     high = torch.tensor([1.0, 1.0], device=device)
# elif number == 2:
#     low = torch.tensor([-3.0, -3.0, -3.0, -3.0, -3.0], device=device)
#     high = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0], device=device)
prior = BoxUniform(low=low, high=high)

with torch.no_grad():
    dummy_theta = torch.randn(256, theta_dim) * 0.1
    dummy_x = torch.randn(256, x_dim) * 0.1
inference_L.append_simulations(dummy_theta, dummy_x)
density_estimator = inference_L.train(stop_after_epochs=0)

density_estimator.load_state_dict(torch.load(nle_ckpt, map_location=device))
density_estimator.to("cpu")
posterior_r = inference_L.build_posterior(prior = prior,density_estimator=density_estimator)


# observation = task.get_observation(num_observation=1)

t_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
# t_list = [1.0]
for t in t_list:
    # if number == 0:
    #     samples, _ = mh_sample(logpost=posterior_r.log_prob, theta0=theta, x_obs=observation, n_samples=10_000, beta=t,
    #                            prop_scale=0.5, seed=SEED)
    #
    # elif number == 1:
    #     samples, info = sample_posterior_two_moons(observation, num_samples=5_000,
    #                                                burnin=10_000, chains=2, thin=2,
    #                                                t_power=t, log_fn=posterior_r.log_prob)
    #
    # elif number == 2:
    #     betas = make_gamma_schedule(T=200, gamma_min=0.01, gamma_max=t, direction="decreasing")
    #     samples, acc = parallel_tempering_rw(
    #         x_obs=observation,
    #         log_likelihood=posterior_r.log_prob,
    #         sample_prior=sample_prior,
    #         D=5,
    #         betas=betas,
    #         n_iters=20000, burn_in=10000,
    #         local_steps_per_iter=5,
    #         step_init=0.2,
    #         swap_interval=5,
    #         seed=42
    #     )



    samples = samples.to(device)
    ckpt_path = save_dir / f"True_Power_posterior_with_t_power={t}_10000.pt"
    reference_samples = torch.load(ckpt_path, map_location=device)
    # reference_samples = task.get_reference_posterior_samples(num_observation=1).to(device)
    c2st_accuracy = to_float(c2st(reference_samples, samples))
    mmd_value = to_float(mmd(reference_samples, samples))
    print(f"c2st:{c2st_accuracy:.4f}")
    print(f"MMD: {mmd_value:.4g}")
    print(samples.shape)
    print(reference_samples.shape)
    plot_posteriors_compare(reference_samples, posterior_samples=samples,
                            title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g},t_power={t}")

    logger.log(
        method=method,
        n=n,
        beta=float(t),
        c2st=float(c2st_accuracy),
        mmd=float(mmd_value),
        seed=SEED,
        device=device,
        ckpt_path=ckpt_path,
        notes=None
    )