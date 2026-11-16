import torch
import numpy as np

from sbi.inference import SNLE
from sbi.utils import BoxUniform

from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from NSEutils import plot_posteriors_compare,to_float,ExperimentLogger
from pathlib import Path


from sbibm import get_task

from RouteB_utils import (
    train_conditional_mdn,
    log_r_hat,
    get_routeB_samples,
    sample_with_rejection,
    save_checkpoint,
    load_checkpoint,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# name = ["gaussian_mixture","two_moons","slcp","sir"]
# number = 0
#
# task = get_task(name[number])
# prior = task.get_prior()
# simulator = task.get_simulator()
# x_obs = task.get_observation(num_observation=1)
# theta_dim, x_dim = task.dim_parameters,task.dim_data
# save_dir = Path(name[number])

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
method = "RouteB_NLE"

with torch.no_grad():
    theta_train = prior(n)
    x_train = simulator(theta_train).to(device)
    theta_train = theta_train.to(device)


inference_L = SNLE(prior=None, density_estimator="mdn")
load_model_NLE = False


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


if not load_model_NLE:
    inference_L = inference_L.append_simulations(theta_train.cpu(), x_train.cpu())

    density_estimator = inference_L.train(training_batch_size=1024,
                                          validation_fraction=0.1,
                                          max_num_epochs=3000,
                                          show_train_summary=False,
                                          stop_after_epochs=50)

    posterior_r = inference_L.build_posterior(prior=prior,density_estimator= density_estimator)

    torch.save(density_estimator.state_dict(), nle_ckpt)
else:
    with torch.no_grad():
        dummy_theta = torch.randn(256, theta_dim) * 0.1
        dummy_x = torch.randn(256, x_dim) * 0.1
    inference_L.append_simulations(dummy_theta, dummy_x)
    density_estimator = inference_L.train(stop_after_epochs=0)

    density_estimator.load_state_dict(torch.load(nle_ckpt, map_location=device))
    density_estimator.to("cpu")
    posterior_r = inference_L.build_posterior(prior = prior,density_estimator=density_estimator)


lamda = 0.9999
change_proposal = False

flow_ckpt = save_dir / f"routeB_NRE_weighted_npe_{n}.pt"


betas_sets = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
betas_sets = [1.0]

load_model_NPE = False

if not load_model_NPE:
    with torch.no_grad():
        log_r_all = log_r_hat(theta_train, x_train, prior=prior, posterior_r=posterior_r)

    theta_train = theta_train.to(device).float()
    x_train = x_train.to(device).float()
    log_r_all = log_r_all.to(device).float()

    theta_joint, x_joint, beta_joint, w_joint = get_routeB_samples(theta_train, x_train, log_r_all,
                                                                   betas_sets, change_proposal=change_proposal,
                                                                   lamda=lamda)

    print("Joint dataset:", theta_joint.shape, x_joint.shape, beta_joint.shape, w_joint.shape)



    model, norm_theta, norm_x, best_val = train_conditional_mdn(
        theta=theta_joint,
        x=x_joint,
        beta=beta_joint,
        weights=w_joint,
        hidden=128,
        n_components=9,
        n_hidden_layers=3,
        batch_size=2048,
        epochs=500,
        lr=1e-3,
        val_fraction=0.1,
        device=device,
        patience=30
    )

    # meta = {
    #     "hidden": 128,
    #     "n_hidden_layers": 3,
    #     "n_components": 9,
    #     "batch_size": 8196
    # }
    # save_checkpoint(str(flow_ckpt), model, norm_theta, norm_x, meta=meta)


else:
    model, norm_theta, norm_x, meta = load_checkpoint(
        str(flow_ckpt), hidden=128, n_hidden_layers=3, device=device
    )

for beta_scalar in betas_sets:
    with torch.no_grad():
        posterior_samples = sample_with_rejection(model, x_obs = x_obs, beta=beta_scalar,
                             norm_theta=norm_theta, norm_x=norm_x,
                             num_samples=10_000,lower=low,upper=high,
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

        # logger.log(
        #     method=method,
        #     n=n,
        #     beta=float(beta_scalar),
        #     c2st=float(c2st_accuracy),
        #     mmd=float(mmd_value),
        #     seed=SEED,
        #     device=device,
        #     ckpt_path=ckpt_path,
        #     notes=None
        # )
