import torch
import numpy as np

from sbi.inference import SNRE_B
from sbi.utils import BoxUniform

from NSEutils import ExperimentLogger,plot_posteriors_compare,to_float
from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd
from pathlib import Path
from sbibm import get_task

from RouteB_utils import (
    train_conditional_mdn,
    log_r_hat,
    get_routeB_samples,
    sample_with_rejection,
    save_checkpoint,
    load_checkpoint
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# name = ["gaussian_mixture","two_moons","slcp","sir"]
# number = 0
# n = 10_000
#
#
# task = get_task(name[number])
# prior = task.get_prior()
# simulator = task.get_simulator()
# theta_dim, x_dim = task.dim_parameters,task.dim_data
#
# x_obs = task.get_observation(num_observation=1)
#
#
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
nre_ckpt = save_dir / f"NRE_ckpt_{n}.pt"
metrics_csv = save_dir / "metrics.csv"
logger = ExperimentLogger(metrics_csv)
method = "RouteB_NRE"

with torch.no_grad():
    theta_train = prior(n)
    x_train = simulator(theta_train).to(device)
    theta_train = theta_train.to(device)


inference_r = SNRE_B(classifier="resnet")
load_model_NRE = True

# if number == 0 :
#     low = torch.tensor([-10.0, -10.0],device = device)
#     high=torch.tensor([10.0, 10.0],device = device)
# elif number == 1:
#     low = torch.tensor([-1.0, -1.0], device=device)
#     high = torch.tensor([1.0, 1.0], device=device)
# elif number == 2:
#     low = torch.tensor([-3.0, -3.0, -3.0, -3.0, -3.0], device=device)
#     high = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0], device=device)


prior = BoxUniform(low= low, high= high)


if not load_model_NRE:
    inference_r = inference_r.append_simulations(theta_train.cpu(), x_train.cpu())

    density_estimator = inference_r.train(training_batch_size=256,
                                          validation_fraction=0.1,
                                          max_num_epochs=3000,
                                          show_train_summary=False,
                                          stop_after_epochs=30)
    density_estimator.eval()
    posterior_r = inference_r.build_posterior(prior=prior,density_estimator= density_estimator)

    torch.save(density_estimator.state_dict(), nre_ckpt)
else:
    with torch.no_grad():
        dummy_theta = torch.randn(256, theta_dim) * 0.1
        dummy_x = torch.randn(256, x_dim) * 0.1

    inference_r.append_simulations(dummy_theta, dummy_x)

    de = inference_r.train(stop_after_epochs=0)
    de.load_state_dict(torch.load(nre_ckpt, map_location=device))
    de.to("cpu").eval()
    posterior_r = inference_r.build_posterior(prior = prior,density_estimator=de)


@torch.no_grad()
def mix_resample_from_joint(theta, x, lam=0.5, n_out=None, generator=None):
    assert 0.0 < lam < 1.0
    N = theta.shape[0]
    if n_out is None: n_out = N

    n_ind = int(round(lam * n_out))
    n_joint = n_out - n_ind


    idx_j = torch.randint(N, (n_joint,), generator=generator)
    theta_j = theta[idx_j]
    x_j = x[idx_j]


    idx_t = torch.randint(N, (n_ind,), generator=generator)
    idx_x = torch.randint(N, (n_ind,), generator=generator)
    theta_i = theta[idx_t]
    x_i = x[idx_x]

    theta_mix = torch.cat([theta_j, theta_i], dim=0)
    x_mix = torch.cat([x_j, x_i], dim=0)

    perm = torch.randperm(n_out, generator=generator)
    return theta_mix[perm], x_mix[perm]


lamda = 0.5
change_proposal = False

if change_proposal:
    theta_train,x_train = mix_resample_from_joint(theta_train,x_train,lam = lamda)

flow_ckpt = save_dir / f"routeB_NRE_weighted_npe_{n}.pt"

load_model_NPE = False
betas_sets = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
betas_sets = [1.0]

if not load_model_NPE:

    with torch.no_grad():
        log_r_all = log_r_hat(theta_train, x_train, prior=prior, posterior_r=posterior_r)

    theta_train = theta_train.to(device).float()
    x_train = x_train.to(device).float()
    log_r_all = log_r_all.to(device).float()

    theta_joint, x_joint, beta_joint, w_joint = get_routeB_samples(theta_train, x_train, log_r_all,
                                                                   betas_sets,change_proposal=change_proposal,lamda=lamda)

    print("Joint dataset:", theta_joint.shape, x_joint.shape, beta_joint.shape, w_joint.shape)

    model, norm_theta, norm_x, best_val= train_conditional_mdn(
        theta=theta_joint,
        x=x_joint,
        beta=beta_joint,
        weights=w_joint,
        hidden=128,
        n_components=8,
        n_hidden_layers=3,
        batch_size=512,
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
    #     "batch_size":32784
    # }
    # save_checkpoint(str(flow_ckpt), model, norm_theta, norm_x,meta = meta)

else:
    model, norm_theta, norm_x, meta = load_checkpoint(
        save_dir, hidden=128,
        n_hidden_layers=5, device=device
    )


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

