import torch
from flow_matching_utils import FMPEConfig,FMPE,PairDataset,plot_losses
from NSEutils import seed_everything,plot_posteriors_compare,to_float
from torch.utils.data import DataLoader,random_split
import sbibm
from pathlib import Path

from sbibm.metrics import c2st
from sbibm.metrics.mmd import mmd

if __name__ == "__main__":
    SEED = 42
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = ["gaussian_mixture", "two_moons", "gaussian_linear", "bernoulli_glm"]
    number = 1
    task = sbibm.get_task(name[number])
    prior = task.get_prior()
    simulator = task.get_simulator()
    x_obs = task.get_observation(num_observation=3).to(device)
    reference_samples = task.get_reference_posterior_samples(num_observation=3).to(device)
    save_dir = Path(name[number])


    theta_dim, x_dim = task.dim_parameters, task.dim_data
    ref_sampler = lambda B: torch.randn(B,theta_dim)
    N = 10_000

    dataset = PairDataset(prior, simulator, N, device)
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    g = torch.Generator().manual_seed(SEED)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=g)
    tr_loader = DataLoader(train_set, batch_size=300, shuffle=True, drop_last=True)
    va_loader = DataLoader(val_set, batch_size=1024, shuffle=False, drop_last=False)

    cfg = FMPEConfig(
        theta_dim=theta_dim, x_dim=x_dim,
        hidden=128, depth=6, lr=1e-3,
        pairing="independent",device=device,
        x_feat_dim=128,
        x_enc_hidden=128, x_enc_depth=0
    )

    fmpe = FMPE(cfg, ref_sampler=lambda B: prior(B).to(device))

    fmpe.fit(tr_loader, va_loader, epochs=400, early_stop_patience=30)
    plot_losses(fmpe.history, save_path=None, show=True)


    with torch.no_grad():
        posterior_samples = fmpe.sample_posterior(x_obs, n_samples=10000, ode_steps=800)
        c2st_accuracy = to_float(c2st(reference_samples, posterior_samples))
        mmd_value = to_float(mmd(reference_samples, posterior_samples))
        print(f"c2st:{c2st_accuracy:.4f}")
        print(f"MMD: {mmd_value:.4g}")

        plot_posteriors_compare(reference_samples, posterior_samples,title=f"c2st={c2st_accuracy:.4f}, mmd={mmd_value:.4g}")

