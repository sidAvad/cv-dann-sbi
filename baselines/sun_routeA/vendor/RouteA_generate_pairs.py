import wandb
from RouteA_utils import *
import sbibm
from torch.utils.data import DataLoader,random_split


device = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
seed_everything(SEED)

# name = ["gaussian_mixture","two_moons","slcp","sir"]
# number = 2
# task = sbibm.get_task(name[number])
# prior = task.get_prior()
# simulator = task.get_simulator()
# observation = task.get_observation(num_observation=1).to(device)
# refernce_samples = task.get_reference_posterior_samples(num_observation=1).to(device)
# save_dir = Path(name[number])
#
# theta_dim, x_dim = task.dim_parameters,task.dim_data

import torch
from lorenz96 import PriorBox, make_prior_fn,make_simulator_fn,Theta

prior_box = PriorBox(1.4, 2.2, 0.1, 1.0, 1.5, 2.5)

prior = make_prior_fn(prior_box, device=device, dtype=torch.float32)
K, T, dt = 8, 20, 3.0/40.0
x0 = torch.ones(K, device=device)
simulator = make_simulator_fn(K, T, dt, R=5,x0=x0, device=device, dtype=torch.float32, out_device="cpu")
save_dir = Path("lorenz96")
theta_dim,x_dim = 3, 840

n = 10_000
batch_size = 1024
learning_rate = 1e-3
epoch_ = 1500
T = 10
gamma_min = 0.01
gamma_max = 1.0


ckpt_path = save_dir / f"Amortize_theta_and_x_sigma({gamma_min},{gamma_max},{T})_{n}.pt"

sigma_list = make_gamma_schedule(T = T, gamma_min= gamma_min, gamma_max = gamma_max)
model = ScoreNet_sigma(theta_dim=theta_dim, x_dim=x_dim,amortize=True ,emb_dim=128)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)

early_stopper = EarlyStopping(
    patience=100,
    min_delta=1e-5,
    mode="min",
    restore_best_weights=True,
    verbose=True
)

best_val = float("inf")

load_model = True
if not load_model:
    wandb.init(project="Generalized Bayes socre matching for pairs",
               name=f"learning rate:{learning_rate}",
               config={"learningrate" : learning_rate,
                       "epoch":epoch_,
                                                   })

    tr_ind, val_ind = random_split(range(n), [0.90, 0.10])
    dataset = PairDataset(prior, simulator,n)  #[theta,x]
    tr_dataloader = DataLoader(dataset[tr_ind], batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(dataset[val_ind], batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


    for epoch in tqdm(range(epoch_)):

        train_loss = train(model, tr_dataloader, sigma_list, optimizer,theta_dim = theta_dim)
        scheduler.step()
        val_loss = validate(model, val_dataloader, sigma_list,theta_dim = theta_dim)

        if epoch > 50:
            wandb.log(
                {"train_loss": float(train_loss), "val_loss": float(val_loss)},
                step=epoch
            )

        if epoch > 500 :
            if val_loss < best_val - 1e-8:
                best_val = float(val_loss)
                torch.save(model.state_dict(), ckpt_path)

                # Early stopping
            if early_stopper.step(float(val_loss), model):
                print(f"[EarlyStopping] Stop at epoch {epoch}. Best val={early_stopper.best_score:.6f}")
                break

else:
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))


n_samples = n

all_pairs = []

beta_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for beta in beta_list:
    with torch.no_grad():
        posterior_samples = Lgmcmc(gammas = sigma_list, prior = prior,simulator=simulator, model = model,
                                   theta_dim = theta_dim,x_dim = x_dim,
                                   n_samples = n_samples,initial_step = 10,
                                   t_power = beta,step_increase = 10,
                                   device=device,
               )
        theta,x = posterior_samples[:,:theta_dim].to(device).float(),posterior_samples[:,theta_dim:].to(device).float()

        beta_col = torch.full((x.size(0), 1), float(beta), device=device, dtype=x.dtype)

        pair = torch.cat([theta, x, beta_col], dim=-1)  # [N, theta_dim + x_dim + 1]
        all_pairs.append(pair)
        ckpt_path = save_dir / f"True_Power_posterior_with_t_power={beta}_10000.pt"
        reference_samples = torch.load(ckpt_path, map_location=device)

        plot_posteriors_compare(reference_samples=reference_samples,posterior_samples=theta,title=f"Beta:{beta:.2f}")


pairs_all = torch.cat(all_pairs, dim=0)  # [len(t_list)*n_samples, theta_dim+x_dim+1]

out_path = save_dir / f"pairs_all_{n_samples}.pt"
torch.save(pairs_all, out_path)