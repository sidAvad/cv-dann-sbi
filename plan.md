# Plan

## Next experiments

| # | Change | Run name | Rationale |
|---|--------|----------|-----------|
| 2.1 | extend to 400 epochs | `exp-v2.1_encoder-lipschitz_dann_flow-maf5` | v2 hit 200 ep still improving (wait=2) |
| 2.2 | higher λ (0.3–0.5) | `exp-v2.2_encoder-lipschitz_dann_flow-maf5` | W1 decreasing slowly at λ=0.1 |
| 2.3 | add Mixup on real beats | `exp-v2.3_encoder-lipschitz_dann_flow-maf5` | more diversity for critic, reduce memorisation |
| 3 | VAE encoder + WDGRL | `exp-v3_encoder-vae_dann_flow-maf5` | smooth/regularized latent |

## WDGRL implementation (v2, implemented)

Three-phase schedule: flow-warmup (enc frozen) → enc-warmup (flow frozen, WDGRL at λ_warm=0.01) → joint (enc+flow+WDGRL, λ ramps to target).

**WassersteinCritic** (`models.py`): MLP (latent→128→128→1), LeakyReLU(0.2), no sigmoid. Estimates W1 via KR duality.

**Gradient penalty**: interpolates `ẑ = ε·z_sim + (1-ε)·z_real`, `gp = ((||∇f(ẑ)||₂ - 1)²).mean()`. `create_graph=True` is critical.

**Critic inner loop** (n_critic=5 per encoder step, encoder detached):
- `L_critic = -(E_sim[f(z)] - E_real[f(z)]) + 10 * gp`
- Critic Adam: β1=0.5

**Encoder step**: `total = task_loss + λ * (E_sim[f(z_sim)] - E_real[f(z_real)])`

**λ schedule**: linear or sigmoid `2/(1+e^{-γt})-1` to `--lambda-target`. Phase 1 uses `--lambda-enc-warmup`.

**CLI**: `--n-critic 5`, `--gp-weight 10`, `--critic-hidden 128`, `--lambda-enc-warmup 0.01`, `--lambda-schedule [linear|sigmoid]`, `--lambda-gamma 10`

**CSV columns**: `task`, `w1_est`, `gp`, `total`, `lambda`, `phase`

## Mixup augmentation spec (planned for v2.3+)

Per batch draw `batch_size` pairs `(i,j)` from 802 real beats, interpolate `x = α·x_i + (1-α)·x_j` with `α ~ Beta(0.4, 0.4)`, optionally add Gaussian noise (σ_wave≈0.03, σ_scalar≈0.01). Expands effective diversity from 802 to ~321k unique pairs per epoch.

## Infra

**Output sync mithril → adamant**: one-way Mutagen sessions configured in `mutagen.yml` (`outputs-mithril-to-adamant`, `dry-runs-mithril-to-adamant`). Start with `mutagen project start`.

**Sim data on mithril**: copy in progress from `/media/pulsar/` to `/home/sa4604/sim_data/` (~114 GB). Once done, use `--sim-data-root ~/sim_data` for mithril runs. Update CLAUDE.md data layout section when confirmed.
