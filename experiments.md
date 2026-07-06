# Experiments

## Results

| # | Encoder | Objective | Run name | Result |
|---|---------|-----------|----------|--------|
| 1 | lipschitz (64-dim) | flow-maf5 | `exp-v1_encoder-lipschitz_dann_flow-maf5` | task=30.24, dom=0.0014 (BCE saturated) |
| 1.1 | lipschitz (128-dim) | flow-maf5 | `exp-v1.1_encoder-lipschitz_dann_flow-maf5` | task=28.22, dom=0.0003 (BCE saturated) |
| 1.3 | lipschitz (64-dim, λ=0.5) | flow-maf5 | `exp-v1.3_encoder-lipschitz_dann_flow-maf5` | task=29.43, dom=0.0002 (BCE saturated) |
| 2 | lipschitz (128-dim) | WDGRL + flow-maf5 | `exp-v2_encoder-lipschitz_dann_flow-maf5` | task=28.95, w1=2.0↓ (hit 200 ep, still improving) |

**v1 finding**: BCE saturates to ~0 regardless of λ; encoder gets near-zero domain gradient. 128-dim latent improves flow quality independently.

**v2 finding**: WDGRL working — W1 non-trivial and slowly decreasing (2.47→1.98 over 200 ep). Task loss 28.95 slightly worse than v1.1 (28.22) because domain gradient is real. Hit 200 ep limit still improving — needs more epochs or higher λ.

## Next runs

| # | Change | Run name |
|---|--------|----------|
| 2.1 | extend to 400 epochs | `exp-v2.1_encoder-lipschitz_dann_flow-maf5` |
| 2.2 | higher λ (0.3–0.5) | `exp-v2.2_encoder-lipschitz_dann_flow-maf5` |
| 2.3 | add Mixup on real beats | `exp-v2.3_encoder-lipschitz_dann_flow-maf5` |
| 3 | VAE encoder + WDGRL | `exp-v3_encoder-vae_dann_flow-maf5` |

## WDGRL implementation notes (v2+)

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
