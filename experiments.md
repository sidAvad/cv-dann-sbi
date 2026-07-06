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

### v2 series — hyperparameter search (100k sims, 802 reals, lipschitz encoder)
Gate for v3: confirm adversarial tension (W1 and task diverge under higher λ) and lock in best λ + epoch count.

| # | Change | Run name | Status |
|---|--------|----------|--------|
| 2.1 | 400 epochs | `exp-v2.1_encoder-lipschitz_dann_flow-maf5` | running (adamant GPU 1) |
| 2.2 | higher λ (0.3–0.5) | `exp-v2.2_encoder-lipschitz_dann_flow-maf5` | pending |

### v3 series — scale up (300k sims, reals augmented to ~300k via Mixup, best v2 hyperparams)
Gate for v4: v3 confirms alignment at scale; Mixup note: interpolated reals add critic diversity but no new physiology.

| # | Change | Run name | Status |
|---|--------|----------|--------|
| 3 | 300k sims + Mixup reals | `exp-v3_encoder-lipschitz_dann_flow-maf5` | pending |

### v4 series — VAE encoder
| # | Change | Run name | Status |
|---|--------|----------|--------|
| 4 | VAE encoder + WDGRL, best v3 hyperparams | `exp-v4_encoder-vae_dann_flow-maf5` | pending |

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

**Sim data on mithril**: done — 110 GB, 139 train + 10 test files at `/home/sa4604/sim_data/`. Use `--sim-data-root ~/sim_data` for mithril runs.
