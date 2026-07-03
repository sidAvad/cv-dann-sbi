# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Simulation-based inference (SBI) over cardiovascular physiology. Given observed waveforms, infer a posterior distribution over the 25 physiological parameters using the [`sbi`](https://sbi-dev.github.io/sbi/) package with pre-simulated (theta, waveform) pairs.

Forked from `cv-inverse-autoencoder` (the surrogate training repo).

## Approach

- **No live simulator**: uses pre-simulated HDF5 data directly via `sbi`'s `append_simulations`
- **Inference**: Neural Posterior Estimation (NPE) via the `sbi` package (v0.26)
- **Observation**: 28 waveforms × 201 time steps per simulation

## Data layout

- HDF5 files under `<data-root>/train/` and `<data-root>/test/`; pass the root via `--data-root`
  - Adamant: `/media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314` (local NVMe — use this, not `/media/pulsar/` which is a slow network share)
- `manifest_train.json` / `manifest_test.json` live at `<data-root>/` (one level above the data dirs)
- Each HDF5 group (`sim_NNNNNN`) holds `parameters/<key>` scalars and `waves/<key>` arrays of length 201
- `norm_stats.json`: wave normalisation stats (not committed)
- 25 variable parameters defined by `pvar_low`/`pvar_high` in manifest config

## Key constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_PARAMS` | 25 | Parameters to infer |
| `N_CHANNELS` | 28 | Total waveform channels (24 continuous + 4 valve) |
| `T` | 201 | Time steps per waveform |

## Run naming convention

Joint training runs: `{type}-v{version}_{encoder-type}_dann_{objective}`

- `type`: `exp` (full run) or `dry` (512 sims, smoke test)
- `v{version}`: version string — use integers for major variants (`v1`, `v2`), decimals for minor tweaks (`v1.1`, `v1.2`); stored verbatim in `run_info_v{version}.json` as `"version": "1.1"`
- `encoder-type`: `encoder-lipschitz` or `encoder-vae`
- `dann`: always present — all train_joint.py runs use domain adversarial training
- `objective`: `flow-maf5`, `flow-nsf8`, or `reconstruction`

Examples:
- `exp-v1_encoder-lipschitz_dann_flow-maf5`
- `exp-v1.1_encoder-lipschitz_dann_flow-maf5`
- `exp-v1_encoder-lipschitz_dann_reconstruction`
- `exp-v1_encoder-vae_dann_flow-maf5`
- `dry-v1_encoder-lipschitz_dann_flow-maf5`

## Experiment tracking

- `outputs/{run_name}/` — full runs (gitignored)
- `dry-runs/{run_name}/` — dry runs (gitignored)
- Each run writes `run_info_v{version}.json` (e.g. `run_info_v1.1.json`) containing git hash, config, architecture, and `"version"` field; also writes `train_log.txt`
- **Always commit before starting an `exp_` run** so `run_info.json` captures the exact code
- **Always confirm the output run name/directory with the user before executing any training run** — never assume the name is correct, especially for variant runs that could overwrite existing results.

## Experiment plan

Goal: joint training of encoder + DANN + inference objective (train_joint.py).

| # | Encoder | Objective | Run name | Status |
|---|---------|-----------|----------|--------|
| 1 | lipschitz | flow-maf5 | `exp-v1_encoder-lipschitz_dann_flow-maf5` | done — task=30.24, dom=0.0014 |
| 1.1 | lipschitz (128-dim) | flow-maf5 | `exp-v1.1_encoder-lipschitz_dann_flow-maf5` | done — task=28.22, dom=0.0003 |
| 1.3 | lipschitz (64-dim, λ=0.5) | flow-maf5 | `exp-v1.3_encoder-lipschitz_dann_flow-maf5` | done — task=29.43, dom=0.0002 |
| 2 | lipschitz | WDGRL + flow-maf5 | `exp-v2_encoder-lipschitz_dann_flow-maf5` | next |

**v1 finding**: BCE domain loss saturates near 0 across all λ settings (equilibrium should be 0.693). Classifier trivially separates sim/real — DANN gradient to encoder is near-zero. 128-dim latent helps flow quality regardless.

### v2 plan: WDGRL (Wasserstein critic replaces BCE domain head)

Full spec in chat history (2026-07-02). Key changes to `train_joint.py` and `models.py`:

**Normalization**: verify sim stats applied identically to real (already done in `load_real_beats()`). Log `mean/std` of `x_all` vs `real_beats` at startup to confirm.

**`WassersteinCritic`** (add to `models.py`): small MLP (128→128→128→1), LeakyReLU(0.2), no sigmoid. Estimates W1 via Kantorovich-Rubinstein duality. Keep small (hidden=128) to avoid memorising 802 real beats.

**Gradient penalty** (add to `train_joint.py`): `gp = ((||∇_ẑ f(ẑ)||₂ - 1)²).mean()` at interpolates `ẑ = ε·z_sim + (1-ε)·z_real`. `create_graph=True` is critical — without it, GP contributes no gradient and critic stops being Lipschitz.

**Critic update loop** (`n_critic=5` per encoder step):
- Critic inner loop: `z = encoder(x).detach()` — no encoder gradient
- `L_critic = -(E_sim[f(z)] - E_real[f(z)]) + gp_weight * gp`  (`gp_weight=10`)
- Critic Adam: `β1=0.5` (WGAN-GP convention)
- Encoder step: `total = task_loss + λ * (E_sim[f(z_sim)] - E_real[f(z_real)])` — encoder minimizes W1

**Schedule**:
- Phase 0 (flow warmup): unchanged
- Phase 1 (enc warmup): BEGIN WDGRL at `--lambda-enc-warmup` (default 0.01)
- Phase 2 (joint sim): unchanged, no domain loss
- Phase 3 (joint+domain): ramp λ linearly or sigmoid `2/(1+e^{-γp})-1` to `--lambda-target`

**New CLI args**: `--n-critic 5`, `--gp-weight 10`, `--critic-hidden 128`, `--lambda-enc-warmup 0.01`, `--lambda-schedule [linear|sigmoid]`, `--lambda-gamma 10`

**New CSV columns**: `w1_est`, `gp` (in addition to existing task/total/lambda/phase)

**Deferred from v2**: Mixup augmentation (see below), reconstruction objective, VAE encoder.

### Other planned improvements
- **Real data augmentation (Mixup)**: on-the-fly Mixup of 802 real beats. Per batch draw `batch_size` pairs `(i,j)`, interpolate `x = λ·x_i + (1-λ)·x_j` with `λ ~ Beta(0.4, 0.4)`, add Gaussian noise (σ_wave≈0.03, σ_scalar≈0.01). Expands diversity from 802 fixed vectors to ~321k unique pairs per epoch. Pair with v2 WDGRL.
- **Sim data on mithril**: copy 10 HDF5 files (≈7.5 GB) covering 100k sims from pulsar to `/media/local/` on mithril for faster loading.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
