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
  - Mithril: `~/sim_data` (local copy, 110 GB, 139 train + 10 test files)
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
| 2 | lipschitz (128-dim) | WDGRL + flow-maf5 | `exp-v2_encoder-lipschitz_dann_flow-maf5` | done — task=28.95, w1=2.0↓ (hit 200 ep, still improving) |

**v1 finding**: BCE saturates to ~0; encoder gets near-zero domain gradient. 128-dim latent improves flow quality.

**v2 finding**: WDGRL working — W1 non-trivial and slowly decreasing (2.47→1.98 over 200 ep). Task loss 28.95 slightly worse than v1.1 (28.22) because domain gradient is real. Hit 200 ep limit still improving.

See `plan.md` for next experiments and implementation notes.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
