# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

Simulation-based inference (SBI) over cardiovascular physiology. Given 4 right-heart pressure waveforms (Prv/Pra/Pvp/Pap) + 5 scalars (MAP/SBP/DBP/SV/HR) from a cath lab, infer a posterior over 24 physiological parameters using a jointly-trained Lipschitz encoder + MAF5 flow + WDGRL domain adaptation.

## Data layout

- HDF5 files under `<data-root>/train/` and `<data-root>/test/`
  - Adamant: `/media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314` (local NVMe — use this, not `/media/pulsar/`)
  - Mithril: `~/sim_data` (110 GB, 139 train + 10 test files)
- `manifest_train.json` / `manifest_test.json` at `<data-root>/`
- Each HDF5 group (`sim_NNNNNN`): `parameters/<key>` scalars, `waves/<key>` arrays of length 201
- 25 variable parameters defined by `pvar_low`/`pvar_high` in manifest config

### Key files on adamant (not committed)

| File | Purpose |
|------|---------|
| `norm_stats.json` | Wave + parameter stats from full sim pool |
| `norm_stats_v3c.json` | Wave + parameter stats from PCA-nearest 300k subset |
| `manifest_train_v3.3.json` | PCA-nearest 300k sim manifest (50-PC waveform PCA, 1M pool) |
| `real_data/onebeat_300patients/` | 802 real patient H5 files |

## Key constants

| Symbol | Value | Meaning |
|---|---|---|
| `N_PARAMS` | 25 | Total parameters (24 inferred + HR in observation) |
| `N_REDUCED_CHANNELS` | 4 | Observed waveforms: Prv, Pra, Pvp, Pap |
| `N_SCALARS` | 5 | MAP, SBP, DBP, SV, HR |
| `OBS_DIM` | 809 | 4×201 + 5 (or 808 with --no-sv) |
| `T` | 201 | Time steps per waveform |

## Run naming convention

`{type}-v{version}_{encoder-type}_dann_{objective}`

- `type`: `exp` or `dry` (512 sims)
- `encoder-type`: `encoder-lipschitz` or `encoder-vae`
- `objective`: `flow-maf5`, `flow-nsf8`, or `reconstruction`
- **Versions**: major digit (v1→v2→v3), sequential sub-versions (.1/.2/.3), side variants (b/c/d). `v3` = `v3.0`; suffixes explicit after first (v3b, v3.2c).

## Experiment tracking

- `outputs/{run_name}/` — full runs (gitignored)
- `dry-runs/{run_name}/` — dry runs (gitignored)
- Each run writes `run_info_v{version}.json` with git hash, config, version field
- **Always commit before starting an `exp_` run**
- **Always confirm run name/directory with user before executing any training run**

## Current experiment status

**See `experiments.csv`** for the full run table — version, run name, branch, git hash, what changed,
and the result. That's the current source of truth; don't duplicate its contents here. Bottom line:
`exp-v3_encoder-lipschitz_dann_flow-maf5` (task=12.84, w1=0.85) remains the best real-patient model.
Later attempts (v3.1, v3.2, v3c — more sims, higher λ, PCA-nearest subset) all improved sim-side metrics
but didn't improve real-patient results, or made them worse (over-sharp posteriors amplify residual
domain shift). DANN judged to have hit its ceiling at this scale — active work moved to `cv-sbi-spin`
(domain translation instead of domain-adversarial alignment).

## Key train_joint.py flags

```
--stats-path PATH          sim norm_stats JSON (default: norm_stats.json)
--manifest-train PATH      manifest JSON (absolute path or relative to sim-data-root)
--no-sv                    drop SV scalar → 808-dim observation
--n-sims N                 number of sims to load (default: N_SIMS_FULL=100k; always set explicitly)
```

## Versioning and branching convention

- **Version numbers** are assigned only when a run survives evaluation and is merged to `main`. They are hard-earned: minor improvements earn `.1`/`.2`, architectural changes earn a new major (v4, v5).
- **Git branches** (`exp/<what-you're-testing>`) are used for any change that requires new code, not just hyperparameters. If reverting requires more than deleting a flag, it's a branch. Abandoned branches are tombstones — do not merge.
- Pure hyperparameter variants (n-sims, epochs, λ) commit directly to `main`; the run name is the record.
- Planned branches: `exp/lip-flow-dann-reconstruct` (reconstruction objective), `exp/spin` (domain translation — likely a separate repo).

## WDGRL implementation notes (v2+)

Three-phase schedule: flow-warmup (enc frozen) → enc-warmup (flow frozen, WDGRL at λ_warm=0.01) → joint
(enc+flow+WDGRL, λ ramps to target).

**WassersteinCritic** (`models.py`): MLP (latent→128→128→1), LeakyReLU(0.2), no sigmoid. Estimates W1 via
KR duality.

**Gradient penalty**: interpolates `ẑ = ε·z_sim + (1-ε)·z_real`, `gp = ((||∇f(ẑ)||₂ - 1)²).mean()`.
`create_graph=True` is critical.

**Critic inner loop** (n_critic=5 per encoder step, encoder detached):
- `L_critic = -(E_sim[f(z)] - E_real[f(z)]) + 10 * gp`
- Critic Adam: β1=0.5

**Encoder step**: `total = task_loss + λ * (E_sim[f(z_sim)] - E_real[f(z_real)])`

**λ schedule**: linear or sigmoid `2/(1+e^{-γt})-1` to `--lambda-target`. Phase 1 uses `--lambda-enc-warmup`.

**CLI**: `--n-critic 5`, `--gp-weight 10`, `--critic-hidden 128`, `--lambda-enc-warmup 0.01`,
`--lambda-schedule [linear|sigmoid]`, `--lambda-gamma 10`

**CSV columns**: `task`, `w1_est`, `gp`, `total`, `lambda`, `phase`

## Mixup augmentation spec (v2.3+)

Per batch draw `batch_size` pairs `(i,j)` from 802 real beats, interpolate `x = α·x_i + (1-α)·x_j` with
`α ~ Beta(0.4, 0.4)`, optionally add Gaussian noise (σ_wave≈0.03, σ_scalar≈0.01). Expands effective
diversity from 802 to ~321k unique pairs per epoch.

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
- Git runs only on local Mac — never commit from adamant.

## Untracked shared docs

**`CLAUDE.md`, `experiments.csv`, and `.gitignore` are untracked from git** — standing instructions and
experiment bookkeeping, not code. Edit them in place; there is one physical copy, unaffected by
`git checkout`, so nothing needs propagating across branches. Untracking `.gitignore` itself doesn't
disable its effect — git reads it from disk regardless of whether it's tracked.
