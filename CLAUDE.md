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

| Run | Status | Notes |
|-----|--------|-------|
| `exp-v3_encoder-lipschitz_dann_flow-maf5` | done | task=12.84, w1=0.85 — **best real-patient model** |
| `exp-v3.1_encoder-lipschitz_dann_flow-maf5` | done | task=-3.13, w1=0.57 — worse on reals (over-sharp posteriors) |
| `exp-v3.2_encoder-lipschitz_dann_flow-maf5` | done | task=-1.70, w1=0.38 |
| `exp-v3_nosv_encoder-lipschitz_dann_flow-maf5` | done | task=14.02, w1=0.84 — 808-dim, no SV scalar |
| `exp-v3b_encoder-lipschitz_dann_flow-maf5` | done | task=13.03, w1=0.760 — zscore norm, worse than v3 |
| `exp-v3c_encoder-lipschitz_dann_flow-maf5` | done | task=10.19, w1=1.00 — not better than v3 on reals |

**Key finding**: 1NN proxy inference massively better than direct inference for SV recovery. Posterior means differ between direct/1NN with comparable stds → residual domain gap shifts volume-parameter posteriors OOD. Pressure/resistance params (SVR, PVR, Ras, Rap) work well on reals; volume params (Vrv/Vlv SV) do not.

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

## Git conventions

- Never add `Co-Authored-By: Claude` or any AI authorship trailer to commit messages.
- Always commit before running a full experiment.
- Git runs only on local Mac — never commit from adamant.
