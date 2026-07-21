# Experiments

## Results

| # | Encoder | Objective | Run name | Result |
|---|---------|-----------|----------|--------|
| 1 | lipschitz (64-dim) | flow-maf5 | `exp-v1_encoder-lipschitz_dann_flow-maf5` | task=30.24, dom=0.0014 (BCE saturated) |
| 1.1 | lipschitz (128-dim) | flow-maf5 | `exp-v1.1_encoder-lipschitz_dann_flow-maf5` | task=28.22, dom=0.0003 (BCE saturated) |
| 1.3 | lipschitz (64-dim, λ=0.5) | flow-maf5 | `exp-v1.3_encoder-lipschitz_dann_flow-maf5` | task=29.43, dom=0.0002 (BCE saturated) |
| 2 | lipschitz (128-dim) | WDGRL + flow-maf5 | `exp-v2_encoder-lipschitz_dann_flow-maf5` | task=28.95, w1=2.0↓ (hit 200 ep) |

**v1 finding**: BCE saturates to ~0 regardless of λ; encoder gets near-zero domain gradient. 128-dim latent improves flow quality independently.

**v2 finding**: WDGRL working — W1 non-trivial and slowly decreasing (2.47→1.98 over 200 ep). Task loss 28.95 slightly worse than v1.1 (28.22) because domain gradient is real. Hit 200 ep limit still improving — needs more epochs or higher λ.

## Next runs

### v2 series — hyperparameter search (100k sims, 802 reals, lipschitz encoder)
Gate for v3: confirm adversarial tension (W1 and task diverge under higher λ) and lock in best λ + epoch count.

| # | Change | Run name | Status |
|---|--------|----------|--------|
| 2.1 | 400 epochs, λ=0.1 | `exp-v2.1_encoder-lipschitz_dann_flow-maf5` | done — task=20.56, w1=1.65 at ep400 (hit max, still improving) |
| 2.2 | λ=0.3, 60-ep ramp | `exp-v2.2_encoder-lipschitz_dann_flow-maf5` | done — task=23.75, w1=1.23 at ep400 |
| 2.3 | λ=0.5, 100-ep ramp | `exp-v2.3_encoder-lipschitz_dann_flow-maf5` | done — task=23.97, w1=0.96 at ep400 |

**v3.1 finding**: Sim metrics improve substantially over v3 (tighter posteriors, task=-3.1 nats). But real patient performance is worse: Rap R² 0.940→0.802, Ras R² 0.968→0.714, Emax_RV collapses (PAH std=3.79, MWU p=0.97). Root cause: flow posteriors are so sharp that residual encoder domain shift now lands in the wrong narrow peak. Lower W1 (0.57 vs 0.85) didn't compensate. Real patient calibration similarly poor in both (90% CI coverage ~10%) — posteriors are calibrated for noiseless sims; real measurement noise falls outside them. **v3 remains the best model for real patients.** For v4: consider sim observation noise augmentation or posterior temperature scaling to improve real-patient robustness.

**v3 finding**: Calibration is excellent (mean 90% CI coverage = 0.919 across all 24 params, range 0.887–0.950). Hemodynamic resistance params (Ras, Rap) near-perfect on sims; Emax_RV, Tmax, τ also strong. Compliance/timing params (Eedref_*, Bla/Blv/Bra/Brv, τ_a) have poor R² (0.04–0.42) — structurally unidentifiable from 4 pressure waves + 5 scalars. Real patient SVR/PVR results strong; Cas/Eap GT unreliable on real data. Sim→real gap for Cas (sim MAPE=1.4% → real 25%) reflects residual domain shift; expect improvement at 1M sims.

### v3 series — scale up (300k sims, reals augmented to ~300k via Mixup, best v2 hyperparams)
Gate for v4: v3 confirms alignment at scale; Mixup note: interpolated reals add critic diversity but no new physiology.

| # | Change | Run name | Status |
|---|--------|----------|--------|
| 3 | 300k sims + Mixup reals, λ=0.5, 100-ep ramp | `exp-v3_encoder-lipschitz_dann_flow-maf5` | done — task=12.84, w1=0.85 at ep400; sim: Ras R²=0.998/MAPE=0.8%, Rap R²=1.000/MAPE=1.4%, coverage mean=0.919 @ 90% CI; real: SVR MAPE=4.4% R²=0.968, PVR MAPE=12.7% R²=0.940 |
| 3.1 | 1M sims + Mixup reals, λ=0.5, 100-ep ramp, 600 epochs | `exp-v3.1_encoder-lipschitz_dann_flow-maf5` | done — task=-3.13, w1=0.57 at ep600; sim: Ras R²=0.999/MAPE=0.6%, Rap R²=1.000/MAPE=1.0%, coverage mean=0.905 @ 90% CI; real: SVR MAPE=10.4% R²=0.714, PVR MAPE=21.0% R²=0.802 |

### v3.2 — stronger adversarial pressure at 1M sims
**Hypothesis**: v3.1 showed that sharper posteriors (1M sims) amplify residual domain shift — flow peaks
become narrow enough that small encoder misalignment lands in the wrong mode on real patients. The fix
is to scale adversarial pressure with sim count: higher λ forces tighter alignment before the flow sharpens.

**Mixup real scaling**: on-the-fly, so automatically ~1M encoder-facing + 5M critic-facing samples/epoch
at 1M sims (vs 300k + 1.5M for v3). All from 802 source patients — denser coverage of the same convex
hull each epoch, not new diversity. This caps how high λ can usefully go before over-aligning to the
802-patient sample.

**λ=2, 400-ep ramp**: same ramp rate as v3 (0.005/ep), continuing to a higher target. Encoder trajectory
directly comparable to v3 for first 100 epochs then keeps climbing; ~150 epochs at full λ=2.

| # | Change | Run name | Status |
|---|--------|----------|--------|
| 3.2 | 1M sims, λ=2, 400-ep ramp, 600 ep | `exp-v3.2_encoder-lipschitz_dann_flow-maf5` | done — task=-1.70, w1=0.38 at ep600; real patient eval pending (run `v3-series_real-patient-alignment.ipynb`) |
| 3_nosv | 300k sims, no SV scalar (808-dim), legacy norm | `exp-v3_nosv_encoder-lipschitz_dann_flow-maf5` | done — task=14.02, w1=0.84 |
| 3b | 300k sims, λ=0.5, 100-ep ramp, real inputs z-scored with real stats | `exp-v3b_encoder-lipschitz_dann_flow-maf5` | done — task=13.03, w1=0.760 |
| 3c | PCA-nearest 300k sims + real norm stats + v3c scalar stats, λ=0.5, 100-ep ramp | `exp-v3c_encoder-lipschitz_dann_flow-maf5` | running |

**v3b finding**: task=13.03 (near v3's 12.84), but W1=0.760 vs v3's 0.853 — better domain alignment. Proper scalar normalization (zscore mode) removes input-level scale mismatch before WDGRL, so adversarial budget goes to residual distributional differences only.

**v3_nosv finding**: task=14.02 (worse than v3 as expected without SV info). SV ablation shows 1NN proxy massively better than direct inference for SV recovery — residual domain gap shifts posterior means for volume parameters. Posterior means differ between direct/1NN with comparable stds → flow is conditioned OOD for volume params, not decoder brittleness. Affects Vrv/Vlv volume parameters most; pressure/resistance params (SVR, PVR, Ras, Rap) work well on reals.

### v3b — real-data normalisation (DONE)

**Result**: task=13.03, w1=0.760. W1 lower than v3 (0.853) confirming better alignment. Scalar zscore normalization implemented: each summary (MAP, SBP, DBP, SV, HR) z-scored using its own cross-sim (or cross-real) distribution. Sim stats in `norm_stats.json["scalars"]`, real stats in `real_norm_stats.json["scalars"]`.

**Domain gap at input level** (real mean − sim mean, in sim σ units):

| Channel | Sim mean | Real mean | Sim std | Shift (sim σ) |
|---------|----------|-----------|---------|---------------|
| Prv | 19.7 | 27.0 | 20.4 | +0.36σ |
| Pra | 8.2 | 8.5 | 7.1 | +0.04σ |
| Pvp | 16.0 | 13.3 | 11.8 | −0.23σ |
| Pap | 31.6 | 36.6 | 18.0 | +0.28σ |
| Pas scalar | 76.1 | 102.6 | 27.6 | **+0.97σ** |
| HR | 69.4 | 75.9 | 18.9 | +0.34σ |
| SV (÷ Vlv_std) | — | 65.9 mL | Vlv_std=39.5 | scalar=1.67 vs ~1 |

Pas is nearly 1σ shifted (real patients have much higher systemic pressures); SV scalar sits at 1.67
when sims are centred near 1. With `real_norm_stats.json` all real inputs land near zero mean / unit
variance in their own distribution. Same 300k sims, λ=0.5, 100-ep ramp, 400 epochs as v3 — cleanest
possible comparison.

**Implementation**: `--real-norm-stats real_norm_stats.json` flag in `train_joint.py`. Stats computed
by `scripts/compute_real_stats.py` and saved to `real_norm_stats.json` (not committed — generated on
adamant from the 802 real patient H5 files).

### v3c — PCA-nearest sim subset + real norm stats (RUNNING)

**Hypothesis**: PCA-nearest 300k sims (waveform-shape nearest to reals) + real stats normalization + v3c-specific scalar stats. WDGRL handles only residual distributional differences.

**Key detail**: scalar stats for v3c computed from the PCA-nearest 300k subset (`norm_stats_v3c.json`), not the full sim pool — their distributions differ (e.g. map=71.5 vs 76.2 for full 300k).

**Command**:
```bash
python train_joint.py --run exp-v3c_encoder-lipschitz_dann_flow-maf5 --version 3c \
  --objective flow-maf5 \
  --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \
  --manifest-train /home/sa4604/cv-dann-sbi/manifest_train_v3.3.json \
  --stats-path /home/sa4604/cv-dann-sbi/norm_stats_v3c.json \
  --real-data /home/sa4604/real_data/onebeat_300patients --n-sims 300000 \
  --lambda-target 0.5 --lambda-warmup 100 --use-mixup --max-epochs 400 \
  --scalar-norm zscore --real-norm-stats /home/sa4604/cv-dann-sbi/real_norm_stats.json
```

---

### v4 series — VAE encoder
| # | Change | Run name | Status |
|---|--------|----------|--------|
| 4 | VAE encoder + WDGRL, 300k sims, λ=0.5, 100-ep ramp, kl=1e-4 | `exp-v4_encoder-vae_dann_flow-maf5` | done — task=9.57, w1=1.58 at ep400 (hit max); KL unregularized (kl_weight too small) |
| 4.1 | VAE encoder, kl_weight=1e-2, 600 epochs | `exp-v4.1_encoder-vae_dann_flow-maf5` | done — task=3.34, w1=1.75 at ep600 (hit max); KL forced down (955→440), task improved vs v4 but W1 alignment worse; VAE series abandoned (see notes) |

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

## Next steps

- **Prior acceptance rate**: v3 eval shows ~5% acceptance (50/1000 samples in-prior per patient). Not a problem — 50 samples gives stable posterior means. Try evaluating without prior filter and compare scatter plots to see if it changes results meaningfully.

- **Stochastic encoder + parameter decoder (implicit posterior)**: replace the flow with a stochastic encoder x → (μ_z, σ_z) and a deterministic MLP decoder z → θ. Posterior at inference: sample z₁...z_N ~ q(z|x), push through decoder — {θ̂_i} is the posterior. Training objective is MSE/NLL on θ directly (no reconstruction of x). Needs mild KL(q(z|x) ∥ N(0,I)) regularisation to prevent σ collapse; WDGRL alignment applies in z-space as before. Calibration check is identical to current approach (coverage of true θ in percentile intervals of {θ̂_i}). Simpler and faster than flow; posterior is implicit but fully sampleable.

### Post-hoc corrections

Real-patient posteriors are miscalibrated — the posterior is sharper than warranted because the encoder/flow were trained on sim data that does not match the noise and variability of real waveforms. Two post-hoc fixes that do not require retraining:

- **Temperature scaling**: divide the flow log-prob by a temperature T before sampling, which broadens posteriors uniformly. T cannot be estimated without real ground-truth θ, but SVR and PVR can be derived directly from cath lab measurements (mean arterial pressure / cardiac output via thermodilution). Calibrate T by minimising the calibration error on SVR and PVR across the 802 real patients, then apply the same T to all 25 parameters.

- **Post-hoc calibration fixes**: after temperature scaling, per-parameter recalibration (isotonic regression or Platt scaling on the percentile ranks) can correct residual asymmetries. Requires the same 2 ground-truth quantities (SVR/PVR) as a calibration set.

### Data-level domain adaptation

Address the sim-to-real gap at the observation level rather than in latent space:

- **SPIN domain translation (v5)**: train G_sr (sim→real) and G_rs (real→sim) generators on raw observations. Cycle x_sim → G_sr → G_rs = x_srs retains θ labels; information-preservation loss = MI(θ; x_srs) via flow log-prob. At test time: x_real → G_rs → encoder → flow → posterior. No real labels needed. Start with frozen v3 encoder+flow (v5a), then try with v4 VAE encoder (v5b). Keep generators small (1D ResNet, 3–4 blocks) given only 802 real patients for discriminator. Gate: if v5a fails, 802 reals is likely too few for the discriminator — v5b won't help.

- **Realistic noise model**: characterise the noise and artefact structure in real patient waveforms (sensor noise, catheter ringing, respiration drift) and add matched noise to sim observations at training time. This narrows the domain gap before WDGRL or SPIN are applied. Start with a simple parameterised model (additive Gaussian + low-frequency sinusoidal drift) fit to real waveform residuals.

- **Learning noise parameters jointly**: eventually learn the noise parameters (noise amplitude, drift frequency/amplitude) as latent variables alongside θ, with a prior that regularises them toward the characterised noise model. Requires the noise model to be differentiable so gradients flow back through the noise process into the flow.

## Infra

**Output sync mithril → adamant**: one-way Mutagen sessions configured in `mutagen.yml` (`outputs-mithril-to-adamant`, `dry-runs-mithril-to-adamant`). Start with `mutagen project start`.

**Sim data on mithril**: done — 110 GB, 139 train + 10 test files at `/home/sa4604/sim_data/`. Use `--sim-data-root ~/sim_data` for mithril runs.
