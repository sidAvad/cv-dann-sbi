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
| 3b | 300k sims, λ=0.5, 100-ep ramp, zscore scalar norm (exp branch — abandoned) | `exp-v3b_encoder-lipschitz_dann_flow-maf5` | done — task=13.03, w1=0.760 — worse than v3 |
| 3c | PCA-nearest 300k sims, legacy norm, λ=0.5, 100-ep ramp | `exp-v3c_encoder-lipschitz_dann_flow-maf5` | done — task=10.19, w1=1.00; not obviously better than v3 on reals |

**v3b finding (ABANDONED)**: task=13.03, W1=0.760 — worse than v3 overall. Zscore normalization using separate sim/real stats introduced a domain gap at the input level: the same physical measurement (e.g. SBP=120 mmHg) maps to different normalized values depending on which population's stats are used. Legacy normalization is correct because it uses the same constants for both sims and reals — same physical value → same input to the encoder.

**v3_nosv finding**: task=14.02 (worse than v3 as expected without SV info). SV ablation shows 1NN proxy massively better than direct inference for SV recovery — residual domain gap shifts posterior means for volume parameters. Posterior means differ between direct/1NN with comparable stds → flow is conditioned OOD for volume params, not decoder brittleness. Affects Vrv/Vlv volume parameters most; pressure/resistance params (SVR, PVR, Ras, Rap) work well on reals.

### v3c — PCA-nearest sim subset, legacy norm (RUNNING)

**Result**: task=10.19, w1=1.00. Better sim-side NLL than v3 (10.19 vs 12.84) but domain alignment plateaued at w1≈1.0 from ep270 onward — worse than v3's 0.85. Real patient results not obviously better. PCA selection tightened the training distribution but created a harder-to-align latent gap. **v3 remains best real-patient model. DANN approach at this scale appears to have hit its ceiling — moving to SPIN.**

**Command**:
```bash
python train_joint.py --run exp-v3c_encoder-lipschitz_dann_flow-maf5 --version 3c \
  --objective flow-maf5 \
  --sim-data-root /media/local/SimData/hdf5/cv8/simset_10M_cv8Eed_20260314 \
  --manifest-train /home/sa4604/cv-dann-sbi/manifest_train_v3.3.json \
  --stats-path /home/sa4604/cv-dann-sbi/norm_stats_v3c.json \
  --real-data /home/sa4604/real_data/onebeat_300patients --n-sims 300000 \
  --lambda-target 0.5 --lambda-warmup 100 --use-mixup --max-epochs 400
```

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

## Planned branches

### exp/lip-flow-dann-reconstruct — reconstruction objective

Replace the MAF5 flow with a reconstruction head (decoder x̂ = dec(z), MSE/NLL on x). Everything else identical to v3 (Lipschitz encoder, WDGRL, 300k sims, legacy norm). Tests whether the flow objective is load-bearing or if a simpler reconstruction loss gives comparable alignment.

### exp/spin — domain translation (likely separate repo)

SPIN domain translation: train G_sr (sim→real) and G_rs (real→sim) generators on raw observations. Cycle x_sim → G_sr → G_rs = x_srs retains θ labels; information-preservation loss = MI(θ; x_srs) via flow log-prob. At test time: x_real → G_rs → encoder → flow → posterior. No real labels needed. Start with frozen v3 encoder+flow, then retrain jointly. Keep generators small (1D ResNet, 3–4 blocks) given only 802 real patients for discriminator. Likely warrants a separate git repo given scope.

## Next steps

- **Prior acceptance rate**: v3 eval shows ~5% acceptance (50/1000 samples in-prior per patient). Not a problem — 50 samples gives stable posterior means. Try evaluating without prior filter and compare scatter plots to see if it changes results meaningfully.

- **Extended v3 run**: v3 was still improving at ep400 (task trending 12.4–12.8 in final 50 ep, patience=30 never fired). Consider rerunning to 600–800 epochs with identical config.


## Infra

**Output sync mithril → adamant**: one-way Mutagen sessions configured in `mutagen.yml` (`outputs-mithril-to-adamant`, `dry-runs-mithril-to-adamant`). Start with `mutagen project start`.

**Sim data on mithril**: done — 110 GB, 139 train + 10 test files at `/home/sa4604/sim_data/`. Use `--sim-data-root ~/sim_data` for mithril runs.
