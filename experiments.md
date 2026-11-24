# Gibbs-NPE experiments

Rationale and tuning history. Code comments stay present-tense; the "why we changed X
after run Y showed Z" narrative lives here.

## Framing: two tiers

**Deployable** — everything before flow training uses only what a real patient offers:
the observation and the lofi bank. No patient theta. The metric (`ell`, `s_wave`,
`s_scal`), `w`, and any noise correction must be estimable from nearest-neighbour (or
OT-matched) residuals.

**Oracle** — uses theta-paired lofi/hifi residuals. Legitimate as a *ceiling* and as a
source of hyperparameters, because the pairing is sim-to-sim: we can run both fidelities
at any theta we choose, so no real-world labels are involved. Not deployable only because
a real patient has no theta-matched lofi partner, so the residual must be estimated by
matching instead.

Calibrating on the evaluation inputs is **not** leakage: the calibration never sees theta,
and the score is against theta. This is the ordinary transductive setting and it matches
deployment, where all patients both calibrate and are scored. Verified empirically --
swapping the benchmark index convention (905 of 1000 scored cases moving in or out of the
fit pool) changes the correction by `cos(mu_r) = 0.9998`, subspace overlap 0.954, max
`a_i` change 4.4%. The old `keep`-mask exclusion is therefore dropped; it existed only to
serve an exclusion that was never needed, and the `RandomState(0)` vs `default_rng(0)`
mismatch between the fit scripts and `eval_pretrained_lofihifi` was its only consequence.

## Negative levers (do not re-try)

**Oracle metric-shape refit.** Refitting `ell`/`s_wave`/`s_scal` on exact theta-paired
residuals moves the participation ratio only 26.5/156 -> 28.1/140. The residual stays
strongly anisotropic in the best member of the Matern family fitted on the truth, so the
family cannot express the discrepancy's structure. This is the ceiling on what any
matching improvement can buy for whitening, and it retroactively motivates the low-rank
`Sigma_c`: that correction does work the metric provably cannot.

**Limited capacity as the explanation for Route A's conservatism.** An arm using Route A's
exact mixture head (133,640 density params) on our CNN encoder sharpens past both Route A
(width95 0.729) and the 3.79M-param bounded NSF (0.727) by epoch 25, peaks at ep150
(CRPS skill +0.2325) and then degrades for 650 epochs to +0.156 with IS skill -1.004 and
224/1500 degenerate posteriors -- i.e. to roughly vanilla NPE's calibration. Route A's
width does not come from its size.

**Recentring the w window.** `wpaired` keeps `Sigma_c = I` and moves the window onto the
theta-paired w (0.713), window [0.071, 7.13]. Result is indistinguishable from the
baseline (CRPS +0.179 vs +0.192, IS -0.337 vs -0.328, cov 0.871 vs 0.875). The *location*
of w is not a lever. Window *placement relative to the useful w* still is -- see below.

**The measurement model, for this dataset.** `make_datasets.py` generated every set with
`measurement_model: "off"`, which looked like a serious realism gap. Measured instead:
white noise in the real beats is 0.002-0.017 mmHg (2nd-difference MAD) against signal RMS
0.77-20.4 mmHg, i.e. 0.1-0.3%; two independent estimators disagree 3-6x, so what little is
in the spectral tail is residual smooth structure, not noise. `calibrate_lowpass` puts the
real harmonic roll-off close to *unfiltered* hifi (mean log-spectral distance 0.667
unfiltered vs 0.537 at the best grid point), and the best filter makes Pvp eight times
worse while helping the other three, with one shared catheter across channels. Every real
MAP/SBP/DBP is an integer. The real beats are clean, rounded, clinician-processed traces.
Adding the layer would move hifi away from them. `offset_cath` is additionally
unidentifiable (confounded with the true population means and with structural bias) and is
absorbed by `mu_r` anyway.

Consequence: `PR = 26/156` is **not** a benchmark artefact. The anisotropy is genuine
structural discrepancy, so the `I` in `Sigma_c = I + sum a_i u_i u_i^T` is genuinely
misspecified.

**Per-bin w at inference: the likelihood gain does not reach the posterior.** The
conditional noise model's dominant term is the per-bin SCALE (held-out log-lik +16.1 nats,
against +1.2 for the per-bin offset). The scale needs no retraining: training draws
log w over the window independent of theta and conditions on it, so the Bayes-optimal
conditional is the Gibbs posterior at whatever w is supplied, and plugging in w_b at
evaluation IS the x-dependent-scale noise model's target. Tested on four already-trained
arms, same weights and cases, only w changed (w_i = W_PER_SET[level] * s_b, anchored so
the harmonic mean reproduces the calibrated scalar -- harmonic because 1/w_global =
mean_b(1/w_b)):

| arm | CRPS | IS | cov |
|---|---|---|---|
| lr29full ep400 | +0.220 -> +0.230 | -0.104 -> -0.099 | 0.905 -> 0.905 |
| v2realcal ep400 | +0.192 -> +0.198 | -0.328 -> -0.343 | 0.876 -> 0.874 |
| bnsf400 real-pt | +0.214 -> +0.228 | +0.019 -> +0.003 | 0.925 -> 0.923 |
| bnsf800 real-pt | +0.220 -> +0.232 | -0.001 -> -0.022 | 0.923 -> 0.921 |

About +0.01 CRPS for about -0.02 IS, coverage flat -- the same sharpness-for-calibration
trade seen everywhere else, not a win. On hifi_v2strong it is worse (bnsf800 IS
-0.285 -> -0.353). The clamping caveat did materialise (34.6% / 41.9% of cases clamped on
the real-pt arms, whose window floors at 0.755) but the UNCLAMPED v2realcal arms show the
same pattern, so clamping is not hiding a win; nor is the flow failing to respond to w
(width moved 0.727 -> 0.709, coverage held). A 16-nat improvement in how well the noise
model describes the residual distribution is close to orthogonal to posterior quality.

Since the scale is the big term and it is obtainable free, and the offset term is 13x
smaller on the same measure, this argues against a bin-specific training run. The
`--correction` loader supports a per-bin offset (bin read off the observation, Sigma_c
global) and `--mu-scale-range` amortizes the offset MAGNITUDE over c ~ U[lo,hi] with
`mu = c * mu_hat_b`, both committed but used by no run. Revive only with a reason the
scale result does not already cover.

**Training dynamics: the hifi/lofi trajectory is invariant to learning rate, architecture
and capacity.** Hypothesis was that 980k sims at lr 1e-4 with 3.79M flow params forces an
over-long schedule, and that a higher lr with adaptive decay would land somewhere better
by epoch 400-800. Tested by plotting hifi CRPS against lofi CRPS parametrically along each
arm's epoch ladder, which removes "how far along" from the comparison:

| arm (lr, arch) | lofi | hifi |
|---|---|---|
| MDN ep100 (1e-3, mixture, 133k) | 0.359 | +0.225 |
| lr29full ep400 (1e-4, NSF, 3.79M) | 0.395 | +0.220 |
| MDN ep400 | 0.401 | +0.202 |
| MDN ep800 | **0.415** | **+0.168** |
| wpaired ep400 (1e-4, NSF) | **0.415** | **+0.179** |
| lr29full ep800 | 0.432 | +0.160 |
| wpaired ep800 | 0.442 | +0.129 |

At lofi 0.415 the MDN gives +0.168 and wpaired +0.179 -- within 0.011 on a 0.10 range,
across a 10x learning-rate difference, a different flow family and 28x fewer density
parameters. The arms walk one curve at different speeds; the MDN reaching ep800's position
by ep150 and then continuing down it is the same fact. So lr buys ITERATION SPEED, not
quality, and a higher lr with cosine decay is worth having for that reason alone -- every
question this session was gated on 13-19 hour runs.

What does move the curve is the metric's calibration source: the real-patient arms sit
about +0.06 higher in hifi at matched lofi (bnsf800 at lofi 0.427 reaches +0.220 where the
v2realcal family reaches about +0.160). That remains the only lever that shifts the curve
rather than sliding along it.

## Positive findings

**Low-rank `Sigma_c` works, full beats projected.** `lr29full` (rank 29, unprojected)
against its matched baseline -- same frozen v2realcal metric, same architecture, only
`Sigma_c` differs -- improves all three hifi-realistic metrics at once: CRPS skill
+0.192 -> +0.220, IS skill -0.328 -> -0.104, cov95 0.875 -> 0.905. `lr29proj` (projected)
gets +0.185 / -0.229 / 0.888. Fit on theta-paired residuals, so the deployable version
needs the same fit from NN or OT residuals.

**What controls hifi degradation is the metric's calibration source, not the flow and not
`Sigma_c`.** Delta(ep400 -> ep800) on hifi realistic, swept over six w values spanning both
families' windows:

| family | Delta CRPS | Delta IS | Delta cov |
|---|---|---|---|
| real-patient-calibrated (MAF and NSF alike) | +0.008 .. -0.004 | -0.002 .. -0.080 | ~0.000 .. -0.013 |
| v2realcal (incl. `lr29*`, `wpaired`) | -0.050 .. -0.076 | -0.359 .. -0.460 | -0.042 .. -0.058 |

The real-patient family is flat-to-improving on hifi while improving on lofi -- the
non-degradation property -- at **every** w in its window, so this is not an artefact of
evaluating near the window edge (the hypothesis the sweep was built to test, and refuted).
MAF (305k flow params) and NSF (3.79M) have near-identical slopes within each family and
wildly different slopes across them, so it is a property of the training target.

**Open anomaly.** The metric calibrated on *real patients* does not degrade on *hifi sims*,
while the metric calibrated on the hifi stand-ins does. That inverts "calibrate on what you
target" and is unexplained. Cheap discriminator: the two metrics' fitted parameters
(real-pt `ell=26.13, k_t=44, d_eff=180, w_hat=7.554`, window [0.755, 75.5]; v2realcal
`ell=30.86, k_t=38, d_eff=156, w_hat=1.863`, window [0.186, 18.63]) -- a larger `ell` would
suggest smoother whitening simply generalises better. Definitive test is a window-swap 2x2
(each metric under the other's window), two runs.

**The useful w is below the real-patient window's floor.** Within that family, IS skill and
coverage improve monotonically as w falls, and the best values sit at the lowest w testable
(w=0.8, at the 0.755 floor): CRPS +0.207, IS **+0.035**, cov **0.928** -- the first arm to
beat Route A (+0.199 / +0.009 / 0.881) on all three at once. The optimum is still falling at
the floor, so it is clipped rather than found.

**The discrepancy is strongly heteroscedastic in the operating point.** This is the largest
misspecification measured. Exact theta-paired residual, tercile contrast in `E||r||^2_M`
against per-observable permutation nulls (n=8832, d_eff=156, global w=0.713):

| stratifier | max/min | z | R^2 | implied w by tercile |
|---|---|---|---|---|
| `Prv max` (observable) | **7.50** | 95.0 | 0.061 | 2.75 / 0.89 / 0.37 |
| `Pap max` (observable) | 6.98 | 87.7 | 0.062 | 2.54 / 0.93 / 0.37 |
| `Pap mean` (observable) | 6.21 | 77.7 | 0.046 | 2.35 / 0.88 / 0.38 |
| `Rap` (theta) | 2.70 | 25.0 | -- | 1.21 / 0.87 / 0.45 |

Three things matter here. The **observable beats the hidden parameter** (7.50x vs 2.70x),
so an x-conditional noise model is not a compromise but an improvement, and needs no theta.
The shape rotates too, not just the scale: principal-subspace overlap between extreme
terciles falls below the finite-sample floor (two random halves of the same population) for
`Rap`, `Eap`, `Emax_LV`, though not for `Ras` or `Cas`. And the mean offset varies more than
itself -- between-tercile `||d mu||^2 = 17.0` against global `||mu_r||^2 = 8.72`.

Independent confirmation from per-case w: `sd(log w) = 1.584` against a chi^2 sampling floor
of 0.113, i.e. ~14x the variation pure sampling noise would give, 5th/95th percentiles 0.178
and 32.8.

`R^2` stays 0.02-0.13 across 20+ candidate observables (and 0.103 from all 24 theta with
quadratics), so the signal is **regime-level only**. Per-patient fits of `mu` and `Sigma`
are ruled out for a more basic reason: one observation gives one residual vector, so
`Sigma_i` is undefined and `mu_i = r_i` explains the whole residual as bias and collapses
the posterior. Pool at regime level; `s_b` must come from pooled `||r||^2` within a bin, not
from averaging per-case w, since `w_i` is heavily right-skewed (pooled 0.697 vs per-case
median 3.69).

**The w-conditioning is under-weighted, and it is a gain deficiency rather than a support
limit.** Measured on wpaired ep800 -- the only arm whose window ([0.0713, 7.13]) reaches
low w unclamped, and Sigma_c = I so the uncorrected Laplace curve applies directly -- over
a w grid matched between the two:

| w | Laplace width95 | flow width95 | flow/true |
|---|---|---|---|
| 0.08 | 0.918 | 0.695 | 0.757 |
| 0.60 | 0.792 | 0.647 | 0.817 |
| 1.033 | 0.755 | 0.634 | 0.839 |
| 7.0 | 0.617 | 0.596 | 0.966 |

True compression 1.488x, flow 1.167x; the shortfall grows smoothly from 3% at w=7 to 24%
at w=0.08. Two observations rule out a representational ceiling: the flow's width does not
plateau (it keeps widening to w=0.08), and the prior reference width is 0.950 while the
flow sits at 0.695, so a quarter of the range is unused. A bounded-spline support limit
would show saturation pressed against 0.950.

So the conditioner under-uses w. log w is a SCALAR, so a wider embedding is the wrong fix
-- 16 dims is already over-complete for a 1-D input. The plausible mechanism is routing:
w occupies 16 of 144 conditioning dims against 128 encoder features carrying nearly all
the variance, so the flow reaches most of its likelihood reduction without it. Hence
`--w-cond film`, a multiplicative gate that cannot be routed around. The rank-29
correction barely moves the Laplace target (1.472x vs 1.488x), so none of this depends on
the correction question.

## Conditional noise model: the design, and why

Target the x-conditional Gibbs posterior
`q(theta|x) ~ pi(theta) exp(-w(x)/2 ||phi(x) - phi(f(theta))||^2_{Sigma_c(x)^-1})`.
Everything varying with x is constant in theta and absorbed into the normaliser, so this
**is** exactly a Gibbs posterior. Conditioning the noise on `f(theta)` instead -- the only
thing one can actually sample from -- introduces a `|Sigma_c(f(theta))|^(-1/2)` factor that
multiplies the prior, which (a) breaks the Gibbs form, since the log-det enters with a `1/w`
relative weight so varying w is no longer pure tempering, and (b) down-weights
high-discrepancy regimes precisely because we admitted the simulator is worse there.
Determinant-normalising `Sigma_c` removes the term: *shape* in `Sigma_c(x)`, *scale* in
`w_eff`, Gibbs structure intact.

Per-bin table, cut on `Prv max` terciles: `(s_b, mu_b, {a_i, u_i}_b)` per bin, eigenvalues
kept above the Marchenko-Pastur edge `(1+sqrt(d/n_b))^2`, determinant-normalised, shrunk
toward the global estimate by one cross-validated weight. Shrinkage is load-bearing: three
bins of ~1000 against `d_eff` 156-180 will fit noise, and the resolvable rank falls as bins
narrow.

Training needs **no architecture change**: look the bin up from the sim's own observables,
set `w_eff = w * s_b`, condition the existing scalar conditioner on `log w_eff`. The flow
never sees `mu_b` or `Sigma_c^(b)` -- they are implicit in the data it trains on.

**Why end-to-end learning of the noise cannot work on the NLL.** For any fixed noise the
best achievable loss is `H(theta | x~)`; adding noise destroys information about theta and
so *raises* that floor. The optimum over learnable noise parameters is exactly zero noise.
More fundamentally the lofi sims contain no information about the sim-to-real gap, so no
objective computed on them can identify the noise model. Pinning the scale (fixed-trace
`Sigma_c`) removes the trivial collapse but not the misalignment: NLL then pushes noise into
the Jacobian's near-null space, where it costs least -- anti-conservative.

A data-grounded second term is required. Its gradient must reach **only** the noise
parameters (detached from NLL), otherwise the tug-of-war returns with `lambda` silently
setting the calibration. MMD or energy distance is preferable to a critic at this sample
size and avoids the adversarial instability this project has already hit with `G_rs`.

**Matching marginals is weaker than matching conditionals**, and the risk is not capacity:
unpaired matching cannot separate simulator discrepancy from a shift in the theta
population along directions where `f` is locally invertible -- the same theta-tangent
subspace NN matching projects out. That confound is structural for real patients, whose
theta are not drawn from our box prior, and absent for the hifi stand-ins, which are sims.
Within the constrained "offset plus low-rank Gaussian" class the gross
transport/scrambling failure is unavailable anyway: only the deterministic `mu(x)` can
relocate samples, and the stochastic part can only spread.

**OT instead of NN for matching: tested, no benefit.** The reasoning was that NN lets
every residual sit at its individual minimum and lets observations collapse onto shared
bank sims, so a mass-balanced coupling should be less biased; minimum-cost optimality
(Brenier: gradient of a convex function) also cannot permute arbitrarily, unlike a learned
map. Implemented as an injective min-cost bipartite matching on each observation's top-50
candidate graph (the m=1 capacity-constrained transport problem, which keeps the bank
large, unlike balanced OT against a bank subsampled to n_obs). Measured against the
theta-paired truth, 4000 observations, 250k bank:

| residual | E\|\|r\|\|^2 | vs truth | implied w | cos to truth | Pap contrast |
|---|---|---|---|---|---|
| truth | 220.37 | 1.000 | 0.708 | 1.000 | 6.220 |
| NN | 93.20 | 0.423 | 1.674 | 0.396 | 7.652 |
| OT | 93.43 | 0.424 | 1.670 | 0.396 | 7.646 |

Identical on every measure. The diagnostic explains it: only 129 of 4000 observations
shared a bank sim under NN, so with the bank ~60x the observation count the injectivity
constraint is essentially never binding and the joint optimum equals the individual
optima. Mass balance only helps when observations *compete* for the same bank points.

The corrected understanding: the 2.36x magnitude underestimate is **not** caused by
collisions, it is the projection -- minimum-cost matching still minimises, just jointly --
so every matching-by-proximity scheme inherits it, balanced or not. Do not revisit OT for
residual estimation.

What this does establish is the size of the oracle/deployable gap, which matters more than
the magnitude bias alone: `cos to truth = 0.396` means the matched residual points only
~40% along the true discrepancy, so a noise model fitted from it estimates a substantially
rotated object. Matching also inflates the heteroscedasticity contrast by ~23%
(6.22 -> 7.65 on `Pap mean` at matched sample size). Any deployable conditional noise model
carries all three distortions and there is currently no known way to remove them.

**Catheter-filter magnitude, for the record.** Applying `catheter_filter` post-hoc to the
existing hifi beats, the filter term is 30-80% of the structural RMS per channel depending
on `(fn, zeta)` and HR -- not negligible. But `calibrate_lowpass` puts the real spectra
closest to *unfiltered* hifi, so a perturbation that size would move the benchmark away
from the reals. Both results agree: do not add the layer.

## The Laplace width diagnostic: the flows have not reached their own target

Every score we had measures the flow against data. None of them answers whether the flow
has reached the Gibbs posterior it was *trained* to target, and that distinction carries
the whole interpretation of the hifi degradation:

- **(a)** the flow overshoots past the target -- the target is fine, the fit is not;
- **(b)** the target is itself too sharp for hifi -- the fit is fine, the target is not.

(a) and (b) have identical signatures in validation NLL, in lofi score, and in anything
else computed against the target, which is why "no likelihood-based rule can detect the
turn" (see *Non-findings*). Linearising the target breaks the tie. Near `theta_true`,

    A(w) = w J^T Sigma_c^-1 J + 12 I,    J = d phi(f(theta)) / d u,    width95 = 3.92 sd

with `12 = 1/Var(U[0,1])`, the box prior moment-matched. This uses the **lofi** simulator
-- the one the Gibbs target is defined with -- and the validation sims' own `theta`, so it
needs no labels and no hifi model. `J` costs `2 x n_params` solves per case: 2,880 calls
for 60 cases, 0 failed, under a minute on 48 workers, cached once. Per-epoch cost after
that is one flow sampling pass. `J` is metric-dependent, so a cache is valid only for the
metric that built it, and `train_gibbs_npe.py` refuses a mismatched one.

**First pass was invalid.** It compared flow widths measured on 1,000 **hifi** cases
against a Laplace built on 60 **lofi** cases and read hypothesis (a) off the result
(ratios 0.918 / 0.831 / 0.758, all below 1, concluding every flow overshoots a sound
target). `scripts/laplace_ladder.py` re-measures the flow on exactly the cache's own
cases. The ordering survived the error; the absolute level, which is the whole question,
did not.

**Result at `w = 1.033` in the real-patient metric, matched cases: the arms bracket the
target, and hifi improves monotonically with distance ABOVE it.**

| arm | flow width95 | ratio | ratio_inf | hifi IS |
|---|---|---|---|---|
| bnsf800, Lipschitz ceiling 2.0 | 0.747 | 0.967 | **1.277** | -0.022 |
| bnsf800, no ceiling | 0.674 | 0.872 | **1.033** | -0.232 |
| bnsf1600, no ceiling | 0.614 | 0.795 | **0.880** | -0.819 |

Target `width95 = 0.7725`. The arm sitting essentially *on* its Gibbs posterior
(nolip800, ratio_inf 1.033) scores IS -0.232; the best arm is 28% **wider** than its own
target; the only arm that genuinely overshoots is bnsf1600 at long training.

**This is (b): the Gibbs target is too sharp for hifi.** Being at the target is not
optimal and being past it is bad, so the useful operating point is deliberately *above*
it. The ratio remains a usable unsupervised instrument -- it is the only non-likelihood
signal we have, and it ranks the arms correctly -- but the rule is "hold ratio_inf near
1.28", not "converge to 1". That number is empirical and tied to this metric and `w`.

**The Lipschitz ceiling works by keeping the flow wide**, which is the mechanism behind
the -0.21 IS it was known to be worth. Note the consequence: *tightening* the ceiling is
the right direction, and loosening or removing it is what lets the flow reach and pass the
target.

**The flow under-responds to `w`, badly.** `ratio_inf` for the Lipschitz arm over
`w = 1.033 / 7.55 / 75.5` is 1.277 / 1.392 / 1.904, and the all-parameter ratio crosses 1
inside the window (0.967 -> 1.199). The true target moves 0.7725 -> 0.4590 over that range
while the flow moves 0.747 -> 0.550. A single-`w` ratio is therefore not a property of an
arm, and the w-conditioning itself is a defect worth attacking directly.

**Epoch trend, bnsf800-Lipschitz.** `ratio_inf` 1.310 / 1.295 / 1.287 / 1.277 at
ep500/600/700/800: monotone, approaching the target from **above** at roughly 0.01 per 100
epochs. Training does converge toward the Gibbs posterior; it is simply far from it, and on
this evidence arriving would be a regression.

**Read `ratio_informative`, not `ratio`.** Per-parameter target widths at `w = 1.033` span
0.187 to 1.083 against a prior width of 1.1316, so 11 of 24 parameters are barely
constrained and sit at the prior under both the target and any sane flow. The two columns
disagree on the sign of the finding -- all-parameter ratios read "too sharp" (0.967) where
the informative subset reads "too wide" (1.277) -- so the subset is the one to trust. The
informative count is itself `w`-dependent: 13/24 at `w = 1.033`, 23/24 at `w = 75.5`.

**Caveats.** The width is a local linearisation at the validation sims' true `theta`,
worth perhaps 10-20% in absolute terms; with the operating point now *defined* by an
absolute ratio (~1.28) rather than by proximity to 1, that uncertainty propagates directly
into the rule. It measures width only: right width with the wrong centre scores 1.0.

**Owed: SMC on hifi realistic.** This is now the decisive run, not a validation detail.
The claim is that the Gibbs target is too sharp for hifi; SMC with the lofi simulator at
the same metric and `w`, on hifi-realistic observations, *is* the exact Gibbs posterior
there, so scoring it gives the ceiling of the whole Gibbs approach directly. The
prediction: its hifi IS lands near the on-target arm's -0.232, well short of the
deliberately-wide -0.022. A second, cheaper SMC run on clean lofi would separately
validate the Laplace's absolute level, which the ~1.28 operating point now depends on.

## Bugs fixed

**`export_hifi_theta.py` row order.** `load_real_beats` iterates
`sorted(data_dir.glob("*.h5"))` -- lexicographic on filenames, so `sim_1000.h5` precedes
`sim_200.h5`. The exporter sorted indices numerically, so every row of
`hifi_v2_real3k_theta.npz` labelled the wrong beat. Caught by an alignment guard:
`corr(mean Pap, Rap)` was -0.010 where the guaranteed-aligned paired set gives +0.280.
After the fix, +0.255. Nothing was trained on it; it was diagnostic only. Commit `e73afb5`.

**`resolve_gibbs_ckpt` resolving to incomplete checkpoints.** Some runs have later
timestamped dirs holding a metric but no `flow_net.pt` (relaunches that died during the sim
load). Now skips dirs without a flow.

**`check_noise_consistency` never ran on the frozen-metric path.** It was called only
where the metric is ESTIMATED, so every `--metric-dir` run skipped it -- the lr29proj,
lr29full and wpaired arms, and both real-patient-calibrated runs. A correction fitted in a
different metric's phi (wrong d_eff, wrong U shape) would have gone uncaught until the
injected noise was silently wrong. Now runs on both paths; it passes at 1.4% relative
error for the d_eff 180 refit. No evidence any earlier run was actually wrong, but none
of them was verified.

**The rank-29 correction could not be reused across metrics.** `cor_r29_full.npz` holds
`U` as (156, 29) in the v2realcal metric's phi; the real-patient metric has d_eff 180, so
it will not load at all. Refitting at d_eff 180 (theta_paired_gate then build_correction,
both with `--ck` pointing at the real-patient checkpoint) gives rank **32**,
`a_i` in [3.73, 53.51], `||mu_r||^2 = 7.578`, self-consistent w 3.928, and theta-subspace
energy 35.5% against 35.3% at d_eff 156 -- the refit agrees with the original geometry.
A correction is tied to the metric that defined its phi; it is not a portable artifact.

## Non-findings worth recording

Early stopping cannot rescue the degrading families. Simulated on the real validation
curve: patience 10 stops at ep359, patience 20 at ep760, patience 30+ never, and the best
validation is ep798. Those stops ride on noise (ep600-800 drift 1.49 nats against a
per-epoch step sd of 0.207; 332 of 799 steps go up). Validation NLL and lofi score are both
monotone while hifi degrades, so no likelihood-based rule can detect the turn. Validation
NLL is also **not comparable across arms**, since each trains against a different noise
model and so a different target density.
