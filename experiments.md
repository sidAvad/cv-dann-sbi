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

## Non-findings worth recording

Early stopping cannot rescue the degrading families. Simulated on the real validation
curve: patience 10 stops at ep359, patience 20 at ep760, patience 30+ never, and the best
validation is ep798. Those stops ride on noise (ep600-800 drift 1.49 nats against a
per-epoch step sd of 0.207; 332 of 799 steps go up). Validation NLL and lofi score are both
monotone while hifi degrades, so no likelihood-based rule can detect the turn. Validation
NLL is also **not comparable across arms**, since each trains against a different noise
model and so a different target density.
