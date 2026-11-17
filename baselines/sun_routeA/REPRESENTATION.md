# Route A: representation decisions

Phase 0 output. Everything here is read off the vendored source at `84049e98`, not from
the paper. Sections 1-3 need a decision from Sid before phase 1; 4-7 are settled and
recorded so phase 1 does not re-litigate them.

---

## 1. Upstream's beta grid is entirely <= 1: flattening only, no sharpening

`RouteA_generate_pairs.py:106-107`:

```python
beta_list = [0.1,0.3,0.5,0.7,0.9,1.0,1.1,1.3,1.5]
beta_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]   # <- this one wins
```

The second assignment silently overrides the first. The grid actually run is
`0.1 ... 1.0`, so Route A as released can only produce posteriors **at or flatter than**
the nominal one. It has no sharpening lever at all. (The dead first line did reach 1.5.)

This matters for the comparison. Gibbs NPE's retrained window is `w in [0.05, 50]`:
roughly 150x flatter to 7x sharper than nominal. If Route A runs upstream's grid as-is,
the two methods are not given comparable post-hoc calibration range -- one can only
flatten, the other can do both.

**Decision needed.** Three options:

- **(a) Upstream grid verbatim**, `0.1..1.0`. Maximum fidelity. Report the one-sidedness
  as a property of the released method. Risk: a reviewer reads the sharpening gap as our
  experimental choice rather than theirs.
- **(b) Upstream grid + the dead line's top end**, `0.1..1.0` plus `1.1, 1.3, 1.5`. Still
  entirely values the authors wrote down; just un-shadows the overridden line. 13 betas.
- **(c) Extend further**, e.g. to beta = 5-10, to match Gibbs NPE's sharpening range.
  Furthest from upstream and the point where the raw-beta FiLM conditioning in
  `ConditionalMDN` (`Linear(1, hidden)` on an unnormalised beta) starts to look fragile.

Recommendation: **(b)**. It costs one extra config line, every value is the authors' own,
and it gives the arm a sharpening range without us inventing numbers. Betas are
independent, so (c) remains additive later -- new betas only need their own synthesis, not
a redo.

**Decided 2026-11-16: (b).** The grid is
`[0.1, 0.2, ..., 1.0, 1.1, 1.3, 1.5]`, 13 values.

---

## 2. The released code has no Eq. 6 prior correction; it tempers the joint score

The spec assumed a prior-corrected Langevin step. `Lgmcmc` (`RouteA_utils.py:239-240`) is:

```python
theta_new = theta_view + t_power * a_theta * score_theta + torch.sqrt(2.0*a_theta) * eps_theta
x_new     = x_view     + t_power * a_x     * score_x     + torch.sqrt(2.0*a_x)     * eps_x
```

`prior` is used **only to initialise** (`theta = prior(n_samples)`, line 202) and never
enters the drift. `t_power` (= beta) multiplies the *joint* (θ,x) score, so the target is
`p_sigma(θ,x)^beta`, i.e. the prior is tempered along with the likelihood.

**For our prior this is benign on-support, and that is a real piece of luck.** On the
support a box-uniform prior is a constant `c`, so

```
[pi(θ) p(x|θ)]^beta  ∝  c^beta p(x|θ)^beta  ∝  p(x|θ)^beta  ∝  pi(θ) p(x|θ)^beta
```

-- identical to the intended power posterior up to normalisation. No correction term is
needed and none is missing.

**Verified empirically**, since the argument rests on the prior actually being flat in the
Langevin coordinates: per-dimension KS tests against U(0,1) on the unit-cube theta pass at
a Bonferroni-corrected family-wise 1% for all 24 dims. The worst statistic scales as
1/sqrt(n) (tau_a: D = 0.01194 at n = 20k -> 0.00747 at n = 50k, against 0.00755 predicted
by pure sampling noise) and the identity of the worst dim changes with n, both of which
rule out a real distortion -- in particular there is no detectable survivorship bias from
the bank holding only successful solves. `verify_phase1.py --ks-report`.

**Off-support it is undefined.** `pi = 0` there, so the intended target is zero, but the
score net is an extrapolating MLP and ULA has no reflecting barrier or projection. Nothing
stops a chain leaving the prior box. So the correctness of this arm rests entirely on how
much mass leaks out, which is why out-of-box fraction is logged per beta from phase 4 on
even though it is no longer a hard gate.

Two consequences:

- The prior must be **uniform in the Langevin coordinates**, or the tempering genuinely
  does distort it. In native units our prior is log-uniform for `Emax_RV`, `Eap`, `Rap`,
  so tempering the joint would bias those three by `θ^-beta` instead of `θ^-1`. In
  unit-cube coordinates (section 3) the prior is exactly uniform and the issue vanishes.
- The pair set's **θ-marginal is still tilted** for beta != 1: it is
  `∝ ∫ pi(θ) p(x|θ)^beta dx`, not `pi(θ)`. The NPE learns the conditional, which is what
  we want, but only where that tilted marginal has coverage. Gibbs NPE's θ-marginal is
  exactly the prior at every w. This asymmetry stands and belongs in the writeup.

---

## 3. Langevin must run in unit-cube theta coordinates (forced by their step sizes)

`alpha_from_snr_blocks` clamps the step to `[cmin*sigma^2, cmax*sigma^2]` with
`cmin_theta=2e-5`, `cmax_theta=1e-4` -- **absolute** numbers in θ's own units. Upstream's
θ was 3-dimensional with all ranges O(1): `PriorBox(1.4, 2.2, 0.1, 1.0, 1.5, 2.5)`
(`RouteA_generate_pairs.py:25`).

Our 24 native parameters span from O(0.1) to `Rap` in 10..1200. A fixed absolute step of
~1e-4 is a rounding error for `Rap` and a large move for the smallest parameters, so the
clamps carry no consistent meaning in native units.

**Decision: run the Langevin in the unit-cube coordinates `bounded_nsf.py` already
defines** -- the prior box mapped to `[0,1]^24`, log-scaled for the three log-sampled
dimensions. This does three things at once: the prior becomes exactly uniform (section 2),
every θ dimension becomes O(1) so upstream's step clamps transfer with their intended
meaning, and the box becomes a cube so out-of-box detection is a trivial range check.

This is a reparameterisation of their algorithm, not a change to it, and it reuses a
mapping already in the repo rather than introducing a new one. Our x is already z-scored
and O(1), matching what `cmin_x=1e-2, cmax_x=0.5` assumes.

**theta_dim and x_dim are not merely shape arguments here.** `alpha_from_snr_blocks`
computes the step as `0.5 * d * tau^2 / (g2_median * beta^2)` before clamping, so
`theta_dim` enters the step size directly: going from their 3 to our 24 makes the
unclamped theta step 8x larger, and 840 -> 809 makes the x step ~4% smaller. Whether that
matters depends on which side of the clamp the step lands on, which upstream never has to
ask because its dimensions are fixed. **Log, per noise level, the fraction of steps that
are clamp-bound vs SNR-bound.** If clamp-bound, the dimension change is irrelevant and
the absolute clamps govern everything (section 3's argument for unit-cube coordinates is
then the whole story); if SNR-bound, the 8x is real and the step schedule is effectively
different from the one they tuned. Nothing else in the three library files we call
hard-codes a dimension -- verified by grep at the pinned commit.

---

## 4. x is the identical 809-dim vector, with 808 noised and hr_z left clean

Both arms see the same input. Gibbs's `noisy_obs` perturbs the 804 waveform dims plus
`sc[:, :4]` (MAP/SBP/DBP/SV) and passes `sc[:, 4] = hr_z` through untouched, so HR simply
rides along as an un-noised dimension.

That exclusion is by design, not an oversight, and three things in the code agree: `e =
torch.randn(B, 4)` draws only four scalar noises; `s_scal` has exactly four entries, so
the metric has no HR component; and `d_eff = 180 = 4 waves x k_t 44 + 4 scalars` does not
count it. The reason is that **HR is not inferred** -- `PARAM_KEYS_INFER` drops it, so it
is a known covariate the simulator is conditioned on, carried in x so the network knows
which heart rate the waveforms belong to. It is not a noisy observable to be matched, and
the Gibbs metric has nothing to say about it. Route A's DSM noises the same 808 and leaves
`hr_z` clean. The spec's separate HR-carrying machinery is dropped -- it was solving a
problem that does not exist.

**Implementation cost, accepted deliberately.** Upstream's `score_matching_loss` and
`Lgmcmc` both touch all of x, and `vendor/` is not edited, so keeping `hr_z` clean means
maintaining our own masked copies of those two functions in `routeA_noise.py`. This arm is
therefore *not* bit-faithful to their code. The reason it is still the right call: `hr_z`
is exact in the real observations both arms are evaluated on, so a noised `hr_z` would give
Route A's synthesized pairs a wider spread in that dimension than the data it is tested
against -- a train/test mismatch introduced by the method's own noise model rather than by
the inference problem. Masking removes a confound instead of creating one. The step-size
logic (`alpha_from_snr_blocks`) is imported from vendor unmodified; `x_dim` is still passed
as 809 rather than 808 to stay closer to upstream, a ~0.1% difference in the x step.

A second, smaller deviation lives in the same file: chains are initialised from a tensor
(a slice of the lofi bank) rather than by calling `prior`/`simulator`, which is what makes
the zero-new-simulations budget match in section 5 possible. Distributionally identical,
since the bank is prior-drawn (theta, simulator(theta)) pairs.

**One irreducible mismatch:** Gibbs noises SV *multiplicatively*
(`sc[:,3:4] * exp(e*s_scal[3]*w^-0.5)`, because the metric works on log SV), while DSM is
additive on all 808 (`noise_x = x + sigma * eps_x`, `RouteA_utils.py:52`). The two noise
models therefore differ in form on exactly one dimension even when matched in scale. Not
fixable without altering one method; stated, not patched.

---

## 5. Budget: the existing 980k bank partitions exactly, so zero new simulations

`Lgmcmc` calls `simulator(theta)` once per invocation to initialise x (line 203), so each
beta's synthesis needs `n_samples` prior-drawn (θ, x) pairs -- which is precisely what our
980k lofi bank already is. With 10 betas, `980000 / 10 = 98000` chains per beta partitions
the bank exactly and disjointly.

Net effect: Route A's simulation cost equals Gibbs NPE's to the sim, because it is the
same 980k sims. The score net trains on the same bank. Upstream's own run was far smaller
(10k for the score net + 10 x 10k inits = 110k sims total).

Remaining confound, unfixable and to be reported: Gibbs NPE redraws noise every epoch, so
its effective training set is unbounded, whereas Route A trains on a fixed synthesized
set. At matched sims that is a regularisation difference, not a data-volume one.

---

## 6. Three corrections to earlier assumptions

- **Their d_x was 840, not "far smaller" than ours.** `RouteA_generate_pairs.py:32` sets
  `theta_dim, x_dim = 3, 840` for Lorenz-96. Our 809 is slightly *smaller*. Input
  dimensionality is not the stress point I previously flagged.
- **The real gap is theta_dim: 3 upstream vs 24 here.** With `ConditionalMDN`'s
  **diagonal** per-component covariances and 8 components, a correlated 24-dim posterior
  has to be tiled by 8 axis-aligned blobs. That, not d_x, is where their density estimator
  is being asked for something it was not exercised on.
- **`ScoreNet_sigma` is not a bare MLP on raw inputs.** It has `ResidualMLP` encoders on
  θ and on x separately (both to `emb_dim=128`), plus a `SigmaEmbedding`, concatenated
  into a final `ResidualMLP` producing a `theta_dim + x_dim` score. It is structurally
  reasonable at d_x = 809.

## 7. Settled mechanics worth not rediscovering

- `SigmaEmbedding` normalises by `sigma_min=1e-2, sigma_max=1`, which are **constructor
  defaults on `ScoreNet_sigma`**, independent of `make_gamma_schedule`'s arguments. They
  happen to agree upstream. Changing the gamma schedule without passing matching
  `sigma_min/sigma_max` would silently mis-scale the sigma embedding.
- Step schedule: `initial_step=10`, `+10` per noise level, 10 levels descending =
  **550 Langevin steps** per chain.
- `per_sample_stepsize=False` by default: the step size is a **batch median** over the
  score norms, so all chains in a batch share it and are statistically coupled. Seeds do
  not give independent chains within a batch.
- `ConditionalMDN` has unbounded support over z-scored θ, hence upstream's
  `sample_with_rejection`. Rejection rate is logged per beta; at low beta the posterior
  approaches the prior and piles at the box edges, so that is where it should be worst.
- `train()` in `RouteA_utils.py:87` accumulates `loss_k / B` although `F.mse_loss` already
  averages, so the reported loss scales with batch size. Harmless to the gradient, but
  the logged number is not comparable across batch sizes.

---

## 8. Phase-2 result: the sampler does not temper theta (measured, analytic toy)

Run `verify_phase2.py`. Checks C/D feed the **exact analytic score** of a linear-Gaussian
joint to upstream's `Lgmcmc`, so any failure is the sampler's, not a score-network error.

| beta | cov ratio theta | cov ratio x | mean \|dtheta\| | mean \|dx\| | clamp-bound theta/x |
|---|---|---|---|---|---|
| 0.5 | 0.497 | 0.819 | 0.0875 | 4.83 | 0.80 / 1.00 |
| 1.0 | 0.993 | 0.997 | 0.0871 | 4.02 | 0.80 / 0.91 |
| 2.0 | 1.981 | 1.382 | 0.0876 | 3.46 | 0.80 / 0.80 |

Ratio 1.0 = correct (target covariance is `S/beta`). At beta=0.5 the target is `2S` and the
ratio 0.497 means the empirical covariance is `0.99 S`; at beta=2 the target is `S/2` and
1.981 again means `0.99 S`. **theta's covariance is the untempered one at every beta**, and
`mean |dtheta|` is beta-independent to three digits (0.0875 / 0.0871 / 0.0876). Cause is the
last column: 80% of theta steps are pinned at the `[2e-5 sigma^2, 1e-4 sigma^2]` clamp, which
erases the `1/beta^2` term in `alpha_from_snr_blocks` and leaves a drift too small to move
anything. x does respond but undershoots (64% of the required spread at beta=0.5, 69% of the
required shrinkage at beta=2). beta=1 is correct to 0.3-0.7%, which validates our wiring.
Reproduced identically at 150 and 600 score-net epochs, as expected for an analytic-score check.

### Why a frozen theta is a defect, correcting section 2

Section 2 reports the tilted theta-**marginal** as Route A's asymmetry versus Gibbs. That is
the wrong object to worry about: an NPE learns the **conditional**, so a tilted marginal only
costs training coverage, not correctness. The real problem is the opposite one.

With theta pinned, x is drawn from the tempered likelihood *renormalised for that theta*,
`p(x|theta)^beta / Z_beta(theta)`. The realised pair joint is then
`pi(theta) p(x|theta)^beta / Z_beta(theta)`, whose conditional is

```
q(theta|x)  ∝  pi(theta) p(x|theta)^beta / Z_beta(theta)
            =  [power posterior] x |Sigma(theta)|^((beta-1)/2)
```

since `Z_beta(theta) ∝ |Sigma(theta)|^((1-beta)/2)` for a Gaussian likelihood. Dividing by
`Z_beta(theta)` discards exactly the quantity that tempering is supposed to use to reweight
theta. Concretely: with `x|A ~ N(0,1)` and `x|B ~ N(0,0.1^2)`, observing x=0 gives a beta=1
odds ratio of 10 and a correct beta=0.5 ratio of 3.16, but the frozen-theta construction
returns **10** -- the untempered answer. That is the same thing the table measures.

For Route A `Sigma(theta) ~ sigma^2 (I + J J^T)` with J the simulator Jacobian, so the tilt is
`|I + J^T J|^((beta-1)/2)` (Weinstein-Aronszajn turns the 808x808 determinant into a 24x24
one). It vanishes only at beta=1, and for beta<1 the exponent is negative, so it
**under-weights the theta where the simulator is most sensitive** -- exactly where the
posterior is normally most informative.

Practical consequence: **Route A's beta knob is largely inert for theta at this scale**, so the
NPE's amortisation over beta has little to learn and post-hoc recalibration through beta
would not work. That is worse than a bias and it sits upstream of phases 4-6.

**Gibbs NPE is structurally immune, for a reason worth stating:** its noise covariance is
`M/w` with M frozen and theta-independent, so `Z_w` is constant, no tilt arises, and the pair
conditional *is* the Gibbs posterior by construction. The vulnerability is specific to
Route A's covariance inheriting the simulator's local geometry.

### Cheapest confirmation in the real setting

After synthesis, check whether the pair sets at different beta actually differ in theta|x --
e.g. theta spread at fixed x, per beta. If it is flat across beta, the knob is confirmed inert
here and not just in the toy. That is far cheaper than the FD Jacobian, which is then only
needed to quantify the tilt's magnitude (the spread of `log|I + J^T J|` across the prior).

Caveat: the toy is 2+5 dimensional. `alpha_from_snr_blocks` scales the unclamped step with
`theta_dim`, so our 24-dim theta gets a 12x larger unclamped step -- but 80% of steps were
already at the clamp *ceiling*, and a larger unclamped value pushes further past it, so expect
more clamping in the real setting, not less. Verify rather than assume.

## 9. Phase-2 gate outcome: FAILED as specified, diagnosed as upstream's loss weighting

`verify_phase2.py` checks A/B require relative L2 score error < 0.10 at **every** sigma.
Measured worst case: **0.1010** (unmasked) and **0.1105** (masked). The gate failed on the
criterion set before the run; the threshold was deliberately not relaxed afterwards.

Budget scaling (`--epochs 150 / 600 / 1500`, unmasked, theta block):

| sigma | 150 ep | 600 ep | 1500 ep |
|---|---|---|---|
| 0.01 | 0.1459 | 0.1077 | **0.1010** |
| 0.0167 | 0.1413 | 0.0936 | 0.0964 |
| 0.3594 | 0.1304 | 0.0569 | 0.0691 |
| 1.00 | 0.0925 | 0.0262 | **0.0186** |

Large sigma converges at roughly 1/sqrt(steps); small sigma flattens onto a floor near 0.10
(2.5x more epochs bought 6%). The masked variant was at its floor from 150 epochs
(0.1102 / 0.1087 / 0.1105 at sigma=0.01).

**Diagnosis.** `score_matching_loss` samples `t_idx` uniformly and weights every noise level
equally in the MSE, while the fittable signal at level sigma is O(sigma). The gradient signal
for small sigma is therefore ~sigma^2 weaker, so the optimiser allocates capacity to mid and
large sigma and lets the small-sigma score drift. This predicts all three observations: flat
in budget at small sigma, converging at large sigma, and the same floor masked or not. It is a
property of upstream's unweighted DSM -- sigma-weighted (likelihood-weighted) DSM is the
standard remedy and upstream does not use it. More epochs or data will not move it.

**Independent evidence the port is correct**, which is what the gate existed to establish:
large-sigma error reaches 1.9% and converges at the right rate; the clean dim's learned score
is 0.0000 against an analytic 0.8058, exactly as section 4's derivation predicts; the sampler
reproduces the beta=1 joint to 0.5%; and the error is smooth and monotone in sigma, which a
wiring fault would not produce.

**Practical weight:** low. The sampler's steps at sigma=0.01 are clamped to ~1e-8, so the
small-sigma score barely influences the trajectory -- the same clamping that causes the far
more serious section-8 problem. A sigma-weighted gate would be the better criterion for any
future rerun; it is not retrofitted here.

Open if certainty is wanted: a capacity/data sweep at fixed epochs (hidden_dim 128 -> 512, or
n 200k -> 800k). A floor that stays at 0.10 confirms the diagnosis as structural.

## 10. The one port change: network width must scale with d_x (and it is not optional)

Trained at upstream's own `emb_dim = hidden_dim = 128`, the score net converged to val
0.9521 after 789 epochs (early-stopped by upstream's own rule). Measured on held-out pairs
in the regime it was trained on, that net explains **4.7% of the noise variance on x and 4.0%
on theta, flat across every sigma**, with an output norm 21% of the target's -- MSE shrinkage
toward zero. The baseline for predicting identically zero is 0.9988, so 789 epochs bought 3.4
percentage points.

Flatness in sigma is the diagnostic. At sigma=0.01 the noised x sits essentially on the data
manifold and eps is highly predictable, so a working net's R^2 must rise sharply as sigma
falls. It did not move at all, which is a representational ceiling rather than an
optimization one: `ScoreNet_sigma` embeds x through `ResidualMLP(809 -> 128)` and must emit an
809-dim score field from a 384-dim concatenated bottleneck.

**Why this is a port artifact and not a property of their method.** Upstream's `d_x` is ~10,
so 128 sits an order of magnitude *above* their input dimension and the embedding is nowhere
near binding. At `d_x = 809` the same constant sits far *below* it. Scaling width to exceed
d_x restores the relationship their setting had; it changes no part of the method. Measured at
matched epochs, same 8 s/epoch and 4-9 GB of a 32 GB card at every width:

| emb=hidden | val @20 | @40 | @60 | best ever |
|---|---|---|---|---|
| 128 (upstream) | 0.9712 | 0.9684 | 0.9645 | 0.9521 (789 ep) |
| 512 | 0.9080 | 0.8910 | 0.8813 | -- |
| 1024 | 0.9625 | 0.8335 | 0.8200 | -- |

1024 passes the 128-wide net's 789-epoch converged best by about epoch 25. Width was free in
both time and memory, so the 128-wide result was not a compute/quality tradeoff -- it was
simply too small a model.

**Consequence for the sampler, which is why this blocked phase 4b.** Upstream's step rule
`a_x = 0.5*d_x*tau^2/(g^2*beta^2)` clamped to `[1e-2*sigma^2, 0.5*sigma^2]` is self-calibrating
*provided the score has roughly the right magnitude*: with `g_x^2 = d_x/sigma^2` it returns
`0.125*sigma^2/beta^2`, comfortably inside the clamp. At a norm ratio of 0.215 the 128-wide
net's `g^2` is ~20x too small, the formula overshoots by ~1200x, and `a_x` pins to
`cmax_x = 0.5*sigma^2` on every step -- full-strength injected noise against a restoring drift
only a fifth as strong as it should be, compounding over the 10 levels. Synthesis blew up
accordingly, including at beta=1 where the chain starts at the simulator's own joint and must
reproduce it: covariance-trace ratio 4.6e11 on x, 4.5e5 on theta. The arm fails its own
control before any beta comparison is possible. The clamp values are upstream's
(`vendor/RouteA_utils.py:195`; `RouteA_generate_pairs.py` overrides nothing), so no clamp
retuning is warranted -- the clamp only binds because the score is too small.

The criterion for synthesis stability is therefore a property of the trained net, checkable in
minutes before committing hours to sampling: **the x-block norm ratio `|s_hat|/|eps|` must
reach ~0.5** (R^2_x ~ 0.25) for `a_x` to fall inside the clamp unaided.

**Correction to section 9.** That section closed by proposing a capacity/data sweep and
predicting a floor that "stays at 0.10", i.e. that the DSM error was structural in upstream's
unweighted loss and that "more epochs or data will not move it". Run, the sweep refutes the
strong form of that claim: width moved the real-data net from 4.7% to 48% explained variance
on theta. The weaker form survives and is now visible rather than masked -- at 1024 wide,
R^2_theta decays from 0.48 at sigma=1 to ~0.00 at sigma=0.01, which is the sigma profile
section 9 attributed to uniform sigma weighting. Both effects are real; capacity dominates and
was hiding the other. No sigma-weighting change is made here, since it would be a change to
their method rather than to the port.

## 11. Upstream's x step clamp diverges at d_x=809; theta is inert at every clamp

With the section-10 width fix the score net reaches val 0.7829 (early-stopped at epoch 1032),
x norm ratio 0.47 and R^2_x 0.222 -- roughly a 2x improvement on both over 128 wide. Synthesis
at upstream's call-site clamp nevertheless **diverged harder, not less**: x covariance-trace
ratio 9.9e23 at beta=1, against 4.6e11 for the 128-wide net. The mechanism is not a
contradiction. The 128-wide net output approximately zero, so its chain merely random-walked;
the 1024-wide net has a far larger Lipschitz constant off-distribution, so once a step pushes
x off the data manifold its score is both large and arbitrary. Capacity cannot fix a step-size
problem.

Measured scan, 2000 chains, this score net, cmin_x = min(1e-2, cmax_x/10):

| cmax_x | beta | covratio_x | covratio_theta | mean abs dx | mean abs dtheta |
|---|---|---|---|---|---|
| 5e-1 | 0.5 | 2.48e11 | 1.9e6 | 2.2e4 | 9.74880 |
| 5e-1 | 1.0 | 9.86e23 | 5.80e18 | 1.68e10 | 6.58e6 |
| 1e-1 | 0.5 | 4.612 | 1.059 | 1.5284 | 0.05582 |
| 1e-1 | 1.0 | 4.628 | 1.053 | 1.5484 | 0.05549 |
| 1e-2 | 0.5 | 1.422 | 1.055 | 0.5226 | 0.05593 |
| 1e-2 | 1.0 | 1.374 | 1.056 | 0.4916 | 0.05520 |
| 1e-3 | 1.0 | 1.044 | 1.062 | 0.1686 | 0.05532 |
| 1e-4 | 1.0 | 1.005 | 1.059 | 0.0549 | 0.05570 |

**Which clamp is "theirs" is genuinely ambiguous in their code.** `alpha_from_snr_blocks`
ships `cmin_x=2e-5, cmax_x=1e-4` -- identical to theta's, and stable here --
while `Lgmcmc`'s signature overrides them to `1e-2, 0.5`, and the `RouteA_generate_pairs`
call site passes nothing, so the diverging pair is what actually runs. The headline run uses
`[1e-3, 1e-2]`: the largest window that keeps x's covariance within ~40% of the joint the
chains start from, so beta has the best available chance to show an effect.

Note that even `cmax_x = 1e-1`, which is about what the SNR formula itself wants with an
accurate score (`0.125*sigma^2` at beta=1), inflates x by 4.6x. A score at 47% of true
magnitude supplies 47% of the restoring drift against full-strength injected noise, so the
stationary spread is wider than p_sigma by roughly that factor at each level, and the later
levels' smaller steps cannot contract it back. Keeping x at the right scale requires a step
far below what the formula asks for, at which point x barely moves. There is no window in
which this sampler both moves x materially and holds the correct scale at this d_x.

**The substantive result, and it does not depend on any of the above:** `covratio_theta` is
1.05-1.06 and `mean abs dtheta` is 0.0555 at *every* clamp and *every* beta, identical to
three decimals, excluding only the diverged rows. Beta does nothing to theta. This reproduces
section 8's analytic-toy finding on real data, now independent of both score quality and step
size, and it is the finding that matters for the comparison: Route A's sampler does not temper
theta, so the conditional MDN trained on these pairs has no beta-dependence in theta to learn.

A second observation worth recording, visible in the section-10 numbers: the 1024-wide net has
**lower** R^2_theta (0.087 at sigma=1) than the 512-wide net (0.483), despite much better
aggregate loss. Upstream's DSM loss is an unweighted mean over the joint [theta | x], so with
d_x=809 against d_theta=24, x carries 97% of it. Given more capacity the optimizer spends it
where the loss is, and theta's score -- the only part inference needs -- is traded away. This
is a dimension-ratio artifact of the unweighted joint loss, not of the width.

## 12. Theta is beta-inert at every clamp, because its trajectory is pure diffusion

Section 11 established that theta is beta-inert at upstream's clamp. The obvious objection is
that this is an artifact of the clamp rather than of the method, so the clamp was scanned over
four orders of magnitude, 2000 chains per cell, x clamp held at the section-11 value:

| cmax_theta | covratio_theta, beta=0.5 / 1.0 / 1.5 | pure-diffusion prediction |
|---|---|---|
| 1e-4 (upstream) | 1.058 / 1.055 / 1.058 | 1.055 |
| 1e-3 | 1.563 / 1.568 / 1.579 | 1.552 |
| 1e-2 | 6.739 / 6.595 / 6.507 | 6.52 |
| 3e-2 | 17.62 / 17.64 / 17.27 | 17.6 |
| 1e-1 | 56.66 / 55.87 / 55.33 | 56.2 |
| 5e-1 | 308 / 309 / 323 | 277 |

The prediction is the variance a driftless random walk would add, divided by the prior's:
`1 + 2*cmax_theta*sum_t(sigma_t^2 * steps_t) / Var(prior)` = `1 + 552*cmax_theta` for this
schedule and the unit cube. It matches measurement to within a few percent at every step size.
Theta's trajectory is therefore **diffusion with no effective drift**.

This is what makes the beta-inertness structural rather than a tuning artifact. Beta enters the
update only through the drift term (`t_power * a * score`); if the drift is numerically
irrelevant next to the diffusion, beta cannot act, at any step size. Tightening the clamp makes
the diffusion negligible -- upstream's setting, where theta simply stays at its prior draw and
the marginal is preserved by accident. Loosening it makes the diffusion destructive (100% of
theta outside the unit cube at cmax_theta=0.5). No value in between hands control to the drift.

**Why a relaxed-theta arm was not run.** It was proposed, and this scan is what it would have
tested. The result is conclusive at 2000 chains per cell, so the full arm would only have
produced 980k rows of random-walked theta and an MDN trained on them. The relaxation remains
exposed as `--cmin-theta/--cmax-theta` for anyone who wants to reproduce the scan.

**Root cause, one level down.** Theta's score is not learned: R^2_theta is 0.087 at sigma=1 and
0.0007 at sigma=0.01 (section 11). Upstream's `score_matching_loss` is an unweighted mean over
the concatenated `[theta | x]` residual, so with d_x=809 against d_theta=24, x carries 97% of
the loss. Capacity and gradient signal go where the loss is; theta's score never develops; the
sampler's theta block degenerates to diffusion. The two failures in sections 11 and 12 are the
same cause seen from two sides.

Both plausible remedies -- weighting the two blocks by dimension, or sigma-weighted DSM -- are
changes to their loss, not to the port, and are not made here. Recording them as the identified
fix is the useful output.
