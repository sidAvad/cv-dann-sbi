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

---

## 4. x is the identical 809-dim vector, with 808 noised and hr_z left clean

Both arms see the same input. Gibbs's `noisy_obs` perturbs the 804 waveform dims plus
`sc[:, :4]` (MAP/SBP/DBP/SV) and passes `sc[:, 4] = hr_z` through untouched, so HR simply
rides along as an un-noised dimension. Route A's DSM noises the same 808 and leaves
`hr_z` clean. The spec's separate HR-carrying machinery is dropped -- it was solving a
problem that does not exist.

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
