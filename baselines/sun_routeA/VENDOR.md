# Vendored upstream source

| | |
|---|---|
| Repo | https://github.com/Komorebiww/amortized-generalized-bayes |
| Commit | `84049e9872fdfe5c1ec1dcddf63a197322a395c9` |
| Paper | "Amortized Generalized Bayes via Power Posteriors" (ICML) |
| Retrieved | 2026-11-16 via `codeload.github.com/.../tar.gz/<commit>` |
| Tarball sha256 | `774cad8af2085a2f83472bf3c0263b58b301504a4fe508be99f18170f144ceed` |

`vendor/` is the upstream tree at that commit, byte-for-byte, minus upstream's
`.gitignore` (dropped so it cannot affect this repo's ignore rules). **Nothing under
`vendor/` is edited.** Our code imports from it; fixes and adaptations live in this
directory, outside `vendor/`.

To re-verify the pin:

```bash
curl -sSL "https://codeload.github.com/Komorebiww/amortized-generalized-bayes/tar.gz/84049e9872fdfe5c1ec1dcddf63a197322a395c9" \
  | shasum -a 256   # expect 774cad8a...
```

## What Route A actually consists of

| File | Role |
|---|---|
| `NSEutils.py` | `ScoreNet_sigma` (joint score over (θ,x), amortized in σ), `PairDataset`, `SigmaEmbedding`, `ResidualMLP`, `EarlyStopping`, `seed_everything` |
| `RouteA_utils.py` | `make_gamma_schedule`, `score_matching_loss`, `train`/`validate`, `alpha_from_snr_blocks`, `Lgmcmc` (the annealed Langevin synthesizer) |
| `RouteA_generate_pairs.py` | upstream driver: trains the score net, then synthesizes pairs per β |
| `RouteB_utils.py` | `ConditionalMDN`, `train_conditional_mdn`, `sample_with_rejection` — the NPE Route A uses |
| `RouteA_NPE.py` | upstream driver: trains/evaluates the NPE on the synthesized pairs |

Everything else (`True_*.py`, `lorenz96.py`, `RouteB_*`, `NLE_POWER.py`, `NRE_POWER.py`,
`flowmatching/`, `C-nsf.py`, `diffusion.py`, `generate_image.py`) is other benchmarks or
other routes, vendored for completeness but unused here.

Upstream's two drivers are research scripts with the benchmark hard-coded at the top and
`load_model*` flags controlling whether anything trains. We do not run them; we call the
library functions above from our own driver so the configuration is explicit and logged.
