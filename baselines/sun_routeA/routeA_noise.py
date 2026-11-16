"""Route A's DSM loss and annealed Langevin, with two documented deviations.

Upstream's `score_matching_loss` and `Lgmcmc` live in `vendor/RouteA_utils.py` and are not
edited. These are line-for-line mirrors of them differing only in:

1. **`hr_z` (x dim 808) is never noised and never moved.** Upstream noises all of x. Our
   808 waveform+scalar dims carry the Gibbs metric's noise model, while `hr_z` is exact in
   both the Gibbs training input and the real observations both arms are evaluated on. If
   the score net noised it, Route A's synthesized pairs would carry a wider `hr_z` spread
   than the x it is tested on -- a train/test mismatch that is Route A's own artifact
   rather than a property of the inference problem. See REPRESENTATION.md section 4.
2. **Chains are initialised from the existing lofi bank** instead of calling
   `prior`/`simulator`. The bank is exactly prior-drawn (theta, simulator(theta)) pairs, so
   this is distributionally identical, and it keeps Route A's simulation budget equal to
   Gibbs NPE's to the sim. See REPRESENTATION.md section 5.

`alpha_from_snr_blocks` is imported from vendor and used unmodified.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))
from RouteA_utils import alpha_from_snr_blocks       # noqa: E402  (vendored, unmodified)

HR_Z_INDEX = 808          # WAVE_LEN(804) + scalar 4; see dataset.py / train_gibbs_npe.py
CLEAN_X_DIMS = (HR_Z_INDEX,)


def _clean_mask(x_dim, clean_x_dims, device, dtype):
    """1.0 on dims that may be noised/moved, 0.0 on dims held exact."""
    m = torch.ones(1, x_dim, device=device, dtype=dtype)
    if clean_x_dims:
        m[0, list(clean_x_dims)] = 0.0
    return m


def score_matching_loss(score_net, theta, x, t_idx, gammas,
                        clean_x_dims=CLEAN_X_DIMS):
    """Mirror of vendor score_matching_loss; eps is zeroed on clean_x_dims.

    Zeroing eps there makes the regression target 0 for those coordinates, i.e. the net is
    told the score is zero along them, which is the consistent signal for a coordinate that
    carries no noise.
    """
    p = next(score_net.parameters())
    device, dtype = p.device, p.dtype

    theta = theta.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype)
    t_idx = t_idx.to(device=device, dtype=torch.long)

    gammas = torch.as_tensor(gammas, device=device, dtype=dtype)
    sigma = gammas[t_idx].unsqueeze(-1)

    eps_theta = torch.randn_like(theta, dtype=dtype)
    noise_theta = theta + sigma * eps_theta

    eps_x = torch.randn_like(x, dtype=dtype) * _clean_mask(
        x.shape[1], clean_x_dims, device, dtype)
    noise_x = x + sigma * eps_x

    eps = torch.cat([eps_theta, eps_x], dim=-1)
    s_hat = score_net(noise_theta, noise_x, sigma) * sigma
    return F.mse_loss(s_hat, -eps)


@torch.no_grad()
def Lgmcmc(gammas, model, theta_dim, x_dim, init_theta_and_x,
           initial_step=10, t_power=1.0, step_increase=10,
           device="cuda" if torch.cuda.is_available() else "cpu",
           tau_theta=0.5, tau_x=0.5,
           cmin_theta=2e-5, cmax_theta=1e-4,
           cmin_x=1e-2, cmax_x=0.5,
           per_sample_stepsize=False,
           clean_x_dims=CLEAN_X_DIMS,
           collect_diagnostics=True):
    """Mirror of vendor Lgmcmc, initialised from a tensor and holding clean_x_dims exact.

    Returns (samples, diagnostics). `diagnostics` records, per noise level, whether the
    step size came from the SNR formula or from the clamp, and the fraction of theta rows
    outside the unit cube -- the two things REPRESENTATION.md sections 2-3 say decide
    whether this arm is sound. Pass collect_diagnostics=False to skip.
    """
    model.eval().to(device)
    T = len(gammas)

    theta_and_x = init_theta_and_x.to(device).float()
    n_samples = theta_and_x.shape[0]
    assert theta_and_x.shape[1] == theta_dim + x_dim, theta_and_x.shape
    xmask = _clean_mask(x_dim, clean_x_dims, device, theta_and_x.dtype)

    step = initial_step
    diags = []

    for t in reversed(range(T)):
        sigma = gammas[t]
        clamp_lo_t, clamp_hi_t = cmin_theta * float(sigma) ** 2, cmax_theta * float(sigma) ** 2
        clamp_lo_x, clamp_hi_x = cmin_x * float(sigma) ** 2, cmax_x * float(sigma) ** 2
        n_clamped_t = n_clamped_x = 0

        for _ in range(step):
            theta_view = theta_and_x[:, :theta_dim]
            x_view = theta_and_x[:, theta_dim:]

            sig_batch = torch.full((n_samples, 1), float(sigma),
                                   device=device, dtype=theta_and_x.dtype)
            score = model(theta_view, x_view, sig_batch)

            a_theta, a_x = alpha_from_snr_blocks(
                score, theta_dim, x_dim, sigma, t_power,
                tau_theta=tau_theta, tau_x=tau_x,
                cmin_theta=cmin_theta, cmax_theta=cmax_theta,
                cmin_x=cmin_x, cmax_x=cmax_x,
                per_sample=per_sample_stepsize)

            if collect_diagnostics:
                # clamp-bound iff the returned step sits on a clamp boundary
                n_clamped_t += int(((a_theta <= clamp_lo_t * (1 + 1e-9)) |
                                    (a_theta >= clamp_hi_t * (1 - 1e-9))).all())
                n_clamped_x += int(((a_x <= clamp_lo_x * (1 + 1e-9)) |
                                    (a_x >= clamp_hi_x * (1 - 1e-9))).all())

            score_theta = score[:, :theta_dim]
            score_x = score[:, theta_dim:]

            eps_theta = torch.randn_like(score_theta)
            eps_x = torch.randn_like(score_x) * xmask

            theta_new = (theta_view + t_power * a_theta * score_theta
                         + torch.sqrt(2.0 * a_theta) * eps_theta)
            x_new = (x_view + (t_power * a_x * score_x
                               + torch.sqrt(2.0 * a_x) * eps_x) * xmask)

            theta_and_x = torch.cat([theta_new, x_new], dim=-1)

        if collect_diagnostics:
            th = theta_and_x[:, :theta_dim]
            diags.append({
                "sigma": float(sigma), "steps": step,
                "frac_steps_clamp_bound_theta": n_clamped_t / step,
                "frac_steps_clamp_bound_x": n_clamped_x / step,
                "frac_theta_out_of_box": float(((th < 0) | (th > 1)).any(dim=1)
                                               .float().mean()),
                "max_abs_excursion": float((th - th.clamp(0, 1)).abs().max()),
            })
        step += step_increase

    return theta_and_x.detach().clone(), diags
