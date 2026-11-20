"""Conditional mixture-density head, interface-compatible with BoundedNSF.

Architecture mirrors the Route A baseline's ConditionalMDN (see
baselines/sun_routeA/vendor/RouteB_utils.py), which is left untouched: a linear input
projection, residual blocks with a zero-initialised second layer, FiLM modulation from a
scalar conditioning input, and diagonal-Gaussian mixture heads. Reimplemented here rather
than imported so the class is resolvable from this package at torch.load time and carries
no plotting/tqdm dependencies.

Two differences from an nflows flow, both deliberate:
  - theta is modelled in z-scored coordinates with the standardisation folded into
    log_prob's Jacobian, matching the baseline's ZScore.fit(theta) treatment.
  - support is unbounded, so draws can land outside the prior box. Callers reject them;
    the accept rate is a reported quantity, not an error.

The conditioning vector arrives already concatenated as [features, scalar]; `split_at`
says where the features end, and everything after it is routed to FiLM.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOGSIGMA_MIN = -7.0
LOGSIGMA_MAX = 10.0


class ResBlock(nn.Module):
    def __init__(self, hidden, activation=nn.ReLU):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.fc1 = nn.Linear(hidden, hidden)
        self.act = activation()
        self.fc2 = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(self.ln(x))))


class ConditionalMDN(nn.Module):
    def __init__(self, theta_dim, feature_dim, beta_dim=1, hidden=128,
                 n_components=8, n_hidden_layers=2, theta_mu=None, theta_std=None,
                 activation=nn.ReLU):
        super().__init__()
        self.theta_dim = theta_dim
        self.split_at = feature_dim
        self.beta_dim = beta_dim
        self.K = n_components

        self.in_proj = nn.Sequential(nn.Linear(feature_dim, hidden), activation())
        self.blocks = nn.ModuleList([ResBlock(hidden, activation)
                                     for _ in range(max(n_hidden_layers - 1, 0))])
        self.film = nn.Sequential(nn.Linear(beta_dim, hidden), activation(),
                                  nn.Linear(hidden, hidden * 2))
        self.pi_head = nn.Linear(hidden, self.K)
        self.mu_head = nn.Linear(hidden, self.K * theta_dim)
        self.log_sigma_head = nn.Linear(hidden, self.K * theta_dim)

        mu = torch.zeros(theta_dim) if theta_mu is None else torch.as_tensor(theta_mu).float()
        std = torch.ones(theta_dim) if theta_std is None else torch.as_tensor(theta_std).float()
        self.register_buffer("theta_mu", mu)
        self.register_buffer("theta_std", std)

    def _params(self, condition):
        x, beta = condition[:, :self.split_at], condition[:, self.split_at:]
        h = self.in_proj(x.float())
        for blk in self.blocks:
            h = blk(h)
        gamma, shift = torch.chunk(self.film(beta.float()), 2, dim=-1)
        h = torch.tanh(gamma) * 2.0 * h + shift
        logit_pi = self.pi_head(h)
        mu = self.mu_head(h).view(-1, self.K, self.theta_dim)
        log_sigma = self.log_sigma_head(h).view(-1, self.K, self.theta_dim).clamp(
            LOGSIGMA_MIN, LOGSIGMA_MAX)
        return logit_pi, mu, log_sigma

    def log_prob(self, inputs, condition):
        z = (inputs.float() - self.theta_mu) / self.theta_std
        logit_pi, mu, log_sigma = self._params(condition)
        quad = -0.5 * (((z.unsqueeze(1) - mu) ** 2) * torch.exp(-2.0 * log_sigma)).sum(-1)
        log_norm = (-0.5 * self.theta_dim * math.log(2 * math.pi)
                    - log_sigma.sum(-1) + quad)
        lp = torch.logsumexp(F.log_softmax(logit_pi, dim=-1) + log_norm, dim=-1)
        return lp - torch.log(self.theta_std).sum()      # change of variables to theta

    @torch.no_grad()
    def sample(self, sample_shape, condition):
        n = (int(sample_shape[0]) if isinstance(sample_shape, (tuple, list, torch.Size))
             else int(sample_shape))
        logit_pi, mu, log_sigma = self._params(condition)
        B = mu.shape[0]
        k = torch.multinomial(F.softmax(logit_pi, dim=-1), n, replacement=True)  # (B, n)
        gather = k.unsqueeze(-1).expand(B, n, self.theta_dim)
        m = torch.gather(mu, 1, gather)
        s = torch.gather(log_sigma, 1, gather).exp()
        z = m + s * torch.randn_like(m)
        theta = z * self.theta_std + self.theta_mu
        return theta.transpose(0, 1)                     # (n, B, theta_dim)


class WPassthrough(nn.Module):
    """Stands in for WEmbed when the density head conditions on log w directly."""

    def forward(self, log_w):
        return log_w


def build_mdn(theta_dim, feature_dim, theta_all, hidden=128, n_components=8,
              n_hidden_layers=2):
    t = torch.as_tensor(theta_all).float()
    return ConditionalMDN(theta_dim=theta_dim, feature_dim=feature_dim, beta_dim=1,
                          hidden=hidden, n_components=n_components,
                          n_hidden_layers=n_hidden_layers,
                          theta_mu=t.mean(0), theta_std=t.std(0) + 1e-8)
