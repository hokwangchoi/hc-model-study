"""Lab 06: a diffusion model from scratch on 2-D toy data

Goal
----
Implement the pieces every diffusion model shares (DDPM training, DDIM
sampling) and train a tiny MLP denoiser on a 2-D distribution: 8 Gaussian
blobs on a ring. If everything is right, pure noise turns into points on
the 8 blobs. The same maths drives DiT and Stable Diffusion; only the
network (a transformer on image latents) is bigger.

The maths, step by step
-----------------------
T = 1000 noise levels, t = 0 .. T-1 (t = 0 is almost clean).

  1. Noise schedule (DDPM / DiT linear schedule)
       beta_t     = linspace(1e-4, 0.02, T)[t]          noise added at step t
       alpha_t    = 1 - beta_t
       abar_t     = alpha_0 * alpha_1 * ... * alpha_t   (cumulative product)
     abar_t falls from ~1 (clean) to ~4e-5 (pure noise).
  2. Forward noising in closed form, q(x_t | x_0):
       x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps,   eps ~ N(0, I)
     No loop over t: any noise level is one line away.
  3. Training objective (epsilon prediction):
       sample x_0 from the data, t ~ Uniform{0..T-1}, eps ~ N(0, I)
       loss = || eps_theta(x_t, t) - eps ||^2   (mean squared error)
     The network learns to name the noise that was added.
  4. DDIM sampling (deterministic, sigma = 0), on a subsequence of steps,
     e.g. 50 steps: t = 999, 979, ..., 19 and then "t = -1" with abar = 1:
       eps_hat = eps_theta(x_t, t)
       x0_hat  = (x_t - sqrt(1 - abar_t) * eps_hat) / sqrt(abar_t)
       x_next  = sqrt(abar_next) * x0_hat + sqrt(1 - abar_next) * eps_hat
     Start from x_T ~ N(0, I); the last step (abar_next = 1) returns x0_hat.

  Worked example from the guide (one pixel): x_0 = 0.8, eps = -1.2, t = 500:
       sqrt(abar) = 0.2789, sqrt(1 - abar) = 0.9603, x_500 = -0.9292,
       and one DDIM step to t = 250 with the true eps gives x_250 = -0.2525.

Read alongside
--------------
  ../diffusion/diffusion_guide.html   "The Core Idea: Iterative Denoising",
                                      1. Forward Process (1.1 the noise schedule,
                                      #noise-widget), 3.1 DDIM, 5. Code Walkthrough

Run:  python labs/lab06_diffusion.py   (or: python labs/check.py 6)
The last step trains a model for about 5-20 s on a CPU.
Hints are graded (Hint 1 first). Solution: labs/solutions/lab06_diffusion.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: the noise schedule
# ---------------------------------------------------------------------------
def linear_beta_schedule(T: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    """betas [T] (float32), linearly spaced from beta_start to beta_end inclusive."""
    # Hint 1: torch.linspace includes both end points.
    raise NotImplementedError  # TODO


def alpha_bars_from_betas(betas: torch.Tensor) -> torch.Tensor:
    """abar [T]: abar_t = prod_{s <= t} (1 - beta_s)."""
    # Hint 1: torch.cumprod(..., dim=0)
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 2: forward noising q(x_t | x_0)
# ---------------------------------------------------------------------------
def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
             alpha_bars: torch.Tensor) -> torch.Tensor:
    """x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) noise.

    x0, noise: [B, ...] (any trailing shape: [B, 2] points or [B, C, H, W] images)
    t:         [B] int64, one noise level per sample
    """
    # Hint 1: alpha_bars[t] is [B]; it must broadcast against [B, ...].
    # Hint 2: ab = alpha_bars[t].view(-1, *([1] * (x0.dim() - 1)))   -> [B, 1, ...]
    # Hint 3: return ab.sqrt() * x0 + (1 - ab).sqrt() * noise
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 3: the epsilon-prediction loss
# ---------------------------------------------------------------------------
def epsilon_loss(model: nn.Module, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor,
                 alpha_bars: torch.Tensor) -> torch.Tensor:
    """Scalar MSE between model(x_t, t) and the noise that produced x_t."""
    # Hint 1: build x_t with your q_sample, run the model, compare with the noise.
    # Hint 2: return F.mse_loss(model(q_sample(x0, t, noise, alpha_bars), t), noise)
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Given: the denoiser network and the toy data
# ---------------------------------------------------------------------------
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of integer timesteps: [B] -> [B, dim] (as in DiT)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class Denoiser(nn.Module):
    """eps_theta(x_t, t): a 3-hidden-layer MLP on [x_t, embed(t)]. Tiny stand-in for a DiT."""

    def __init__(self, dim: int = 2, hidden: int = 128, t_dim: int = 32):
        super().__init__()
        self.t_dim = t_dim
        self.layers = nn.ModuleList([nn.Linear(dim + t_dim, hidden), nn.Linear(hidden, hidden),
                                     nn.Linear(hidden, hidden), nn.Linear(hidden, dim)])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t [B, dim], t [B] int64 -> predicted noise [B, dim]"""
        h = torch.cat([x_t, timestep_embedding(t, self.t_dim)], dim=-1)
        for layer in self.layers[:-1]:
            h = F.silu(layer(h))
        return self.layers[-1](h)


def make_ring_data(n: int, generator: torch.Generator = None, n_modes: int = 8,
                   radius: float = 2.0, std: float = 0.1) -> torch.Tensor:
    """n points [n, 2]: a mixture of n_modes Gaussians evenly spaced on a circle."""
    k = torch.randint(0, n_modes, (n,), generator=generator)
    angle = 2 * math.pi * k / n_modes
    centres = torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)
    return centres + std * torch.randn(n, 2, generator=generator)


# ---------------------------------------------------------------------------
# Step 4: the training loop
# ---------------------------------------------------------------------------
def train_denoiser(model: nn.Module, data: torch.Tensor, alpha_bars: torch.Tensor,
                   steps: int = 3000, batch_size: int = 512, lr: float = 2e-3,
                   seed: int = 0) -> list:
    """Train `model` with epsilon_loss on random minibatches of `data` [N, 2].

    Each step: pick batch_size random rows of data, t ~ Uniform{0..T-1},
    noise ~ N(0, I), loss = epsilon_loss(...), Adam update.
    Use a torch.Generator seeded with `seed` for every random draw.
    Returns the list of per-step losses (Python floats).
    """
    # Hint 1: g = torch.Generator().manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Hint 2: per step: idx = torch.randint(0, len(data), (batch_size,), generator=g)
    #         t = torch.randint(0, len(alpha_bars), (batch_size,), generator=g)
    #         noise = torch.randn(batch_size, data.shape[1], generator=g)
    # Hint 3: loss = epsilon_loss(model, data[idx], t, noise, alpha_bars)
    #         opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 5: DDIM sampling
# ---------------------------------------------------------------------------
def ddim_timesteps(T: int, num_steps: int) -> torch.Tensor:
    """Descending int64 timesteps with stride T // num_steps, ending at stride - 1.

    ddim_timesteps(1000, 50) -> [999, 979, 959, ..., 39, 19]  (50 values)
    ddim_timesteps(1000, 4)  -> [999, 749, 499, 249]
    """
    # Hint 1: stride = T // num_steps; the values are k * stride - 1 for k = num_steps .. 1.
    # Hint 2: torch.arange(num_steps, 0, -1) * stride - 1
    raise NotImplementedError  # TODO


def ddim_step(x_t: torch.Tensor, eps_hat: torch.Tensor, abar_t, abar_next) -> torch.Tensor:
    """One deterministic DDIM update from level abar_t to level abar_next.

    abar_t, abar_next: floats or 0-dim tensors. Returns x_next (same shape as x_t).
    """
    # Hint 1: two lines: estimate x0, then re-noise it to the next level using the
    #         SAME eps_hat (that is what makes DDIM deterministic).
    # Hint 2: abar_t = torch.as_tensor(abar_t) lets you call .sqrt() on floats too.
    raise NotImplementedError  # TODO


@torch.no_grad()
def ddim_sample(model: nn.Module, x_T: torch.Tensor, alpha_bars: torch.Tensor,
                num_steps: int = 50) -> torch.Tensor:
    """Run DDIM from x_T [n, ...] (pure noise) to x_0.

    Visit t in ddim_timesteps(T, num_steps). At each t call
    model(x, t_batch) with t_batch = an int64 tensor [n] filled with t;
    the next level is the next timestep in the list, and after the last
    one it is abar = 1.0 (clean).
    """
    # Hint 1: ts = ddim_timesteps(len(alpha_bars), num_steps); loop with enumerate.
    # Hint 2: t_batch = torch.full((x.shape[0],), int(t), dtype=torch.long)
    # Hint 3: abar_next = alpha_bars[ts[i + 1]] if i + 1 < len(ts) else 1.0
    #         x = ddim_step(x, model(x, t_batch), alpha_bars[t], abar_next)
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
