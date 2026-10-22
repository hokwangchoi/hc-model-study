"""Lab 10: flow matching for multimodal action chunks (pi0-style action head) (reference solution)

Goal
----
A VLA such as pi0 outputs a CHUNK of H future actions at once, and it
generates the chunk with flow matching: start from Gaussian noise and follow
a learned velocity field for ~10 Euler steps. You implement the
interpolation path, the velocity target, the loss, the training loop, the
Euler sampler, and the two chunk utilities (cutting demonstrations into
chunks, and executing chunks with re-planning). Then you train a tiny
velocity MLP on demonstrations with TWO valid behaviours (pass an obstacle
on the left or on the right) and check that samples land on one of the two
modes, not on their average (which would drive into the obstacle).

The maths, step by step (this guide's convention)
-------------------------------------------------
a: an action chunk [H, d_a] from the data (H actions of d_a numbers);
eps ~ N(0, I) of the same shape;
t in [0, 1]:  t = 0 is pure noise, t = 1 is data.

  1. Path (straight line between noise and data):
       x_t = (1 - t) * eps + t * a
  2. Velocity along the path (constant in t):
       v = d x_t / dt = a - eps
  3. Loss: t ~ Uniform(0, 1) per sample (pi0 uses a Beta distribution that
     favours noisy t; uniform is fine here):
       L = || v_theta(x_t, t) - (a - eps) ||^2
     The network sees only x_t and t, so it learns the AVERAGE of a - eps over
     all (eps, a) pairs that pass through x_t: a velocity field whose flow
     carries the noise distribution onto the data distribution.
  4. Sampling with Euler, K steps (n_steps in the code), dt = 1 / K:
       x = eps;  for i = 0 .. K-1:  t_i = i / K;  x = x + v_theta(x, t_i) * dt
     After the last step x is a sample at t = 1.
  5. Chunks: a demonstration of T_demo steps gives T_demo - H + 1 training chunks
     (sliding windows). At run time the policy predicts a chunk, the robot
     executes the first n_exec actions, then the policy is called again.

  (openpi's code writes the same ODE backwards: x_t = t eps + (1 - t) a,
  t from 1 to 0. Same model, same samples, mirrored time.)

Read alongside
--------------
  ../vla/vla_guide.html         "The Core Idea: Instruction In, Action Chunk Out",
                                2. Flow Matching from Scratch (#fm-widget, and
                                "Conventions: the same ODE written three ways")
  ../diffusion/diffusion_guide.html   3.2 Flow matching, the modern alternative
  ../vla/vla.py                 euler_integrate, Pi0.compute_loss / generate_action

Run:  python labs/lab10_flow_matching.py   (or: python labs/check.py 10)
The training step takes about 5-20 s on a CPU.
Hints are graded (Hint 1 first). Solution: labs/solutions/lab10_flow_matching.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: path, target and loss
# ---------------------------------------------------------------------------
def interpolate(noise: torch.Tensor, actions: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """x_t = (1 - t) * noise + t * actions.

    noise, actions: [B, H, d_a];  t: [B] floats in [0, 1] (one time per sample).
    """
    t = t.view(-1, *([1] * (actions.dim() - 1)))          # [B, 1, 1]
    return (1 - t) * noise + t * actions


def velocity_target(noise: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """v = d x_t / dt = actions - noise."""
    return actions - noise


def flow_matching_loss(model: nn.Module, actions: torch.Tensor, noise: torch.Tensor,
                       t: torch.Tensor) -> torch.Tensor:
    """Scalar MSE between model(x_t, t) and the velocity target."""
    x_t = interpolate(noise, actions, t)
    return F.mse_loss(model(x_t, t), velocity_target(noise, actions))


# ---------------------------------------------------------------------------
# Given: the velocity network and the demonstration data
# ---------------------------------------------------------------------------
def time_embedding(t: torch.Tensor, dim: int = 32, max_period: float = 1000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous t in [0, 1]: [B] -> [B, dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device) / half)
    args = (t[:, None] * 1000.0) * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class VelocityMLP(nn.Module):
    """v_theta(x_t, t) for a whole chunk: flatten [H, d_a], concatenate a time embedding, MLP."""

    def __init__(self, horizon: int = 8, action_dim: int = 2, hidden: int = 256, t_dim: int = 32):
        super().__init__()
        self.horizon, self.action_dim, self.t_dim = horizon, action_dim, t_dim
        n = horizon * action_dim
        self.layers = nn.ModuleList([nn.Linear(n + t_dim, hidden), nn.Linear(hidden, hidden),
                                     nn.Linear(hidden, hidden), nn.Linear(hidden, n)])

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t [B, H, d_a], t [B] -> velocity [B, H, d_a]"""
        h = torch.cat([x_t.flatten(1), time_embedding(t, self.t_dim)], dim=-1)
        for layer in self.layers[:-1]:
            h = F.silu(layer(h))
        return self.layers[-1](h).view(-1, self.horizon, self.action_dim)


def mode_chunks(horizon: int = 8) -> torch.Tensor:
    """The two clean behaviours, [2, H, 2]: velocity (vx, vy) per step.
    vx = 1 (drive forward); vy = +/- a smooth bump (swerve left or right around an obstacle)."""
    h = torch.arange(horizon)
    vy = torch.sin(math.pi * (h + 0.5) / horizon)
    left = torch.stack([torch.ones(horizon), vy], dim=-1)
    right = torch.stack([torch.ones(horizon), -vy], dim=-1)
    return torch.stack([left, right])


def make_demo_chunks(n: int, generator: torch.Generator = None, horizon: int = 8,
                     std: float = 0.05) -> torch.Tensor:
    """n demonstration chunks [n, H, 2]: half swerve left, half right, plus small noise."""
    which = torch.randint(0, 2, (n,), generator=generator)
    return mode_chunks(horizon)[which] + std * torch.randn(n, horizon, 2, generator=generator)


# ---------------------------------------------------------------------------
# Step 2: training loop
# ---------------------------------------------------------------------------
def train_flow_matching(model: nn.Module, chunks: torch.Tensor, steps: int = 2000,
                        batch_size: int = 256, lr: float = 1e-3, seed: int = 0) -> list:
    """Train with flow_matching_loss on random minibatches of chunks [N, H, d_a].

    Each step: batch_size random chunks, noise ~ N(0, I) of the same shape,
    t ~ Uniform(0, 1) [batch_size], Adam update. Use a torch.Generator seeded
    with `seed` for the random draws. Returns the per-step losses (floats).
    """
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for _ in range(steps):
        a = chunks[torch.randint(0, len(chunks), (batch_size,), generator=g)]
        noise = torch.randn(a.shape, generator=g)
        t = torch.rand(batch_size, generator=g)
        loss = flow_matching_loss(model, a, noise, t)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


# ---------------------------------------------------------------------------
# Step 3: Euler sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def euler_sample(model: nn.Module, noise: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
    """Integrate dx/dt = model(x, t) from t = 0 (noise [B, H, d_a]) to t = 1.

    Step i (i = 0 .. n_steps - 1) evaluates the model at t_i = i / n_steps, passed
    as a float tensor [B], and moves x by velocity * (1 / n_steps).
    """
    x = noise
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((x.shape[0],), i * dt, dtype=x.dtype, device=x.device)
        x = x + model(x, t) * dt
    return x


# ---------------------------------------------------------------------------
# Step 4: action chunks
# ---------------------------------------------------------------------------
def chunk_trajectory(traj: torch.Tensor, horizon: int, stride: int = 1) -> torch.Tensor:
    """Sliding windows of a demonstration: traj [T_demo, d_a] -> [num_chunks, horizon, d_a].

    Window c starts at step c * stride; num_chunks = (T_demo - horizon) // stride + 1.
    """
    return traj.unfold(0, horizon, stride).transpose(1, 2)


def rollout_chunks(policy, n_actions: int, n_exec: int):
    """Receding-horizon execution.

    policy(step) returns a chunk [H, d_a] planned at time `step` (the number of
    actions executed so far). Execute its first n_exec actions (fewer at the very
    end), then call the policy again, until n_actions actions are executed.
    Returns (actions [n_actions, d_a], call_steps: list of the `step` values passed
    to the policy).
    """
    pieces, call_steps, step = [], [], 0
    while step < n_actions:
        chunk = policy(step)
        call_steps.append(step)
        take = min(n_exec, n_actions - step, chunk.shape[0])
        pieces.append(chunk[:take])
        step += take
    return torch.cat(pieces, dim=0), call_steps


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
