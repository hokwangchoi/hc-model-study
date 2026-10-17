"""Lab 07: Mamba's selective scan: sequential, parallel and recurrent (reference solution)

Goal
----
Implement the core of a Mamba layer, the selective state-space scan, three
ways that must agree:
  * a sequential loop over time (the definition),
  * a parallel prefix scan in O(log L) rounds (how training runs),
  * a one-token recurrent step with a fixed-size state (how generation runs).
The projections around the scan (in_proj, conv, gating) are ordinary layers
and are left out.

The maths, step by step
-----------------------
b batch, L sequence length, D channels, N state size per channel.
Every channel d runs its own little linear system with an N-dim state.

  Inputs   x [b, L, D]       the sequence (the paper's x_t; mamba.py calls it u)
           delta [b, L, D]   step size, > 0, per token and channel (input-dependent:
                             delta = softplus(Linear(x)) in Mamba)
           A [D, N]          learned, negative (decay rates)
           B [b, L, N]       input -> state, per token (input-dependent)
           C [b, L, N]       state -> output, per token (input-dependent)
           D_skip [D]        direct skip connection (optional)

  1. Discretize (continuous system -> one update per token):
       A_bar[t, d, n] = exp(delta[t, d] * A[d, n])            in (0, 1): how much to keep
       B_bar[t, d, n] = delta[t, d] * B[t, n]                  ("simple", what Mamba's kernel uses)
       or, exact zero-order hold for diagonal A:
       B_bar[t, d, n] = (exp(delta A) - 1) / A * B[t, n]       ("zoh"; equals the simple rule
                                                               to first order in delta)
     Large delta: forget the past, write the input. Small delta: keep the state.
  2. Recurrence (h_0 = 0, or a given h0):
       h_t = A_bar_t * h_{t-1} + B_bar_t * x_t               h: [b, D, N]
       y_t = sum_n C_t[n] * h_t[:, :, n]  + D_skip * x_t      y: [b, D]
  3. Each step is an affine map h -> a * h + c with a = A_bar_t, c = B_bar_t x_t.
     Doing step 1 then step 2 is again affine:
       (a2, c2) o (a1, c1) = (a2 * a1,  a2 * c1 + c2)
     This combine is associative, so all prefixes h_1..h_L can be computed
     in ceil(log2 L) rounds (Hillis-Steele): in round r, every position t
     combines with position t - 2^r. After the last round position t holds
     the composition of steps 1..t, whose c part is h_t (because h_0 = 0).
  4. Generation: carry only h [b, D, N] (constant size, no KV cache) and
     apply the recurrence once per new token.

  Worked numbers (A = -1, B = 1): delta = 0.1 -> A_bar = 0.905; delta = 5 -> 0.0067.

Read alongside
--------------
  ../mamba/mamba_guide.html   "The Core Idea: A Linear Recurrence",
                              1. SSM -> Selective SSM (#scan-widget),
                              2. Two Modes of Computation, 6. Code Walkthrough
  ../mamba/mamba.py           selective_scan_reference / selective_scan_parallel

Run:  python labs/lab07_mamba_scan.py   (or: python labs/check.py 7)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab07_mamba_scan.py
"""

import torch


# ---------------------------------------------------------------------------
# Step 1: discretization
# ---------------------------------------------------------------------------
def discretize(delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, method: str = "simple"):
    """
    Args:
        delta: [b, L, D]   A: [D, N]   B: [b, L, N]
        method: "simple" (B_bar = delta * B) or "zoh" (B_bar = (exp(delta A) - 1) / A * B)
    Returns:
        A_bar, B_bar: each [b, L, D, N]
    """
    dA = delta[..., None] * A                          # [b, L, D, N]
    A_bar = torch.exp(dA)
    if method == "simple":
        B_bar = delta[..., None] * B[:, :, None, :]    # [b, L, D, N]
    elif method == "zoh":
        B_bar = (A_bar - 1.0) / A * B[:, :, None, :]
    else:
        raise ValueError(f"unknown method {method!r}")
    return A_bar, B_bar


# ---------------------------------------------------------------------------
# Step 2: the sequential scan (the definition)
# ---------------------------------------------------------------------------
def selective_scan_sequential(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                              B: torch.Tensor, C: torch.Tensor, D_skip: torch.Tensor = None,
                              h0: torch.Tensor = None):
    """Loop over t: h_t = A_bar_t * h_{t-1} + B_bar_t * x_t;  y_t = <C_t, h_t> + D_skip * x_t.

    Args:
        x, delta: [b, L, D]   A: [D, N]   B, C: [b, L, N]   D_skip: [D] or None
        h0: initial state [b, D, N] or None (zeros)
    Returns:
        y:      [b, L, D]
        h_last: [b, D, N]   the state after the last token (lets you continue later)
    Use discretize(..., method="simple").
    """
    b, L, D = x.shape
    A_bar, B_bar = discretize(delta, A, B)
    h = torch.zeros(b, D, A.shape[1], dtype=x.dtype, device=x.device) if h0 is None else h0
    ys = []
    for t in range(L):
        h = A_bar[:, t] * h + B_bar[:, t] * x[:, t, :, None]     # [b, D, N]
        ys.append((h * C[:, t, None, :]).sum(-1))                # [b, D]
    y = torch.stack(ys, dim=1)                                    # [b, L, D]
    if D_skip is not None:
        y = y + D_skip * x
    return y, h


# ---------------------------------------------------------------------------
# Step 3: the associative combine and the parallel scan
# ---------------------------------------------------------------------------
def scan_combine(a1: torch.Tensor, c1: torch.Tensor, a2: torch.Tensor, c2: torch.Tensor):
    """Compose two affine steps: first (a1, c1), then (a2, c2).

    Applying the result to h must equal a2 * (a1 * h + c1) + c2.
    Returns (a, c), elementwise.
    """
    return a2 * a1, a2 * c1 + c2


def parallel_scan(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """All prefixes of h_t = a_t * h_{t-1} + c_t with h_0 = 0, in O(log L) rounds.

    a, c: [b, L, ...] (time is dim 1). Returns h: [b, L, ...] with h[:, t] = h_{t+1}
    in 1-based maths, i.e. the state AFTER processing element t.
    Do not loop over L; loop over the rounds offset = 1, 2, 4, ... < L.
    """
    L = a.shape[1]
    offset = 1
    while offset < L:
        a_prev = torch.cat([torch.ones_like(a[:, :offset]), a[:, :-offset]], dim=1)
        c_prev = torch.cat([torch.zeros_like(c[:, :offset]), c[:, :-offset]], dim=1)
        a, c = scan_combine(a_prev, c_prev, a, c)    # earlier block first, then this one
        offset *= 2
    return c                                         # c[:, t] = h after step t (h_0 = 0)


def selective_scan_parallel(x: torch.Tensor, delta: torch.Tensor, A: torch.Tensor,
                            B: torch.Tensor, C: torch.Tensor,
                            D_skip: torch.Tensor = None) -> torch.Tensor:
    """Same y as selective_scan_sequential (h0 = 0), via discretize + parallel_scan."""
    A_bar, B_bar = discretize(delta, A, B)
    h = parallel_scan(A_bar, B_bar * x[..., None])               # [b, L, D, N] all states
    y = (h * C[:, :, None, :]).sum(-1)                           # [b, L, D]
    if D_skip is not None:
        y = y + D_skip * x
    return y


# ---------------------------------------------------------------------------
# Step 4: one recurrent step (generation)
# ---------------------------------------------------------------------------
def recurrent_step(h: torch.Tensor, x_t: torch.Tensor, delta_t: torch.Tensor, A: torch.Tensor,
                   B_t: torch.Tensor, C_t: torch.Tensor, D_skip: torch.Tensor = None):
    """Process ONE token.

    Args:
        h: [b, D, N] state   x_t, delta_t: [b, D]   A: [D, N]   B_t, C_t: [b, N]
    Returns:
        y_t: [b, D],  h_new: [b, D, N]
    """
    A_bar = torch.exp(delta_t[..., None] * A)                    # [b, D, N]
    B_bar = delta_t[..., None] * B_t[:, None, :]                 # [b, D, N]
    h_new = A_bar * h + B_bar * x_t[..., None]
    y_t = (h_new * C_t[:, None, :]).sum(-1)                      # [b, D]
    if D_skip is not None:
        y_t = y_t + D_skip * x_t
    return y_t, h_new


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
