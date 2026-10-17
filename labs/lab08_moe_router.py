"""Lab 08: Mixture-of-Experts routing and dispatch

Goal
----
An MoE layer replaces the one FFN of a block with E expert FFNs and a
router that sends each token to only k of them. You build the two router
types used today (softmax top-k: Mixtral, Qwen-MoE; sigmoid top-k with a
balancing bias: DeepSeek-V3), the auxiliary load-balancing loss, two ways
to dispatch tokens to experts (a simple per-expert loop, and the
sort-and-scatter layout that grouped-GEMM kernels use), and capacity-limited
dispatch with token dropping.

The maths, step by step
-----------------------
T tokens (batch x sequence, flattened), d width, E experts, k experts per token.

  1. Router logits  s = x W_r^T                        [T, E]
  2a. Softmax top-k (Mixtral):
        p = softmax(s)                                 [T, E]
        idx = the k largest p per token (descending)   [T, k]
        w = p[idx] / sum(p[idx])                       [T, k]   (renormalized)
  2b. Sigmoid top-k (DeepSeek-V3):
        g = sigmoid(s)                                 [T, E]   (independent scores)
        idx = top-k of (g + bias)                      bias [E] only steers the CHOICE
        w = g[idx] / sum(g[idx])                       weights use g, not g + bias
  3. Load-balancing loss (Switch / Mixtral style):
        f_e = (number of the T*k assignments that went to expert e) / (T*k)
        P_e = mean over tokens of p[:, e]
        L_aux = E * sum_e f_e * P_e
     = 1.0 when routing is perfectly uniform, up to E when everything goes to
     one expert. f is a count (no gradient); the gradient flows through P.
  4. Dispatch: y[t] = sum_j w[t, j] * Expert_{idx[t, j]}(x[t])        [T, d]
       loop:    for each expert e, gather its tokens, run it once, scatter back.
       sorted:  flatten the T*k assignments, sort them by expert id, so each
                expert's tokens are one contiguous slice; run every slice,
                weight, and index_add the results back to their tokens.
  5. Capacity: each expert accepts at most
        C = ceil(capacity_factor * T * k / E)
     assignments. Served in order (token 0's choices, then token 1's, ...);
     assignments beyond C are dropped: they add nothing, and the token keeps
     only its residual for that expert.

Read alongside
--------------
  ../moe/moe_guide.html#moe-ffn    1. Dense FFN vs MoE FFN
  ../moe/moe_guide.html#router     2. The Router (2.1 load balancing, #router-viz)
  ../moe/moe_guide.html#inference  5.4 capacity factor and token dropping,
                                   5.5 grouped GEMM: from E tiny matmuls to one kernel
  ../moe/moe.py                    TopKRouter, RoutedExperts.dispatch_loop / dispatch_sorted

Run:  python labs/lab08_moe_router.py   (or: python labs/check.py 8)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab08_moe_router.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: routers
# ---------------------------------------------------------------------------
def softmax_topk_router(logits: torch.Tensor, k: int):
    """Mixtral-style router.

    logits [T, E] -> (weights [T, k], indices [T, k] int64, probs [T, E])
    probs = softmax(logits); indices = the k most probable experts, in
    DESCENDING order of probability; weights = their probs renormalized to sum to 1.
    """
    # Hint 1: torch.softmax, then torch.topk(probs, k, dim=-1) (sorted descending by default).
    # Hint 2: weights = top_p / top_p.sum(dim=-1, keepdim=True)
    raise NotImplementedError  # TODO


def sigmoid_topk_router(logits: torch.Tensor, k: int, bias: torch.Tensor = None):
    """DeepSeek-V3-style router.

    logits [T, E], bias [E] or None -> (weights [T, k], indices [T, k] int64)
    scores = sigmoid(logits); choose the k experts with the largest scores + bias
    (descending); weights = the UNBIASED scores of the chosen experts, renormalized.
    """
    # Hint 1: the bias only changes which experts are chosen, never the weights.
    # Hint 2: _, indices = torch.topk(scores + bias, k); chosen = scores.gather(-1, indices)
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 2: the load-balancing loss
# ---------------------------------------------------------------------------
def load_balancing_loss(probs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """L_aux = E * sum_e f_e * P_e  (see the module docstring).

    probs [T, E] (softmax router probabilities), indices [T, k] -> scalar tensor.
    """
    # Hint 1: f = torch.bincount(indices.flatten(), minlength=E) / indices.numel()
    # Hint 2: P = probs.mean(dim=0); return E * (f * P).sum()
    #         (f must be float; the gradient only flows through P.)
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Given: experts
# ---------------------------------------------------------------------------
class Expert(nn.Module):
    """One expert: a small GELU MLP, d -> d_ff -> d."""

    def __init__(self, d: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


def make_experts(n_experts: int, d: int, d_ff: int) -> nn.ModuleList:
    return nn.ModuleList(Expert(d, d_ff) for _ in range(n_experts))


# ---------------------------------------------------------------------------
# Step 3: dispatch, two ways
# ---------------------------------------------------------------------------
def dispatch_loop(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                  experts: nn.ModuleList) -> torch.Tensor:
    """y[t] = sum_j weights[t, j] * experts[indices[t, j]](x[t]), one expert at a time.

    x [T, d], weights/indices [T, k] -> y [T, d]. Call each expert at most once,
    on all of its tokens together (skip experts that got no tokens).
    """
    # Hint 1: for e in range(len(experts)): find the (token, slot) pairs that chose e:
    #         tok, slot = torch.where(indices == e)
    # Hint 2: if tok.numel() == 0: continue;  out = experts[e](x[tok])   # [n_e, d]
    # Hint 3: y.index_add_(0, tok, out * weights[tok, slot, None])
    #         (index_add_ sums correctly even if the same token appears twice)
    raise NotImplementedError  # TODO


def dispatch_sorted(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                    experts: nn.ModuleList) -> torch.Tensor:
    """Same result as dispatch_loop, via sort -> contiguous slices -> scatter.

    1. flatten the assignments: expert_of = indices.flatten() [T*k], token_of = the
       token each one belongs to, weight_of = weights.flatten()
    2. order = argsort(expert_of): now every expert's assignments are contiguous
    3. counts = bincount(expert_of, minlength=E); run expert e on its slice of
       x[token_of[order]] (this layout is what grouped-GEMM kernels consume)
    4. multiply by the weights and index_add the rows back to their tokens
    """
    # Hint 1: token_of = torch.arange(T).repeat_interleave(k) matches indices.flatten().
    # Hint 2: xs = x[token_of[order]]; chunks = torch.split(xs, counts.tolist())
    # Hint 3: out = torch.cat([experts[e](c) for e, c in enumerate(chunks) if len(c)])  (keep
    #         the order!) then y.index_add_(0, token_of[order], out * weight_of[order, None])
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 4: capacity and token dropping
# ---------------------------------------------------------------------------
def expert_capacity(n_tokens: int, n_experts: int, k: int, capacity_factor: float) -> int:
    """C = ceil(capacity_factor * n_tokens * k / n_experts), as an int."""
    # Hint 1: math.ceil (import math at the top), then int().
    raise NotImplementedError  # TODO


def dispatch_with_capacity(x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor,
                           experts: nn.ModuleList, capacity: int):
    """Like dispatch_loop, but every expert keeps at most `capacity` assignments.

    Assignments are served in token-major order: (token 0, slot 0), (token 0, slot 1),
    (token 1, slot 0), ...  An expert keeps the first `capacity` it receives and
    drops the rest. Returns (y [T, d], kept [T, k] bool).
    """
    # Hint 1: position of each assignment inside its expert's queue: one-hot the
    #         flattened indices [T*k, E], cumsum over dim 0, read your own column.
    # Hint 2: onehot = F.one_hot(indices.flatten(), E); pos = (onehot.cumsum(0) * onehot).sum(-1) - 1
    #         kept = (pos < capacity).view(T, k)
    # Hint 3: dropped assignments must contribute nothing: dispatch_loop with the
    #         weights set to 0 where not kept gives y (an expert may then run on a
    #         token it ignores; real kernels skip it, the result is the same).
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
