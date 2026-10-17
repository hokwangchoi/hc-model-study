"""Lab 01: scaled dot-product attention, from matmul + softmax (reference solution)

Goal
----
Build the operation at the heart of every model in this repo: scaled
dot-product attention with a causal mask, and then multi-head attention
(split the width into heads, attend per head, merge back). You use only
matmul, exp and reshapes; no torch attention function.

The maths, step by step
-----------------------
Notation: B batch, h heads, N_q queries, N_k keys, d_k key/query width,
d_v value width, d = h * d_k model width.

  Inputs   Q [B, h, N_q, d_k]    K [B, h, N_k, d_k]    V [B, h, N_k, d_v]

  1. Scores      S = Q @ K^T / sqrt(d_k)                   [B, h, N_q, N_k]
                 S[i, j] = how well query i matches key j. Dividing by
                 sqrt(d_k) keeps the variance of S near 1, so softmax does
                 not saturate.
  2. Mask        S[i, j] = -inf wherever query i must NOT see key j.
                 Causal: query i sees keys j <= i (no peeking ahead).
  3. Softmax     P = softmax(S, dim=-1)                    [B, h, N_q, N_k]
                 P[i, j] = exp(S[i, j] - m_i) / sum_j' exp(S[i, j'] - m_i),
                 m_i = max_j S[i, j] (subtracting m_i avoids overflow and
                 does not change the result). Each row sums to 1; masked
                 entries get exactly 0.
  4. Output      O = P @ V                                 [B, h, N_q, d_v]
                 Each output row is a weighted average of the value rows.

Multi-head attention on x [B, N, d] with n_heads = h, d_k = d / h:

  q = x W_q^T + b_q,  k = x W_k^T + b_k,  v = x W_v^T + b_v   [B, N, d]
  split:  [B, N, d] -> view [B, N, h, d_k] -> transpose -> [B, h, N, d_k]
  attend: O = attention(q, k, v)                              [B, h, N, d_k]
  merge:  [B, h, N, d_k] -> transpose -> [B, N, h, d_k] -> [B, N, d]
  out = O W_o^T + b_o                                         [B, N, d]

Head i owns columns i*d_k ... (i+1)*d_k - 1 of q, k and v. That is why the
split is "view as [B, N, h, d_k], then transpose", and not "view as
[B, h, N, d_k]" (which would mix tokens and heads).

Read alongside
--------------
  ../transformer/transformer_step_by_step.html      (the step-by-step page)
  ../transformer/transformer_guide.html#worked-example  2a. attention with real numbers
  ../transformer/transformer_guide.html#mha         2c. multi-head data flow
  ../transformer/transformer_guide.html#reshape     2d. [B, N, d] -> [B, h, N, d_k]
  ../transformer/transformer_guide.html#qkt         2e. why Q K^T is [B, h, N, N]
  ../transformer/transformer_guide.html#causal-mask 3.  the causal mask

How to work
-----------
Implement the functions in order. After each one run

    python labs/lab01_attention.py        (or: python labs/check.py 1)

Steps you have not started show TODO; finished steps show PASS. The hints
are graded: read Hint 1 first and only go on if you are stuck. The full
answer is in labs/solutions/lab01_attention.py.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Step 1: the causal mask
# ---------------------------------------------------------------------------
def causal_mask(n_q: int, n_k: int = None, device=None) -> torch.Tensor:
    """Boolean mask of shape [n_q, n_k]; True means "query i may attend to key j".

    With n_k == n_q (the usual case) query i sees keys 0..i, so the mask is
    lower-triangular, diagonal included.

    When there are more keys than queries (n_k > n_q) the queries are the
    LAST n_q positions of the sequence; this happens during generation with a
    KV cache (lab 03). Query i then sits at absolute position i + (n_k - n_q)
    and may attend to keys j <= i + (n_k - n_q).

    Example, n_q = 2, n_k = 4 (2 cached keys + 2 new tokens):
        [[True, True, True, False],
         [True, True, True, True ]]
    """
    if n_k is None:
        n_k = n_q
    offset = n_k - n_q
    q_pos = torch.arange(n_q, device=device)[:, None] + offset  # [n_q, 1] absolute positions
    k_pos = torch.arange(n_k, device=device)[None, :]           # [1, n_k]
    return k_pos <= q_pos                                       # [n_q, n_k]


# ---------------------------------------------------------------------------
# Step 2: a numerically stable softmax
# ---------------------------------------------------------------------------
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """softmax(x)_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x)) along `dim`.

    Must not overflow for large inputs (try x = [1000., 1001., 1002.]) and
    must give exactly 0 for entries equal to -inf (masked positions).
    Do not call torch.softmax / F.softmax here; build it from torch.exp.
    """
    m = x.amax(dim=dim, keepdim=True)   # row max, same rank as x
    e = torch.exp(x - m)                # largest entry becomes exp(0) = 1
    return e / e.sum(dim=dim, keepdim=True)


# ---------------------------------------------------------------------------
# Step 3: scaled dot-product attention
# ---------------------------------------------------------------------------
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: torch.Tensor = None):
    """Attention from matmul + softmax.

    Args:
        q:    [..., N_q, d_k]   (leading dims, e.g. [B, h], are batch dims)
        k:    [..., N_k, d_k]
        v:    [..., N_k, d_v]
        mask: optional bool tensor broadcastable to [..., N_q, N_k];
              True = may attend (same convention as causal_mask()).

    Returns:
        out:     [..., N_q, d_v]
        weights: [..., N_q, N_k]   the softmax probabilities (rows sum to 1)
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / d_k ** 0.5              # [..., N_q, N_k]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))      # blocked -> weight 0
    weights = softmax(scores, dim=-1)                          # rows sum to 1
    out = weights @ v                                          # [..., N_q, d_v]
    return out, weights


# ---------------------------------------------------------------------------
# Step 4: split and merge heads
# ---------------------------------------------------------------------------
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B, N, d] -> [B, h, N, d_k] with h = n_heads and d_k = d // h.

    Head i must hold the columns x[:, :, i*d_k:(i+1)*d_k].
    """
    B, N, d = x.shape
    d_k = d // n_heads
    return x.view(B, N, n_heads, d_k).transpose(1, 2)          # [B, h, N, d_k]


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, h, N, d_k] -> [B, N, h * d_k], the exact inverse of split_heads."""
    B, h, N, d_k = x.shape
    return x.transpose(1, 2).reshape(B, N, h * d_k)            # [B, N, d]


# ---------------------------------------------------------------------------
# Step 5: multi-head attention module
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with separate q/k/v/out projections.

    The layers are created for you. The check copies these weights into
    torch.nn.MultiheadAttention and compares the outputs.
    """

    def __init__(self, d_model: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x: torch.Tensor, causal: bool = False):
        """
        Args:
            x: [B, N, d_model]
            causal: if True, apply causal_mask(N)
        Returns:
            out:     [B, N, d_model]
            weights: [B, h, N, N]  per-head attention probabilities
        """
        B, N, _ = x.shape
        q = split_heads(self.q_proj(x), self.n_heads)          # [B, h, N, d_k]
        k = split_heads(self.k_proj(x), self.n_heads)
        v = split_heads(self.v_proj(x), self.n_heads)
        mask = causal_mask(N, device=x.device) if causal else None
        o, weights = scaled_dot_product_attention(q, k, v, mask)
        out = self.out_proj(merge_heads(o))                    # [B, N, d]
        return out, weights


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
