"""Lab 02: a pre-norm transformer block, and counting its parameters (reference solution)

Goal
----
Assemble one decoder block the way GPT-2 (LayerNorm + GELU MLP) and Llama
(RMSNorm + SwiGLU) build it, then write a closed-form parameter count and
check it against PyTorch's own count. Lab 01 built attention; here it
becomes one sub-layer of a block.

The maths, step by step (x is [B, N, d])
----------------------------------------
LayerNorm, per token, over the last dim (d values):
    mu      = mean(x)                                   [B, N, 1]
    var     = mean((x - mu)^2)       (divide by d, NOT d - 1)
    LN(x)   = gamma * (x - mu) / sqrt(var + eps) + beta     gamma, beta: [d]

RMSNorm (Llama, Mistral, Qwen...): no mean subtraction, no beta:
    RMS(x)  = gamma * x / sqrt(mean(x^2) + eps)              gamma: [d]

Feed-forward network (FFN), applied to every token independently:
    GELU MLP:  W2 GELU(W1 x + b1) + b2            d -> d_ff -> d   (d_ff = 4d in GPT-2)
    SwiGLU:    W_down( SiLU(W_gate x) * (W_up x) )  d -> d_ff (twice) -> d
               SiLU(z) = z * sigmoid(z); the "gate" decides, per hidden unit,
               how much of W_up x passes through.

Self-attention: lab 01's multi-head attention (q/k/v/o projections).

Pre-norm block (every modern LLM and ViT):
    h = x + Attn(Norm1(x))          the residual stream x is never normalised
    y = h + FFN(Norm2(h))           itself; each sub-layer reads a normalised
                                    copy and ADDS its result back.
(The 2017 paper used post-norm, x = Norm(x + Attn(x)), which trains worse
when deep.)

Parameter count (d model width, f = d_ff):
    attention   4 d^2           + 4 d   biases (q, k, v, o), if bias
    GELU MLP    2 d f           + f + d biases, if bias
    SwiGLU      3 d f           (no biases in Llama-style models)
    LayerNorm   2 d  (gamma, beta)       RMSNorm  d  (gamma)
The number of heads does not change the count: heads only split d.

Read alongside
--------------
  ../transformer/transformer_step_by_step.html       (the step-by-step page)
  ../transformer/transformer_guide.html#architecture  the block, expandable parts:
        #blk-ln (LayerNorm), #blk-residual, #blk-ffn
  ../transformer/transformer_guide.html#modern       RMSNorm, SwiGLU, pre-norm
  Preview (covered in the next tutorial, Build an LLM):
  ../build-llm/build_llm_guide.html#model            the same block in a trained LLM

Run:  python labs/lab02_transformer_block.py   (or: python labs/check.py 2)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab02_transformer_block.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: LayerNorm and RMSNorm
# ---------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """LayerNorm over the last dimension, written out (do not call F.layer_norm)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))   # gamma
        self.bias = nn.Parameter(torch.zeros(d))    # beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d] -> [..., d]"""
        mu = x.mean(dim=-1, keepdim=True)                           # [..., 1]
        var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)            # biased variance
        return self.weight * (x - mu) / torch.sqrt(var + self.eps) + self.bias


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension: gamma * x / sqrt(mean(x^2) + eps)."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))   # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d] -> [..., d]"""
        ms = x.pow(2).mean(dim=-1, keepdim=True)                    # mean square, [..., 1]
        return self.weight * x * torch.rsqrt(ms + self.eps)


# ---------------------------------------------------------------------------
# Step 2: the two feed-forward networks
# ---------------------------------------------------------------------------
class GeluMLP(nn.Module):
    """GPT-2 / ViT FFN: fc2(GELU(fc1(x))), d -> d_ff -> d."""

    def __init__(self, d: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff, bias=bias)
        self.fc2 = nn.Linear(d_ff, d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class SwiGLU(nn.Module):
    """Llama FFN: w_down( SiLU(w_gate(x)) * w_up(x) ), d -> d_ff -> d."""

    def __init__(self, d: int, d_ff: int, bias: bool = False):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=bias)
        self.w_up = nn.Linear(d, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ---------------------------------------------------------------------------
# Step 3: self-attention (lab 01 again, as a sub-layer)
# ---------------------------------------------------------------------------
def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """Given: lab 01's scaled_dot_product_attention, as torch's fused kernel.
    q, k, v [B, h, N, d_k] -> [B, h, N, d_k]."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


class SelfAttention(nn.Module):
    """Multi-head self-attention. Same maths as lab 01's MultiHeadAttention."""

    def __init__(self, d: int, n_heads: int, bias: bool = True):
        super().__init__()
        assert d % n_heads == 0
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d, d, bias=bias)
        self.k_proj = nn.Linear(d, d, bias=bias)
        self.v_proj = nn.Linear(d, d, bias=bias)
        self.o_proj = nn.Linear(d, d, bias=bias)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: [B, N, d] -> [B, N, d]  (returns only the output, no weights)"""
        B, N, d = x.shape
        h = self.n_heads

        def split(t):                                               # [B, N, d] -> [B, h, N, d/h]
            return t.view(B, N, h, d // h).transpose(1, 2)

        q, k, v = split(self.q_proj(x)), split(self.k_proj(x)), split(self.v_proj(x))
        o = attend(q, k, v, causal)                                  # [B, h, N, d/h]
        return self.o_proj(o.transpose(1, 2).reshape(B, N, d))


# ---------------------------------------------------------------------------
# Step 4: the pre-norm block
# ---------------------------------------------------------------------------
class Block(nn.Module):
    """Pre-norm transformer block.

    norm: "layernorm" (GPT-2, ViT) or "rmsnorm" (Llama)
    ffn:  "gelu" (GPT-2, ViT)      or "swiglu"  (Llama)
    bias: whether the Linear layers have biases (GPT-2: yes, Llama: no)
    """

    def __init__(self, d: int, n_heads: int, d_ff: int, norm: str = "layernorm",
                 ffn: str = "gelu", bias: bool = True):
        super().__init__()
        Norm = {"layernorm": LayerNorm, "rmsnorm": RMSNorm}[norm]
        self.norm1 = Norm(d)
        self.attn = SelfAttention(d, n_heads, bias=bias)
        self.norm2 = Norm(d)
        self.ffn = GeluMLP(d, d_ff, bias=bias) if ffn == "gelu" else SwiGLU(d, d_ff, bias=bias)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """x: [B, N, d] -> [B, N, d]"""
        x = x + self.attn(self.norm1(x), causal=causal)    # attention sub-layer
        x = x + self.ffn(self.norm2(x))                    # FFN sub-layer
        return x


# ---------------------------------------------------------------------------
# Step 5: count the parameters in closed form
# ---------------------------------------------------------------------------
def count_block_params(d: int, n_heads: int, d_ff: int, norm: str = "layernorm",
                       ffn: str = "gelu", bias: bool = True) -> int:
    """Number of parameters of Block(d, n_heads, d_ff, norm, ffn, bias), from a formula.

    Do not build the module and call .numel(); work it out from the shapes.
    Sanity anchors:
      GPT-2 small block: d=768, d_ff=3072, layernorm, gelu, bias  -> 7,087,872
      Llama-2-7B block:  d=4096, d_ff=11008, rmsnorm, swiglu, no bias -> 202,383,360
    """
    attn = 4 * d * d + (4 * d if bias else 0)                     # q, k, v, o projections
    if ffn == "gelu":
        mlp = 2 * d * d_ff + ((d_ff + d) if bias else 0)          # fc1 + fc2
    else:
        mlp = 3 * d * d_ff + ((2 * d_ff + d) if bias else 0)      # gate + up + down
    per_norm = 2 * d if norm == "layernorm" else d                 # gamma (+ beta)
    return attn + mlp + 2 * per_norm


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
