"""Lab 03: a tiny GPT, sampling, and a KV cache that equals full recompute

Goal
----
Run a decoder-only transformer (GPT-2 layout) end to end: token ids in,
next-token logits out. Then generate text with greedy decoding and top-k
sampling, first by recomputing the whole sequence at every step, then with
a KV cache. The cached version must produce the same logits and tokens.
The model is untrained (random weights); correctness is what we check.

The maths, step by step
-----------------------
B batch, N tokens in this call, V vocab, d width, h heads, d_k = d / h.
The code keeps nanoGPT's variable names (as build-llm does): T = N, T_past = N_past.

Forward pass
  idx [B, N] (int64)
  x = tok_emb[idx] + pos_emb[pos]                       [B, N, d]
      pos = start, start+1, ..., start+N-1  (start = number of cached tokens)
  for each block (pre-norm, lab 02):
      x = x + Attn(LN1(x))
      x = x + MLP(LN2(x))
  logits = LN_f(x) @ tok_emb^T                          [B, N, V]
      (the output head shares its weight with the token embedding: "tied")

Causal attention with a KV cache
  q, k, v = split(x W_qkv^T + b)                         each [B, h, N, d_k]
  k_all = cat(k_past, k), v_all = cat(v_past, v)          [B, h, N_past + N, d_k]
  scores = q k_all^T / sqrt(d_k)                          [B, h, N, N_past + N]
  mask: new token i sits at absolute position N_past + i and may see keys
        j <= N_past + i  ->  causal_mask(N, N_past + N) from lab 01
  y = softmax(scores) v_all  -> merge heads -> W_o        [B, N, d]
  return y and the new cache (k_all, v_all)

Why the cache is exact: in a causal model the keys and values of token j
depend only on tokens 0..j. Appending a token never changes them, so they
can be computed once and reused. Each decode step then costs O(N) instead
of re-running all N tokens through the model.

Decoding
  greedy:  next = argmax(logits[:, -1])
  top-k:   keep the k largest logits, divide by the temperature, softmax,
           sample one index from those k (temperature < 1 sharpens,
           > 1 flattens; k = 1 is greedy).

Read alongside
--------------
  ../transformer/transformer_step_by_step.html      (the step-by-step page)
  ../transformer/transformer_guide.html#kv-cache    2.4 generation with a KV cache
  ../transformer/transformer_guide.html#causal-mask 2.3 the causal mask
  Preview (covered in the next tutorial, Build an LLM):
  ../build-llm/build_llm_guide.html#sampling        temperature, top-k, top-p
  ../build-llm/build_llm_guide.html#kvcache         the KV cache in a trained model

Run:  python labs/lab03_gpt_generate.py   (or: python labs/check.py 3)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab03_gpt_generate.py
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 64
    max_len: int = 64
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2


def causal_mask(n_q: int, n_k: int, device=None) -> torch.Tensor:
    """Given (you wrote it in lab 01): [n_q, n_k] bool, True = may attend.
    The n_q queries are the last n_q of the n_k positions."""
    offset = n_k - n_q
    return torch.arange(n_k, device=device)[None, :] <= torch.arange(n_q, device=device)[:, None] + offset


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Given: lab 01's scaled_dot_product_attention (torch's fused kernel).
    q [B, h, T_q, d_k], k and v [B, h, T_k, d_k], mask [T_q, T_k] bool -> [B, h, T_q, d_k]."""
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


# ---------------------------------------------------------------------------
# Step 1: causal self-attention that can read and extend a KV cache
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)   # rows: [W_q; W_k; W_v]
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)      # W_o

    def forward(self, x: torch.Tensor, past_kv=None):
        """
        Args:
            x:       [B, T, d]   the NEW tokens only
            past_kv: None, or a tuple (k_past, v_past), each [B, h, T_past, d_k]
        Returns:
            y:       [B, T, d]
            present: (k_all, v_all), each [B, h, T_past + T, d_k]
        """
        # Hint 1: q, k, v = self.qkv(x).split(d, dim=-1), then split each into heads
        #         ([B, T, d] -> [B, h, T, d_k]) exactly as in lab 01.
        # Hint 2: if past_kv is given, k = torch.cat([past_kv[0], k], dim=2) (same for v).
        #         The mask is then causal_mask(T, k.shape[2], x.device): the new tokens
        #         are the LAST T of the k.shape[2] positions. (torch's is_causal=True
        #         would align the mask to the top-left, wrong once there is a cache.)
        # Hint 3: y = attend(q, k, v, mask)
        #         y = y.transpose(1, 2).reshape(B, T, d); return self.proj(y), (k, v)
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 2: block and model forward
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d = cfg.d_model
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, past_kv=None):
        """Pre-norm block. Returns (x [B, T, d], present) where present comes from attn."""
        # Hint 1: lab 02's block, except that attention also returns its cache.
        # Hint 2: a, present = self.attn(self.ln1(x), past_kv); x = x + a
        #         x = x + self.mlp(self.ln2(x)); return x, present
        raise NotImplementedError  # TODO


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight             # weight tying

    def forward(self, idx: torch.Tensor, past_kvs=None):
        """
        Args:
            idx:      [B, T] token ids (only the NEW tokens when a cache is given)
            past_kvs: None, or a list with one (k, v) tuple per block
        Returns:
            logits:   [B, T, vocab_size]
            presents: list with one (k_all, v_all) tuple per block
        """
        # Hint 1: start = 0 without a cache, else the cached length past_kvs[0][0].shape[2].
        #         The positions of the new tokens are start, ..., start + T - 1.
        # Hint 2: pos = torch.arange(start, start + T, device=idx.device)
        #         x = self.tok_emb(idx) + self.pos_emb(pos)      ([B, T, d] + [T, d] broadcasts)
        # Hint 3: loop over (block, past) pairs, collecting presents; finish with
        #         logits = self.lm_head(self.ln_f(x))
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 3: choosing the next token
# ---------------------------------------------------------------------------
def greedy_next(logits: torch.Tensor) -> torch.Tensor:
    """logits [B, V] -> next token ids [B] (int64): the most likely token."""
    # Hint 1: one call; which dim holds the vocabulary?
    raise NotImplementedError  # TODO


def sample_top_k(logits: torch.Tensor, k: int, temperature: float = 1.0,
                 generator: torch.Generator = None) -> torch.Tensor:
    """Top-k sampling. logits [B, V] -> sampled token ids [B] (int64).

    1. keep the k largest logits of every row,
    2. divide them by `temperature` and softmax over those k,
    3. draw one of the k with torch.multinomial(..., generator=generator),
    4. map the drawn position (0..k-1) back to a vocabulary id.
    """
    # Hint 1: torch.topk(logits, k, dim=-1) returns (values, indices), both [B, k].
    # Hint 2: probs = torch.softmax(values / temperature, dim=-1)
    #         choice = torch.multinomial(probs, 1, generator=generator)    # [B, 1] in 0..k-1
    # Hint 3: return indices.gather(-1, choice).squeeze(-1)
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 4: generation, with and without a KV cache
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, use_cache: bool = True,
             top_k: int = None, temperature: float = 1.0,
             generator: torch.Generator = None) -> torch.Tensor:
    """Append max_new_tokens tokens to idx [B, T0]; returns [B, T0 + max_new_tokens].

    top_k=None means greedy decoding; otherwise use sample_top_k(..., top_k,
    temperature, generator).

    use_cache=False: every step runs the WHOLE sequence through the model
                     and uses the last position's logits.
    use_cache=True:  run the prompt once (the "prefill"), keep the returned
                     presents, then feed only the newly chosen token each step
                     together with the cache (the "decode" steps).
    """
    # Hint 1: write a small helper pick(last_logits) that returns greedy_next(...) or
    #         sample_top_k(...). The last position's logits are logits[:, -1].
    # Hint 2 (no cache): for _ in range(max_new_tokens):
    #             logits, _ = model(idx); idx = torch.cat([idx, pick(logits[:, -1])[:, None]], 1)
    # Hint 3 (cache): logits, past = model(idx) once; then each step pick the token,
    #         append it to idx, and call logits, past = model(token[:, None], past).
    #         (You do not need the model call after the very last token.)
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
