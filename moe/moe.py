"""
Mixture-of-Experts (MoE) LLM — Reference Implementation
Sparse-MoE decoder in the style of DeepSeek-V3 / Qwen3 / gpt-oss, plus
Multi-head Latent Attention (MLA) with a compressed KV cache.

Papers this file follows:
    Shazeer et al. 2017, "Outrageously Large Neural Networks"  (sparsely-gated MoE)
    Fedus et al. 2021, "Switch Transformers"                    (aux load-balancing loss, capacity factor)
    Dai et al. 2024, "DeepSeekMoE"                              (fine-grained + shared experts)
    Wang et al. 2024, "Auxiliary-Loss-Free Load Balancing"      (per-expert routing bias)
    DeepSeek-AI 2024, "DeepSeek-V2" / "DeepSeek-V3"             (MLA, sigmoid router, aux-free balancing)

Usage:
    python moe.py          # runs the tests + prints the model comparison (CPU, ~10 s)

Architecture (one decoder block):
    x [B, T, d]
      ↓ RMSNorm → MLA attention (KV cache = one d_c latent + one d_R RoPE key per token)
      ↓ + residual
      ↓ RMSNorm → MoE FFN:
      │     router: scores = sigmoid(x·W_r) (or softmax)   [T, E]
      │     pick top-k experts per token (bias b_i only affects the CHOICE)
      │     y = Σ_shared FFN_s(x) + Σ_{i∈topk} g_i · FFN_i(x)
      ↓ + residual
    x [B, T, d]

KEY INSIGHTS this file tries to teach:

1. TOTAL vs ACTIVE parameters. All E experts must be stored, but each token
   only runs k of them. FLOPs and decode-time weight reads scale with ACTIVE
   params; memory capacity scales with TOTAL params.

2. THE ROUTER is a tiny linear layer + top-k. Load balancing keeps experts
   evenly used: either an auxiliary loss (Switch) or a per-expert bias that is
   nudged up/down after each step (DeepSeek-V3, no extra gradient).

3. DISPATCH. The naive "for each expert: gather its tokens" loop and the
   sort/scatter "grouped" dispatch compute exactly the same thing; the grouped
   form turns E small matmuls into one batched GEMM (what MegaBlocks/DeepGEMM do).

4. MLA caches a small latent c_KV (+ a shared RoPE key) instead of full K/V.
   At decode the up-projections can be absorbed into W_Q and W_O, so attention
   runs directly against the latent cache.
"""
import math
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Config
# ============================================================================

@dataclass
class MoEConfig:
    vocab_size: int = 512
    d_model: int = 64
    n_layers: int = 4
    n_dense_layers: int = 1          # first layers keep a dense FFN (DeepSeek-V3: 3, Kimi K2: 1)
    d_ff_dense: int = 128
    # --- MLA attention ---
    n_heads: int = 4
    q_lora_rank: int = 48            # d'_c  (query compression)
    kv_lora_rank: int = 32           # d_c   (the cached KV latent)
    qk_nope_head_dim: int = 16       # d_h   (per-head dim of the non-RoPE part)
    qk_rope_head_dim: int = 8        # d_R   (decoupled RoPE key, shared by all heads)
    v_head_dim: int = 16
    rope_theta: float = 10000.0
    # --- MoE FFN ---
    n_routed_experts: int = 16       # E
    n_shared_experts: int = 1        # S (always active)
    top_k: int = 4                   # k
    d_expert: int = 32               # hidden size of ONE expert (fine-grained = small)
    scoring: str = "sigmoid"         # "sigmoid" (DeepSeek-V3) or "softmax" (Mixtral/Qwen3)
    balance: str = "aux_free"        # "aux_free" | "aux_loss" | "none"
    aux_loss_coef: float = 0.01      # Switch Transformer used 1e-2
    bias_update_speed: float = 0.001 # gamma; DeepSeek-V3 used 0.001
    routed_scaling: float = 1.0      # DeepSeek-V3 multiplies routed gates by 2.5
    capacity_factor: Optional[float] = None   # None = dropless (inference default)


# ============================================================================
# Small building blocks
# ============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope_cos_sin(positions: torch.Tensor, dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables for rotate-half RoPE. positions: [T] -> [T, dim]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    ang = positions.float()[:, None] * inv_freq[None, :]            # [T, dim/2]
    ang = torch.cat([ang, ang], dim=-1)                              # [T, dim]
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [..., T, dim] (T is the second-to-last dim)."""
    x1, x2 = x.chunk(2, dim=-1)
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class SwiGLU(nn.Module):
    """Dense FFN / shared expert: down(silu(gate(x)) * up(x)). Params = 3·d·f."""

    def __init__(self, d: int, f: int):
        super().__init__()
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ============================================================================
# 1. Router: top-k gating + load balancing
# ============================================================================

class TopKRouter(nn.Module):
    """
    scores  = sigmoid(x W_r^T)  or  softmax(x W_r^T)          [T, E]
    choice  = scores + bias     (bias only when balance == "aux_free")
    idx     = topk(choice, k)                                  [T, k]
    weights = scores[idx] / sum(scores[idx]) * routed_scaling  [T, k]

    The bias changes WHICH experts are picked, never the gate VALUE. That is the
    whole trick of auxiliary-loss-free balancing: no extra gradient touches the
    language-model loss.
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.E, self.k = cfg.n_routed_experts, cfg.top_k
        self.scoring, self.balance = cfg.scoring, cfg.balance
        self.aux_coef, self.gamma = cfg.aux_loss_coef, cfg.bias_update_speed
        self.scale = cfg.routed_scaling
        self.weight = nn.Parameter(torch.randn(self.E, cfg.d_model) * cfg.d_model ** -0.5)
        # Not a gradient parameter: updated by a rule, stored in the checkpoint.
        self.register_buffer("bias", torch.zeros(self.E))

    def forward(self, x: torch.Tensor):
        """x: [T, d] -> (idx [T,k], weights [T,k], aux_loss scalar, load [E])"""
        logits = x @ self.weight.t()                                  # [T, E]
        scores = logits.softmax(-1) if self.scoring == "softmax" else logits.sigmoid()
        choice = scores + self.bias if self.balance == "aux_free" else scores
        idx = choice.topk(self.k, dim=-1).indices                     # [T, k]
        w = scores.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True) * self.scale                  # normalise over the k picked
        load = torch.bincount(idx.flatten(), minlength=self.E)        # tokens per expert

        aux = x.new_zeros(())
        if self.balance == "aux_loss" and self.training:
            # Switch-style: alpha * E * sum_i f_i * P_i
            #   f_i = fraction of routed slots that went to expert i (no gradient)
            #   P_i = mean normalised router probability of expert i (has gradient)
            T = x.shape[0]
            f = load.float() / (T * self.k)
            P = (scores / scores.sum(-1, keepdim=True)).mean(0)
            aux = self.aux_coef * self.E * (f * P).sum()               # = aux_coef when perfectly uniform
        return idx, w, aux, load

    @torch.no_grad()
    def update_bias(self, load: torch.Tensor):
        """Aux-loss-free balancing: raise the bias of under-loaded experts, lower over-loaded ones."""
        if self.balance != "aux_free":
            return
        err = load.float().mean() - load.float()                      # > 0 means under-loaded
        self.bias += self.gamma * err.sign()


# ============================================================================
# 2. Experts + two dispatch implementations
# ============================================================================

class RoutedExperts(nn.Module):
    """E SwiGLU experts stored as stacked 3-D weights (like vLLM / HF 'fused' experts)."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        E, d, f = cfg.n_routed_experts, cfg.d_model, cfg.d_expert
        self.E = E
        self.w_gate = nn.Parameter(torch.randn(E, d, f) * d ** -0.5)
        self.w_up = nn.Parameter(torch.randn(E, d, f) * d ** -0.5)
        self.w_down = nn.Parameter(torch.randn(E, f, d) * f ** -0.5)

    def expert(self, e: int, x: torch.Tensor) -> torch.Tensor:
        return (F.silu(x @ self.w_gate[e]) * (x @ self.w_up[e])) @ self.w_down[e]

    # ---- 2a. Naive: loop over experts ------------------------------------
    def dispatch_loop(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """For each expert: find its tokens, run it, scatter-add the weighted result."""
        out = torch.zeros_like(x)
        for e in range(self.E):
            tok, slot = (idx == e).nonzero(as_tuple=True)              # which tokens chose e
            if tok.numel() == 0:
                continue                                               # an idle expert costs nothing
            out.index_add_(0, tok, self.expert(e, x[tok]) * w[tok, slot, None])
        return out

    # ---- 2b. Grouped: sort by expert -> one batched GEMM -> unsort -------
    def dispatch_sorted(self, x: torch.Tensor, idx: torch.Tensor, w: torch.Tensor,
                        capacity: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        1. flatten the T·k (token, expert) assignments and sort them by expert id
        2. copy tokens into a padded [E, C, d] buffer (C = capacity or max load)
        3. ONE batched matmul per weight matrix: [E, C, d] @ [E, d, f]
        4. scatter results back to token order, weighted by the gate
        If `capacity` is set, assignments beyond slot C of an expert are DROPPED
        (that token just gets less/no expert output, the residual carries it).
        Returns (output, n_dropped).
        """
        T, k = idx.shape
        flat_e = idx.flatten()                                         # [T·k]
        flat_tok = torch.arange(T, device=x.device).repeat_interleave(k)
        flat_w = w.flatten()
        order = torch.argsort(flat_e, stable=True)                     # group by expert
        e_sorted, tok_sorted, w_sorted = flat_e[order], flat_tok[order], flat_w[order]

        counts = torch.bincount(e_sorted, minlength=self.E)            # tokens per expert
        starts = torch.cumsum(counts, 0) - counts                      # segment offsets
        pos = torch.arange(T * k, device=x.device) - starts[e_sorted]  # slot within its expert

        C = int(counts.max()) if capacity is None else capacity
        keep = pos < C
        n_dropped = int((~keep).sum())
        e_k, pos_k, tok_k, w_k = e_sorted[keep], pos[keep], tok_sorted[keep], w_sorted[keep]

        buf = x.new_zeros(self.E, C, x.shape[-1])                      # padded "grouped" layout
        buf[e_k, pos_k] = x[tok_k]
        h = F.silu(torch.bmm(buf, self.w_gate)) * torch.bmm(buf, self.w_up)   # [E, C, f]
        y = torch.bmm(h, self.w_down)                                  # [E, C, d]

        out = torch.zeros_like(x)
        out.index_add_(0, tok_k, y[e_k, pos_k] * w_k[:, None])
        return out, n_dropped


class MoEFFN(nn.Module):
    """Shared experts (always on) + top-k routed experts."""

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.router = TopKRouter(cfg)
        self.experts = RoutedExperts(cfg)
        self.shared = SwiGLU(cfg.d_model, cfg.d_expert * cfg.n_shared_experts) if cfg.n_shared_experts else None
        self.mode = "sorted"          # or "loop"
        self.last_load = None
        self.last_dropped = 0

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, d = x.shape
        xt = x.reshape(-1, d)
        idx, w, aux, load = self.router(xt)
        self.last_load = load
        if self.mode == "loop":
            y = self.experts.dispatch_loop(xt, idx, w)
        else:
            cap = None
            if self.cfg.capacity_factor is not None:
                cap = math.ceil(self.cfg.capacity_factor * xt.shape[0] * self.cfg.top_k / self.cfg.n_routed_experts)
            y, self.last_dropped = self.experts.dispatch_sorted(xt, idx, w, cap)
        if self.shared is not None:
            y = y + self.shared(xt)
        return y.view(B, T, d), aux


# ============================================================================
# 3. Multi-head Latent Attention (MLA) with a latent KV cache
# ============================================================================

class MLAttention(nn.Module):
    """
    Per token we cache only:
        c_KV = RMSNorm(W_DKV x)          [d_c]    (content latent, shared by all heads)
        k_R  = RoPE(W_KR x)              [d_R]    (positional key, shared by all heads)
    Keys/values for head h are reconstructed on the fly:
        k_h = [W_UK_h c_KV ; k_R],   v_h = W_UV_h c_KV
    'absorbed' mode never builds k_h / v_h: it folds W_UK into the query and W_UV
    into the output, so attention runs against the d_c-dim latent directly.
    KV-cache elements per token per layer: d_c + d_R (vs 2·n_heads·d_h for MHA).
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        d, nh = cfg.d_model, cfg.n_heads
        self.nh, self.dn, self.dr, self.dv, self.dc = nh, cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
        # query path (low-rank, like DeepSeek-V3)
        self.w_dq = nn.Linear(d, cfg.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(cfg.q_lora_rank)
        self.w_uq = nn.Linear(cfg.q_lora_rank, nh * (self.dn + self.dr), bias=False)
        # kv path: one down-projection produces [c_KV | k_R]
        self.w_dkv = nn.Linear(d, self.dc + self.dr, bias=False)
        self.kv_norm = RMSNorm(self.dc)
        self.w_uk = nn.Linear(self.dc, nh * self.dn, bias=False)
        self.w_uv = nn.Linear(self.dc, nh * self.dv, bias=False)
        self.w_o = nn.Linear(nh * self.dv, d, bias=False)
        self.scale = (self.dn + self.dr) ** -0.5
        self.absorbed = False

    def forward(self, x: torch.Tensor, cache: Optional[Dict[str, torch.Tensor]] = None, start_pos: int = 0):
        B, T, _ = x.shape
        pos = torch.arange(start_pos, start_pos + T)
        cos, sin = rope_cos_sin(pos, self.dr, self.cfg.rope_theta)

        q = self.w_uq(self.q_norm(self.w_dq(x))).view(B, T, self.nh, self.dn + self.dr)
        q_nope, q_rope = q.split([self.dn, self.dr], dim=-1)
        q_rope = apply_rope(q_rope.transpose(1, 2), cos, sin)               # [B, nh, T, dr]

        ckv, k_rope = self.w_dkv(x).split([self.dc, self.dr], dim=-1)
        ckv = self.kv_norm(ckv)                                             # [B, T, dc]
        k_rope = apply_rope(k_rope, cos, sin)                               # [B, T, dr]

        if cache is not None:                                               # append to latent cache
            if "ckv" in cache:
                ckv = torch.cat([cache["ckv"], ckv], dim=1)
                k_rope = torch.cat([cache["k_rope"], k_rope], dim=1)
            cache["ckv"], cache["k_rope"] = ckv, k_rope
        S = ckv.shape[1]                                                    # total keys

        # causal mask: query at absolute position start_pos+t sees keys 0..start_pos+t
        q_pos = torch.arange(start_pos, start_pos + T)[:, None]
        k_pos = torch.arange(S)[None, :]
        mask = k_pos > q_pos                                                # [T, S] True = blocked

        if not self.absorbed:
            # ---- naive: rebuild per-head K and V from the latent ----
            k_nope = self.w_uk(ckv).view(B, S, self.nh, self.dn).transpose(1, 2)    # [B, nh, S, dn]
            v = self.w_uv(ckv).view(B, S, self.nh, self.dv).transpose(1, 2)         # [B, nh, S, dv]
            scores = (q_nope.transpose(1, 2) @ k_nope.transpose(-1, -2)
                      + q_rope @ k_rope[:, None].transpose(-1, -2)) * self.scale    # [B, nh, T, S]
            attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
            o = attn @ v                                                             # [B, nh, T, dv]
        else:
            # ---- absorbed: q_lat = q_nope · W_UK_h  (so q_nope·k_nope = q_lat·c_KV) ----
            W_uk = self.w_uk.weight.view(self.nh, self.dn, self.dc)
            W_uv = self.w_uv.weight.view(self.nh, self.dv, self.dc)
            q_lat = torch.einsum("bthn,hnc->bhtc", q_nope, W_uk)                     # [B, nh, T, dc]
            scores = (q_lat @ ckv[:, None].transpose(-1, -2)
                      + q_rope @ k_rope[:, None].transpose(-1, -2)) * self.scale     # [B, nh, T, S]
            attn = scores.masked_fill(mask, float("-inf")).softmax(-1)
            o_lat = attn @ ckv[:, None]                                              # [B, nh, T, dc]
            o = torch.einsum("bhtc,hvc->bhtv", o_lat, W_uv)                          # [B, nh, T, dv]
        o = o.transpose(1, 2).reshape(B, T, self.nh * self.dv)
        return self.w_o(o)


# ============================================================================
# 4. Decoder block and a tiny MoE decoder
# ============================================================================

class MoEBlock(nn.Module):
    def __init__(self, cfg: MoEConfig, dense: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model)
        self.attn = MLAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model)
        self.dense = dense
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff_dense) if dense else MoEFFN(cfg)

    def forward(self, x, cache=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), cache, start_pos)
        if self.dense:
            return x + self.ffn(self.ffn_norm(x)), x.new_zeros(())
        y, aux = self.ffn(self.ffn_norm(x))
        return x + y, aux


class TinyMoEDecoder(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([MoEBlock(cfg, dense=i < cfg.n_dense_layers) for i in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, ids: torch.Tensor, caches: Optional[List[dict]] = None, start_pos: int = 0):
        x = self.embed(ids)
        aux_total = x.new_zeros(())
        for i, blk in enumerate(self.blocks):
            x, aux = blk(x, None if caches is None else caches[i], start_pos)
            aux_total = aux_total + aux
        return self.lm_head(self.norm(x)), aux_total

    def moe_layers(self) -> List[MoEFFN]:
        return [b.ffn for b in self.blocks if not b.dense]

    def set_mode(self, dispatch: Optional[str] = None, absorbed: Optional[bool] = None):
        for b in self.blocks:
            if dispatch is not None and not b.dense:
                b.ffn.mode = dispatch
            if absorbed is not None:
                b.attn.absorbed = absorbed


# ============================================================================
# 5. Analytic parameter / memory model (same formulas as the guide's calculator)
# ============================================================================

# Real configs, copied from each model's official config.json on Hugging Face.
MODEL_SPECS = {
    "Mixtral-8x7B": dict(d=4096, L=32, n_dense=0, d_ff_dense=0, vocab=32000, tied=False,
                         attn=dict(type="gqa", n_heads=32, n_kv=8, head_dim=128),
                         moe=dict(E=8, k=2, shared=0, d_expert=14336)),
    "DeepSeek-V3": dict(d=7168, L=61, n_dense=3, d_ff_dense=18432, vocab=129280, tied=False,
                        attn=dict(type="mla", n_heads=128, q_lora=1536, kv_lora=512, d_nope=128, d_rope=64, d_v=128),
                        moe=dict(E=256, k=8, shared=1, d_expert=2048, router_bias=True)),
    "Kimi-K2": dict(d=7168, L=61, n_dense=1, d_ff_dense=18432, vocab=163840, tied=False,
                    attn=dict(type="mla", n_heads=64, q_lora=1536, kv_lora=512, d_nope=128, d_rope=64, d_v=128),
                    moe=dict(E=384, k=8, shared=1, d_expert=2048, router_bias=True)),
    "Qwen3-235B-A22B": dict(d=4096, L=94, n_dense=0, d_ff_dense=0, vocab=151936, tied=False,
                            attn=dict(type="gqa", n_heads=64, n_kv=4, head_dim=128, qk_norm=True),
                            moe=dict(E=128, k=8, shared=0, d_expert=1536)),
    "Qwen3-30B-A3B": dict(d=2048, L=48, n_dense=0, d_ff_dense=0, vocab=151936, tied=False,
                          attn=dict(type="gqa", n_heads=32, n_kv=4, head_dim=128, qk_norm=True),
                          moe=dict(E=128, k=8, shared=0, d_expert=768)),
    "gpt-oss-120b": dict(d=2880, L=36, n_dense=0, d_ff_dense=0, vocab=201088, tied=False,
                         attn=dict(type="gqa", n_heads=64, n_kv=8, head_dim=64, qkv_bias=True, o_bias=True,
                                   sinks=True, swa_layers=18, window=128),
                         moe=dict(E=128, k=4, shared=0, d_expert=2880, expert_bias=True, router_bias=True)),
    "gpt-oss-20b": dict(d=2880, L=24, n_dense=0, d_ff_dense=0, vocab=201088, tied=False,
                        attn=dict(type="gqa", n_heads=64, n_kv=8, head_dim=64, qkv_bias=True, o_bias=True,
                                  sinks=True, swa_layers=12, window=128),
                        moe=dict(E=32, k=4, shared=0, d_expert=2880, expert_bias=True, router_bias=True)),
}


def attn_params(d: int, a: dict) -> int:
    if a["type"] == "mla":
        nh, qr, kr, dn, dr, dv = a["n_heads"], a["q_lora"], a["kv_lora"], a["d_nope"], a["d_rope"], a["d_v"]
        q = (d * qr + qr + qr * nh * (dn + dr)) if qr else d * nh * (dn + dr)
        return q + d * (kr + dr) + kr + kr * nh * (dn + dv) + nh * dv * d
    nh, nkv, hd = a["n_heads"], a["n_kv"], a["head_dim"]          # MHA (nkv=nh) / GQA / MQA (nkv=1)
    p = d * nh * hd + 2 * d * nkv * hd + nh * hd * d
    p += (nh * hd + 2 * nkv * hd) if a.get("qkv_bias") else 0
    p += d if a.get("o_bias") else 0
    p += 2 * hd if a.get("qk_norm") else 0
    p += nh if a.get("sinks") else 0
    return p


def kv_elems_per_token(spec: dict, ctx: Optional[int] = None) -> float:
    """KV-cache elements per token summed over layers (sliding-window layers amortised at `ctx`)."""
    a, L = spec["attn"], spec["L"]
    if a["type"] == "mla":
        return L * (a["kv_lora"] + a["d_rope"])
    per_layer = 2 * a["n_kv"] * a["head_dim"]
    swa = a.get("swa_layers", 0)
    if swa and ctx:
        return (L - swa) * per_layer + swa * per_layer * min(ctx, a["window"]) / ctx
    return L * per_layer


def analyze(spec: dict) -> dict:
    d, L, m = spec["d"], spec["L"], spec["moe"]
    E, k, S, f = m["E"], m["k"], m["shared"], m["d_expert"]
    expert = 3 * d * f + ((2 * f + d) if m.get("expert_bias") else 0)
    router = E * d + (E if m.get("router_bias") else 0)
    n_moe = L - spec["n_dense"]
    emb = spec["vocab"] * d
    head = 0 if spec["tied"] else spec["vocab"] * d
    common = emb + head + d + L * (attn_params(d, spec["attn"]) + 2 * d) + spec["n_dense"] * 3 * d * spec["d_ff_dense"]
    total = common + n_moe * (E * expert + S * 3 * d * f + router)
    active = common + n_moe * (k * expert + S * 3 * d * f + router)
    return dict(total=total, active=active, active_noemb=active - emb, expert=expert, n_moe=n_moe, E=E, k=k,
                kv_per_tok=kv_elems_per_token(spec))


def distinct_experts(E: int, k: int, B: int) -> float:
    """Expected number of distinct experts hit in one layer by B tokens, each picking k of E uniformly."""
    return E * (1.0 - (1.0 - k / E) ** B)


def weight_bytes(spec: dict, w_bytes: float, expert_bytes: Optional[float] = None) -> float:
    """Resident weight memory. Routed experts may use their own precision (gpt-oss: MXFP4 = 4.25 bits)."""
    r = analyze(spec)
    eb = w_bytes if expert_bytes is None else expert_bytes
    routed = r["n_moe"] * r["E"] * r["expert"]
    return (r["total"] - routed) * w_bytes + routed * eb


def decode_bytes_per_step(spec: dict, B: int, ctx: int, w_bytes: float, kv_bytes: float,
                          expert_bytes: Optional[float] = None) -> dict:
    """Bytes read from memory for ONE decode step of a batch of B sequences, each with `ctx` cached tokens."""
    r = analyze(spec)
    eb = w_bytes if expert_bytes is None else expert_bytes
    always = r["active_noemb"] - r["n_moe"] * r["k"] * r["expert"]       # attention, shared, dense, head, router
    dx = distinct_experts(r["E"], r["k"], B)
    weights = always * w_bytes + r["n_moe"] * dx * r["expert"] * eb
    kv = B * ctx * kv_elems_per_token(spec, ctx) * kv_bytes
    return dict(distinct=dx, weights=weights, kv=kv, total=weights + kv, per_token=(weights + kv) / B)


def attn_flops_per_token(spec: dict, ctx: int) -> float:
    """FLOPs one new token spends attending over `ctx` cached tokens (scores + weighted sum), all layers.
    MLA is counted in absorbed form: scores over d_c + d_R, values over d_c, for every query head."""
    a, L = spec["attn"], spec["L"]
    if a["type"] == "mla":
        return L * 2 * a["n_heads"] * (2 * a["kv_lora"] + a["d_rope"]) * ctx
    swa = a.get("swa_layers", 0)
    span = (L - swa) * ctx + swa * min(ctx, a.get("window", ctx))
    return 4 * a["n_heads"] * a["head_dim"] * span


def attn_intensity(spec: dict, kv_bytes: float = 2.0) -> float:
    """Decode attention FLOPs per byte of KV cache read (full-attention layers)."""
    a = spec["attn"]
    if a["type"] == "mla":
        return 2 * a["n_heads"] * (2 * a["kv_lora"] + a["d_rope"]) / ((a["kv_lora"] + a["d_rope"]) * kv_bytes)
    return 4 * a["n_heads"] * a["head_dim"] / (2 * a["n_kv"] * a["head_dim"] * kv_bytes)


def spec_from_config(cfg: MoEConfig) -> dict:
    """Describe the tiny model in the same spec language (used to check the formulas)."""
    return dict(d=cfg.d_model, L=cfg.n_layers, n_dense=cfg.n_dense_layers, d_ff_dense=cfg.d_ff_dense,
                vocab=cfg.vocab_size, tied=False,
                attn=dict(type="mla", n_heads=cfg.n_heads, q_lora=cfg.q_lora_rank, kv_lora=cfg.kv_lora_rank,
                          d_nope=cfg.qk_nope_head_dim, d_rope=cfg.qk_rope_head_dim, d_v=cfg.v_head_dim),
                moe=dict(E=cfg.n_routed_experts, k=cfg.top_k, shared=cfg.n_shared_experts,
                         d_expert=cfg.d_expert, router_bias=True))


def count_parameters(model: nn.Module, include_buffers: bool = True) -> int:
    n = sum(p.numel() for p in model.parameters())
    if include_buffers:   # the router bias is a buffer but lives in the checkpoint
        n += sum(b.numel() for name, b in model.named_buffers() if name.endswith("router.bias"))
    return n


# ============================================================================
# 6. Tests / demos
# ============================================================================

def test_dispatch_equivalence():
    print("\n[1] Dispatch: naive per-expert loop vs sort/scatter grouped GEMM")
    torch.manual_seed(0)
    cfg = MoEConfig()
    moe = MoEFFN(cfg).eval()
    x = torch.randn(3, 20, cfg.d_model)
    with torch.no_grad():
        moe.mode = "loop";   y_loop, _ = moe(x)
        moe.mode = "sorted"; y_sort, _ = moe(x)
    diff = (y_loop - y_sort).abs().max().item()
    assert torch.allclose(y_loop, y_sort, atol=1e-5), diff
    print(f"    60 tokens, E={cfg.n_routed_experts}, k={cfg.top_k}: max |loop - sorted| = {diff:.2e}  OK")

    # capacity factor: dropped tokens, and why inference engines run dropless
    with torch.no_grad():
        moe.cfg = replace(cfg, capacity_factor=1.0)
        y_cap, _ = moe(x)
        dropped = moe.last_dropped
        load = moe.last_load
        cap = math.ceil(1.0 * 60 * cfg.top_k / cfg.n_routed_experts)
        print(f"    capacity_factor=1.0 -> C={cap} slots/expert, load per expert = {load.tolist()}")
        print(f"    dropped {dropped} of {60 * cfg.top_k} (token, expert) assignments")
        # Run every token ALONE and compare with its output inside the batch.
        xt = x.reshape(-1, 1, 1, cfg.d_model)
        alone_cap = torch.cat([moe(t)[0] for t in xt]).view_as(x)          # still capacity 1.0
        moe.cfg = cfg
        alone_free = torch.cat([moe(t)[0] for t in xt]).view_as(x)         # dropless
    changed = lambda a, b: int(((a - b).abs().amax(-1) > 1e-5).sum())
    print(f"    tokens whose output depends on the REST of the batch: "
          f"dropless {changed(alone_free, y_sort)}/60, capacity 1.0 {changed(alone_cap, y_cap)}/60")
    assert changed(alone_free, y_sort) == 0 and changed(alone_cap, y_cap) > 0


def test_mla_cache():
    print("\n[2] MLA: full forward == prefill + token-by-token decode with the latent KV cache")
    torch.manual_seed(0)
    cfg = MoEConfig()
    model = TinyMoEDecoder(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 24))
    with torch.no_grad():
        full, _ = model(ids)
        for absorbed in (False, True):
            model.set_mode(absorbed=absorbed)
            caches = [dict() for _ in range(cfg.n_layers)]
            pieces = [model(ids[:, :8], caches, 0)[0]]                     # prefill 8 tokens
            for t in range(8, ids.shape[1]):                               # decode one at a time
                pieces.append(model(ids[:, t:t + 1], caches, t)[0])
            inc = torch.cat(pieces, dim=1)
            diff = (inc - full).abs().max().item()
            assert torch.allclose(inc, full, atol=1e-4), diff
            print(f"    absorbed={absorbed!s:<5}: max |decode - full| = {diff:.2e}  OK")
        model.set_mode(absorbed=False)
    c = caches[0]
    per_tok = c["ckv"].shape[-1] + c["k_rope"].shape[-1]
    mha = 2 * cfg.n_heads * cfg.qk_nope_head_dim
    print(f"    cache per token per layer: c_KV {c['ckv'].shape[-1]} + k_R {c['k_rope'].shape[-1]} = {per_tok} values "
          f"(MHA with {cfg.n_heads} heads x {cfg.qk_nope_head_dim} would store {mha})")


def test_param_formula():
    print("\n[3] Analytic parameter formula == actual module parameter count")
    cfg = MoEConfig()
    model = TinyMoEDecoder(cfg)
    actual = count_parameters(model)
    r = analyze(spec_from_config(cfg))
    assert actual == r["total"], (actual, r["total"])
    print(f"    tiny model: {actual:,} params (formula {r['total']:,})  active per token {r['active']:,} "
          f"({100 * r['active'] / r['total']:.1f}%)  OK")


def demo_load_balancing(steps: int = 300):
    print("\n[4] Load balancing on a skewed token distribution (router only, E=16, k=2)")
    torch.manual_seed(0)
    base = MoEConfig(n_routed_experts=16, top_k=2, d_model=32, bias_update_speed=0.01)
    # skewed inputs: most tokens share one direction -> a few experts win everything
    def batch():
        return torch.randn(512, base.d_model) * 0.5 + torch.linspace(1, 0, base.d_model)

    def imbalance(load):
        return (load.max() / load.float().mean()).item()

    init = TopKRouter(base).state_dict()
    for mode in ("none", "aux_free", "aux_loss"):
        r = TopKRouter(replace(base, balance=mode, aux_loss_coef=0.1))
        r.load_state_dict(init)
        opt = torch.optim.SGD(r.parameters(), lr=1.0)
        before = imbalance(r(batch())[3])
        for _ in range(steps):
            idx, w, aux, load = r.train()(batch())
            if mode == "aux_loss":
                opt.zero_grad(); aux.backward(); opt.step()
            r.update_bias(load)
        after = imbalance(r.eval()(batch())[3])
        print(f"    balance={mode:<9} max/mean expert load: {before:5.2f} -> {after:5.2f}")


def demo_training(steps: int = 60):
    print("\n[5] Tiny MoE decoder trains (next-token on a repeating pattern) with aux-free bias updates")
    torch.manual_seed(0)
    cfg = MoEConfig(vocab_size=64, bias_update_speed=0.01)
    model = TinyMoEDecoder(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    pattern = torch.randint(0, cfg.vocab_size, (16,))
    t0 = time.time()
    for step in range(steps + 1):
        offs = torch.randint(0, 16, (8,))
        ids = torch.stack([pattern.roll(-int(o)).repeat(3)[:33] for o in offs])
        logits, aux = model(ids[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), ids[:, 1:].reshape(-1))
        opt.zero_grad(); (loss + aux).backward(); opt.step()
        for moe in model.moe_layers():                     # the aux-free rule runs after each step
            moe.router.update_bias(moe.last_load)
        if step % 20 == 0:
            loads = torch.stack([m.last_load for m in model.moe_layers()]).float()
            print(f"    step {step:3d}  loss {loss.item():.3f}  max/mean load {(loads.max(1).values / loads.mean(1)).mean():.2f}")
    print(f"    ({time.time() - t0:.1f} s on CPU)")


def print_model_comparison():
    print("\n" + "=" * 96)
    print("Real MoE configs (from config.json) — analytic counts")
    print("=" * 96)
    print(f"{'Model':<17}{'Total':>9}{'Active':>9}{'Act%':>6}{'E':>5}{'k':>3}{'S':>3}"
          f"{'KV/tok BF16':>13}{'KV @32K':>10}{'Read/tok FP8':>14}{'GFLOP/tok':>11}")
    print("-" * 96)
    for name, spec in MODEL_SPECS.items():
        r = analyze(spec)
        kv_tok = kv_elems_per_token(spec, 32768) * 2
        print(f"{name:<17}{r['total']/1e9:8.1f}B{r['active']/1e9:8.2f}B{100*r['active']/r['total']:5.1f}%"
              f"{r['E']:>5}{r['k']:>3}{spec['moe']['shared']:>3}"
              f"{kv_tok/1024:10.1f} KiB{kv_tok*32768/2**30:7.2f} GiB"
              f"{r['active_noemb']/1e9:11.2f} GB{2*r['active_noemb']/1e9:11.1f}")
    print("  Active includes the input embedding table; 'Read/tok' and FLOPs exclude it (a row lookup).")
    print("  gpt-oss KV is amortised at 32K: half its layers use a 128-token sliding window.")

    print("\nDecode attention at 32K context: extra FLOPs per token, and FLOPs per byte of BF16 KV read")
    for name in ("Mixtral-8x7B", "DeepSeek-V3", "Qwen3-235B-A22B", "gpt-oss-120b"):
        spec = MODEL_SPECS[name]
        print(f"  {name:<17}{attn_flops_per_token(spec, 32768)/1e9:7.1f} GFLOP/token   "
              f"intensity {attn_intensity(spec):6.1f} FLOP/byte")
    print("  (H100 SXM ridge point ~ 990 dense BF16 TFLOPS / 3.35 TB/s ~ 295 FLOP/byte)")

    print("\nKV cache per token at the DeepSeek-V3 shape (61 layers, 128 heads, d_h=128), BF16:")
    L, nh, dh = 61, 128, 128
    rows = [("MHA  (128 KV heads)", L * 2 * nh * dh), ("GQA  (8 KV heads)", L * 2 * 8 * dh),
            ("MQA  (1 KV head)", L * 2 * dh), ("MLA  (d_c=512 + d_R=64)", L * (512 + 64))]
    for n, e in rows:
        print(f"  {n:<26}{e * 2 / 1024:9.1f} KiB/token  {e * 2 * 131072 / 2**30:8.2f} GiB @128K  "
              f"({rows[0][1] / e:5.1f}x smaller than MHA)")

    print("\nDecode step, DeepSeek-V3, FP8 weights, BF16 KV, 4K context per sequence:")
    print(f"  {'batch':>6}{'experts hit/layer':>19}{'weights read':>15}{'KV read':>10}{'bytes/token':>13}")
    for B in (1, 8, 32, 128, 512):
        s = decode_bytes_per_step(MODEL_SPECS["DeepSeek-V3"], B, 4096, 1.0, 2.0)
        print(f"  {B:>6}{s['distinct']:>19.1f}{s['weights']/1e9:>12.1f} GB{s['kv']/1e9:>7.1f} GB{s['per_token']/1e9:>10.2f} GB")
    print("  (uniform routing assumed; real routing is skewed, so fewer distinct experts are hit)")

    g = MODEL_SPECS["gpt-oss-120b"]
    gr = analyze(g)
    routed = gr["n_moe"] * gr["E"] * gr["expert"]
    print(f"\ngpt-oss-120b: routed experts are {100 * routed / gr['total']:.1f}% of params; MXFP4 experts (4.25 bit) "
          f"+ BF16 rest = {weight_bytes(g, 2.0, 4.25 / 8) / 2**30:.1f} GiB resident "
          f"(model card: 60.8 GiB checkpoint); "
          f"{decode_bytes_per_step(g, 1, 0, 2.0, 2.0, 4.25 / 8)['weights'] / 1e9:.2f} GB read per token at B=1")

    print("\nEdge roofline (weights only, B=1): tokens/s <= bandwidth / bytes read per token")
    q = analyze(MODEL_SPECS["Qwen3-30B-A3B"])
    for hw, bw in (("Jetson AGX Orin 64GB", 204.8e9), ("Jetson AGX Thor T5000", 273e9)):
        moe_b = q["active_noemb"] * 0.5
        dense_b = (q["total"] - q["active"] + q["active_noemb"]) * 0.5
        print(f"  {hw:<23} Qwen3-30B-A3B @4-bit: {moe_b/1e9:.2f} GB/token -> <= {bw/moe_b:5.0f} tok/s;"
              f"  a dense 30B @4-bit: <= {bw/dense_b:4.1f} tok/s")
    g128 = q["active_noemb"] * 4.25 / 8      # INT4 with one FP16 scale per 128 weights
    print(f"  (4-bit = 0.5 B/param, scales ignored. With g=128 scales, 4.25 bit: "
          f"{q['total'] * 4.25 / 8 / 2**30:.1f} GiB resident, {g128/1e9:.2f} GB/token, "
          f"<= {204.8e9/g128:.0f} tok/s on AGX Orin, <= {273e9/g128:.0f} on Thor)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Mixture-of-Experts LLM Reference Implementation")
    print("=" * 60)
    cfg = MoEConfig()
    model = TinyMoEDecoder(cfg)
    r = analyze(spec_from_config(cfg))
    print(f"Tiny config: d={cfg.d_model}, layers={cfg.n_layers} ({cfg.n_dense_layers} dense), "
          f"E={cfg.n_routed_experts} routed + {cfg.n_shared_experts} shared, top-{cfg.top_k}, "
          f"MLA d_c={cfg.kv_lora_rank}, d_R={cfg.qk_rope_head_dim}")
    print(f"Params: total {r['total']:,}  active/token {r['active']:,}")

    test_dispatch_equivalence()
    test_mla_cache()
    test_param_formula()
    demo_load_balancing()
    demo_training()
    print_model_comparison()
    print("\nAll checks passed.")
