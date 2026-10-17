#!/usr/bin/env python3
"""
efficient_inference.py: the memory & inference efficiency lab.

Companion to efficiency_guide.html ("Memory & inference efficiency cookbook").
CPU only, torch + the standard library. Every section prints a small table and
asserts what it claims, so a clean run means the lab-derived numbers in the guide hold.

    python efficient_inference.py                  # every section (well under 2 min on a laptop CPU)
    python efficient_inference.py quant spec       # chosen sections only
    python efficient_inference.py --list           # list the sections

Sections (the letter is the technique card in the guide):
    quant    (c) FP32/BF16/FP8/INT8/INT4/FP4 weights by hand, outlier channel, SmoothQuant
    kvcache  (a) KV cache vs recompute: identical tokens, and the timing
    kvsize   (b) KV-cache bytes for MHA / GQA / MQA and an MLA-style latent
    tiled    (d) FlashAttention-style tiled attention with online softmax
    paged    (e) paged KV block allocator vs contiguous max-length allocation
    prefix   (f) prefix caching: compute a shared prompt's KV once, reuse it
    spec     (g) speculative decoding: tiny draft + target, identical to greedy
    fusion   (h) kernel fusion: bytes moved and time, eager vs fused
    sparse   (i) 2:4 structured sparsity, then distillation to recover
    ckpt     (j) activation checkpointing: saved bytes and identical gradients
    train    (j) Adam training-memory calculator, ZeRO stages, LoRA parameter count
    bench    (k) timing done right: warm-up, median, p90 (and the Jetson checklist)
"""
import argparse
import math
import statistics
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

KiB, MiB, GiB = 1024, 1024 ** 2, 1024 ** 3
SUMMARY = []          # (section, one-line result) collected for the final summary


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def header(title):
    print('\n' + '=' * 84 + '\n' + title + '\n' + '=' * 84)


def check(cond, msg):
    """Assert a claim and print it, so the log doubles as a list of verified statements."""
    assert cond, 'FAILED: ' + msg
    print('  [ok] ' + msg)


def fmt_bytes(n):
    for unit, size in (('GiB', GiB), ('MiB', MiB), ('KiB', KiB)):
        if abs(n) >= size:
            return f'{n / size:,.2f} {unit}'
    return f'{n:,.0f} B'


def rel_err(a, b):
    """||a - b|| / ||b||: the relative error of a against the reference b."""
    return ((a - b).norm() / b.norm()).item()


def bench(fn, warmup=2, iters=5):
    """Time fn() the way you should on a Jetson too: one cold call, a warm-up, then the median.

    Returns a dict with the cold first call, the median and the 90th percentile in seconds.
    (On a GPU you would also call torch.cuda.synchronize() before reading each clock.)"""
    t0 = time.perf_counter(); fn(); first = time.perf_counter() - t0
    for _ in range(max(0, warmup - 1)):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return {'first': first, 'median': statistics.median(ts), 'p90': ts[min(len(ts) - 1, int(0.9 * len(ts)))]}


# ---------------------------------------------------------------------------------------------
# a tiny GPT (pre-norm decoder) with an optional KV cache and grouped-query attention
# ---------------------------------------------------------------------------------------------
@dataclass
class GPTConfig:
    vocab: int = 64
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    n_kv_heads: int = 4          # < n_heads gives GQA, 1 gives MQA
    max_len: int = 512
    ffn_mult: int = 4


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_heads % cfg.n_kv_heads == 0
        self.h, self.kv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.d_model // cfg.n_heads
        self.q = nn.Linear(cfg.d_model, self.h * self.hd, bias=False)
        self.k = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.v = nn.Linear(cfg.d_model, self.kv * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, cfg.d_model, bias=False)

    def forward(self, x, past=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.hd).transpose(1, 2)           # [B, h, T, hd]
        k = self.k(x).view(B, T, self.kv, self.hd).transpose(1, 2)          # [B, kv, T, hd]
        v = self.v(x).view(B, T, self.kv, self.hd).transpose(1, 2)
        if past is not None:                                                # KV cache: append, never recompute
            k = torch.cat([past[0], k], dim=2)
            v = torch.cat([past[1], v], dim=2)
        present = (k, v)
        g = self.h // self.kv                                               # query heads per KV head
        k, v = k.repeat_interleave(g, dim=1), v.repeat_interleave(g, dim=1)
        S = k.shape[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)                # [B, h, T, S]
        mask = torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)      # query i sees keys <= its position
        att = att.masked_fill(~mask, float('-inf')).softmax(dim=-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, self.h * self.hd)
        return self.o(y), present


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.fc1 = nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model)
        self.fc2 = nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model)

    def forward(self, x, past=None):
        a, present = self.attn(self.ln1(x), past)
        x = x + a
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x, present


class TinyGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)

    def forward(self, idx, past=None):
        """idx: [B, T] token ids. past: list of (k, v) per layer or None. Returns logits, presents."""
        T = idx.shape[1]
        start = 0 if past is None else past[0][0].shape[2]                  # position of the first new token
        x = self.tok(idx) + self.pos(torch.arange(start, start + T))
        presents = []
        for i, blk in enumerate(self.blocks):
            x, pr = blk(x, None if past is None else past[i])
            presents.append(pr)
        return self.head(self.ln_f(x)), presents


@torch.no_grad()
def generate_recompute(model, prompt, n_new):
    """Greedy decoding WITHOUT a cache: every step re-runs the whole sequence."""
    idx = prompt
    for _ in range(n_new):
        logits, _ = model(idx)
        idx = torch.cat([idx, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
    return idx, logits[:, -1]


@torch.no_grad()
def generate_cached(model, prompt, n_new):
    """Greedy decoding WITH a KV cache: prefill once, then feed one token per step."""
    logits, past = model(prompt)
    idx = prompt
    for i in range(n_new):
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
        if i < n_new - 1:
            logits, past = model(nxt, past)
    return idx, logits[:, -1]                      # the logits that picked the last token


# =============================================================================================
# (c) quantization by hand
# =============================================================================================
def quant_sym(w, bits, dim=None):
    """Symmetric integer quantization. dim=None: one scale per tensor; dim=1: one scale per row
    (per output channel of a Linear weight [out, in]). Returns the dequantized tensor and #scales."""
    qmax = 2 ** (bits - 1) - 1                                   # 127 for INT8, 7 for INT4
    amax = w.abs().amax() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    scale = amax.clamp(min=1e-12) / qmax
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)        # the integers you would store
    return q * scale, scale.numel()


def quant_asym_groups(w, bits, group):
    """Asymmetric (scale + zero-point) quantization of every `group` consecutive weights in a row,
    the layout used by GPTQ/AWQ INT4 checkpoints. Returns dequantized weights and #groups."""
    out_f, in_f = w.shape
    wg = w.reshape(out_f, in_f // group, group)
    lo, hi = wg.amin(-1, keepdim=True), wg.amax(-1, keepdim=True)
    qmax = 2 ** bits - 1                                         # 15 for INT4: integers 0..15
    scale = (hi - lo).clamp(min=1e-12) / qmax
    zp = torch.round(-lo / scale)                                # the integer that represents 0.0
    q = torch.clamp(torch.round(wg / scale) + zp, 0, qmax)
    return ((q - zp) * scale).reshape(out_f, in_f), scale.numel()


E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])    # every non-negative FP4 (E2M1) value


def fp_format_max(e_bits, m_bits, bias, ieee_like):
    """Largest finite value of a small float format. E4M3 (FN variant) reserves only the all-ones
    pattern of exponent AND mantissa for NaN; E5M2 is IEEE-like (all-ones exponent = inf/NaN)."""
    if ieee_like:
        max_exp = (2 ** e_bits - 2) - bias
        return (2 - 2 ** -m_bits) * 2 ** max_exp
    max_exp = (2 ** e_bits - 1) - bias
    return (2 - 2 * 2 ** -m_bits) * 2 ** max_exp


def quant_fp4_blocks(w, block=16, fp8_scales=True):
    """NVFP4-style: every 16 consecutive weights share a scale (stored in FP8 E4M3), each weight is
    one of the 15 FP4 E2M1 values {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}."""
    out_f, in_f = w.shape
    wb = w.reshape(out_f, in_f // block, block)
    scale = wb.abs().amax(-1, keepdim=True).clamp(min=1e-12) / 6.0     # map the block max to 6
    if fp8_scales and hasattr(torch, 'float8_e4m3fn'):
        scale = scale.to(torch.float8_e4m3fn).to(torch.float32)           # the scale itself is 8-bit
    x = (wb / scale).abs().clamp(max=6.0)
    idx = (x.unsqueeze(-1) - E2M1_GRID).abs().argmin(-1)                   # nearest grid point
    q = E2M1_GRID[idx] * wb.sign()
    return (q * scale).reshape(out_f, in_f), scale.numel()


def quant_fp8_tensor(w):
    """FP8 E4M3 with one per-tensor scale (amax -> 448), via torch's float8 dtype when available."""
    if not hasattr(torch, 'float8_e4m3fn'):
        return None
    scale = w.abs().amax() / 448.0
    return (w / scale).to(torch.float8_e4m3fn).to(torch.float32) * scale


def section_quant():
    header('(c) Quantization by hand: scale, zero-point, granularity, outliers, formats')

    # --- 1. the worked example from the guide: 4 numbers -> 4-bit asymmetric -------------------
    x = torch.tensor([-1.2, 0.0, 0.5, 2.6])
    scale = (x.max() - x.min()) / 15                             # 3.8 / 15 = 0.2533
    zp = torch.round(-x.min() / scale)                           # round(4.737) = 5
    q = torch.clamp(torch.round(x / scale) + zp, 0, 15)
    xh = (q - zp) * scale
    print(f'  hand example x = {x.tolist()}  scale = {scale:.4f}  zero-point = {int(zp)}')
    print(f'  q = {q.int().tolist()}   dequantized = {[round(v, 3) for v in xh.tolist()]}')
    check(q.int().tolist() == [0, 5, 7, 15], 'hand example: q = [0, 5, 7, 15] with scale 0.2533 and zero-point 5')
    check(abs(xh[2].item() - 0.507) < 1e-3 and xh[1].item() == 0.0, '0.0 is exact (zero-point) and 0.5 comes back as 0.507')
    check((xh - x).abs().max().item() <= scale.item() / 2 + 1e-6, 'error never exceeds half a step (scale / 2)')

    # --- 2. the formats themselves -------------------------------------------------------------
    e4m3, e5m2 = fp_format_max(4, 3, 7, False), fp_format_max(5, 2, 15, True)
    print(f'  FP8 E4M3 max = {e4m3:g}, FP8 E5M2 max = {e5m2:g}, FP4 E2M1 values = {E2M1_GRID.tolist()}')
    check(e4m3 == 448 and e5m2 == 57344, 'FP8 E4M3 tops out at 448 (precision) and E5M2 at 57,344 (range)')
    if hasattr(torch, 'float8_e4m3fn'):
        check(torch.finfo(torch.float8_e4m3fn).max == 448 and torch.finfo(torch.float8_e5m2).max == 57344,
              'torch.finfo agrees with the bit-level formula')

    # --- 3. a toy GPT layer with one outlier output channel ------------------------------------
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=64, d_model=256, n_layers=1, n_heads=4, n_kv_heads=4, ffn_mult=4)
    block = Block(cfg).eval()
    linears = {n: m for n, m in block.named_modules() if isinstance(m, nn.Linear)}
    OUTLIER_ROW, OUTLIER_GAIN = 7, 20.0
    with torch.no_grad():
        for n, m in linears.items():
            # heavy-tailed weights (Student-t, 4 d.o.f.): like trained LLM weights, mostly small with rare big ones
            t = torch.distributions.StudentT(4.0).sample(m.weight.shape)
            m.weight.copy_(t / t.std() * m.weight.std())
            if n.split('.')[-1] in ('v', 'o', 'fc1', 'fc2'):     # (q, k left alone: x20 there would saturate softmax)
                m.weight[OUTLIER_ROW] *= OUTLIER_GAIN            # one output channel 20x larger than the rest
    x_in = torch.randn(4, 64, cfg.d_model)
    with torch.no_grad():
        y_ref, _ = block(x_in)
    w_ref = {n: m.weight.detach().clone() for n, m in linears.items()}
    n_w = sum(w.numel() for w in w_ref.values())

    # bytes per format: data bits + stored scales (FP16 scale, FP16 zero-point where used)
    schemes = {
        'FP32':                     (lambda w: (w.clone(), 0), 32, 0),
        'BF16':                     (lambda w: (w.to(torch.bfloat16).float(), 0), 16, 0),
        'FP8 E4M3, per-tensor':     (lambda w: (quant_fp8_tensor(w), 1), 8, 4),
        'INT8, per-tensor':         (lambda w: quant_sym(w, 8, None), 8, 2),
        'INT8, per-channel':        (lambda w: quant_sym(w, 8, 1), 8, 2),
        'INT4, per-channel':        (lambda w: quant_sym(w, 4, 1), 4, 2),
        'INT4, groups of 64 (+zp)': (lambda w: quant_asym_groups(w, 4, 64), 4, 4),
        'FP4 E2M1, blocks of 16':   (lambda w: quant_fp4_blocks(w, 16), 4, 1),
    }
    if not hasattr(torch, 'float8_e4m3fn'):
        schemes.pop('FP8 E4M3, per-tensor')
    results = {}
    print(f'\n  toy GPT layer: d=256, ffn=1024, 6 Linear weights = {n_w:,} params, heavy-tailed; '
          f'output channel {OUTLIER_ROW} of v, o, fc1, fc2 scaled x{OUTLIER_GAIN:g}')
    print(f'  {"weights":28s} {"bytes":>10s} {"bits/w":>7s} {"vs FP32":>8s} {"layer out err":>14s} {"err, normal rows":>17s}')
    for name, (fn, bits, scale_bytes) in schemes.items():
        n_scales, w_err_num, w_err_den = 0, 0.0, 0.0
        with torch.no_grad():
            for n, m in linears.items():
                wq, ns = fn(w_ref[n])
                n_scales += ns
                keep = torch.ones(wq.shape[0], dtype=torch.bool); keep[OUTLIER_ROW] = False
                w_err_num += (wq[keep] - w_ref[n][keep]).norm().item() ** 2
                w_err_den += w_ref[n][keep].norm().item() ** 2
                m.weight.copy_(wq)
            y_q, _ = block(x_in)
            for n, m in linears.items():
                m.weight.copy_(w_ref[n])                          # restore
        nbytes = n_w * bits / 8 + n_scales * scale_bytes
        results[name] = (nbytes, rel_err(y_q, y_ref), math.sqrt(w_err_num / w_err_den))
        print(f'  {name:28s} {nbytes:>10,.0f} {8 * nbytes / n_w:>7.2f} {nbytes / (4 * n_w):>8.3f} '
              f'{results[name][1]:>14.2e} {results[name][2]:>17.2e}')

    pt, pc = results['INT8, per-tensor'], results['INT8, per-channel']
    check(pt[2] > 10 * pc[2], f'outlier channel: per-tensor INT8 error on the normal rows is '
          f'{pt[2] / pc[2]:.0f}x the per-channel error')
    check(pt[1] > 3 * pc[1], 'and the layer output error is also clearly worse per-tensor')
    g4, c4 = results['INT4, groups of 64 (+zp)'], results['INT4, per-channel']
    f4 = results['FP4 E2M1, blocks of 16']
    check(g4[1] < c4[1] / 1.5, f'INT4: groups of 64 beat one scale per row (output error {g4[1]:.3f} vs {c4[1]:.3f})')
    check(f4[1] < g4[1], f'FP4 with small blocks of 16 does better still here ({f4[1]:.3f}) at the same 4.5 bits')
    check(results['BF16'][1] < pc[1] < g4[1], 'error grows as bits shrink: BF16 < INT8 per-channel < INT4 g64')
    check(abs(8 * g4[0] / n_w - 4.5) < 1e-9 and abs(8 * results['FP4 E2M1, blocks of 16'][0] / n_w - 4.5) < 1e-9,
          'INT4 g64 with FP16 scale+zero and FP4 with an FP8 scale per 16 both cost 4.5 bits per weight')
    SUMMARY.append(('quant', f'INT8 per-channel out err {pc[1]:.1e} vs per-tensor {pt[1]:.1e} '
                             f'(outlier row x{OUTLIER_GAIN:g}); INT4 g64 {g4[1]:.3f}, 4.5 bits/w'))

    # --- 4. W8A8: an activation outlier and the SmoothQuant fix --------------------------------
    torch.manual_seed(1)
    d_in, d_out = 256, 256
    W = torch.randn(d_out, d_in) / math.sqrt(d_in)
    X = torch.randn(512, d_in)
    X[:, 13] *= 40.0                                             # one input channel with huge activations
    Y = X @ W.T

    def w8a8(Xa, Wa):                                            # INT8 activations per-tensor, weights per-channel
        Xq, _ = quant_sym(Xa, 8, None)
        Wq, _ = quant_sym(Wa, 8, 1)
        return Xq @ Wq.T

    alpha = 0.5
    s = X.abs().amax(0).pow(alpha) / W.abs().amax(0).pow(1 - alpha)      # one factor per input channel j
    Xs, Ws = X / s, W * s                                                 # X diag(s)^-1 and diag(s) W
    check(torch.allclose(Xs @ Ws.T, Y, atol=1e-4), 'smoothing is exact in full precision: (X/s)(sW)^T = XW^T')
    e_plain, e_smooth = rel_err(w8a8(X, W), Y), rel_err(w8a8(Xs, Ws), Y)
    e_wonly = rel_err(X @ quant_sym(W, 8, 1)[0].T, Y)
    print(f'  W8A8 with a 40x activation outlier: error {e_plain:.4f} plain, {e_smooth:.4f} after SmoothQuant '
          f'(alpha=0.5); weight-only INT8: {e_wonly:.4f}')
    check(e_smooth < e_plain / 3, f'SmoothQuant cuts the W8A8 error {e_plain / e_smooth:.1f}x by moving the outlier into W')
    SUMMARY.append(('quant', f'W8A8 err {e_plain:.4f} -> {e_smooth:.4f} with SmoothQuant'))


# =============================================================================================
# (a) KV cache vs recompute
# =============================================================================================
def section_kvcache():
    header('(a) KV cache vs recompute: same tokens, far fewer token-forwards')
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=256, d_model=256, n_layers=4, n_heads=8, n_kv_heads=8, max_len=512)
    model = TinyGPT(cfg).eval()
    P, N = 32, 160
    prompt = torch.randint(0, cfg.vocab, (1, P))
    out_r, logit_r = generate_recompute(model, prompt, N)
    out_c, logit_c = generate_cached(model, prompt, N)
    check(torch.equal(out_r, out_c), f'{N} greedy tokens are identical with and without the cache')
    diff = (logit_r - logit_c).abs().max().item()
    check(diff < 1e-4, f'final logits match (max abs diff {diff:.1e})')
    # work: recompute runs the model over P, P+1, ..., P+N-1 tokens; the cache runs P, then 1 per step
    work_r = sum(P + i for i in range(N))
    work_c = P + (N - 1)
    t_r = bench(lambda: generate_recompute(model, prompt, N), warmup=1, iters=3)['median']
    t_c = bench(lambda: generate_cached(model, prompt, N), warmup=1, iters=3)['median']
    print(f'  token-forwards: recompute {work_r:,}  vs  cached {work_c:,}  ({work_r / work_c:.0f}x fewer)')
    print(f'  wall time (median of 3): recompute {t_r * 1e3:.0f} ms  vs  cached {t_c * 1e3:.0f} ms  ({t_r / t_c:.1f}x)')
    check(t_c < t_r, 'the cached loop is faster')
    per_tok = 2 * cfg.n_layers * cfg.n_kv_heads * (cfg.d_model // cfg.n_heads) * 4
    print(f'  price: the cache holds {per_tok:,} B per token here (FP32) = {fmt_bytes(per_tok * (P + N))} at {P + N} tokens')
    SUMMARY.append(('kvcache', f'identical tokens; {work_r / work_c:.0f}x fewer token-forwards, {t_r / t_c:.1f}x faster on CPU'))


# =============================================================================================
# (b) KV-cache size: MHA / GQA / MQA / MLA
# =============================================================================================
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem=2):
    """K and V, for every layer and KV head: 2 * L * n_kv * d_head * bytes."""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem


def mla_bytes_per_token(n_layers, d_latent, d_rope, bytes_per_elem=2):
    """MLA caches one latent (plus a small shared RoPE key) per layer, shared by all heads."""
    return n_layers * (d_latent + d_rope) * bytes_per_elem


class LatentKVAttention(nn.Module):
    """MLA-style attention without RoPE: cache c = x W_dkv (d_c numbers per token);
    every head rebuilds its K and V from c. The 'absorbed' path never materialises K or V."""

    def __init__(self, d, n_heads, d_c):
        super().__init__()
        self.h, self.hd = n_heads, d // n_heads
        self.wq = nn.Linear(d, d, bias=False)
        self.w_dkv = nn.Linear(d, d_c, bias=False)                         # down-projection (cached output)
        self.w_uk = nn.Parameter(torch.randn(n_heads, d_c, self.hd) / math.sqrt(d_c))   # up: latent -> K
        self.w_uv = nn.Parameter(torch.randn(n_heads, d_c, self.hd) / math.sqrt(d_c))   # up: latent -> V

    def forward(self, x, absorbed=False):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)        # [B, h, T, hd]
        c = self.w_dkv(x)                                                  # [B, T, d_c]  <- the whole cache
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        if not absorbed:
            k = torch.einsum('btc,hcd->bhtd', c, self.w_uk)                # rebuild per-head K, V
            v = torch.einsum('btc,hcd->bhtd', c, self.w_uv)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
            att = att.masked_fill(~mask, float('-inf')).softmax(-1)
            return att @ v, c
        q_lat = torch.einsum('bhtd,hcd->bhtc', q, self.w_uk)               # fold W_uk into the query
        att = (q_lat @ c.unsqueeze(1).transpose(-2, -1)) / math.sqrt(self.hd)
        att = att.masked_fill(~mask, float('-inf')).softmax(-1)
        o_lat = att @ c.unsqueeze(1)                                       # [B, h, T, d_c]
        return torch.einsum('bhtc,hcd->bhtd', o_lat, self.w_uv), c         # fold W_uv on the way out


def section_kvsize():
    header('(b) KV-cache size: MHA vs GQA vs MQA vs an MLA-style latent')
    torch.manual_seed(0)
    T, H = 100, 8
    for kv in (8, 4, 2, 1):
        cfg = GPTConfig(vocab=64, d_model=256, n_layers=3, n_heads=H, n_kv_heads=kv)
        model = TinyGPT(cfg).eval()
        with torch.no_grad():
            _, past = model(torch.randint(0, 64, (1, T)))
        measured = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in past)
        formula = T * kv_bytes_per_token(cfg.n_layers, kv, cfg.d_model // H, 4)
        name = {8: 'MHA', 1: 'MQA'}.get(kv, 'GQA')
        check(measured == formula, f'{name:3s} n_kv={kv}: cache tensors hold {measured:,} B = '
              f'2 x L x n_kv x d_head x 4 B x {T} tokens')

    # MLA-style latent: same outputs, much smaller cache
    torch.manual_seed(0)
    d, h, d_c = 256, 8, 64
    mla = LatentKVAttention(d, h, d_c).eval()
    x = torch.randn(2, 40, d)
    with torch.no_grad():
        o1, c = mla(x, absorbed=False)
        o2, _ = mla(x, absorbed=True)
    check(torch.allclose(o1, o2, atol=1e-5), 'MLA-style: the absorbed form (attend on the latent) equals rebuilding K and V')
    check(c.shape[-1] == d_c, f'MLA-style cache: {d_c} numbers per token per layer vs {2 * d} for MHA ({2 * d // d_c}x smaller)')

    # real shapes, BF16 (2 bytes)
    llama = dict(n_layers=32, head_dim=128)                                  # Llama-3.1-8B config.json
    rows = [('MHA, 32 KV heads (hypothetical)', kv_bytes_per_token(n_kv_heads=32, **llama)),
            ('GQA, 8 KV heads (actual)', kv_bytes_per_token(n_kv_heads=8, **llama)),
            ('MQA, 1 KV head (hypothetical)', kv_bytes_per_token(n_kv_heads=1, **llama))]
    print('\n  Llama-3.1-8B shape (32 layers, 32 query heads, d_head 128), BF16 cache:')
    print(f'  {"attention":34s} {"per token":>10s} {"8K ctx":>10s} {"32K ctx":>10s} {"128K ctx":>10s}')
    for name, b in rows:
        print(f'  {name:34s} {fmt_bytes(b):>10s} {fmt_bytes(b * 8192):>10s} {fmt_bytes(b * 32768):>10s} {fmt_bytes(b * 131072):>10s}')
    check(rows[1][1] == 128 * KiB and rows[1][1] * 8192 == 1 * GiB, 'Llama-3.1-8B GQA: 128 KiB per token, 1 GiB per 8K-token sequence')
    ds_mla = mla_bytes_per_token(61, 512, 64)
    ds_mha = kv_bytes_per_token(61, 128, 128)
    check(ds_mla == 70272 and abs(ds_mha / ds_mla - 56.9) < 0.05,
          f'DeepSeek-V3 MLA: {ds_mla:,} B = {ds_mla / KiB:.1f} KiB per token, {ds_mha / ds_mla:.1f}x smaller than MHA at its shape')
    SUMMARY.append(('kvsize', 'formula = measured cache bytes for MHA/GQA/MQA; MLA absorbed == explicit; '
                              'Llama-3.1-8B 128 KiB/token, DeepSeek-V3 MLA 68.6 KiB/token'))


# =============================================================================================
# (d) FlashAttention-style tiling with online softmax
# =============================================================================================
def attention_tiled(q, k, v, tile=32):
    """One head, no mask. Visit K/V one tile at a time and keep, per query row, a running max m,
    a running denominator l and an unnormalised output acc. The N x N score matrix never exists."""
    N, d = q.shape
    m = torch.full((N,), float('-inf'))
    l = torch.zeros(N)
    acc = torch.zeros(N, v.shape[1])
    for j in range(0, k.shape[0], tile):
        s = (q @ k[j:j + tile].T) / math.sqrt(d)            # [N, tile]: the only scores we hold
        m_new = torch.maximum(m, s.amax(-1))
        p = torch.exp(s - m_new[:, None])
        corr = torch.exp(m - m_new)                          # rescale what we summed so far
        l = l * corr + p.sum(-1)
        acc = acc * corr[:, None] + p @ v[j:j + tile]
        m = m_new
    return acc / l[:, None]


def section_tiled():
    header('(d) FlashAttention idea: tiles + online softmax give the exact same output')
    # hand example from the guide: x = [1, 2, 3, 4] in two tiles
    m1 = 2.0; l1 = math.exp(1 - 2) + math.exp(0)                       # tile [1, 2]
    m2 = 4.0; l2 = l1 * math.exp(m1 - m2) + math.exp(3 - 4) + math.exp(0)
    direct = sum(math.exp(v - 4) for v in (1, 2, 3, 4))
    print(f'  online softmax on [1,2 | 3,4]: l after tile 1 = {l1:.4f}, after tile 2 = {l2:.4f}; direct = {direct:.4f}')
    check(abs(l2 - direct) < 1e-12, 'rescaling by e^(m_old - m_new) makes the running sum exact')
    torch.manual_seed(0)
    N, d = 512, 64
    q, k, v = torch.randn(N, d), torch.randn(N, d), torch.randn(N, d)
    ref = torch.softmax(q @ k.T / math.sqrt(d), -1) @ v
    for tile in (16, 64, 512):
        out = attention_tiled(q, k, v, tile)
        check(torch.allclose(out, ref, atol=1e-5), f'tile={tile:3d}: tiled output equals softmax(QK^T/sqrt d)V')
    n, heads = 8192, 32
    full = heads * n * n * 2
    stats = heads * n * 2 * 4
    print(f'  N={n}, {heads} heads, FP16: the score matrix would be {fmt_bytes(full)} per layer; '
          f'the running stats are {fmt_bytes(stats)}')
    check(full == 4 * GiB, 'naive attention at N=8192 x 32 heads writes a 4 GiB score matrix per layer')
    SUMMARY.append(('tiled', 'tiled online-softmax attention == reference for tiles 16/64/512'))


# =============================================================================================
# (e) paged KV cache and (f) prefix caching
# =============================================================================================
class BlockAllocator:
    """A free list of fixed-size physical blocks with reference counts (for prefix sharing)."""

    def __init__(self, num_blocks):
        self.free = list(range(num_blocks - 1, -1, -1))
        self.ref = [0] * num_blocks

    def alloc(self):
        if not self.free:
            raise MemoryError('KV pool exhausted')
        b = self.free.pop()
        self.ref[b] = 1
        return b

    def share(self, b):
        self.ref[b] += 1

    def release(self, b):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)


class PagedKVCache:
    """K (or V) vectors for many sequences in one pool of [num_blocks, block_size, dim].
    Each sequence owns a block table: logical block i -> physical block id."""

    def __init__(self, num_blocks, block_size, dim):
        self.bs = block_size
        self.pool = torch.zeros(num_blocks, block_size, dim)
        self.alloc = BlockAllocator(num_blocks)
        self.tables, self.lens = {}, {}

    def append(self, seq, vec):
        table = self.tables.setdefault(seq, [])
        n = self.lens.get(seq, 0)
        if n % self.bs == 0:                                   # last block full (or none yet): grab one
            table.append(self.alloc.alloc())
        self.pool[table[-1], n % self.bs] = vec
        self.lens[seq] = n + 1

    def gather(self, seq):
        n = self.lens[seq]
        return self.pool[self.tables[seq]].reshape(-1, self.pool.shape[-1])[:n]

    def free_seq(self, seq):
        for b in self.tables.pop(seq):
            self.alloc.release(b)
        self.lens.pop(seq)


def section_paged():
    header('(e) PagedAttention: fixed-size blocks instead of one max-length slab per request')
    g = torch.Generator().manual_seed(0)
    max_len, bs, n_req = 2048, 16, 12
    lengths = torch.randint(30, 700, (n_req,), generator=g).tolist()       # prompt + output, mixed lengths
    used = sum(lengths)
    contiguous = n_req * max_len
    blocks = sum(math.ceil(n / bs) for n in lengths)
    paged = blocks * bs
    print(f'  {n_req} requests, lengths {lengths}')
    print(f'  contiguous (reserve max_len={max_len} each): {contiguous:,} slots for {used:,} tokens '
          f'-> {100 * (1 - used / contiguous):.1f}% wasted')
    print(f'  paged (blocks of {bs}): {blocks} blocks = {paged:,} slots -> {100 * (1 - used / paged):.1f}% wasted '
          f'(only inside each last block)')
    check(paged - used < n_req * bs, 'paged waste is under one block per request')
    check(1 - used / contiguous > 0.75, 'contiguous max-length reservation wastes most of the KV memory here')
    per_tok = kv_bytes_per_token(32, 8, 128)                               # Llama-3.1-8B, BF16
    print(f'  at Llama-3.1-8B size (128 KiB/token): contiguous {fmt_bytes(contiguous * per_tok)} vs paged '
          f'{fmt_bytes(paged * per_tok)} for the same {n_req} requests')

    # the block table really works: write K vectors token by token, interleaved across sequences
    torch.manual_seed(0)
    dim = 8
    cache = PagedKVCache(num_blocks=blocks, block_size=bs, dim=dim)
    ref = {i: torch.randn(n, dim) for i, n in enumerate(lengths)}
    for t in range(max(lengths)):                                           # decode steps, all requests in flight
        for i, n in enumerate(lengths):
            if t < n:
                cache.append(i, ref[i][t])
    check(all(torch.equal(cache.gather(i), ref[i]) for i in ref), 'gathering through the block tables returns every sequence exactly')
    check(len(cache.alloc.free) == 0, f'the pool of exactly {blocks} blocks was enough: no slab per request needed')
    tables_scattered = any(sorted(t) != list(range(min(t), min(t) + len(t))) for t in cache.tables.values())
    check(tables_scattered, 'physical blocks of one sequence are not contiguous (and do not need to be)')
    cache.free_seq(0)
    check(len(cache.alloc.free) == math.ceil(lengths[0] / bs), 'a finished request returns its blocks to the pool at once')

    # how many requests fit in a fixed budget (continuous batching admits them as memory frees up)
    budget = 8 * max_len                                                    # room for 8 max-length slabs
    fit_paged, acc = 0, 0
    for n in lengths:
        need = math.ceil(n / bs) * bs
        if acc + need > budget:
            break
        acc += need; fit_paged += 1
    print(f'  with a budget of {budget:,} token slots: contiguous admits {budget // max_len} requests, '
          f'paged admits {fit_paged} (all {n_req} if they fit)')
    check(fit_paged > budget // max_len, 'paging lets more requests run at once in the same memory')
    SUMMARY.append(('paged', f'waste {100 * (1 - used / contiguous):.0f}% contiguous vs '
                             f'{100 * (1 - used / paged):.1f}% paged; block tables verified'))


def section_prefix():
    header('(f) Prefix caching: compute the shared prompt once, reuse its KV blocks')
    bs = 16
    system = list(range(100, 164))                                          # 64-token shared system prompt
    users = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12, 13], list(range(20, 45))]
    alloc = BlockAllocator(64)
    index, computed, reused = {}, 0, 0                                      # block hash -> physical block
    for u in users:
        toks, parent = system + u, None
        for i in range(0, len(toks), bs):
            chunk = tuple(toks[i:i + bs])
            if len(chunk) < bs:                                             # partial last block: not shareable
                computed += len(chunk); alloc.alloc(); continue
            key = hash((parent, chunk))                                     # depends on the WHOLE prefix
            if key in index:
                alloc.share(index[key]); reused += bs
            else:
                index[key] = alloc.alloc(); computed += bs
            parent = key
    total = sum(len(system) + len(u) for u in users)
    print(f'  3 requests share a {len(system)}-token system prompt: {total} prompt tokens, '
          f'{computed} computed, {reused} served from cache')
    check(reused == 2 * len(system), 'requests 2 and 3 reuse all 4 system-prompt blocks')
    check(computed + reused == total, 'every prompt token is either computed once or reused')

    # numerically: logits from (cached prefix KV + suffix) equal a full forward pass
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=200, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2, max_len=256)
    model = TinyGPT(cfg).eval()
    prefix = torch.tensor([system])
    with torch.no_grad():
        _, prefix_past = model(prefix)                                       # prefill the shared part ONCE
        for u in users:
            suffix = torch.tensor([u])
            logits_cached, _ = model(suffix, prefix_past)
            logits_full, _ = model(torch.cat([prefix, suffix], 1))
            assert torch.allclose(logits_cached, logits_full[:, prefix.shape[1]:], atol=1e-5)
    check(True, 'for every request, suffix logits on top of the cached prefix KV equal a full forward pass')
    SUMMARY.append(('prefix', f'{reused} of {total} prompt tokens reused; logits identical to full prefill'))


# =============================================================================================
# (g) speculative decoding
# =============================================================================================
def synthetic_batch(B, T, V, tables, g, noise=0.2):
    """A toy 'language'. After a token below V/2 the next token is easy (a fixed successor A[prev]);
    after the others it depends on the last TWO tokens (B[prev2, prev]), which a tiny model learns
    only partly. 20% of tokens are random noise."""
    A, Bt = tables
    x = torch.randint(0, V, (B, T), generator=g)
    for t in range(2, T):
        easy = x[:, t - 1] < V // 2
        rule = torch.where(easy, A[x[:, t - 1]], Bt[x[:, t - 2], x[:, t - 1]])
        rnd = torch.randint(0, V, (B,), generator=g)
        keep = torch.rand(B, generator=g) > noise
        x[:, t] = torch.where(keep, rule, rnd)
    return x


def train_lm(model, steps, tables, V, seed, lr=3e-3, B=64, T=48):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    for _ in range(steps):
        x = synthetic_batch(B, T, V, tables, g)
        logits, _ = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def speculative_greedy(target, draft, prompt, n_new, gamma=4):
    """Greedy speculative decoding. The draft guesses gamma tokens one by one; the target checks
    all of them in ONE forward pass; we keep the longest prefix the target agrees with, plus the
    target's own next token. The result is exactly the target's greedy output."""
    idx = prompt
    calls = accepted = rejected = 0
    while idx.shape[1] - prompt.shape[1] < n_new:
        d_idx = idx
        for _ in range(gamma):                                               # 1) draft proposes
            nxt = draft(d_idx)[0][:, -1].argmax(-1, keepdim=True)
            d_idx = torch.cat([d_idx, nxt], 1)
        guesses = d_idx[:, idx.shape[1]:]                                    # [1, gamma]
        logits = target(d_idx)[0]; calls += 1                                # 2) target verifies, one pass
        want = logits[:, idx.shape[1] - 1:].argmax(-1)                       # target's choice at each slot: [1, gamma+1]
        n_ok = 0
        while n_ok < gamma and guesses[0, n_ok] == want[0, n_ok]:            # 3) accept the agreeing prefix
            n_ok += 1
        accepted += n_ok
        rejected += n_ok < gamma                                             # at most one rejection per pass
        idx = torch.cat([idx, guesses[:, :n_ok], want[:, n_ok:n_ok + 1]], 1)  # + 1 token from the target
    return idx[:, :prompt.shape[1] + n_new], {'calls': calls, 'accepted': accepted, 'rejected': rejected}


def section_spec():
    header('(g) Speculative decoding: a tiny draft guesses, the target verifies in one pass')
    torch.manual_seed(0)
    V = 32
    g = torch.Generator().manual_seed(42)
    tables = (torch.randint(0, V, (V,), generator=g), torch.randint(0, V, (V, V), generator=g))
    target = TinyGPT(GPTConfig(vocab=V, d_model=128, n_layers=2, n_heads=4, n_kv_heads=4, max_len=256))
    draft = TinyGPT(GPTConfig(vocab=V, d_model=32, n_layers=1, n_heads=2, n_kv_heads=2, max_len=256))
    n_t = sum(p.numel() for p in target.parameters()); n_d = sum(p.numel() for p in draft.parameters())
    t0 = time.perf_counter()
    lt = train_lm(target, 300, tables, V, seed=1)
    ld = train_lm(draft, 250, tables, V, seed=2)
    target.eval(); draft.eval()
    print(f'  target {n_t:,} params (loss {lt:.2f}), draft {n_d:,} params = {n_d / n_t:.1%} of the target '
          f'(loss {ld:.2f}); both trained in {time.perf_counter() - t0:.1f} s')
    n_new, gamma, n_prompts = 96, 4, 4
    tot = {'calls': 0, 'accepted': 0, 'rejected': 0}
    for s in range(n_prompts):
        prompt = synthetic_batch(1, 8, V, tables, torch.Generator().manual_seed(100 + s))
        ref, _ = generate_recompute(target, prompt, n_new)
        out, st = speculative_greedy(target, draft, prompt, n_new, gamma)
        assert torch.equal(out, ref), 'speculative output differs from greedy'
        for k in tot:
            tot[k] += st[k]
    check(True, f'{n_prompts} prompts x {n_new} tokens: speculative output == target greedy output, token for token')
    alpha = tot['accepted'] / (tot['accepted'] + tot['rejected'])          # P(draft token accepted | reached)
    calls = tot['calls'] / n_prompts
    tok_per_call = n_new / calls
    expect = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    print(f'  acceptance rate alpha = {alpha:.1%} (accepted / checked draft tokens, gamma = {gamma})')
    print(f'  target passes: {calls:.1f} instead of {n_new} -> {tok_per_call:.2f} tokens per target pass '
          f'(formula (1 - a^(g+1)) / (1 - a) = {expect:.2f})')
    c = n_d / n_t                                                           # cost of a draft step relative to the target
    print(f'  rough speedup if cost ~ params read: {tok_per_call:.2f} / (1 + {gamma} x {c:.3f}) = '
          f'{tok_per_call / (1 + gamma * c):.2f}x')
    check(calls < n_new / 2, f'the draft is right often enough to halve the target passes ({tok_per_call:.2f} tokens per pass)')
    check(abs(expect - tok_per_call) < 0.35, 'and the i.i.d. formula predicts the measured tokens per pass to within 0.35')
    SUMMARY.append(('spec', f'identical to greedy; alpha {alpha:.0%}, {tok_per_call:.2f} tokens per target pass (gamma={gamma})'))


# =============================================================================================
# (h) kernel fusion
# =============================================================================================
def gelu_tanh_unfused(x):
    """GELU (tanh form) written as 9 separate elementwise kernels: each reads and writes a full tensor."""
    a = x * x
    a = a * x
    a = 0.044715 * a
    a = x + a
    a = 0.7978845608028654 * a                    # sqrt(2 / pi)
    a = torch.tanh(a)
    a = 1.0 + a
    b = 0.5 * x
    return b * a


def section_fusion():
    header('(h) Kernel fusion: one pass over memory instead of nine')
    torch.manual_seed(0)
    x = torch.randn(4 * 1024 * 1024)                                       # 16 MiB of FP32
    ref = F.gelu(x, approximate='tanh')
    check(torch.allclose(gelu_tanh_unfused(x), ref, atol=1e-5), 'the 9-kernel version computes the same GELU')
    reads, writes = 2 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 2, 9                   # tensor-sized reads/writes per kernel
    nbytes = x.numel() * 4
    print(f'  traffic: unfused {reads} reads + {writes} writes = {(reads + writes) * nbytes / MiB:.0f} MiB; '
          f'fused 1 read + 1 write = {2 * nbytes / MiB:.0f} MiB ({(reads + writes) / 2:.0f}x less)')
    t_u = bench(lambda: gelu_tanh_unfused(x), warmup=2, iters=7)['median']
    t_f = bench(lambda: F.gelu(x, approximate='tanh'), warmup=2, iters=7)['median']
    print(f'  CPU time (median): unfused {t_u * 1e3:.1f} ms, fused {t_f * 1e3:.1f} ms ({t_u / t_f:.1f}x)')
    check(t_f < t_u, 'the fused kernel is faster: it is memory-bound, so fewer bytes = less time')
    if '--compile' in sys.argv and hasattr(torch, 'compile'):
        fn = torch.compile(gelu_tanh_unfused)
        b = bench(lambda: fn(x), warmup=2, iters=7)
        print(f'  torch.compile of the unfused code: first call {b["first"]:.1f} s (compiling), '
              f'then {b["median"] * 1e3:.1f} ms')
    SUMMARY.append(('fusion', f'{(reads + writes) / 2:.0f}x fewer bytes, {t_u / t_f:.1f}x faster on CPU'))


# =============================================================================================
# (i) 2:4 sparsity and distillation
# =============================================================================================
def mask_2to4(w):
    """Keep the 2 largest-magnitude weights in every group of 4 consecutive inputs."""
    out_f, in_f = w.shape
    g = w.abs().reshape(out_f, in_f // 4, 4)
    keep = torch.zeros_like(g, dtype=torch.bool).scatter_(-1, g.topk(2, dim=-1).indices, True)
    return keep.reshape(out_f, in_f)


def section_sparse():
    header('(i) 2:4 sparsity, then distillation to win back accuracy')
    torch.manual_seed(0)
    d, hidden, n_cls = 128, 512, 64
    mix = torch.randn(d, d) * torch.logspace(0, -1.5, d)[:, None]           # correlated inputs, like real activations
    data = lambda n: torch.randn(n, d) @ mix
    teacher = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_cls)).eval()
    with torch.no_grad():
        teacher[2].weight.mul_(4.0)                                          # a confident teacher, like a trained one
    student = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_cls))
    student.load_state_dict(teacher.state_dict())
    masks = {}
    with torch.no_grad():
        for name, m in student.named_modules():
            if isinstance(m, nn.Linear):
                masks[name] = mask_2to4(m.weight)
                m.weight.mul_(masks[name])
    w = student[0].weight
    groups = (w.reshape(w.shape[0], -1, 4) != 0).sum(-1)
    check(bool((groups <= 2).all()), 'every group of 4 weights has at most 2 non-zeros')
    dense16 = w.numel() * 2
    sparse16 = w.numel() // 2 * 2 + w.numel() // 2 * 2 / 8                  # kept FP16 values + 2-bit index each
    print(f'  storage for a {tuple(w.shape)} FP16 weight: dense {dense16:,} B, 2:4 compressed {sparse16:,.0f} B '
          f'({sparse16 / dense16:.4f} of dense)')
    check(abs(sparse16 / dense16 - 0.5625) < 1e-9, '2:4 stores half the values plus 2 bits of position each: 56.25% of FP16')

    X_eval = data(4096)
    T = 2.0

    def score():
        with torch.no_grad():
            zt, zs = teacher(X_eval), student(X_eval)
            agree = (zt.argmax(-1) == zs.argmax(-1)).float().mean().item()
            kl = F.kl_div(F.log_softmax(zs / T, -1), F.softmax(zt / T, -1), reduction='batchmean').item()
        return agree, kl

    a0, k0 = score()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    for _ in range(300):                                                    # distill: match the teacher's soft labels
        x = data(256)
        with torch.no_grad():
            zt = teacher(x)
        loss = (T * T) * F.kl_div(F.log_softmax(student(x) / T, -1), F.softmax(zt / T, -1), reduction='batchmean')
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                               # stay 2:4 sparse
            for name, m in student.named_modules():
                if isinstance(m, nn.Linear):
                    m.weight.mul_(masks[name])
    a1, k1 = score()
    print(f'  student = 2:4-pruned teacher: top-1 agreement {a0:.1%} -> {a1:.1%}, KL {k0:.4f} -> {k1:.4f} '
          f'after 300 distillation steps')
    check(a1 > 1.8 * a0 and k1 < k0 / 5, 'distillation doubles the agreement and cuts the KL over 5x, and the student stays 2:4')
    SUMMARY.append(('sparse', f'2:4 = 56.25% of FP16 bytes; distillation: agreement {a0:.0%} -> {a1:.0%}'))


# =============================================================================================
# (j) activation checkpointing
# =============================================================================================
class MLPBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(x)))


def saved_activation_bytes(fn, params):
    """Bytes autograd keeps for the backward pass, excluding the parameters themselves."""
    pids = {p.data_ptr() for p in params}
    seen = {}

    def pack(t):
        if t.data_ptr() not in pids:
            seen[(t.data_ptr(), t.numel())] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    return out, sum(seen.values())


def section_ckpt():
    header('(j) Activation checkpointing: store layer inputs only, recompute the rest')
    torch.manual_seed(0)
    L, B, d = 8, 256, 128
    model = nn.ModuleList(MLPBlock(d) for _ in range(L))
    params = list(model.parameters())
    x0 = torch.randn(B, d)

    def run(mode):
        x = x0.clone().requires_grad_(True)
        h = x
        for blk in model:
            if mode == 'none':
                h = blk(h)
            else:
                h = checkpoint(blk, h, use_reentrant=(mode == 'reentrant'))
        return h

    out_n, bytes_none = saved_activation_bytes(lambda: run('none'), params)
    out_c, bytes_ckpt = saved_activation_bytes(lambda: run('reentrant'), params)
    # analytic: fc1 saves its input (B*d), GELU its input (B*4d), fc2 its input (B*4d): 9*B*d per block
    pred_none, pred_ckpt = L * 9 * B * d * 4, L * B * d * 4
    print(f'  {L} blocks of x + fc2(gelu(fc1(x))), batch {B}, d {d}, FP32:')
    print(f'  saved for backward: no checkpointing {fmt_bytes(bytes_none)} (formula 9*L*B*d*4 = {fmt_bytes(pred_none)}); '
          f'checkpointed {fmt_bytes(bytes_ckpt)} (formula L*B*d*4 = {fmt_bytes(pred_ckpt)})')
    check(bytes_none == pred_none and bytes_ckpt == pred_ckpt, 'measured saved bytes equal the analytic count: 9x less with checkpointing')

    grads = {}
    for mode in ('none', 'reentrant', 'non-reentrant'):
        model.zero_grad()
        run(mode).pow(2).sum().backward()
        grads[mode] = [p.grad.clone() for p in params]
    same = all(torch.equal(a, b) for m in ('reentrant', 'non-reentrant') for a, b in zip(grads['none'], grads[m]))
    check(same, 'gradients are bit-identical with torch.utils.checkpoint (both implementations)')
    # transformer scale (Korthikanti et al. 2022): sbh(34 + 5as/h) bytes per layer in 16-bit, 2sbh with full recompute
    s, b, h, a, Lm = 4096, 1, 4096, 32, 32
    per_layer = s * b * h * (34 + 5 * a * s / h)
    no_scores = s * b * h * 34
    print(f'  GPT layer at s=4096, h=4096, 32 heads: {per_layer / 1e9:.2f} GB/layer naive, {no_scores / 1e9:.2f} GB without the '
          f'stored score matrices (FlashAttention), {2 * s * b * h / 1e6:.1f} MB with full recompute; x {Lm} layers = '
          f'{Lm * no_scores / 1e9:.1f} GB vs {Lm * 2 * s * b * h / 1e9:.2f} GB')
    SUMMARY.append(('ckpt', f'saved bytes {fmt_bytes(bytes_none)} -> {fmt_bytes(bytes_ckpt)} (9x), grads bit-identical'))


# =============================================================================================
# (j) training memory: Adam, ZeRO, LoRA
# =============================================================================================
# bytes per parameter for (weights, gradients, optimizer states incl. any FP32 master copy)
RECIPES = {
    'FP32 weights + Adam':                     (4, 4, 8),
    'Mixed precision BF16 + FP32 master + Adam': (2, 2, 12),
    'Mixed precision + 8-bit Adam':            (2, 2, 4 + 2),
    'Pure BF16 (no master copy)':              (2, 2, 4),
}


def train_bytes_per_param(recipe):
    return sum(RECIPES[recipe])


def zero_bytes_per_param(stage, n_gpus, w=2, g=2, k=12):
    """ZeRO (Rajbhandari et al. 2020) for mixed-precision Adam: shard optimizer (1), + grads (2), + weights (3)."""
    return {0: w + g + k, 1: w + g + k / n_gpus, 2: w + (g + k) / n_gpus, 3: (w + g + k) / n_gpus}[stage]


def lora_params(shapes, r):
    """LoRA adds A [r, d_in] and B [d_out, r] next to each frozen W [d_out, d_in]: r * (d_in + d_out)."""
    return sum(r * (d_in + d_out) for d_out, d_in in shapes)


def llama_linear_shapes(d=4096, n_heads=32, n_kv=8, hd=128, ffn=14336, attn_only=False):
    att = [(n_heads * hd, d), (n_kv * hd, d), (n_kv * hd, d), (d, n_heads * hd)]     # q, k, v, o as [out, in]
    mlp = [(ffn, d), (ffn, d), (d, ffn)]                                               # gate, up, down
    return att if attn_only else att + mlp


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                                   # frozen
        self.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))      # zero init: starts as the base model
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def section_train():
    header('(j) Training memory: Adam bytes per parameter, ZeRO sharding, LoRA')
    print(f'  {"recipe":44s} {"w":>3s} {"g":>3s} {"opt":>4s} {"B/param":>8s} {"1.7B":>9s} {"8B":>9s} {"70B":>9s}')
    for name, (w, g, o) in RECIPES.items():
        bpp = w + g + o
        print(f'  {name:44s} {w:>3d} {g:>3d} {o:>4d} {bpp:>8d} ' +
              ' '.join(f'{p * bpp / 1e9:>7.0f}GB' for p in (1.721e9, 8.03e9, 70.55e9)))
    check(train_bytes_per_param('Mixed precision BF16 + FP32 master + Adam') == 16, 'mixed-precision Adam = 2 + 2 + (4 master + 4 m + 4 v) = 16 B/param')
    check(train_bytes_per_param('FP32 weights + Adam') == 16, 'plain FP32 Adam is also 16 B/param: mixed precision saves activation memory and time, not state')
    psi, n = 7.5e9, 64                                                     # the ZeRO paper's Figure 1 example
    z = [zero_bytes_per_param(s, n) * psi / 1e9 for s in range(4)]
    print(f'  ZeRO, 7.5B params on 64 GPUs: per GPU {z[0]:.1f} GB (none), {z[1]:.1f} (stage 1), {z[2]:.1f} (stage 2), {z[3]:.2f} (stage 3)')
    check([round(v, 1) for v in z] == [120.0, 31.4, 16.6, 1.9], 'matches the ZeRO paper: 120 / 31.4 / 16.6 / 1.9 GB')

    r = 16
    lora_att = lora_params(llama_linear_shapes(attn_only=True), r) * 32
    lora_all = lora_params(llama_linear_shapes(), r) * 32
    print(f'  LoRA r={r} on Llama-3.1-8B: q,k,v,o only {lora_att / 1e6:.1f}M params; all 7 linears {lora_all / 1e6:.1f}M '
          f'({lora_all / 8.03e9:.2%} of 8.03B)')
    check(lora_att == 13_631_488 and lora_all == 41_943_040, 'LoRA counts: 13,631,488 (attention) and 41,943,040 (all linears)')
    base_q, base_bf16 = 8.03e9 * (4 + 0.127) / 8, 8.03e9 * 2
    lora_state = lora_all * 16
    print(f'  fine-tuning memory before activations: full {8.03e9 * 16 / 1e9:.0f} GB; LoRA on a BF16 base '
          f'{(base_bf16 + lora_state) / 1e9:.1f} GB; QLoRA on an NF4 base {(base_q + lora_state) / 1e9:.1f} GB')

    # verify the formula on a real module tree, and that LoRA starts as (and merges back into) the base
    torch.manual_seed(0)
    cfg = GPTConfig(vocab=64, d_model=128, n_layers=2, n_heads=4, n_kv_heads=2)
    model = TinyGPT(cfg)
    x = torch.randint(0, 64, (2, 16))
    with torch.no_grad():
        before, _ = model(x)
    model.requires_grad_(False)                                            # freeze the whole base model
    shapes = []
    for blk in model.blocks:
        for name in ('q', 'k', 'v', 'o'):
            lin = getattr(blk.attn, name)
            shapes.append(tuple(lin.weight.shape))
            setattr(blk.attn, name, LoRALinear(lin, r=4))                   # only A and B are trainable
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(trainable == lora_params(shapes, 4), f'toy GPT: {trainable:,} trainable LoRA params = sum r*(d_in + d_out)')
    with torch.no_grad():
        after, _ = model(x)
    check(torch.equal(before, after), 'B = 0 at init, so the LoRA model starts exactly at the base model')
    lin = model.blocks[0].attn.q
    with torch.no_grad():
        lin.B.normal_()
        xin = torch.randn(3, cfg.d_model)
        merged = lin.base.weight + lin.scale * lin.B @ lin.A
        check(torch.allclose(lin(xin), xin @ merged.T, atol=1e-4), 'after training, W + (alpha/r) B A merges into one matrix: zero inference overhead')
    SUMMARY.append(('train', 'Adam 16 B/param; ZeRO 120/31.4/16.6/1.9 GB; LoRA r=16 all-linear on 8B = 41.9M params'))


# =============================================================================================
# (k) measuring
# =============================================================================================
def section_bench():
    header('(k) Measure like you mean it: cold call, warm-up, median and p90')
    torch.manual_seed(0)
    model = TinyGPT(GPTConfig(vocab=512, d_model=256, n_layers=4, n_heads=8, n_kv_heads=8)).eval()
    x = torch.randint(0, 512, (8, 128))
    with torch.no_grad():
        r = bench(lambda: model(x), warmup=3, iters=20)
    print(f'  fresh TinyGPT, batch 8 x 128 tokens: cold first call {r["first"] * 1e3:.1f} ms, '
          f'then median {r["median"] * 1e3:.1f} ms, p90 {r["p90"] * 1e3:.1f} ms over 20 runs')
    print('  (on a GPU the cold call also pays for CUDA context, kernel loading and autotuning: never time it)')
    check(r['median'] > 0 and r['p90'] >= r['median'], 'report the median and a tail percentile, never a single run')
    print('  On a Jetson, before you trust a number:')
    for line in ('sudo nvpmodel -q                 # which power mode am I in?',
                 'sudo nvpmodel -m <id>            # pick one (Orin Nano Super: 2 = MAXN_SUPER)',
                 'sudo jetson_clocks               # pin CPU/GPU/EMC clocks to the mode\'s maximum',
                 'tegrastats                       # watch RAM, GPU load, clocks and power while it runs',
                 'trtexec --loadEngine=m.engine --warmUp=500 --iterations=200   # warm-up, then many runs'):
        print('    ' + line)
    SUMMARY.append(('bench', f'median {r["median"] * 1e3:.1f} ms, p90 {r["p90"] * 1e3:.1f} ms, cold {r["first"] * 1e3:.1f} ms'))


SECTIONS = {
    'quant': section_quant, 'kvcache': section_kvcache, 'kvsize': section_kvsize, 'tiled': section_tiled,
    'paged': section_paged, 'prefix': section_prefix, 'spec': section_spec, 'fusion': section_fusion,
    'sparse': section_sparse, 'ckpt': section_ckpt, 'train': section_train, 'bench': section_bench,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('sections', nargs='*', help='sections to run (default: all)')
    ap.add_argument('--list', action='store_true', help='list the sections and exit')
    ap.add_argument('--compile', action='store_true', help='also try torch.compile in the fusion section (slow first call)')
    args = ap.parse_args()
    if args.list:
        print('\n'.join(SECTIONS))
        return
    chosen = args.sections or list(SECTIONS)
    unknown = [s for s in chosen if s not in SECTIONS]
    if unknown:
        ap.error(f'unknown section(s) {unknown}; choose from {list(SECTIONS)}')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    t0 = time.perf_counter()
    for name in chosen:
        SECTIONS[name]()
    header(f'SUMMARY: {len(chosen)} section(s), all checks passed in {time.perf_counter() - t0:.1f} s')
    for name, line in SUMMARY:
        print(f'  {name:8s} {line}')


if __name__ == '__main__':
    main()
