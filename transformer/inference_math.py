"""
Inference maths for transformer_guide.html: prints every worked number on the page.

Companion to the guide's prefill, decode, comparison, memory and training sections. The
page's tables were copied from this output; rerun it after changing a config below.

Run:
    python3 inference_math.py          # print every table, in the guide's section order
    python3 inference_math.py --json   # the same numbers as JSON

Standard library only. If torch is installed, it also checks its prefill FLOPs and the
GPT-2 small parameter count against compute_flops() and Transformer in transformer.py.

Conventions (the same as the guide, step_by_step.py and transformer.py):
    [m x k] @ [k x n] costs 2*m*k*n FLOPs (one multiply + one add per term)
    B batch, N prompt length (prefill), t cached tokens a decode step attends to,
    d model width, h query heads, n_kv K/V heads, d_k = d / h, d_ff FFN width,
    L layers, V vocabulary, b bytes per element (2 = FP16)
    GB = 1e9 bytes (bandwidth maths), MiB / GiB = 2**20 / 2**30 bytes (tensor sizes)
Reference config: the primer's 7B-class model (step_by_step.py, step 19): GPT-3 6.7B
shape, d = 4096, L = 32, h = 32, d_k = 128, d_ff = 4d, V = 50257 (GPT-2 vocabulary),
a 2048-row learned position table, tied LM head, FP16. Like the primer, parameter
counts ignore biases and norm gains (under 0.01% here).
Byte model for prefill ops ("unfused"): each op reads its inputs from DRAM and writes
its output once; weights are read once per op. Decode bytes = all weights once + the
K and V cache read once (the primer's convention). Times are lower bounds at 100% of
peak FLOP/s and bandwidth: t = max(FLOPs / peak, bytes / bandwidth).
"""

import argparse
import json
import math

GB, MB = 1e9, 1e6
KiB, MiB, GiB = 2**10, 2**20, 2**30

DEVICES = {
    # Jetson AGX Orin 64GB: 204.8 GB/s LPDDR5; dense FP16 = 170 sparse INT8 TOPS / 2 / 2.
    "agx": dict(name="Jetson AGX Orin 64GB", bw=204.8e9, peak=42.5e12, mem=64e9),
    # Orin Nano 8GB in Super mode (JetPack 6.2+): 102 GB/s, 17 FP16 TFLOPS.
    "nano": dict(name="Jetson Orin Nano Super 8GB", bw=102e9, peak=17e12, mem=8e9),
}
AGX = DEVICES["agx"]


def config(name, d, L, h, V, n_pos, n_kv=None, d_ff=None, b=2):
    n_kv = h if n_kv is None else n_kv
    return dict(name=name, d=d, L=L, h=h, n_kv=n_kv, d_k=d // h, d_ff=d_ff or 4 * d, V=V,
                n_pos=n_pos, b=b)


REF = config("7B-class (primer step 19)", d=4096, L=32, h=32, V=50257, n_pos=2048)
GQA8 = dict(REF, name="7B-class + GQA (n_kv = 8)", n_kv=8)
MQA = dict(REF, name="7B-class + MQA (n_kv = 1)", n_kv=1)
GPT2 = config("GPT-2 small", d=768, L=12, h=12, V=50257, n_pos=1024)
N_REF = 4096        # prefill prompt length and decode cache length used for the worked numbers


def r(x, n=2):
    return round(x, n)


def params(c):
    """Parameter counts (no biases, no norms; tied LM head)."""
    d, dk = c["d"], c["d_k"]
    q = d * c["h"] * dk
    kv = 2 * d * c["n_kv"] * dk
    o = c["h"] * dk * d
    ffn = 2 * d * c["d_ff"]
    block = q + kv + o + ffn
    emb, pos = c["V"] * d, c["n_pos"] * d
    total = c["L"] * block + emb + pos
    return dict(W_Q=q, W_KV=kv, W_QKV=q + kv, W_O=o, W1=d * c["d_ff"], W2=c["d_ff"] * d,
                attn=q + kv + o, ffn=ffn, block=block, blocks=c["L"] * block, emb=emb, pos=pos,
                norms_per_block=4 * d, total=total,
                matmul=c["L"] * block + emb)          # weights that take part in a matmul


def kv_bytes_per_token(c, b=None):
    return 2 * c["L"] * c["n_kv"] * c["d_k"] * (c["b"] if b is None else b)


# ----------------------------------------------------------------------------
# Prefill: the whole prompt [B, N] in one pass
# ----------------------------------------------------------------------------
def prefill_ops(c, N, B=1):
    """One layer's ops (unfused byte model), plus the ops that run once."""
    d, h, nkv, dk, dff, V, b = c["d"], c["h"], c["n_kv"], c["d_k"], c["d_ff"], c["V"], c["b"]
    BN = B * N
    qkv_w = d * (h + 2 * nkv) * dk
    ops = [
        # name, FLOPs, bytes, matmul?
        ("LN1", 7 * BN * d, 2 * b * BN * d, False),
        ("QKV projection", 2 * BN * qkv_w, b * (BN * d + qkv_w + BN * (h + 2 * nkv) * dk), True),
        ("head split (view)", 0, 0, False),
        ("scores Q·Kᵀ", 2 * B * h * N * N * dk, b * (B * h * N * dk + B * nkv * N * dk + B * h * N * N), True),
        ("scale + mask", B * h * N * N, 2 * b * B * h * N * N, False),
        ("softmax", 5 * B * h * N * N, 2 * b * B * h * N * N, False),
        ("A·V", 2 * B * h * N * N * dk, b * (B * h * N * N + B * nkv * N * dk + B * h * N * dk), True),
        ("merge heads (copy)", 0, 2 * b * BN * d, False),
        ("W_O", 2 * BN * h * dk * d, b * (BN * d + h * dk * d + BN * d), True),
        ("residual add 1", BN * d, 3 * b * BN * d, False),
        ("LN2", 7 * BN * d, 2 * b * BN * d, False),
        ("FFN up W1", 2 * BN * d * dff, b * (BN * d + d * dff + BN * dff), True),
        ("GELU", 0, 2 * b * BN * dff, False),
        ("FFN down W2", 2 * BN * dff * d, b * (BN * dff + dff * d + BN * d), True),
        ("residual add 2", BN * d, 3 * b * BN * d, False),
    ]
    flash = ("FlashAttention (fused scores..A·V)", 4 * B * h * N * N * dk + 6 * B * h * N * N,
             b * (2 * B * h * N * dk + 2 * B * nkv * N * dk), True)
    once = [
        ("embedding + positions", BN * d, b * (BN * d + N * d + BN * d), False),
        ("LN_f (last row)", 7 * B * d, 2 * b * B * d, False),
        ("LM head (last row)", 2 * B * d * V, b * (B * d + V * d + B * V), True),
    ]
    lm_all = ("LM head (all N rows)", 2 * BN * d * V, b * (BN * d + V * d + BN * V), True)
    return ops, flash, once, lm_all


def op_time(flops, nbytes, dev=AGX):
    tc, tm = flops / dev["peak"], nbytes / dev["bw"]
    return tc, tm, max(tc, tm), ("-" if tc == tm == 0 else "compute" if tc >= tm else "memory")


def prefill(c, N, B=1, dev=AGX):
    ops, flash, once, lm_all = prefill_ops(c, N, B)
    rows = []
    for name, f, by, mm in ops + [flash] + once + [lm_all]:
        tc, tm, t, bound = op_time(f, by, dev)
        rows.append(dict(op=name, flops=f, bytes=by, ai=(f / by if by else None), t_compute_ms=tc * 1e3,
                         t_memory_ms=tm * 1e3, t_ms=t * 1e3, bound=bound, matmul=mm))
    attn_names = {"scores Q·Kᵀ", "scale + mask", "softmax", "A·V"}
    layer_naive = [x for x in rows[:len(ops)]]
    layer_flash = [x for x in layer_naive if x["op"] not in attn_names] + [rows[len(ops)]]
    d, L = c["d"], c["L"]
    linear = 24 * B * N * d * d if c["n_kv"] == c["h"] and c["d_ff"] == 4 * d else None
    mm_layer = sum(x["flops"] for x in layer_naive if x["matmul"])
    attn_mm = 4 * B * N * N * d
    head_last = 2 * B * d * c["V"]
    total_mm = L * mm_layer + head_last
    res = dict(
        N=N, B=B, rows=rows,
        per_layer_matmul=mm_layer, per_layer_linear=mm_layer - attn_mm, per_layer_attn=attn_mm,
        linear_check=linear,
        total_matmul=total_mm, lm_head_last=head_last, lm_head_all=2 * B * N * d * c["V"],
        attn_share=attn_mm / mm_layer, ffn_share=16 * B * N * d * d / mm_layer if linear else None,
        layer_ms_naive=sum(x["t_ms"] for x in layer_naive),
        layer_ms_flash=sum(x["t_ms"] for x in layer_flash),
        attn_ms_naive=sum(x["t_ms"] for x in layer_naive if x["op"] in attn_names),
        attn_ms_flash=rows[len(ops)]["t_ms"],
        ttft_flops_s=total_mm / dev["peak"],
        weight_read_s=params(c)["total"] * c["b"] / dev["bw"],
        kv_written=B * kv_bytes_per_token(c) * N,
        kv_written_per_layer=B * kv_bytes_per_token(c) * N / L,
    )
    once_ms = sum(x["t_ms"] for x in rows[len(ops) + 1:len(ops) + 4])
    res["ttft_roofline_naive_s"] = (L * res["layer_ms_naive"] + once_ms) / 1e3
    res["ttft_roofline_flash_s"] = (L * res["layer_ms_flash"] + once_ms) / 1e3
    res["ttft_s"] = max(res["ttft_flops_s"], res["weight_read_s"])
    res["per_layer_elementwise"] = sum(x["flops"] for x in layer_naive if not x["matmul"])
    # GEMM arithmetic intensity with M = B*N rows: AI = 2M / (b + 4M/d)
    M = B * N
    res["gemm_ai"] = 2 * M / (c["b"] + 4 * M / d)
    return res


# ----------------------------------------------------------------------------
# Decode: one new token per sequence against t cached tokens
# ----------------------------------------------------------------------------
def decode(c, t, B=1, wb=None, kvb=None, dev=AGX):
    """One decode step. wb / kvb = bytes per weight / per cached number (default c['b'])."""
    wb = c["b"] if wb is None else wb
    kvb = c["b"] if kvb is None else kvb
    P = params(c)
    d, L, h, nkv, dk, dff, V = c["d"], c["L"], c["h"], c["n_kv"], c["d_k"], c["d_ff"], c["V"]
    per_layer = [
        ("QKV projection", 2 * B * d * (h + 2 * nkv) * dk, wb * P["W_QKV"], 0),
        ("scores q·Kᵀ", 2 * B * h * t * dk, 0, kvb * B * nkv * t * dk),
        ("A·V", 2 * B * h * t * dk, 0, kvb * B * nkv * t * dk),
        ("W_O", 2 * B * h * dk * d, wb * P["W_O"], 0),
        ("FFN up W1", 2 * B * d * dff, wb * P["W1"], 0),
        ("FFN down W2", 2 * B * dff * d, wb * P["W2"], 0),
    ]
    once = [("LM head (E, tied)", 2 * B * d * V, wb * P["emb"], 0),
            ("position table P (counted whole, as in the primer)", 0, wb * P["pos"], 0)]
    rows = [dict(op=n, flops=f, w_bytes=w, kv_bytes=k, scope="per layer") for n, f, w, k in per_layer]
    rows += [dict(op=n, flops=f, w_bytes=w, kv_bytes=k, scope="once") for n, f, w, k in once]
    w_flops = B * 2 * P["matmul"]
    a_flops = B * 4 * L * t * h * dk
    w_bytes = wb * P["total"]
    kv_bytes = B * kvb * kv_bytes_per_token(c, b=1) * t
    flops, nbytes = w_flops + a_flops, w_bytes + kv_bytes
    tc, tm, tt, bound = op_time(flops, nbytes, dev)
    assert abs(sum(x["flops"] * (L if x["scope"] == "per layer" else 1) for x in rows) - flops) < 1
    assert abs(sum((x["w_bytes"] + x["kv_bytes"]) * (L if x["scope"] == "per layer" else 1)
                   for x in rows) - nbytes) < 1
    return dict(t=t, B=B, wb=wb, kvb=kvb, rows=rows, w_flops=w_flops, a_flops=a_flops, flops=flops,
                w_bytes=w_bytes, kv_bytes=kv_bytes, bytes=nbytes, ai=flops / nbytes,
                attn_ai=a_flops / kv_bytes,                  # = h / n_kv for an FP16 cache
                gemm_ai=2 * B / (wb + 4 * B / d),
                t_compute_ms=tc * 1e3, t_memory_ms=tm * 1e3, tpot_ms=tt * 1e3, bound=bound,
                tok_s=B / tt, mem=w_bytes + kv_bytes)


def ridge_batch(c, dev=AGX):
    """Smallest M with 2M / (b + 4M/d) >= ridge (GEMMs only, KV reads ignored)."""
    ridge = dev["peak"] / dev["bw"]
    return ridge / (1 - 2 * ridge / c["d"]) * c["b"] / 2


# ----------------------------------------------------------------------------
# Small hand-checkable examples
# ----------------------------------------------------------------------------
def softmax(xs):
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [v / s for v in e]


def matvec(x, W, cols):
    return [sum(x[i] * W[i][j] for i in range(len(x))) for j in cols]


def toy_reshape():
    """Multi-head split with the primer's numbers (step_by_step.py --json: keys xh, W_Q, Q,
    concat, W_O, attn_out). Inputs are the rounded values the page shows."""
    xh = [-1.59, 0.65, 1.03, -0.09]                      # x-hat of "sat" (LN1 output)
    W_Q = [[-0.2, -0.1, 0.3, 0.4], [0.9, 0.3, -0.2, 0.0], [0.4, 0.1, 0.1, 0.0], [0.1, -0.6, 0.4, 0.9]]
    head0 = matvec(xh, W_Q, [0, 1])
    head1 = matvec(xh, W_Q, [2, 3])
    terms0 = [xh[i] * W_Q[i][0] for i in range(4)]
    lossy0 = matvec(xh[:2], [row[:2] for row in W_Q[:2]], [0, 1])  # slicing x first (not used):
    lossy1 = matvec(xh[2:], [row[2:] for row in W_Q[2:]], [0, 1])  # = a block-diagonal W_Q
    Q_primer = [1.31, 0.51, -0.54, -0.72]                # step_by_step.py --json, Q[2]
    assert all(abs(r(a) - e) < 1e-9 for a, e in zip(head0 + head1, Q_primer))
    concat = [1.5, -0.08, 0.34, -1.57]                   # concat[2]: [head 0 | head 1] of "sat"
    W_O = [[0.2, 0.4, 1.0, -0.2], [-0.5, -0.6, 0.3, -0.7], [-0.5, -0.2, 0.1, -0.5], [0.2, 1.0, -0.5, -0.4]]
    out0 = [sum(concat[i] * W_O[i][j] for i in range(0, 2)) for j in range(4)]
    out1 = [sum(concat[i] * W_O[i][j] for i in range(2, 4)) for j in range(4)]
    attn_out_primer = [-0.15, -0.99, 2.29, 0.21]         # attn_out[2], from unrounded inputs
    assert all(abs(a + b_ - e) < 0.011 for a, b_, e in zip(out0, out1, attn_out_primer))
    return dict(xh=xh, head0=[r(v) for v in head0], head1=[r(v) for v in head1],
                col0_terms=[r(v, 3) for v in terms0], col0=r(sum(terms0), 3),
                Q=[r(v) for v in head0 + head1], lossy_head0=[r(v) for v in lossy0],
                lossy_head1=[r(v) for v in lossy1],
                wo_head0=[r(v) for v in out0], wo_head1=[r(v) for v in out1],
                attn_out_primer=attn_out_primer,
                params_split=dict(per_head=4 * 2, h=2, total=2 * 4 * 2, fused=4 * 4))


def attention_example():
    """The guide's one-head example (q of "sat", d_k = 4); labs/check.py uses it too."""
    q = [1.0, 0.0, 1.0, 0.0]
    K = [[0.0, 1.0, -0.5, 0.0], [0.9, 0.1, 0.8, 0.2], [0.1, 0.0, 0.2, 1.0]]
    Vv = [[0.1, 0.9, -0.2, 0.4], [0.6, -0.1, 0.8, 0.3], [0.3, 0.5, 0.1, -0.2]]
    s = [sum(a * b_ for a, b_ in zip(q, k)) for k in K]
    sc = [x / 2 for x in s]
    e = [math.exp(x) for x in sc]
    w = [x / sum(e) for x in e]
    o = [sum(w[j] * Vv[j][c] for j in range(3)) for c in range(4)]
    return dict(scores=[r(x) for x in s], scaled=[r(x) for x in sc], exp=[r(x) for x in e],
                exp_sum=r(sum(e)), weights=[r(x) for x in w], out=[r(x) for x in o])


def scale_example():
    raw = [8.0, 0.0]
    z = [1.2, -0.3, 0.5, -1.1, 0.8, 0.1, -0.6, 0.3]       # the sqrt(d_k) widget's scaled scores
    widget = {dk: r(max(softmax([x * math.sqrt(dk) for x in z]))) for dk in (4, 16, 64, 128, 512)}
    return dict(unscaled=[r(x, 4) for x in softmax(raw)], scaled=[r(x) for x in softmax([1.0, 0.0])],
                std_dk={dk: r(math.sqrt(dk), 1) for dk in (64, 128, 512)},
                widget_unscaled_max=widget, widget_scaled_max=r(max(softmax(z))))


def online_softmax_example():
    S, Vv = [1.0, 3.0, 2.0, 5.0], [10.0, 20.0, 30.0, 40.0]
    m, l, acc, steps = -math.inf, 0.0, 0.0, []
    for i in (0, 2):
        s_blk, v_blk = S[i:i + 2], Vv[i:i + 2]
        m_new = max(m, max(s_blk))
        alpha = math.exp(m - m_new) if m != -math.inf else 0.0
        p = [math.exp(x - m_new) for x in s_blk]
        l = l * alpha + sum(p)
        acc = acc * alpha + sum(pi * vi for pi, vi in zip(p, v_blk))
        m = m_new
        steps.append(dict(m=m, alpha=r(alpha, 3), p=[r(x, 3) for x in p], l=r(l, 3), acc=r(acc), out=r(acc / l)))
    w = softmax(S)
    direct = sum(wi * vi for wi, vi in zip(w, Vv))
    assert abs(direct - acc / l) < 1e-12
    return dict(steps=steps, direct_w=[r(x, 3) for x in w], direct=r(direct))


def apple_counts(n_prompt=5, n_steps=15):
    """Token-rows through the GEMMs and causal score pairs (per head, per layer)."""
    tri = lambda n: n * (n + 1) // 2
    lens = [n_prompt + k for k in range(n_steps + 1)]          # sequence length at each pass
    return dict(rows_nocache=sum(lens), rows_cache=n_prompt + n_steps,
                pairs_nocache=sum(tri(n) for n in lens), pairs_cache=tri(n_prompt) + sum(lens[1:]),
                last_pairs_nocache=tri(lens[-1]), last_pairs_cache=lens[-1], final_len=lens[-1])


# ----------------------------------------------------------------------------
# Everything the guide prints
# ----------------------------------------------------------------------------
def compute():
    J = {}
    Pr, Pg, P8 = params(REF), params(GPT2), params(GQA8)
    J["devices"] = {k: dict(v, ridge=r(v["peak"] / v["bw"], 1)) for k, v in DEVICES.items()}
    J["ref"] = dict(REF, params=Pr, total_b=r(Pr["total"] / 1e9), weight_gb=r(Pr["total"] * 2 / GB),
                    weight_gib=r(Pr["total"] * 2 / GiB), check_12Ld2_Vd=12 * REF["L"] * REF["d"] ** 2 + REF["V"] * REF["d"])
    gpt2_total = Pg["total"] + REF["L"] * 0                     # analytic, without biases/norms
    # transformer.py's GPT-2 small also has FFN biases (d_ff + d per block) and LayerNorms (4d per
    # block + 2d final): count_parameters() = 124,402,944.
    gpt2_full = Pg["total"] + GPT2["L"] * (GPT2["d_ff"] + GPT2["d"] + 4 * GPT2["d"]) + 2 * GPT2["d"]
    J["gpt2"] = dict(GPT2, params=Pg, total_no_bias=gpt2_total, total_transformer_py=gpt2_full,
                     tying_saves=Pg["emb"])
    J["gqa8_params"] = P8["total"]
    J["gqa8_saved"] = Pr["total"] - P8["total"]

    # §2 prefill
    pf = prefill(REF, N_REF)
    J["prefill"] = pf
    J["prefill_by_N"] = []
    for N in (128, 512, 1024, 2048, 4096, 8192):
        p = prefill(REF, N)
        J["prefill_by_N"].append(dict(N=N, tflop=r(p["total_matmul"] / 1e12, 2), attn_share=r(100 * p["attn_share"], 1),
                                      t_compute_ms=r(p["ttft_flops_s"] * 1e3, 0), t_weights_ms=r(p["weight_read_s"] * 1e3, 0),
                                      bound="compute" if p["ttft_flops_s"] >= p["weight_read_s"] else "memory",
                                      gemm_ai=r(p["gemm_ai"], 0)))
    g = prefill(GPT2, 1024)
    J["gpt2_prefill"] = dict(linear_gflop=r(g["per_layer_linear"] / 1e9, 1), attn_gflop=r(g["per_layer_attn"] / 1e9, 1),
                             layer_gflop=r(g["per_layer_matmul"] / 1e9, 1), layers_gflop=r(12 * g["per_layer_matmul"] / 1e9, 1),
                             lm_all_gflop=r(g["lm_head_all"] / 1e9, 1),
                             lm_share=r(100 * g["lm_head_all"] / (12 * g["per_layer_matmul"] + g["lm_head_all"]), 0))
    J["ffn_share_N2048"] = r(100 * 16 * 4096 / (24 * 4096 + 4 * 2048), 1)
    J["flash_stats"] = dict(row_stats_mib=r(REF["h"] * N_REF * 4 / MiB, 2), O_mib=r(N_REF * REF["d"] * 2 / MiB))

    # §3 decode
    dec = decode(REF, N_REF)
    J["decode"] = dec
    J["decode_variants"] = []
    for label, c, wb, kvb in (("FP16 weights, MHA (reference)", REF, 2, 2), ("INT8 weights", REF, 1, 2),
                              ("INT4 weights", REF, 0.5, 2), ("INT4 weights + GQA (n_kv = 8)", GQA8, 0.5, 2),
                              ("INT4 + GQA-8 + INT8 KV cache", GQA8, 0.5, 1)):
        x = decode(c, N_REF, wb=wb, kvb=kvb)
        J["decode_variants"].append(dict(label=label, w_gb=r(x["w_bytes"] / GB), kv_gb=r(x["kv_bytes"] / GB),
                                         gb=r(x["bytes"] / GB), tpot_ms=r(x["tpot_ms"], 1), tok_s=r(x["tok_s"], 1)))
    J["weights_only_tok_s"] = dict(fp16=r(AGX["bw"] / (Pr["total"] * 2), 1), int4=r(AGX["bw"] / (Pr["total"] * 0.5), 1))
    J["batching"] = []
    for c in (REF, GQA8):
        for B in (1, 4, 8, 16, 32):
            x = decode(c, N_REF, B=B)
            J["batching"].append(dict(cfg="MHA" if c is REF else "GQA-8", B=B, gb=r(x["bytes"] / GB),
                                      ai=r(x["ai"], 1), tpot_ms=r(x["tpot_ms"], 1), tok_s=r(x["tok_s"], 0),
                                      mem_gb=r(x["mem"] / GB, 1), fits_agx=x["mem"] <= AGX["mem"]))
    J["max_batch_agx"] = {k: int((AGX["mem"] - params(c)["total"] * 2) // (kv_bytes_per_token(c) * N_REF))
                          for k, c in (("MHA", REF), ("GQA-8", GQA8))}
    J["ridge_batch"] = r(ridge_batch(REF), 0)
    J["batch_ai_limit"] = {k: r(decode(c, N_REF)["flops"] / decode(c, N_REF)["kv_bytes"], 1)   # B -> infinity
                           for k, c in (("MHA", REF), ("GQA-8", GQA8))}
    J["attn_ai"] = {k: r(decode(c, N_REF)["a_flops"] / decode(c, N_REF)["kv_bytes"], 1)
                    for k, c in (("MHA", REF), ("GQA-8", GQA8), ("MQA", MQA))}
    # end-to-end latency for a 4096-token prompt and 100 output tokens
    n_out = 100
    tpots = [decode(REF, N_REF + k)["tpot_ms"] for k in range(1, n_out)]
    J["latency"] = dict(N=N_REF, n_out=n_out, ttft_s=r(pf["ttft_s"], 2), decode_s=r(sum(tpots) / 1e3, 2),
                        decode_pct=r(100 * sum(tpots) / 1e3 / (pf["ttft_s"] + sum(tpots) / 1e3), 0),
                        mean_tpot_ms=r(sum(tpots) / len(tpots), 1), total_s=r(pf["ttft_s"] + sum(tpots) / 1e3, 2))
    J["apple"] = apple_counts()

    # §4 roofline
    d = REF["d"]
    J["roofline"] = dict(ridge_agx=r(AGX["peak"] / AGX["bw"], 1), ridge_nano=r(DEVICES["nano"]["peak"] / DEVICES["nano"]["bw"], 1),
                         ai={M: r(2 * M / (2 + 4 * M / d), 2) for M in (1, 8, 512, 4096)},
                         ai_limit=d // 2,
                         attainable_m1_tflops=r(2 / (2 + 4 / d) * AGX["bw"] / 1e12, 3),
                         pct_peak_m1=r(100 * 2 / (2 + 4 / d) * AGX["bw"] / AGX["peak"], 2),
                         decode_vs_compute=r(dec["t_memory_ms"] / dec["t_compute_ms"], 0))

    # §5 memory
    b, N, h, V = REF["b"], N_REF, REF["h"], REF["V"]
    J["memory"] = dict(
        weights_gb={"FP16": r(Pr["total"] * 2 / GB), "INT8": r(Pr["total"] * 1 / GB), "INT4": r(Pr["total"] * 0.5 / GB)},
        weights_gib_fp16=r(Pr["total"] * 2 / GiB),
        kv_per_token_kib={k: kv_bytes_per_token(c) / KiB for k, c in (("MHA", REF), ("GQA-8", GQA8), ("MQA", MQA))},
        kv_at_t_mib={k: kv_bytes_per_token(c) * N / MiB for k, c in (("MHA", REF), ("GQA-8", GQA8), ("MQA", MQA))},
        kv_at_t_gb=r(kv_bytes_per_token(REF) * N / GB),
        act_prefill_mib=dict(X=N * d * b / MiB, QKV=3 * N * d * b / MiB, scores=h * N * N * b / MiB,
                             ffn_hidden=4 * N * d * b / MiB, logits_all=r(N * V * b / MiB, 1),
                             logits_last_kib=r(V * b / KiB, 1)),
        peak_naive_gib=r((N * d * b + 3 * N * d * b + 2 * h * N * N * b) / GiB, 2),
        peak_flash_mib=(2 * N * d * b + 2 * 4 * N * d * b) / MiB,
        act_decode_kib=dict(x=d * b / KiB, scores=h * N * b / KiB),
        logits_2048_mib=r(2048 * V * b / MiB, 1),
    )
    fits = []
    for label, c, wb in (("FP16, MHA", REF, 2), ("INT4, MHA", REF, 0.5), ("INT4, GQA-8", GQA8, 0.5)):
        need = params(c)["total"] * wb + kv_bytes_per_token(c) * N
        fits.append(dict(label=label, gb=r(need / GB), agx_pct=r(100 * need / AGX["mem"], 0),
                         nano_pct=r(100 * need / DEVICES["nano"]["mem"], 0)))
    J["memory"]["fits"] = fits
    J["kv_examples"] = dict(
        toy_bytes=2 * 2 * 3 * 2 * 4 * 2,
        gpt2_1024_mib=kv_bytes_per_token(GPT2) * 1024 / MiB,
        llama31_8b_per_token_kib=2 * 32 * 8 * 128 * 2 / KiB,
        llama31_8b_8192_gib=2 * 32 * 8 * 128 * 2 * 8192 / GiB,
        llama2_7b_per_token_kib=2 * 32 * 32 * 128 * 2 / KiB,
        llama2_7b_4096_gib=2 * 32 * 32 * 128 * 2 * 4096 / GiB,
        gqa8_4096_mib=2 * 32 * 8 * 128 * 2 * 4096 / MiB,
    )

    # §7 training
    tok = 1e9
    J["training"] = dict(
        flop_per_token_gflop=r(6 * Pr["total"] / 1e9, 1), fwd_per_token_gflop=r(2 * Pr["matmul"] / 1e9, 1),
        model_states_gb=r(16 * Pr["total"] / GB, 1),
        act_layer_naive_gb=r(N * d * (34 + 5 * h * N / d) / GB, 2), act_layer_noscores_gb=r(N * d * 34 / GB, 2),
        act_layer_ckpt_mb=r(2 * N * d / MB, 1),
        act_all_naive_gb=r(REF["L"] * N * d * (34 + 5 * h * N / d) / GB, 1),
        act_all_noscores_gb=r(REF["L"] * N * d * 34 / GB, 1), act_all_ckpt_gb=r(REF["L"] * 2 * N * d / GB, 2),
        gpt2_flop_per_token_gflop=r(6 * gpt2_full / 1e9, 2),
        gpt2_1b_tokens_flop=6 * gpt2_full * tok, gpt2_1b_tokens_hours=r(6 * gpt2_full * tok / AGX["peak"] / 3600, 1),
    )

    # numbers quoted in the guide's "GPU kernel deep dive" drop-downs (reference config, B = 1)
    N, d, b, dff, L = N_REF, REF["d"], REF["b"], REF["d_ff"], REF["L"]
    nt = N // 128                                              # 128 x 128 attention tiles
    unfused_ops = [x for x in prefill_ops(REF, N)[0] if x[0] != "head split (view)"]
    J["kernels"] = dict(
        qkv_fused_saves_mib=2 * N * d * b / MiB,               # X-hat read once instead of 3 times
        w_qkv_mib={"FP16": Pr["W_QKV"] * 2 / MiB, "INT4": Pr["W_QKV"] * 0.5 / MiB},
        contiguous_copy_mib=2 * N * d * b / MiB,               # read + write one [B, N, d] tensor
        ln_values_per_lane=d // 32,                            # one row per warp of 32 lanes
        add_norm_saves_mib=N * d * b / MiB,                    # one read of the residual stream
        add_norm_saves_decode_kib=d * b / KiB,
        wo_residual_epilogue_saves_mib=2 * N * d * b / MiB,    # Y is never written and re-read
        gelu_epilogue_saves_mib=2 * N * dff * b / MiB,         # H is never written and re-read
        causal_tiles=dict(tile=128, total=nt * nt, needed=nt * (nt + 1) // 2, diagonal=nt,
                          skipped_pct=r(100 * (1 - (nt * (nt + 1) // 2) / (nt * nt)), 1)),
        decode_attn_blocks=dict(MHA=REF["h"], GQA8_per_kv_head=GQA8["n_kv"], agx_sms=16, a100_sms=108),
        kv_per_head_layer_mib=N_REF * REF["d_k"] * b / MiB,    # one head's K at t = 4096
        ffn_share_of_weight_bytes=r(100 * L * Pr["ffn"] / Pr["total"], 1),
        lm_head_share_of_decode_bytes=r(100 * Pr["emb"] * 2 / dec["bytes"], 1),
        launches=dict(unfused=len(unfused_ops) * L + 3, fused=7 * L + 3),
    )

    # toy examples
    J["reshape"] = toy_reshape()
    J["attention_example"] = attention_example()
    J["scale_example"] = scale_example()
    J["online_softmax"] = online_softmax_example()
    J["selfcheck"] = dict(d_k=4096 // 32, scores_shape=[2, 32, 128, 128],
                          head3_columns=[3 * REF["d_k"], 4 * REF["d_k"] - 1], head3_shape=[REF["d"], REF["d_k"]])
    return J


def fmt_bytes(x):
    for div, u in ((GiB, "GiB"), (MiB, "MiB"), (KiB, "KiB")):
        if x >= div:
            return f"{x / div:.2f} {u}"
    return f"{x:.0f} B"


def fmt_flops(x):
    for div, u in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f} {u}"
    return f"{x:.0f}"


def show(J):
    P = J["ref"]["params"]
    line = lambda s="": print(s)
    head = lambda s: print("\n" + s + "\n" + "-" * len(s))
    agx, nano = J["devices"]["agx"], J["devices"]["nano"]
    head("Devices")
    for dv in (agx, nano):
        line(f"  {dv['name']}: {dv['bw'] / 1e9:g} GB/s, {dv['peak'] / 1e12:g} TFLOPS FP16 dense, "
             f"ridge = {dv['ridge']} FLOP/B, {dv['mem'] / 1e9:g} GB")

    head("§1 Reference config and parameters (no biases or norms; tied LM head)")
    c = J["ref"]
    line(f"  d={c['d']} L={c['L']} h={c['h']} n_kv={c['n_kv']} d_k={c['d_k']} d_ff={c['d_ff']} V={c['V']} "
         f"positions={c['n_pos']} b={c['b']}")
    for k in ("W_Q", "W_KV", "W_QKV", "W_O", "W1", "W2", "attn", "ffn", "block", "norms_per_block",
              "blocks", "emb", "pos", "total", "matmul"):
        line(f"  {k:<16}{P[k]:>16,}")
    line(f"  12*L*d^2 + V*d = {c['check_12Ld2_Vd']:,}   total = {c['total_b']} B params, "
         f"FP16 weights = {c['weight_gb']} GB = {c['weight_gib']} GiB")
    g = J["gpt2"]
    line(f"  GPT-2 small: {g['total_no_bias']:,} without biases/norms; transformer.py count_parameters() = "
         f"{g['total_transformer_py']:,}; weight tying saves V*d = {g['tying_saves']:,}")
    line(f"  GQA (n_kv = 8) variant: {J['gqa8_params']:,} params ({J['gqa8_saved']:,} fewer: W_K, W_V are [d, n_kv*d_k])")

    head("§2 Toy multi-head split (primer numbers, d = 4, h = 2, d_k = 2)")
    t = J["reshape"]
    line(f"  x̂[sat] = {t['xh']}")
    line(f"  column 0 terms = {t['col0_terms']} -> sum {t['col0']}")
    line(f"  head 0 = x̂·W_Q[:, 0:2] = {t['head0']}; head 1 = x̂·W_Q[:, 2:4] = {t['head1']}; Q[sat] = {t['Q']}")
    line(f"  lossy (slice x first = block-diagonal W_Q): head 0 = x̂[0:2]·W_Q[0:2, 0:2] = {t['lossy_head0']} "
         f"(ignores x̂[2], x̂[3]); head 1 = x̂[2:4]·W_Q[2:4, 2:4] = {t['lossy_head1']}")
    line(f"  output side: [1.50, -0.08]·W_O[0:2] = {t['wo_head0']}, [0.34, -1.57]·W_O[2:4] = {t['wo_head1']}, "
         f"sum ≈ attn_out[sat] = {t['attn_out_primer']}")
    line(f"  params: h·d·d_k = 2·4·2 = {t['params_split']['total']} = d² = {t['params_split']['fused']}")

    head("§2 One-head attention example (q of 'sat', d_k = 4)")
    a = J["attention_example"]
    line(f"  scores {a['scores']} -> /2 {a['scaled']} -> exp {a['exp']} (sum {a['exp_sum']}) -> "
         f"weights {a['weights']} -> out {a['out']}")
    s = J["scale_example"]
    line(f"  scale: softmax([8, 0]) = {s['unscaled']}; softmax([1, 0]) = {s['scaled']}; "
         f"std = sqrt(d_k): {s['std_dk']}")
    line(f"  sqrt(d_k) widget: max weight unscaled {s['widget_unscaled_max']}, scaled {s['widget_scaled_max']}")
    o = J["online_softmax"]
    for i, st in enumerate(o["steps"], 1):
        line(f"  online softmax tile {i}: m={st['m']:g} alpha={st['alpha']} p={st['p']} l={st['l']} acc={st['acc']} acc/l={st['out']}")
    line(f"  direct: softmax = {o['direct_w']} -> {o['direct']}")

    pf = J["prefill"]
    head(f"§2 Prefill, reference config, B = 1, N = {pf['N']}, FP16, {agx['name']} (unfused byte model)")
    line(f"  {'op':<36}{'FLOPs':>11}{'bytes':>12}{'AI':>9}{'t_c ms':>9}{'t_m ms':>9}  bound")
    for x in pf["rows"]:
        ai = f"{x['ai']:.0f}" if x["ai"] and x["ai"] >= 10 else (f"{x['ai']:.2f}" if x["ai"] else "-")
        line(f"  {x['op']:<36}{fmt_flops(x['flops']):>11}{fmt_bytes(x['bytes']):>12}{ai:>9}"
             f"{x['t_compute_ms']:>9.2f}{x['t_memory_ms']:>9.2f}  {x['bound']}")
    line(f"  per layer: matmuls {fmt_flops(pf['per_layer_matmul'])}FLOP = linear 24·N·d² {fmt_flops(pf['per_layer_linear'])} "
         f"+ attention 4·N²·d {fmt_flops(pf['per_layer_attn'])} (attention share {100 * pf['attn_share']:.1f}%, "
         f"FFN share {100 * pf['ffn_share']:.1f}%)")
    line(f"  all L layers + LM head on the last row: {pf['total_matmul'] / 1e12:.2f} TFLOP "
         f"(LM head last row {pf['lm_head_last'] / 1e9:.2f} GFLOP; all rows {pf['lm_head_all'] / 1e12:.2f} TFLOP)")
    line(f"  elementwise ops (LN, scale + mask, softmax, residual adds) per layer: {pf['per_layer_elementwise'] / 1e9:.2f} GFLOP")
    ffn_ms = sum(x["t_ms"] for x in pf["rows"] if x["op"].startswith("FFN"))
    lm = next(x for x in pf["rows"] if x["op"] == "LM head (last row)")
    line(f"  FFN W1 + W2: {ffn_ms:.1f} ms per layer; LM head reads all of E: {fmt_bytes(lm['bytes'])} (≈ {lm['bytes'] / MiB:.0f} MiB)")
    line(f"  FP16 overflows above 65,504 = exp({math.log(65504):.1f}); max-subtracted scores are <= 0, so exp() <= 1")
    line(f"  TTFT >= FLOPs / peak = {pf['ttft_flops_s']:.2f} s; weights read once = {pf['weight_read_s'] * 1e3:.0f} ms "
         f"-> compute-bound; GEMM AI with M = {pf['N']}: {pf['gemm_ai']:.0f}")
    line(f"  roofline sum per layer: naive attention {pf['layer_ms_naive']:.1f} ms (attention ops {pf['attn_ms_naive']:.1f} ms), "
         f"FlashAttention {pf['layer_ms_flash']:.1f} ms (attention {pf['attn_ms_flash']:.1f} ms)")
    line(f"  TTFT roofline estimate: naive {pf['ttft_roofline_naive_s']:.2f} s, FlashAttention {pf['ttft_roofline_flash_s']:.2f} s")
    line(f"  KV cache written: {fmt_bytes(pf['kv_written_per_layer'])} per layer, {fmt_bytes(pf['kv_written'])} "
         f"= {pf['kv_written'] / GB:.2f} GB in all")
    fs = J["flash_stats"]
    line(f"  FlashAttention row statistics (m, l in FP32): {fs['row_stats_mib']} MiB each per layer; O = {fs['O_mib']} MiB")
    line(f"  {'N':>6}{'TFLOP':>9}{'attn %':>8}{'t_compute ms':>14}{'t_weights ms':>14}  bound   GEMM AI")
    for x in J["prefill_by_N"]:
        line(f"  {x['N']:>6}{x['tflop']:>9}{x['attn_share']:>8}{x['t_compute_ms']:>14.0f}{x['t_weights_ms']:>14.0f}  {x['bound']:<8}{x['gemm_ai']:.0f}")
    gp = J["gpt2_prefill"]
    line(f"  GPT-2 small, N = 1024: per layer {gp['linear_gflop']} (linear) + {gp['attn_gflop']} (attention) = "
         f"{gp['layer_gflop']} GFLOP; x12 = {gp['layers_gflop']}; LM head on all rows {gp['lm_all_gflop']} GFLOP "
         f"({gp['lm_share']:.0f}% of the forward pass)")
    line(f"  FFN share at N = 2048, d = 4096: {J['ffn_share_N2048']}%")

    d = J["decode"]
    head(f"§3 Decode, reference config, B = 1, t = {d['t']} cached tokens, FP16")
    line(f"  {'op (per layer unless noted)':<52}{'FLOPs':>11}{'weight B':>12}{'KV B':>12}")
    for x in d["rows"]:
        line(f"  {x['op'] + (' [once]' if x['scope'] == 'once' else ''):<52}{fmt_flops(x['flops']):>11}"
             f"{fmt_bytes(x['w_bytes']) if x['w_bytes'] else '-':>12}{fmt_bytes(x['kv_bytes']) if x['kv_bytes'] else '-':>12}")
    ffn_dec = sum(x["flops"] for x in d["rows"] if x["op"].startswith("FFN"))
    line(f"  FFN per layer: {ffn_dec / 1e6:.2f} MFLOP over {fmt_bytes(sum(x['w_bytes'] for x in d['rows'] if x['op'].startswith('FFN')))}")
    line(f"  FLOPs = 2·params_matmul {d['w_flops'] / 1e9:.2f} G + 4·L·t·d {d['a_flops'] / 1e9:.2f} G = {d['flops'] / 1e9:.2f} GFLOP")
    line(f"  bytes = weights {d['w_bytes'] / GB:.2f} GB + KV {d['kv_bytes'] / GB:.2f} GB = {d['bytes'] / GB:.2f} GB; "
         f"AI = {d['ai']:.2f} FLOP/B; GEMV AI = {d['gemm_ai']:.2f}")
    line(f"  t_compute = {d['t_compute_ms']:.2f} ms, t_memory = {d['t_memory_ms']:.1f} ms -> TPOT >= {d['tpot_ms']:.1f} ms, "
         f"{d['tok_s']:.1f} tok/s ({d['bound']}-bound; memory/compute time = {J['roofline']['decode_vs_compute']:.0f}x)")
    line(f"  weights-only ceiling: FP16 {J['weights_only_tok_s']['fp16']} tok/s, INT4 {J['weights_only_tok_s']['int4']} tok/s")
    line("  variants at B = 1, t = 4096:")
    for x in J["decode_variants"]:
        line(f"    {x['label']:<34} weights {x['w_gb']:>5} GB + KV {x['kv_gb']:>4} GB = {x['gb']:>5} GB -> "
             f"TPOT {x['tpot_ms']:>5} ms, {x['tok_s']:>5} tok/s")
    line("  batching, every sequence at t = 4096:")
    for x in J["batching"]:
        line(f"    {x['cfg']:<6} B={x['B']:>3}: bytes {x['gb']:>6} GB, AI {x['ai']:>5}, TPOT {x['tpot_ms']:>6} ms, "
             f"{x['tok_s']:>5.0f} tok/s total, memory {x['mem_gb']:>5} GB {'fits' if x['fits_agx'] else 'does NOT fit'} in 64 GB")
    line(f"  max B at t = 4096 in 64 GB (no OS reserve): {J['max_batch_agx']}")
    line(f"  rows per weight read to reach the AGX ridge (GEMMs only): M ≈ {J['ridge_batch']:.0f}; "
         f"whole-step AI as B grows without bound (FLOPs / KV bytes per sequence): {J['batch_ai_limit']}")
    line(f"  attention-over-cache AI (FP16) = h / n_kv: {J['attn_ai']}")
    la = J["latency"]
    line(f"  latency, N = {la['N']} prompt, {la['n_out']} output tokens: TTFT {la['ttft_s']} s + 99 decode steps "
         f"{la['decode_s']} s (mean TPOT {la['mean_tpot_ms']} ms) = {la['total_s']} s, {la['decode_pct']:.0f}% decode")
    ap = J["apple"]
    line(f"  'What color is apple?' (5 prompt tokens, 15 decode steps, {ap['final_len']} tokens at the end): "
         f"GEMM token-rows {ap['rows_nocache']} -> {ap['rows_cache']}, causal pairs {ap['pairs_nocache']:,} -> "
         f"{ap['pairs_cache']}; last step {ap['last_pairs_nocache']} -> {ap['last_pairs_cache']}")

    rf = J["roofline"]
    head("§4 Roofline (one d x d GEMM, d = 4096, FP16: AI = 2M / (b + 4M/d))")
    line(f"  ridge: AGX Orin {rf['ridge_agx']}, Orin Nano Super {rf['ridge_nano']} FLOP/B; AI(M) = {rf['ai']}; "
         f"limit d/2 = {rf['ai_limit']}")
    line(f"  M = 1: attainable {rf['attainable_m1_tflops']} TFLOPS = {rf['pct_peak_m1']}% of 42.5")

    m = J["memory"]
    head("§5 Memory, reference config")
    line(f"  weights: {m['weights_gb']} GB (FP16 = {m['weights_gib_fp16']} GiB)")
    line(f"  KV per token (KiB): {m['kv_per_token_kib']}; at t = 4096 (MiB): {m['kv_at_t_mib']} = {m['kv_at_t_gb']} GB for MHA")
    line(f"  prefill activations, N = 4096, one layer (MiB): {m['act_prefill_mib']}")
    line(f"  live set: naive (X, Q|K|V, S, A) {m['peak_naive_gib']} GiB; FlashAttention, inside the FFN "
         f"(X, LN2 out, H, GELU(H)) {m['peak_flash_mib']:.0f} MiB")
    line(f"  decode activations (KiB): {m['act_decode_kib']}; full logits at N = 2048: {m['logits_2048_mib']} MiB")
    for x in m["fits"]:
        line(f"  {x['label']:<12} weights + KV at t = 4096: {x['gb']} GB = {x['agx_pct']:.0f}% of AGX Orin 64 GB, "
             f"{x['nano_pct']:.0f}% of Orin Nano 8 GB")
    k = J["kv_examples"]
    line(f"  KV examples: toy {k['toy_bytes']} B; GPT-2 small at 1024: {k['gpt2_1024_mib']:g} MiB; "
         f"Llama 3.1 8B {k['llama31_8b_per_token_kib']:g} KiB/token, {k['llama31_8b_8192_gib']:g} GiB at 8192; "
         f"Llama 2 7B {k['llama2_7b_per_token_kib']:g} KiB/token, {k['llama2_7b_4096_gib']:g} GiB at 4096 "
         f"(GQA-8: {k['gqa8_4096_mib']:g} MiB)")

    kd = J["kernels"]
    head("§2 GPU kernel deep dives (reference config, B = 1, N = t = 4096)")
    line(f"  fused QKV saves re-reading X̂ twice: {kd['qkv_fused_saves_mib']:g} MiB per layer; W_QKV {kd['w_qkv_mib']} MiB")
    line(f"  .contiguous() copy of one [B, N, d] tensor: {kd['contiguous_copy_mib']:g} MiB; LN one row per warp: "
         f"{kd['ln_values_per_lane']} values per lane")
    line(f"  fused add + norm saves one stream read: {kd['add_norm_saves_mib']:g} MiB (decode {kd['add_norm_saves_decode_kib']:g} KiB); "
         f"residual in W_O's epilogue saves {kd['wo_residual_epilogue_saves_mib']:g} MiB; GELU in W1's epilogue saves "
         f"{kd['gelu_epilogue_saves_mib']:g} MiB per layer")
    line(f"  causal tiles (128 x 128): {kd['causal_tiles']}")
    line(f"  decode attention thread blocks at B = 1: {kd['decode_attn_blocks']}; one head's K at t = 4096: "
         f"{kd['kv_per_head_layer_mib']:g} MiB")
    line(f"  FFN share of weight bytes {kd['ffn_share_of_weight_bytes']}%; LM head share of a decode step's bytes "
         f"{kd['lm_head_share_of_decode_bytes']}%; kernel launches per decode step: {kd['launches']}")

    sc = J["selfcheck"]
    line(f"\n  self-check: d_k = {sc['d_k']}; scores {sc['scores_shape']}; head 3's W^Q is {sc['head3_shape']}, "
         f"columns {sc['head3_columns'][0]}–{sc['head3_columns'][1]} of W_Q")
    tr = J["training"]
    head("§7 Training, reference config")
    line(f"  6·params = {tr['flop_per_token_gflop']} GFLOP per token (forward 2·params_matmul = {tr['fwd_per_token_gflop']})")
    line(f"  mixed-precision Adam: 16 B/param = {tr['model_states_gb']} GB")
    line(f"  activations per layer, N = 4096, B = 1 (N·B·d·(34 + 5·h·N/d) bytes): {tr['act_layer_naive_gb']} GB, "
         f"without stored scores {tr['act_layer_noscores_gb']} GB, checkpointed {tr['act_layer_ckpt_mb']} MB")
    line(f"  x32 layers: {tr['act_all_naive_gb']} GB naive, {tr['act_all_noscores_gb']} GB without scores, "
         f"{tr['act_all_ckpt_gb']} GB checkpointed")
    line(f"  GPT-2 small: {tr['gpt2_flop_per_token_gflop']} GFLOP per training token; 1e9 tokens = "
         f"{tr['gpt2_1b_tokens_flop']:.2e} FLOP = {tr['gpt2_1b_tokens_hours']} h at 42.5 TFLOPS")


def torch_checks(J):
    """Optional cross-checks against transformer.py (needs torch)."""
    try:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import transformer as T
    except Exception as e:                                   # torch missing: skip quietly
        return f"skipped ({type(e).__name__})"
    c, N = REF, N_REF
    f = T.compute_flops(1, N, c["d"], c["h"], c["d_ff"], c["L"], c["V"])
    pf = J["prefill"]
    mm = f["total"] - f["softmax"] - f["lm_head"]
    assert mm == c["L"] * pf["per_layer_matmul"], (mm, pf["per_layer_matmul"])
    assert f["lm_head"] == pf["lm_head_all"]
    m = T.Transformer(vocab_size=50257, d_model=768, n_heads=12, n_layers=12, max_seq_len=1024)
    assert m.count_parameters() == J["gpt2"]["total_transformer_py"], m.count_parameters()
    return "ok: prefill matmul FLOPs == compute_flops(); GPT-2 small count == count_parameters()"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true", help="print the numbers as JSON")
    args = ap.parse_args()
    J = compute()
    if args.json:
        print(json.dumps(J, indent=1, ensure_ascii=False, default=str))
        return
    show(J)
    print("\ntransformer.py cross-check:", torch_checks(J))


if __name__ == "__main__":
    main()
