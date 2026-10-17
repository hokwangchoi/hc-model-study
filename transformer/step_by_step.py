"""
Transformer, step by step: follow 3 tokens through a tiny GPT.

Companion to transformer_step_by_step.html. It builds the page's toy model with the
exact same weights and prints every number the page shows, in the same order and
rounding (weights have 1 decimal, everything computed is shown with 2 decimals).

Run:
    python step_by_step.py          # print steps 1-18
    python step_by_step.py --json   # the same numbers as JSON (used to check the page)

The toy model (one block, pre-norm, causal, like transformer.py but tiny):
    vocab V = 8 words, d_model = 4, 2 heads of d_k = 2, d_ff = 8, max length 8.
Simplifications vs transformer.py, to keep the numbers hand-checkable:
    - no biases anywhere, LayerNorm with gamma = 1 and beta = 0 (their initial values)
    - ReLU in the FFN (transformer.py and GPT-2 use GELU, a smooth ReLU)
    - an untied LM head U (transformer.py ties it to the embedding table: U = E^T)
The weights are multiples of 0.1, picked by a small random search so that greedy
decoding continues "the cat sat" with "on the mat". Nothing here is trained.
Needs only torch (>= 2.1) and the standard library. It computes in float64; float32
(PyTorch's default, as in the page's snippets) gives the same 2-decimal numbers.
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)  # exact enough that 2-decimal rounding is stable

# ----------------------------------------------------------------------------
# Config and weights (every weight is a multiple of 0.1)
# ----------------------------------------------------------------------------
VOCAB = [".", "cat", "dog", "mat", "on", "ran", "sat", "the"]
PROMPT = "the cat sat"
D, H, DK, DFF, NMAX = 4, 2, 2, 8, 8      # d_model, heads, d_k = D // H, d_ff, max length
V = len(VOCAB)
EPS = 1e-5                               # LayerNorm epsilon (PyTorch default)
N_NEW = 3                                # tokens to generate in step 17

E = torch.tensor([                       # token embedding table [V=8, D=4]
    [0.2, 0.4, 0.5, -0.6],               # .
    [0.0, -0.5, -0.6, 0.2],              # cat
    [-0.4, 0.0, 1.0, -0.4],              # dog
    [0.0, -0.6, 0.4, 0.5],               # mat
    [-0.4, 0.5, -0.5, 0.1],              # on
    [-0.2, 1.0, 0.3, -0.2],              # ran
    [-0.6, 0.3, 0.9, 0.2],               # sat
    [-0.2, -0.8, -0.3, -0.4],            # the
])
P = torch.tensor([                       # learned position table [NMAX=8, D=4]
    [-0.1, 0.2, 0.4, -0.3],
    [0.0, -0.5, 0.2, 0.4],
    [-0.1, 0.2, -0.2, -0.1],
    [0.4, 0.2, 0.1, -0.1],
    [0.0, -0.5, 0.1, -0.2],
    [-0.6, 0.1, -0.2, -0.1],
    [0.0, 0.2, 0.0, 0.0],
    [-0.2, -0.2, -0.2, 0.0],
])
W_Q = torch.tensor([[-0.2, -0.1, 0.3, 0.4], [0.9, 0.3, -0.2, 0.0], [0.4, 0.1, 0.1, 0.0], [0.1, -0.6, 0.4, 0.9]])
W_K = torch.tensor([[0.8, -0.1, 0.4, 0.2], [0.4, 0.2, -0.5, -0.3], [0.2, -0.4, -0.2, -0.2], [1.0, -0.2, 0.0, 0.4]])
W_V = torch.tensor([[0.6, 0.2, -0.7, 0.9], [-1.0, 0.3, 0.4, -0.3], [0.3, 0.5, -0.3, -0.7], [0.5, 0.2, -0.3, 0.4]])
W_O = torch.tensor([[0.2, 0.4, 1.0, -0.2], [-0.5, -0.6, 0.3, -0.7], [-0.5, -0.2, 0.1, -0.5], [0.2, 1.0, -0.5, -0.4]])
W1 = torch.tensor([                      # FFN expand [D=4, DFF=8]
    [-0.9, -1.0, -0.7, -1.0, -0.4, -1.0, -0.1, -1.0],
    [0.7, -0.2, 1.0, 0.2, -0.3, 0.2, 0.2, -0.6],
    [-0.6, 0.4, -0.7, -0.9, -0.1, -0.3, 0.4, -0.1],
    [0.0, 0.1, -0.1, -0.1, 0.2, -0.2, -0.1, 0.1],
])
W2 = torch.tensor([                      # FFN shrink [DFF=8, D=4]
    [-0.4, 0.1, 0.8, 0.5],
    [0.1, 0.4, -0.4, 0.0],
    [0.5, -0.3, -0.1, 0.4],
    [0.2, 0.1, 0.0, -1.0],
    [-0.1, -0.1, -0.6, -0.8],
    [-0.8, 0.5, 0.0, 0.6],
    [-0.3, -0.3, 0.1, 0.2],
    [0.1, -0.6, 0.2, 0.0],
])
U = torch.tensor([                       # LM head [D=4, V=8]: logits = norm(y) @ U
    [-0.1, 0.0, -0.1, 1.0, -1.0, -0.1, 0.3, -0.2],
    [0.2, -0.2, 0.3, -0.7, -0.7, 0.2, -0.1, 0.4],
    [0.3, 0.0, 0.6, 0.7, 1.0, 0.2, -0.2, -1.0],
    [-0.1, 0.0, -0.1, 0.2, -0.5, 0.3, 0.1, 0.8],
])
WEIGHTS = dict(E=E, P=P, W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O, W1=W1, W2=W2, U=U)


# ----------------------------------------------------------------------------
# The model, written out op by op
# ----------------------------------------------------------------------------
def layer_norm(x):
    """(x - mean) / sqrt(var + eps) over the last dim; gamma = 1, beta = 0."""
    mu = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mu) / torch.sqrt(var + EPS)


def split_heads(M):
    """[N, D] -> [H, N, DK]: head h owns columns h*DK .. h*DK+DK-1."""
    return M.view(M.shape[0], H, DK).transpose(0, 1)


def forward(ids):
    """Full forward pass over a list of token ids. Returns every intermediate."""
    n = len(ids)
    r = {}
    r["tok"] = E[ids]                                # step 2: pick rows of E
    r["pos"] = P[:n]                                 # step 3
    r["x"] = r["tok"] + r["pos"]
    r["xh"] = layer_norm(r["x"])                     # pre-norm (explained in step 13)
    r["Q"], r["K"], r["V"] = r["xh"] @ W_Q, r["xh"] @ W_K, r["xh"] @ W_V   # steps 4-5
    Qh, Kh, Vh = split_heads(r["Q"]), split_heads(r["K"]), split_heads(r["V"])
    r["S_raw"] = Qh @ Kh.transpose(-2, -1)           # step 6: [H, n, n]
    r["S_scaled"] = r["S_raw"] / math.sqrt(DK)       # step 7
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))
    r["S_masked"] = r["S_scaled"].masked_fill(~causal, float("-inf"))  # step 8
    r["A"] = torch.softmax(r["S_masked"], dim=-1)   # step 9
    r["A_nomask"] = torch.softmax(r["S_scaled"], dim=-1)  # what leaks without the mask
    r["O"] = r["A"] @ Vh                             # step 10: [H, n, DK]
    r["concat"] = r["O"].transpose(0, 1).reshape(n, D)  # step 11
    r["attn_out"] = r["concat"] @ W_O
    r["h"] = r["x"] + r["attn_out"]                  # step 12: residual
    r["hh"] = layer_norm(r["h"])                     # step 13
    r["u"] = r["hh"] @ W1                            # step 14
    r["a"] = torch.relu(r["u"])
    r["f"] = r["a"] @ W2
    r["y"] = r["h"] + r["f"]                         # step 15: second residual
    r["yh"] = layer_norm(r["y"])                     # step 16: final norm
    r["logits"] = r["yh"] @ U
    r["probs"] = torch.softmax(r["logits"][-1], dim=-1)
    return r


def decode_step(token_id, pos, cache):
    """One generation step with a KV cache: only the NEW token goes through the block."""
    x = E[token_id] + P[pos]                         # [D]
    xh = layer_norm(x)
    q, k, v = xh @ W_Q, xh @ W_K, xh @ W_V           # [D] each: one row, not n rows
    cache["K"] = torch.cat([cache["K"], k[None]])    # append; old rows are reused as-is
    cache["V"] = torch.cat([cache["V"], v[None]])
    qh = q.view(H, 1, DK)                            # [H, 1, DK]
    Kh, Vh = split_heads(cache["K"]), split_heads(cache["V"])
    w = torch.softmax(qh @ Kh.transpose(-2, -1) / math.sqrt(DK), dim=-1)  # no mask needed:
    o = (w @ Vh).transpose(0, 1).reshape(D)          # the cache holds only the past
    h = x + o @ W_O
    y = h + torch.relu(layer_norm(h) @ W1) @ W2
    logits = layer_norm(y) @ U
    return dict(q=q, k=k, v=v, w=w, logits=logits)


# ----------------------------------------------------------------------------
# Formatting: one function decides how every number looks (page and printout)
# ----------------------------------------------------------------------------
def rnd(v, dec=2):
    if v is None or (isinstance(v, float) and math.isinf(v)):
        return None
    return round(float(v), dec) + 0.0  # + 0.0 turns -0.0 into 0.0


def tolist(t, dec=2):
    if isinstance(t, torch.Tensor):
        t = t.tolist()
    if isinstance(t, list):
        return [tolist(v, dec) for v in t]
    return rnd(t, dec)


def fmt(v, dec=2):
    v = rnd(v, dec)
    return "-inf" if v is None else f"{v:.{dec}f}"


def show(title, M, rows=None, cols=None, dec=2):
    M = M.tolist() if isinstance(M, torch.Tensor) else M
    if not isinstance(M[0], list):
        M = [M]
    print(f"  {title}")
    w = max(6, dec + 4)
    if cols:
        print("  " + " " * 6 + "".join(f"{c:>{w}}" for c in cols))
    for i, row in enumerate(M):
        lab = rows[i] if rows else ""
        print("  " + f"{lab:>5} " + "".join(f"{fmt(v, dec):>{w}}" for v in row))


def dot_line(name, a, b, dec_a=2, dec_b=1, result=None):
    """'name = a1*b1 + a2*b2 + ... = result' using the ROUNDED shown numbers.
    If the shown terms do not add up to the shown result in the last digit
    (rounding), the line uses '~' instead of '='."""
    terms = [f"{fmt(x, dec_a)}*{fmt(y, dec_b)}" for x, y in zip(a, b)]
    exact = sum(float(x) * float(y) for x, y in zip(a, b)) if result is None else result
    shown_sum = sum(rnd(x, dec_a) * rnd(y, dec_b) for x, y in zip(a, b))
    eq = "=" if fmt(shown_sum) == fmt(exact) else "~"
    return f"{name} = " + " + ".join(terms) + f" {eq} {fmt(exact)}", eq


def eq_sign(shown, exact, dec=2):
    """'=' if numbers recomputed from the SHOWN (rounded) inputs round to the shown
    results, else '~' (the page prints it as the approx sign)."""
    shown = shown if isinstance(shown, list) else [shown]
    exact = exact if isinstance(exact, list) else [exact]
    return "=" if all(fmt(a, dec) == fmt(b, dec) for a, b in zip(shown, exact)) else "~"


def section(n, title):
    print()
    print("=" * 72)
    print(f"Step {n}. {title}" if n is not None else title)
    print("=" * 72)


# ----------------------------------------------------------------------------
# Main: compute everything, check it, print it in page order
# ----------------------------------------------------------------------------
def build():
    words = PROMPT.split()
    ids = [VOCAB.index(w) for w in words]
    r = forward(ids)
    n = len(ids)
    J = {"vocab": VOCAB, "words": words, "ids": ids,
         "config": dict(V=V, D=D, H=H, DK=DK, DFF=DFF, NMAX=NMAX)}
    for k, t in WEIGHTS.items():
        J[k] = tolist(t, 1)
    for k in ["tok", "pos", "x"]:
        J[k] = tolist(r[k], 1)                       # exact sums of 1-decimal numbers
    for k in ["xh", "Q", "K", "V", "S_raw", "S_scaled", "S_masked", "A", "A_nomask",
              "O", "concat", "attn_out", "h", "hh", "u", "a", "f", "y", "yh", "logits"]:
        J[k] = tolist(r[k])
    J["probs"] = tolist(r["probs"])
    J["logits_last_exact"] = [round(float(v), 6) for v in r["logits"][-1]]  # for the temperature slider
    J["next_id"] = int(r["probs"].argmax())
    J["next_word"] = VOCAB[J["next_id"]]

    # --- assertions ---------------------------------------------------------
    # LayerNorm as written == PyTorch's
    assert torch.allclose(r["xh"], F.layer_norm(r["x"], (D,), eps=EPS))
    # each attention row sums to 1; masked (future) weights are exactly 0
    assert torch.allclose(r["A"].sum(-1), torch.ones(H, n))
    future = ~torch.tril(torch.ones(n, n, dtype=torch.bool))
    assert torch.all(r["A"][:, future] == 0)
    # our attention == PyTorch's fused scaled_dot_product_attention
    Qh, Kh, Vh = split_heads(r["Q"]), split_heads(r["K"]), split_heads(r["V"])
    sdpa = F.scaled_dot_product_attention(Qh[None], Kh[None], Vh[None], is_causal=True)[0]
    assert torch.allclose(r["O"], sdpa, atol=1e-12), (r["O"], sdpa)
    assert torch.allclose(r["attn_out"], sdpa.transpose(0, 1).reshape(n, D) @ W_O, atol=1e-12)

    # step 4: one output cell written out as a dot product (Q row "sat", column 0)
    i, j = 2, 0
    line, eq = dot_line(f"Q[{words[i]}, {j}]", r["xh"][i], W_Q[:, j], result=float(r["Q"][i, j]))
    J["mm_example"] = dict(i=i, j=j, line=line, eq=eq)

    # step 7: softmax of one score row with / without the 1/sqrt(d_k) scale.
    # "d_k = 128": same row, raw scores sqrt(128/2) = 8x bigger (each extra dim adds a term)
    raw = r["S_raw"][0, 2]                            # head 0, query "sat"
    big = raw * math.sqrt(128 / DK)
    J["scale_demo"] = {
        "raw2": tolist(raw), "raw128": tolist(big),
        "dk2_on": tolist(torch.softmax(raw / math.sqrt(DK), -1)),
        "dk2_off": tolist(torch.softmax(raw, -1)),
        "dk128_on": tolist(torch.softmax(big / math.sqrt(128), -1)),
        "dk128_off": tolist(torch.softmax(big, -1)),
        "sqrt2": rnd(math.sqrt(2)), "sqrt128": rnd(math.sqrt(128)),
        "scaled2": tolist(raw / math.sqrt(DK)), "scaled128": tolist(big / math.sqrt(128)),
    }
    assert torch.allclose(torch.softmax(raw / math.sqrt(DK), -1),            # with the scale, the
                          torch.softmax(big / math.sqrt(128), -1))           # head size drops out

    # step 8: how much attention "the" puts on later tokens when the mask is removed
    J["leak_row0"] = rnd(r["A_nomask"][0, 0, 1:].sum())

    # step 9: softmax by hand for row "sat" of head 0
    s = r["S_masked"][0, 2]
    ex = torch.exp(s)
    J["softmax_demo"] = dict(s=tolist(s), exp=tolist(ex), sum=rnd(ex.sum()), w=tolist(ex / ex.sum()))
    # (the page's slider starts from the rounded scores; for these weights they give the same
    #  rounded softmax, which the page check verifies)

    # step 13: LayerNorm and RMSNorm of row "sat" of h
    hs = r["h"][2]
    mu, var = hs.mean(), hs.var(unbiased=False)
    rms = torch.sqrt((hs ** 2).mean() + EPS)
    J["ln_demo"] = dict(h=tolist(hs), mean=rnd(mu), var=rnd(var), std=rnd(torch.sqrt(var + EPS)),
                        centered=tolist(hs - mu), out=tolist(layer_norm(hs)),
                        rms=rnd(rms), rms_out=tolist(hs / rms))
    assert torch.allclose(r["hh"], F.layer_norm(r["h"], (D,), eps=EPS))

    # step 16: temperature
    J["temp_demo"] = {f"T{int(T * 10):02d}": tolist(torch.softmax(r["logits"][-1] / T, -1))
                      for T in (0.5, 1.0, 2.0)}                # keys T05, T10, T20
    assert J["temp_demo"]["T10"] == J["probs"]

    # step 17: generation with a KV cache == full recompute
    seq = list(ids)
    cache = {"K": r["K"].clone(), "V": r["V"].clone()}     # prefill fills the cache
    gen = [dict(tokens=[VOCAB[t] for t in seq], cache_len=n, new=None,
                K=tolist(cache["K"]), V=tolist(cache["V"]),
                probs=tolist(r["probs"]), next=VOCAB[int(r["probs"].argmax())])]
    nxt = int(r["probs"].argmax())
    for _ in range(N_NEW - 1):
        seq.append(nxt)
        st = decode_step(nxt, len(seq) - 1, cache)
        full = forward(seq)                                 # recompute everything
        assert torch.allclose(st["logits"], full["logits"][-1], atol=1e-12)
        assert torch.allclose(cache["K"], full["K"], atol=1e-12)
        assert torch.allclose(cache["V"], full["V"], atol=1e-12)
        p = torch.softmax(st["logits"], -1)
        gen.append(dict(tokens=[VOCAB[t] for t in seq], cache_len=len(seq), new=VOCAB[nxt],
                        q=tolist(st["q"]), k=tolist(st["k"]), v=tolist(st["v"]),
                        K=tolist(cache["K"]), V=tolist(cache["V"]),
                        probs=tolist(p), next=VOCAB[int(p.argmax())]))
        nxt = int(p.argmax())
    seq.append(nxt)
    J["gen"] = gen
    J["generated"] = " ".join(VOCAB[t] for t in seq)

    # step 18: counting
    n_attn = 4 * D * D
    n_ffn = 2 * D * DFF
    n_norm = 3 * 2 * D                                  # 3 norms x (gamma, beta)
    counts = dict(emb=V * D, pos=NMAX * D, attn=n_attn, ffn=n_ffn, norms=n_norm, head=D * V)
    counts["total"] = sum(counts.values())
    assert counts["total"] == sum(t.numel() for t in WEIGHTS.values()) + n_norm
    N = n
    flops = dict(qkv=3 * 2 * N * D * D, scores=2 * N * N * D, mix=2 * N * N * D,
                 out=2 * N * D * D, ffn=2 * 2 * N * D * DFF, head=2 * N * D * V)
    flops["total"] = sum(flops.values())
    # a 7B-class model with the same formulas (GPT-3 6.7B shape: d_ff = 4d, tied head)
    d, L, Vb, Nctx = 4096, 32, 50257, 2048
    big_params = 12 * L * d * d + Vb * d + Nctx * d
    big = dict(d=d, L=L, V=Vb, Nctx=Nctx, per_block=12 * d * d, blocks=12 * L * d * d,
               emb=Vb * d, pos=Nctx * d, total=big_params, total_b=rnd(big_params / 1e9),
               gflops_per_token=rnd(2 * (12 * L * d * d + Vb * d) / 1e9, 1),
               attn_gflops_per_token_at_ctx=rnd(4 * L * Nctx * d / 1e9, 2),
               weight_gb_fp16=rnd(big_params * 2 / 1e9, 1),
               kv_bytes_per_token=2 * L * d * 2,
               kv_gib_2048=rnd(2 * L * d * 2 * 2048 / 2**30, 1),
               kv_gib_4096=rnd(2 * L * d * 2 * 4096 / 2**30, 1))
    J["counts"] = counts
    J["flops"] = flops
    J["flops_rule"] = dict(per_token_matmul=2 * (n_attn + n_ffn + D * V), attn=4 * N * N * D)
    assert flops["total"] == N * J["flops_rule"]["per_token_matmul"] + J["flops_rule"]["attn"]
    J["toy_kv_numbers_per_token"] = 2 * 1 * D
    J["toy_weight_bytes_fp16"] = counts["total"] * 2
    J["big"] = big
    pct = lambda a, b: rnd(100 * a / b, 1)
    J["share"] = dict(
        toy=dict(emb=pct(counts["emb"] + counts["pos"] + counts["head"], counts["total"]),
                 attn=pct(n_attn, counts["total"]), ffn=pct(n_ffn, counts["total"]),
                 norms=pct(n_norm, counts["total"])),
        big=dict(emb=pct(big["emb"] + big["pos"], big_params), attn=pct(4 * L * d * d, big_params),
                 ffn=pct(8 * L * d * d, big_params)))

    # "=" or "~" for every by-hand line on the page (recomputed from the rounded inputs)
    R = lambda v, d=2: rnd(v, d)
    sd, sm, ln = J["scale_demo"], J["softmax_demo"], J["ln_demo"]
    a0, v0 = J["A"][0][2], [row[:2] for row in J["V"]]
    J["eq"] = {
        "s3": eq_sign([R(a, 1) + R(b, 1) for a, b in zip(J["tok"][0], J["pos"][0])], J["x"][0], 1),
        "s4": J["mm_example"]["eq"],
        "s6": dot_line("", J["Q"][2][:2], J["K"][1][:2], 2, 2, float(r["S_raw"][0, 2, 1]))[1],
        "s7": eq_sign(sd["raw2"][1] / sd["sqrt2"], J["S_scaled"][0][2][1]),
        "s9exp": eq_sign([math.exp(v) for v in sm["s"]], sm["exp"]),
        "s9sum": eq_sign(sum(sm["exp"]), sm["sum"]),
        "s9w": eq_sign([e / sm["sum"] for e in sm["exp"]], sm["w"]),
        "s10": eq_sign([sum(a0[t] * v0[t][k] for t in range(3)) for k in range(2)], J["O"][0][2]),
        "s12": eq_sign([a + b for a, b in zip(J["x"][2], J["attn_out"][2])], J["h"][2]),
        "s13c": eq_sign([v - ln["mean"] for v in ln["h"]], ln["centered"]),
        "s13o": eq_sign([v / ln["std"] for v in ln["centered"]], ln["out"]),
        "s13r": eq_sign([v / ln["rms"] for v in ln["h"]], ln["rms_out"]),
        "s15": eq_sign([a + b for a, b in zip(J["h"][2], J["f"][2])], J["y"][2]),
    }
    J["eq"] = {k: ("=" if v == "=" else "≈") for k, v in J["eq"].items()}

    # numbers used in the self-check answers
    J["quiz"] = dict(
        std_dk128=rnd(math.sqrt(128)),
        params_dff16=counts["total"] - n_ffn + 2 * D * 16,
        kv_numbers_5_tokens=5 * 2 * D,
        self_score=D)                                   # |x_hat|^2 = d after LayerNorm
    assert abs(float((r["xh"] ** 2).sum(-1).mean()) - D) < 1e-3
    return J, r


def print_all(J):
    w = J["words"]
    S = lambda key: "~" if J["eq"][key] == "≈" else "="    # by-hand lines: = or ~ (rounding)
    vec = lambda xs, d=2: "[" + ", ".join(fmt(v, d) for v in xs) + "]"
    hd = lambda h: [f"h{h}d{k}" for k in range(DK)]
    dims = [f"d{k}" for k in range(D)]
    section(0, "Big picture")
    print(f"  vocab V = {V}, d_model = {D}, heads = {H} (d_k = {DK}), d_ff = {DFF}, "
          f"max length = {NMAX}, 1 block")
    print(f"  '{PROMPT}'  ->  next-token probabilities  ->  '{J['next_word']}' "
          f"(p = {fmt(J['probs'][J['next_id']])})")

    section(1, "Text -> token IDs")
    print("  vocab: " + ", ".join(f"{i}:{t!r}" for i, t in enumerate(VOCAB)))
    print(f"  {w} -> ids {J['ids']}")

    section(2, "Embedding = pick rows from a table")
    show("E  [8 vocab x 4 dims]  (learned weights)", J["E"], VOCAB, dims, 1)
    show(f"tok = E[{J['ids']}]  [3 x 4]", J["tok"], w, dims, 1)
    print(f"  (GPT-2 small: E is [50,257 x 768] = {50257 * 768:,} = {50257 * 768 / 1e6:.1f} M numbers)")

    section(3, "Add position information")
    show("P[0:3]  (rows 0-2 of the [8 x 4] position table)", J["pos"], ["p0", "p1", "p2"], dims, 1)
    show("x = tok + P[0:3]  [3 x 4]", J["x"], w, dims, 1)
    print(f"  x[the] = {vec(J['tok'][0], 1)} + {vec(J['pos'][0], 1)} {S('s3')} {vec(J['x'][0], 1)}")

    section(4, "Matrix multiply refresher: Q = x_hat @ W_Q")
    show("x_hat = LayerNorm(x)  [3 x 4]  (step 13 explains the normalization)", J["xh"], w, dims)
    show("W_Q  [4 x 4]", J["W_Q"], dims, ["q0", "q1", "q2", "q3"], 1)
    show("Q = x_hat @ W_Q  [3 x 4]", J["Q"], w, hd(0) + hd(1))
    print("  " + J["mm_example"]["line"])

    section(5, "Q, K, V projections")
    show("W_K  [4 x 4]", J["W_K"], dims, ["k0", "k1", "k2", "k3"], 1)
    show("W_V  [4 x 4]", J["W_V"], dims, ["v0", "v1", "v2", "v3"], 1)
    show("K = x_hat @ W_K  [3 x 4]", J["K"], w, hd(0) + hd(1))
    show("V = x_hat @ W_V  [3 x 4]", J["V"], w, hd(0) + hd(1))

    section(6, "Scores = Q K^T  (per head, [3 x 3]; rows = query, cols = key)")
    for h in range(H):
        show(f"head {h}: S = Q_h K_h^T", J["S_raw"][h], w, w)
    q, k = J["Q"][2][:2], J["K"][1][:2]
    print("  " + dot_line("score(sat -> cat), head 0", q, k, 2, 2, J["S_raw"][0][2][1])[0])

    section(7, "Scale by 1/sqrt(d_k)")
    for h in range(H):
        show(f"head {h}: S / sqrt({DK})", J["S_scaled"][h], w, w)
    sd = J["scale_demo"]
    print(f"  sqrt(2) = {fmt(sd['sqrt2'])};  {fmt(sd['raw2'][1])} / {fmt(sd['sqrt2'])} {S('s7')} "
          f"{fmt(J['S_scaled'][0][2][1])}")
    show("demo row (head 0, query 'sat'), raw scores, d_k = 2", sd["raw2"], ["raw"], w)
    show("same row if d_k were 128 (raw scores 8x bigger)", sd["raw128"], ["raw"], w)
    show("after the scale (d_k = 2: / 1.41; d_k = 128: / " + fmt(sd["sqrt128"]) + ")",
         [sd["scaled2"], sd["scaled128"]], ["dk=2", "dk=128"], w)
    for key, lab in [("dk2_on", "d_k=2,   scaled"), ("dk2_off", "d_k=2,   NOT scaled"),
                     ("dk128_on", "d_k=128, scaled"), ("dk128_off", "d_k=128, NOT scaled")]:
        print(f"  softmax  {lab:<20} " + "  ".join(fmt(v) for v in sd[key]))

    section(8, "Causal mask")
    for h in range(H):
        show(f"head {h}: masked scores", J["S_masked"][h], w, w)
    for h in range(H):
        show(f"head {h}: weights WITHOUT the mask (future leaks in)", J["A_nomask"][h], w, w)
    print(f"  without the mask, 'the' (head 0) puts {fmt(J['leak_row0'])} of its attention on later tokens")

    section(9, "Softmax: each row -> weights that sum to 1")
    for h in range(H):
        show(f"head {h}: A = softmax(masked scores)", J["A"][h], w, w)
    sm = J["softmax_demo"]
    print("  row 'sat', head 0: scores " + ", ".join(fmt(v) for v in sm["s"]))
    print(f"  exp {S('s9exp')} " + ", ".join(fmt(v) for v in sm["exp"]) + f"; sum {S('s9sum')} {fmt(sm['sum'])}")
    print(f"  weights = exp / sum {S('s9w')} " + ", ".join(fmt(v) for v in sm["w"]))

    section(10, "Weighted sum of V")
    for h in range(H):
        show(f"head {h}: V_h", [row[h * DK:(h + 1) * DK] for row in J["V"]], w, hd(h))
        show(f"head {h}: O_h = A_h V_h  [3 x 2]", J["O"][h], w, hd(h))
    a, v = J["A"][0][2], [row[:2] for row in J["V"]]
    terms = " + ".join(f"{fmt(a[t])}*[{fmt(v[t][0])}, {fmt(v[t][1])}]" for t in range(3))
    print(f"  o[sat], head 0 = {terms} {S('s10')} [{fmt(J['O'][0][2][0])}, {fmt(J['O'][0][2][1])}]")

    section(11, "Multi-head: concatenate the heads, then W_O")
    show("concat = [O_0 | O_1]  [3 x 4]", J["concat"], w, hd(0) + hd(1))
    show("W_O  [4 x 4]", J["W_O"], hd(0) + hd(1), dims, 1)
    show("attn_out = concat @ W_O  [3 x 4]", J["attn_out"], w, dims)

    section(12, "Residual: h = x + attn_out")
    show("h  [3 x 4]", J["h"], w, dims)
    print(f"  h[sat] = {vec(J['x'][2], 1)} + {vec(J['attn_out'][2])} {S('s12')} {vec(J['h'][2])}")

    section(13, "LayerNorm (and RMSNorm), row 'sat' of h")
    ln = J["ln_demo"]
    print("  h[sat]         = " + ", ".join(fmt(v) for v in ln["h"]))
    print(f"  mean = {fmt(ln['mean'])}, variance = {fmt(ln['var'])}, std = sqrt(var + eps) = {fmt(ln['std'])}")
    print(f"  h - mean       {S('s13c')} " + ", ".join(fmt(v) for v in ln["centered"]))
    print(f"  (h - mean)/std {S('s13o')} " + ", ".join(fmt(v) for v in ln["out"]))
    print(f"  RMSNorm: rms = {fmt(ln['rms'])}; h / rms {S('s13r')} " + ", ".join(fmt(v) for v in ln["rms_out"]))
    show("h_hat = LayerNorm(h)  [3 x 4]", J["hh"], w, dims)

    section(14, "FFN: expand 4 -> 8, ReLU, shrink 8 -> 4 (each token on its own)")
    show("W1  [4 x 8]", J["W1"], dims, [f"u{k}" for k in range(DFF)], 1)
    show("u = h_hat @ W1  [3 x 8]", J["u"], w, [f"u{k}" for k in range(DFF)])
    show("a = ReLU(u)  [3 x 8]", J["a"], w, [f"u{k}" for k in range(DFF)])
    show("W2  [8 x 4]", J["W2"], [f"u{k}" for k in range(DFF)], dims, 1)
    show("f = a @ W2  [3 x 4]", J["f"], w, dims)

    section(15, "One full block: y = h + FFN(LN(h))")
    show("y  [3 x 4]  (block output, same shape as x)", J["y"], w, dims)
    print(f"  y[sat] = h[sat] + f[sat] = {vec(J['h'][2])} + {vec(J['f'][2])} {S('s15')} {vec(J['y'][2])}")

    section(16, "Final norm -> LM head -> softmax -> next token")
    show("y_hat = LayerNorm(y)  [3 x 4]", J["yh"], w, dims)
    show("U (LM head)  [4 x 8]", J["U"], dims, VOCAB, 1)
    show("logits = y_hat @ U  [3 x 8]", J["logits"], w, VOCAB)
    for T, key in (("0.5", "T05"), ("1.0", "T10"), ("2.0", "T20")):
        print(f"  probs (last row, T = {T}): " + "  ".join(f"{t}:{fmt(p)}" for t, p in zip(VOCAB, J["temp_demo"][key])))
    print(f"  greedy pick: '{J['next_word']}'")

    section(17, "Generate: the loop and the KV cache")
    for g in J["gen"]:
        if g["new"] is None:
            print(f"  prefill {g['tokens']}: K, V cache = {g['cache_len']} rows -> next '{g['next']}' "
                  f"(p = {fmt(max(g['probs']))})")
        else:
            print(f"  decode '{g['new']}' (pos {g['cache_len'] - 1}): q = {vec(g['q'])}, "
                  f"new k = {vec(g['k'])}, new v = {vec(g['v'])}")
            print(f"     cache = {g['cache_len']} rows -> next '{g['next']}' (p = {fmt(max(g['probs']))})")
        show("K cache", g["K"], g["tokens"], hd(0) + hd(1))
        show("V cache", g["V"], g["tokens"], hd(0) + hd(1))
    print(f"  text: '{J['generated']}'   (cached logits == full recompute: OK)")

    section(18, "Counting parameters, FLOPs and memory")
    c, fl, b = J["counts"], J["flops"], J["big"]
    print(f"  toy params: embedding {c['emb']} + positions {c['pos']} + attention {c['attn']} + "
          f"FFN {c['ffn']} + norms {c['norms']} + LM head {c['head']} = {c['total']}")
    print(f"  toy FLOPs (3 tokens): QKV {fl['qkv']} + scores {fl['scores']} + mix {fl['mix']} + "
          f"W_O {fl['out']} + FFN {fl['ffn']} + LM head {fl['head']} = {fl['total']}")
    print(f"     = 3 tokens x {J['flops_rule']['per_token_matmul']} (2 x matmul weights) + "
          f"{J['flops_rule']['attn']} (attention 4*N^2*d)")
    print(f"  toy KV cache: {J['toy_kv_numbers_per_token']} numbers per token; "
          f"weights in FP16: {J['toy_weight_bytes_fp16']} bytes")
    print(f"  7B-class, GPT-3 6.7B shape (d={b['d']}, L={b['L']}, V={b['V']}, context {b['Nctx']}):")
    print(f"     per block 12d^2 = {b['per_block']:,}; x L = {b['blocks']:,}")
    print(f"     + embedding V*d = {b['emb']:,} + positions = {b['pos']:,} -> {b['total']:,} params "
          f"({fmt(b['total_b'])} B)")
    print(f"     FLOPs per token ~ 2 x params = {b['gflops_per_token']} GFLOPs "
          f"(+ attention 4*L*N*d = {b['attn_gflops_per_token_at_ctx']} GFLOPs at N = {b['Nctx']})")
    print(f"     weights FP16 = {b['weight_gb_fp16']} GB; KV cache = 2*L*d*2 B = "
          f"{b['kv_bytes_per_token']:,} B = {b['kv_bytes_per_token'] // 1024} KiB per token; "
          f"{b['kv_gib_2048']} GiB at 2048 tokens, {b['kv_gib_4096']} GiB at 4096")
    sh = J["share"]
    print(f"  share of params, toy: embeddings + positions + LM head {sh['toy']['emb']}%, attention "
          f"{sh['toy']['attn']}%, FFN {sh['toy']['ffn']}%, norms {sh['toy']['norms']}%")
    print(f"  share of params, 7B-class: embeddings {sh['big']['emb']}%, attention {sh['big']['attn']}%, "
          f"FFN {sh['big']['ffn']}%")

    section(None, "Self-check answers (numbers)")
    qz = J["quiz"]
    print(f"  std of a raw score for d_k = 128: sqrt(128) = {fmt(qz['std_dk128'])}")
    print(f"  after LayerNorm every |x_hat|^2 = d = {qz['self_score']}, so x_hat_i . x_hat_j <= x_hat_i . x_hat_i")
    print(f"  toy params with d_ff = 16: {qz['params_dff16']}")
    print(f"  toy KV cache after 5 tokens: 5 x (4 + 4) = {qz['kv_numbers_5_tokens']} numbers; "
          f"7B-class at 4096 tokens: {J['big']['kv_gib_4096']} GiB")
    print()
    print("All checks passed: rows sum to 1, future weights are 0, attention == "
          "F.scaled_dot_product_attention, KV-cached generation == full recompute.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true", help="print the page data as JSON")
    args = ap.parse_args()
    J, _ = build()
    if args.json:
        print(json.dumps(J, ensure_ascii=False))
    else:
        print_all(J)
