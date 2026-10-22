#!/usr/bin/env python3
"""Checker for the hands-on labs.

    python labs/check.py              check every lab in labs/
    python labs/check.py 3            check lab 3 only (also: 03, lab03, or several numbers)
    python labs/check.py --solutions  run the same checks against labs/solutions/
    python labs/check.py 3 -v         also print the full traceback when a step fails

Running a lab file directly (python labs/lab03_gpt_generate.py) checks that file.

Every lab is split into steps, and each step prints one line:
    PASS  your code matches the reference
    TODO  your code still raises NotImplementedError (not started yet)
    FAIL  your code runs but gives a wrong answer, or crashes; the message says why

The checks compare your code with independent references wherever possible:
torch's own implementations (F.scaled_dot_product_attention,
nn.MultiheadAttention with copied weights, F.conv2d, F.grid_sample, ...),
closed-form maths, worked examples from the guides, and brute-force Python
loops. Everything is seeded, so a result is reproducible.

Requires only torch (2.1 or newer) and the standard library.
"""

import argparse
import importlib.util
import math
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LABS_DIR = Path(__file__).resolve().parent
SOLUTIONS_DIR = LABS_DIR / "solutions"


# ===========================================================================
# Framework: registry, assertions, runner
# ===========================================================================
class Lab:
    def __init__(self, num, filename, title):
        self.num = num
        self.filename = filename
        self.title = title
        self.steps = []


LABS = {}


def lab(num, filename, title):
    LABS[num] = Lab(num, filename, title)


def step(num, name):
    """Register the decorated function as the next step of lab `num`."""
    def deco(fn):
        LABS[num].steps.append((name, fn))
        return fn
    return deco


lab(1, "lab01_attention.py", "Attention")
lab(2, "lab02_transformer_block.py", "Transformer block")
lab(3, "lab03_gpt_generate.py", "GPT + generation")
lab(4, "lab04_vit.py", "Vision Transformer")
lab(5, "lab05_vlm_fusion.py", "VLM fusion")
lab(6, "lab06_diffusion.py", "Diffusion")
lab(7, "lab07_mamba_scan.py", "Mamba selective scan")
lab(8, "lab08_moe_router.py", "MoE routing")
lab(9, "lab09_bev_projection.py", "BEV projection")
lab(10, "lab10_flow_matching.py", "Flow matching")


class CheckFailure(AssertionError):
    """A check found a wrong answer; the message explains what was expected."""


_PASSED = [0]   # comparisons passed so far in the current step (reported with a TODO)


def fail(msg):
    raise CheckFailure(msg)


def expect(cond, msg):
    if not cond:
        fail(msg)


def _fmt_shape(shape):
    return "[" + ", ".join(str(s) for s in shape) + "]"


def expect_tensor(x, what):
    if not isinstance(x, torch.Tensor):
        fail(f"{what}: expected a torch.Tensor, got {type(x).__name__}"
             + (" (did you forget a `return`?)" if x is None else ""))


def expect_shape(x, shape, what):
    expect_tensor(x, what)
    if tuple(x.shape) != tuple(shape):
        fail(f"{what}: shape mismatch\n"
             f"    expected {_fmt_shape(shape)}\n"
             f"    got      {_fmt_shape(x.shape)}")


def _preview(t, n=6):
    flat = t.detach().reshape(-1)[:n].tolist()
    if isinstance(flat, (bool, int)) or (flat and isinstance(flat[0], bool)):
        body = ", ".join(str(v) for v in flat)
    else:
        body = ", ".join(f"{v:.4g}" for v in flat)
    more = ", ..." if t.numel() > n else ""
    return f"[{body}{more}]"


def _unravel(flat_idx, shape):
    idx = []
    for size in reversed(tuple(shape)):
        idx.append(flat_idx % size)
        flat_idx //= size
    return tuple(reversed(idx))


def expect_close(got, want, what, atol=1e-5, rtol=1e-4, hint=None):
    """Compare a tensor with a reference. Prints the worst entry on failure."""
    expect_shape(got, want.shape, what)
    g = got.detach().to(torch.float64).cpu()
    w = want.detach().to(torch.float64).cpu()
    bad_g = ~torch.isfinite(g)
    if bad_g.any() and not (~torch.isfinite(w)).any():
        n_nan = int(torch.isnan(g).sum())
        n_inf = int(torch.isinf(g).sum())
        fail(f"{what}: your result contains {n_nan} NaN and {n_inf} inf values "
             f"(of {g.numel()})" + (f"\n    hint: {hint}" if hint else ""))
    diff = (g - w).abs()
    diff = torch.where(torch.isfinite(w) | torch.isfinite(g), diff, torch.zeros_like(diff))
    diff = torch.nan_to_num(diff, nan=float("inf"))
    tol = atol + rtol * w.abs().nan_to_num(0.0)
    if (diff > tol).any():
        flat_idx = int((diff - tol).reshape(-1).argmax())
        idx = _unravel(flat_idx, diff.shape)
        msg = (f"{what}: values differ from the reference\n"
               f"    worst entry {idx}: got {g[idx].item():.6g}, expected {w[idx].item():.6g} "
               f"(|diff| = {diff[idx].item():.3g}, tolerance {tol[idx].item():.3g})\n"
               f"    got      {_preview(g)}\n"
               f"    expected {_preview(w)}")
        if hint:
            msg += f"\n    hint: {hint}"
        fail(msg)
    _PASSED[0] += 1


def is_close(got, want, atol=1e-5, rtol=1e-4):
    """True if `got` has the same shape as `want` and matches it (used to diagnose mistakes)."""
    if not isinstance(got, torch.Tensor) or tuple(got.shape) != tuple(want.shape):
        return False
    return torch.allclose(got.detach().double().cpu(), want.detach().double().cpu(),
                          atol=atol, rtol=rtol)


def expect_equal(got, want, what, hint=None):
    """Exact equality (bool masks, integer indices)."""
    expect_shape(got, want.shape, what)
    if want.dtype == torch.bool and got.dtype != torch.bool:
        fail(f"{what}: expected a bool tensor, got dtype {got.dtype} "
             f"(comparisons such as a <= b return bools)")
    g = got.detach().cpu()
    w = want.detach().cpu()
    neq = g != w
    if neq.any():
        idx = tuple(int(i) for i in neq.nonzero()[0])
        msg = (f"{what}: {int(neq.sum())} of {neq.numel()} entries are wrong\n"
               f"    first wrong entry {idx}: got {g[idx].item()}, expected {w[idx].item()}\n"
               f"    got      {_preview(g, 10)}\n"
               f"    expected {_preview(w, 10)}")
        if hint:
            msg += f"\n    hint: {hint}"
        fail(msg)
    _PASSED[0] += 1


def expect_int(got, want, what, hint=None):
    if isinstance(got, torch.Tensor) and got.numel() == 1:
        got = got.item()
    if isinstance(got, float) and got.is_integer():
        got = int(got)
    if not isinstance(got, int) or isinstance(got, bool):
        fail(f"{what}: expected an int, got {type(got).__name__} ({got!r})")
    if got != want:
        msg = f"{what}: expected {want:,}, got {got:,} (difference {got - want:+,})"
        if hint:
            msg += f"\n    hint: {hint}"
        fail(msg)
    _PASSED[0] += 1


def load_module(path):
    name = "labcheck_" + "_".join(path.with_suffix("").parts[-2:])
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Result:
    def __init__(self, status, message="", seconds=0.0, tb=None):
        self.status = status
        self.message = message
        self.seconds = seconds
        self.tb = tb


def _frames_in(tb, path):
    path = os.path.realpath(path)
    return [f for f in traceback.extract_tb(tb) if os.path.realpath(f.filename) == path]


def _qualname(tb, path):
    """Name of the innermost function of `path` in the traceback, as Class.method if it is one."""
    path = os.path.realpath(path)
    name = "?"
    while tb is not None:
        frame = tb.tb_frame
        if os.path.realpath(frame.f_code.co_filename) == path:
            name = frame.f_code.co_name
            owner = frame.f_locals.get("self")
            if owner is not None:
                name = f"{type(owner).__name__}.{name}"
        tb = tb.tb_next
    return name


def _short(path):
    """Path relative to the current directory if it lies below it, else absolute."""
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return str(path)
    return str(path) if rel.startswith("..") else rel


def run_step(fn, module, path):
    torch.manual_seed(0)
    _PASSED[0] = 0
    t0 = time.perf_counter()
    try:
        info = fn(module)
        return Result("PASS", info or "", time.perf_counter() - t0)
    except NotImplementedError as e:
        dt = time.perf_counter() - t0
        all_frames = traceback.extract_tb(e.__traceback__)
        mine = _frames_in(e.__traceback__, path)
        if mine and all_frames and os.path.realpath(all_frames[-1].filename) == os.path.realpath(path):
            f = mine[-1]
            where = f"{Path(f.filename).name}:{f.lineno}"
            msg = f"implement {_qualname(e.__traceback__, path)}()  ({where})"
            if _PASSED[0]:
                n = _PASSED[0]
                msg += f"; {n} earlier check{'s' if n > 1 else ''} in this step passed"
            return Result("TODO", msg, dt)
        return Result("FAIL", f"NotImplementedError raised outside your code: {e}", dt,
                      traceback.format_exc())
    except CheckFailure as e:
        return Result("FAIL", str(e), time.perf_counter() - t0, traceback.format_exc())
    except Exception as e:  # the learner's code crashed, or returned something unusable
        dt = time.perf_counter() - t0
        mine = _frames_in(e.__traceback__, path)
        if mine:
            f = mine[-1]
            msg = (f"your code raised {type(e).__name__}: {e}\n"
                   f"    at {Path(f.filename).name}:{f.lineno} in {f.name}():  {f.line}")
        else:
            last = traceback.extract_tb(e.__traceback__)[-1]
            msg = (f"{type(e).__name__} while checking your result: {e}\n"
                   f"    (in {Path(last.filename).name}:{last.lineno}; check the type, shape and "
                   f"dtype of what your function returns)")
        return Result("FAIL", msg, dt, traceback.format_exc())


class Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def __call__(self, text, code):
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def status(self, status):
        return self(f"{status:<4}", {"PASS": "32;1", "TODO": "33;1", "FAIL": "31;1"}[status])


def check_lab(num, path, style, verbose=False):
    """Run every step of one lab against the file at `path`. Returns (status, n_pass, n_steps, seconds)."""
    spec = LABS[num]
    print(style(f"Lab {num:02d}  {spec.title}", "1") + f"   ({_short(path)})")
    t0 = time.perf_counter()
    n = len(spec.steps)
    if not path.exists():
        print(f"  {style.status('FAIL')}  file not found: {path}")
        return "FAIL", 0, n, 0.0
    try:
        module = load_module(path)
    except Exception as e:
        print(f"  {style.status('FAIL')}  could not import {path.name}: {type(e).__name__}: {e}")
        if verbose:
            traceback.print_exc()
        return "FAIL", 0, n, time.perf_counter() - t0
    results = []
    width = max(len(name) for name, _ in spec.steps)
    for name, fn in spec.steps:
        r = run_step(fn, module, path)
        results.append(r)
        line = f"  {style.status(r.status)}  {name:<{width}}"
        if r.status == "PASS" and r.message:
            line += style(f"  [{r.message}]", "2")
        elif r.status == "TODO":
            line += style(f"  -> {r.message}", "2")
        if r.seconds > 2.0:
            line += style(f"  ({r.seconds:.1f} s)", "2")
        print(line)
        if r.status == "FAIL":
            for msg_line in r.message.splitlines():
                print("        " + msg_line)
            if verbose and r.tb:
                print(style("        traceback:", "2"))
                for tb_line in r.tb.rstrip().splitlines():
                    print("        " + tb_line)
    seconds = time.perf_counter() - t0
    n_pass = sum(r.status == "PASS" for r in results)
    if n_pass == n:
        status = "PASS"
    elif any(r.status == "FAIL" for r in results):
        status = "FAIL"
    else:
        status = "TODO"
    print(f"  ==> {style.status(status)}  {n_pass}/{n} steps pass  ({seconds:.1f} s)")
    if status == "FAIL" and not verbose:
        print(style(f"      (rerun with: python labs/check.py {num} -v  for the full traceback)", "2"))
    print()
    return status, n_pass, n, seconds


def _parse_lab_numbers(items):
    nums = []
    for item in items:
        s = item.lower().removeprefix("lab").split("_")[0]
        if not s.isdigit() or int(s) not in LABS:
            raise SystemExit(f"check.py: unknown lab {item!r} (expected a number 1-{len(LABS)})")
        nums.append(int(s))
    return nums or sorted(LABS)


def _limit_threads():
    """The models here are tiny; more than a few CPU threads only adds overhead."""
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _run(nums, folder, verbose, label):
    _limit_threads()
    style = Style(sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    summary = []
    for num in nums:
        path = folder / LABS[num].filename
        summary.append((num,) + check_lab(num, path, style, verbose))
    if len(summary) > 1:
        print(style(f"Summary ({label})", "1"))
        for num, status, n_pass, n, seconds in summary:
            print(f"  lab{num:02d}  {LABS[num].title:<22} {style.status(status)}  "
                  f"{n_pass:>2}/{n} steps  {seconds:5.1f} s")
        counts = {s: sum(1 for x in summary if x[1] == s) for s in ("PASS", "TODO", "FAIL")}
        total = sum(x[4] for x in summary)
        print(f"  {counts['PASS']} PASS, {counts['TODO']} TODO, {counts['FAIL']} FAIL   "
              f"(total {total:.1f} s)")
    return 0 if all(x[1] == "PASS" for x in summary) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Check the hands-on labs (PASS / TODO / FAIL per step).")
    parser.add_argument("labs", nargs="*", help="lab numbers to check (default: all)")
    parser.add_argument("--solutions", action="store_true",
                        help="check the reference solutions in labs/solutions/ instead")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print the full traceback of failing steps")
    args = parser.parse_args(argv)
    nums = _parse_lab_numbers(args.labs)
    folder = SOLUTIONS_DIR if args.solutions else LABS_DIR
    label = ("reference solutions in " if args.solutions else "your code in ") + _short(folder) + "/"
    return _run(nums, folder, args.verbose, label)


def run_file(path):
    """Entry point used by `python labs/labNN_*.py`: check that one file."""
    path = Path(path).resolve()
    num = int(path.name[3:5])
    _limit_threads()
    style = Style(sys.stdout.isatty() and "NO_COLOR" not in os.environ)
    status = check_lab(num, path, style, verbose="-v" in sys.argv[1:])[0]
    return 0 if status == "PASS" else 1


# ===========================================================================
# Shared references (independent of the lab files)
# ===========================================================================
def ref_causal_allowed(n_q, n_k=None):
    """Brute-force causal mask with a Python loop: True = may attend."""
    n_k = n_q if n_k is None else n_k
    off = n_k - n_q
    return torch.tensor([[j <= i + off for j in range(n_k)] for i in range(n_q)])


def copy_into_torch_mha(q_proj, k_proj, v_proj, o_proj, n_heads):
    """Build nn.MultiheadAttention (batch_first) holding the given projection weights."""
    d = q_proj.weight.shape[0]
    bias = q_proj.bias is not None
    mha = nn.MultiheadAttention(d, n_heads, bias=bias, batch_first=True)
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], 0))
        mha.out_proj.weight.copy_(o_proj.weight)
        if bias:
            mha.in_proj_bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], 0))
            mha.out_proj.bias.copy_(o_proj.bias)
    return mha.eval()


def torch_mha_forward(mha, x, causal):
    """Run nn.MultiheadAttention as self-attention; returns (out, per-head weights)."""
    n = x.shape[1]
    attn_mask = ~ref_causal_allowed(n) if causal else None  # torch: True = NOT allowed
    out, w = mha(x, x, x, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
    return out, w


def randomize_(module, std=0.2):
    """Fill every parameter of `module` with random values (so ones/zeros init cannot hide bugs)."""
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0.0, std)
    return module


# ===========================================================================
# Lab 01: attention
# ===========================================================================
@step(1, "causal_mask (also with more keys than queries)")
def _check_causal_mask(lab):
    for n_q, n_k in [(1, 1), (4, 4), (6, 6), (2, 4), (1, 5), (3, 7)]:
        got = lab.causal_mask(n_q) if n_q == n_k else lab.causal_mask(n_q, n_k)
        hint = None
        if n_k > n_q:
            hint = ("with n_k > n_q the queries are the LAST n_q positions: "
                    "query i is at absolute position i + (n_k - n_q)")
        expect_equal(got, ref_causal_allowed(n_q, n_k), f"causal_mask({n_q}, {n_k})", hint=hint)


@step(1, "softmax: stable, exact zeros for -inf")
def _check_softmax(lab):
    x = torch.randn(3, 5, 7) * 3
    expect_close(lab.softmax(x, dim=-1), torch.softmax(x, dim=-1), "softmax(x, dim=-1)")
    expect_close(lab.softmax(x, dim=0), torch.softmax(x, dim=0), "softmax(x, dim=0)")
    big = torch.tensor([[1000.0, 1001.0, 1002.0], [-1000.0, -1001.0, -1002.0]])
    expect_close(lab.softmax(big, dim=-1), torch.softmax(big, dim=-1),
                 "softmax([1000, 1001, 1002])",
                 hint="exp(1000) overflows to inf; subtract the row max before exp")
    masked = torch.tensor([[0.5, float("-inf"), 1.5, float("-inf")]])
    got = lab.softmax(masked, dim=-1)
    expect_close(got, torch.softmax(masked, dim=-1), "softmax with -inf entries")
    expect(bool((got[0, 1] == 0) & (got[0, 3] == 0)),
           "softmax: entries equal to -inf must get exactly 0 probability")


@step(1, "scaled_dot_product_attention vs torch SDPA")
def _check_sdpa(lab):
    B, h, N, d_k, d_v = 2, 3, 5, 8, 6
    q, k, v = torch.randn(B, h, N, d_k), torch.randn(B, h, N, d_k), torch.randn(B, h, N, d_v)
    res = lab.scaled_dot_product_attention(q, k, v)
    expect(isinstance(res, tuple) and len(res) == 2,
           "scaled_dot_product_attention must return a tuple (out, weights)")
    out, w = res
    ref_w = torch.softmax(q.double() @ k.double().transpose(-2, -1) / math.sqrt(d_k), -1)
    hint = None
    if is_close(w, torch.softmax(q @ k.transpose(-2, -1), -1)):
        hint = "the scores are not scaled: divide Q K^T by sqrt(d_k)"
    elif is_close(w, torch.softmax(q @ k.transpose(-2, -1) / d_k, -1)):
        hint = "divide by sqrt(d_k), not by d_k"
    elif is_close(w, torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(d_k), -2)):
        hint = "softmax over the KEY axis (dim=-1), so every row sums to 1"
    expect_close(w, ref_w, "attention weights (no mask)", hint=hint)
    expect_close(out, F.scaled_dot_product_attention(q, k, v), "attention output (no mask)")

    out, w = lab.scaled_dot_product_attention(q, k, v, ref_causal_allowed(N))
    expect_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True),
                 "attention output (causal mask)",
                 hint="mask=True means 'may attend'; fill the scores where mask is False with -inf")
    upper = ~ref_causal_allowed(N)
    expect(bool((w[..., upper] == 0).all()),
           "causal weights must be exactly 0 above the diagonal (query i must not see key j > i)")
    expect_close(w.sum(-1), torch.ones(B, h, N), "row sums of the causal weights")

    # More keys than queries (the KV-cache situation of lab 03).
    n_q, n_k = 3, 7
    q2 = torch.randn(B, h, n_q, d_k)
    k2, v2 = torch.randn(B, h, n_k, d_k), torch.randn(B, h, n_k, d_v)
    m = ref_causal_allowed(n_q, n_k)
    out, w = lab.scaled_dot_product_attention(q2, k2, v2, m)
    expect_close(out, F.scaled_dot_product_attention(q2, k2, v2, attn_mask=m),
                 "attention output with 3 queries over 7 keys")

    # The worked example from transformer_guide.html#worked-example ("The cat sat").
    q_sat = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    keys = torch.tensor([[0.0, 1.0, -0.5, 0.0], [0.9, 0.1, 0.8, 0.2], [0.1, 0.0, 0.2, 1.0]])
    vals = torch.tensor([[0.1, 0.9, -0.2, 0.4], [0.6, -0.1, 0.8, 0.3], [0.3, 0.5, 0.1, -0.2]])
    scores = [sum(a * b for a, b in zip(q_sat[0].tolist(), kk)) / 2.0 for kk in keys.tolist()]
    e = [math.exp(s) for s in scores]
    probs = [x / sum(e) for x in e]
    want_out = [sum(p * vv[c] for p, vv in zip(probs, vals.tolist())) for c in range(4)]
    out, w = lab.scaled_dot_product_attention(q_sat, keys, vals)
    expect_close(w, torch.tensor([probs]), "guide worked example: weights of 'sat'")
    expect_close(out, torch.tensor([want_out]), "guide worked example: output of 'sat'")
    return "weights for 'sat' = " + ", ".join(f"{p:.2f}" for p in probs)


@step(1, "split_heads / merge_heads")
def _check_split_merge(lab):
    B, N, h, d_k = 2, 5, 3, 4
    x = torch.randn(B, N, h * d_k)
    s = lab.split_heads(x, h)
    hint = None
    if is_close(s, x.reshape(B, h, N, d_k)):
        hint = ("you reshaped [B, N, d] straight to [B, h, N, d_k], which mixes tokens and heads; "
                "view as [B, N, h, d_k] first, then transpose(1, 2)")
    want = torch.stack([x[:, :, i * d_k:(i + 1) * d_k] for i in range(h)], dim=1)
    expect_close(s, want, "split_heads(x, 3)", atol=0, rtol=0, hint=hint)
    y = torch.randn(B, h, N, d_k)
    want_m = torch.cat([y[:, i] for i in range(h)], dim=-1)
    expect_close(lab.merge_heads(y), want_m, "merge_heads(y)", atol=0, rtol=0)
    expect_close(lab.merge_heads(lab.split_heads(x, h)), x, "merge_heads(split_heads(x))",
                 atol=0, rtol=0)


@step(1, "MultiHeadAttention vs nn.MultiheadAttention")
def _check_mha(lab):
    d, h, B, N = 16, 4, 2, 6
    mine = lab.MultiHeadAttention(d, h).eval()
    ref = copy_into_torch_mha(mine.q_proj, mine.k_proj, mine.v_proj, mine.out_proj, h)
    x = torch.randn(B, N, d)
    with torch.no_grad():
        for causal in (False, True):
            res = mine(x, causal=causal)
            expect(isinstance(res, tuple) and len(res) == 2,
                   "MultiHeadAttention.forward must return (out, weights)")
            out, w = res
            ref_out, ref_w = torch_mha_forward(ref, x, causal)
            tag = "causal" if causal else "bidirectional"
            expect_close(w, ref_w, f"per-head attention weights ({tag})")
            expect_close(out, ref_out, f"MultiHeadAttention output ({tag})")
    return "out and per-head weights match, causal and bidirectional"


# ===========================================================================
# Lab 02: transformer block
# ===========================================================================
def ref_rmsnorm(x, weight, eps):
    xd = x.double()
    return (weight.double() * xd * torch.rsqrt(xd.pow(2).mean(-1, keepdim=True) + eps)).float()


def ref_norm(mod, x):
    """Reference for the lab's LayerNorm / RMSNorm module, using F.layer_norm or the formula."""
    if hasattr(mod, "bias"):
        return F.layer_norm(x, (x.shape[-1],), mod.weight, mod.bias, mod.eps)
    return ref_rmsnorm(x, mod.weight, mod.eps)


def ref_ffn(mod, x):
    if hasattr(mod, "fc1"):
        return F.linear(F.gelu(F.linear(x, mod.fc1.weight, mod.fc1.bias)), mod.fc2.weight, mod.fc2.bias)
    gate = F.linear(x, mod.w_gate.weight, mod.w_gate.bias)
    up = F.linear(x, mod.w_up.weight, mod.w_up.bias)
    return F.linear(F.silu(gate) * up, mod.w_down.weight, mod.w_down.bias)


@step(2, "LayerNorm vs F.layer_norm")
def _check_layernorm(lab):
    d = 16
    ln = randomize_(lab.LayerNorm(d))
    x = torch.randn(2, 5, d) * 2 + 0.5
    with torch.no_grad():
        got = ln(x)
        want = F.layer_norm(x, (d,), ln.weight, ln.bias, ln.eps)
        hint = None
        mu = x.mean(-1, keepdim=True)
        unbiased = ln.weight * (x - mu) / torch.sqrt(x.var(-1, keepdim=True) + ln.eps) + ln.bias
        if is_close(got, unbiased):
            hint = "LayerNorm divides the variance by d, not d - 1: use unbiased=False"
        expect_close(got, want, "LayerNorm(x)", hint=hint)
        x_off = torch.randn(4, d) + 300.0
        expect_close(ln(x_off), F.layer_norm(x_off, (d,), ln.weight, ln.bias, ln.eps),
                     "LayerNorm(x + 300)", atol=2e-4, rtol=1e-3,
                     hint="compute the variance as mean((x - mu)^2); E[x^2] - E[x]^2 "
                          "cancels catastrophically in float32 when the mean is large")


@step(2, "RMSNorm: formula, scale invariance")
def _check_rmsnorm(lab):
    d = 16
    rms = randomize_(lab.RMSNorm(d))
    x = torch.randn(2, 5, d) + 0.7
    with torch.no_grad():
        got = rms(x)
        hint = None
        centred = x - x.mean(-1, keepdim=True)
        if is_close(got, ref_rmsnorm(centred, rms.weight, rms.eps)):
            hint = "RMSNorm does NOT subtract the mean (that is what makes it cheaper than LayerNorm)"
        expect_close(got, ref_rmsnorm(x, rms.weight, rms.eps), "RMSNorm(x)", hint=hint)
        expect_close(rms(3.0 * x), got, "RMSNorm(3x) vs RMSNorm(x)", atol=1e-4,
                     hint="RMSNorm output should not change when the input is scaled")


@step(2, "GeluMLP and SwiGLU")
def _check_ffn(lab):
    d, f = 12, 40
    x = torch.randn(2, 5, d)
    mlp = randomize_(lab.GeluMLP(d, f))
    with torch.no_grad():
        got = mlp(x)
        hint = None
        tanh = F.linear(F.gelu(F.linear(x, mlp.fc1.weight, mlp.fc1.bias), approximate="tanh"),
                        mlp.fc2.weight, mlp.fc2.bias)
        if is_close(got, tanh, atol=1e-6) and not is_close(got, ref_ffn(mlp, x), atol=1e-6):
            hint = "use the exact GELU (F.gelu default), not approximate='tanh'"
        expect_close(got, ref_ffn(mlp, x), "GeluMLP(x)", hint=hint)
        sw = randomize_(lab.SwiGLU(d, f))
        got = sw(x)
        hint = None
        swapped = F.linear(F.silu(F.linear(x, sw.w_up.weight)) * F.linear(x, sw.w_gate.weight),
                           sw.w_down.weight)
        if is_close(got, swapped):
            hint = "SiLU belongs on the w_gate branch, not on w_up"
        expect_close(got, ref_ffn(sw, x), "SwiGLU(x)", hint=hint)


@step(2, "SelfAttention vs nn.MultiheadAttention")
def _check_block_attention(lab):
    d, h = 16, 4
    x = torch.randn(2, 6, d)
    for bias in (True, False):
        att = randomize_(lab.SelfAttention(d, h, bias=bias)).eval()
        ref = copy_into_torch_mha(att.q_proj, att.k_proj, att.v_proj, att.o_proj, h)
        with torch.no_grad():
            for causal in (True, False):
                expect_close(att(x, causal=causal), torch_mha_forward(ref, x, causal)[0],
                             f"SelfAttention(bias={bias}, causal={causal})")


def ref_block_forward(blk, x, causal):
    mha = copy_into_torch_mha(blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj,
                              blk.attn.o_proj, blk.attn.n_heads)
    h = x + torch_mha_forward(mha, ref_norm(blk.norm1, x), causal)[0]
    return h + ref_ffn(blk.ffn, ref_norm(blk.norm2, h))


@step(2, "Block: pre-norm, vs a torch reference")
def _check_block(lab):
    d, h, f = 16, 4, 48
    x = torch.randn(2, 7, d)
    for norm, ffn, bias in (("layernorm", "gelu", True), ("rmsnorm", "swiglu", False)):
        blk = randomize_(lab.Block(d, h, f, norm=norm, ffn=ffn, bias=bias)).eval()
        with torch.no_grad():
            for p in (blk.norm1.weight, blk.norm2.weight):
                p.add_(1.0)
            got = blk(x, causal=True)
            hint = None
            mha = copy_into_torch_mha(blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj,
                                      blk.attn.o_proj, h)
            post_h = ref_norm(blk.norm1, x + torch_mha_forward(mha, x, True)[0])
            post = ref_norm(blk.norm2, post_h + ref_ffn(blk.ffn, post_h))
            if is_close(got, post):
                hint = ("this is POST-norm (norm after the add). Pre-norm: "
                        "x = x + attn(norm1(x)); x = x + ffn(norm2(x))")
            expect_close(got, ref_block_forward(blk, x, True),
                         f"Block({norm}, {ffn}, bias={bias}) output", hint=hint)
    return "LayerNorm+GELU (GPT-2) and RMSNorm+SwiGLU (Llama) variants"


@step(2, "count_block_params vs numel, GPT-2, Llama-2")
def _check_param_count(lab):
    def breakdown(d, f, norm, ffn, bias):
        attn = 4 * d * d + (4 * d if bias else 0)
        mlp = (2 * d * f + ((f + d) if bias else 0)) if ffn == "gelu" \
            else (3 * d * f + ((2 * f + d) if bias else 0))
        norms = 2 * (2 * d if norm == "layernorm" else d)
        return f"attention {attn:,} + ffn {mlp:,} + norms {norms:,}"

    small = [(16, 4, 48, "layernorm", "gelu", True), (16, 2, 48, "layernorm", "gelu", False),
             (24, 3, 64, "rmsnorm", "swiglu", False), (24, 6, 64, "rmsnorm", "swiglu", True),
             (32, 4, 128, "rmsnorm", "gelu", True), (32, 8, 80, "layernorm", "swiglu", False)]
    for d, h, f, norm, ffn, bias in small:
        want = sum(p.numel() for p in lab.Block(d, h, f, norm=norm, ffn=ffn, bias=bias).parameters())
        expect_int(lab.count_block_params(d, h, f, norm=norm, ffn=ffn, bias=bias), want,
                   f"count_block_params(d={d}, h={h}, d_ff={f}, {norm}, {ffn}, bias={bias})",
                   hint="expected " + breakdown(d, f, norm, ffn, bias))
    gpt2 = lab.count_block_params(768, 12, 3072, norm="layernorm", ffn="gelu", bias=True)
    expect_int(gpt2, 7_087_872, "GPT-2 small block (d=768, d_ff=3072)",
               hint="expected " + breakdown(768, 3072, "layernorm", "gelu", True))
    llama = lab.count_block_params(4096, 32, 11008, norm="rmsnorm", ffn="swiglu", bias=False)
    expect_int(llama, 202_383_360, "Llama-2-7B block (d=4096, d_ff=11008)",
               hint="expected " + breakdown(4096, 11008, "rmsnorm", "swiglu", False))
    total = 32 * llama + 2 * 32000 * 4096 + 4096
    return f"GPT-2 block 7.09M; Llama-2-7B: 32 blocks + embeddings = {total / 1e9:.2f}B"


# ===========================================================================
# Lab 03: GPT forward, sampling, KV cache
# ===========================================================================
def ref_split_qkv(attn, x):
    B, T, d = x.shape
    h = attn.n_heads
    q, k, v = F.linear(x, attn.qkv.weight, attn.qkv.bias).split(d, dim=-1)
    return [t.view(B, T, h, d // h).transpose(1, 2) for t in (q, k, v)]


def ref_gpt_forward(model, idx):
    """Independent full-sequence forward pass built from torch functionals."""
    B, T = idx.shape
    d = model.cfg.d_model
    x = model.tok_emb.weight[idx] + model.pos_emb.weight[:T]
    for blk in model.blocks:
        h = F.layer_norm(x, (d,), blk.ln1.weight, blk.ln1.bias, blk.ln1.eps)
        q, k, v = ref_split_qkv(blk.attn, h)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        x = x + F.linear(y, blk.attn.proj.weight, blk.attn.proj.bias)
        h = F.layer_norm(x, (d,), blk.ln2.weight, blk.ln2.bias, blk.ln2.eps)
        x = x + F.linear(F.gelu(F.linear(h, blk.mlp[0].weight, blk.mlp[0].bias)),
                         blk.mlp[2].weight, blk.mlp[2].bias)
    x = F.layer_norm(x, (d,), model.ln_f.weight, model.ln_f.bias, model.ln_f.eps)
    return F.linear(x, model.tok_emb.weight)


def _gpt(lab, **kw):
    torch.manual_seed(3)
    model = lab.GPT(lab.GPTConfig(**kw)).eval()
    with torch.no_grad():  # non-trivial LayerNorm parameters
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.normal_(1.0, 0.2)
                m.bias.normal_(0.0, 0.2)
    return model


def _unpack_forward(res, what):
    expect(isinstance(res, tuple) and len(res) == 2, f"{what} must return a tuple (logits, presents)")
    return res


@step(3, "CausalSelfAttention: output and (k, v)")
def _check_gpt_attention(lab):
    model = _gpt(lab)
    attn = model.blocks[0].attn
    B, T, d = 2, 7, 32
    x = torch.randn(B, T, d)
    with torch.no_grad():
        res = attn(x)
        expect(isinstance(res, tuple) and len(res) == 2,
               "CausalSelfAttention.forward must return (y, (k, v))")
        y, present = res
        q, k, v = ref_split_qkv(attn, x)
        want = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
        want = F.linear(want, attn.proj.weight, attn.proj.bias)
        expect_close(y, want, "attention output y")
        expect(isinstance(present, (tuple, list)) and len(present) == 2,
               "present must be a tuple (k, v)")
        expect_close(present[0], k, "present k ([B, h, T, d_k], after the head split)")
        expect_close(present[1], v, "present v ([B, h, T, d_k], after the head split)")


@step(3, "GPT.forward: logits vs reference, causal")
def _check_gpt_forward(lab):
    model = _gpt(lab)
    idx = torch.randint(0, 64, (2, 11))
    with torch.no_grad():
        logits, presents = _unpack_forward(model(idx), "GPT.forward")
        want = ref_gpt_forward(model, idx)
        expect_close(logits, want, "logits [B, T, V]",
                     hint="x = tok_emb(idx) + pos_emb(positions); pre-norm blocks; ln_f; lm_head")
        expect(isinstance(presents, (list, tuple)) and len(presents) == model.cfg.n_layers,
               f"presents must be a list with one (k, v) per block ({model.cfg.n_layers})")
        expect_shape(presents[0][0], (2, 4, 11, 8), "presents[0][0] (k of block 0)")
        idx2 = idx.clone()
        idx2[:, 6] = (idx2[:, 6] + 1) % 64
        logits2, _ = model(idx2)
        expect_close(logits2[:, :6], logits[:, :6], "logits before position 6 after changing token 6",
                     hint="changing a later token must not change earlier logits: is the mask causal?")
        expect(not is_close(logits2[:, 6:], logits[:, 6:]),
               "logits at and after position 6 should change when token 6 changes")


@step(3, "KV cache: prefill + decode == full recompute")
def _check_kv_cache(lab):
    model = _gpt(lab)
    idx = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        full, _ = model(idx)
        logits, past = _unpack_forward(model(idx[:, :5]), "GPT.forward")
        expect_close(logits, full[:, :5], "prefill logits (first 5 tokens)")
        for t in range(5, 12):
            logits, past = model(idx[:, t:t + 1], past)
            expect_shape(logits, (2, 1, 64), f"decode-step logits at position {t}")
            expect_close(logits[:, 0], full[:, t], f"cached decode logits at position {t}",
                         hint="the new token's position is the cached length; concatenate k, v "
                              "with the cache along dim=2; mask with causal_mask(T, T_past + T), "
                              "not is_causal=True (torch aligns that mask to the top-left)")
        expect_shape(past[0][0], (2, 4, 12, 8), "cache k after 12 tokens")
        # Chunked prefill: 5 tokens, then 4 at once, then 3 at once.
        _, past = model(idx[:, :5])
        logits, past = model(idx[:, 5:9], past)
        expect_close(logits, full[:, 5:9], "logits for a 4-token chunk after 5 cached tokens",
                     hint="with T_past cached keys use causal_mask(T, T_past + T), not is_causal=True "
                          "(torch's is_causal aligns the mask to the top-left)")
        logits, past = model(idx[:, 9:12], past)
        expect_close(logits, full[:, 9:12], "logits for a 3-token chunk after 9 cached tokens")


@step(3, "greedy_next and sample_top_k")
def _check_sampling(lab):
    logits = torch.tensor([[0.1, 2.0, -1.0, 0.3], [5.0, 4.9, 0.0, -2.0]])
    got = lab.greedy_next(logits)
    expect_equal(got, torch.tensor([1, 0]), "greedy_next")
    rnd = torch.randn(8, 20)
    expect_equal(lab.sample_top_k(rnd, 1), rnd.argmax(-1), "sample_top_k with k=1 (must be greedy)")
    base = torch.tensor([2.0, 1.0, 0.5, 0.0, -1.0, 3.0])
    n = 20000
    for temperature in (1.0, 0.5):
        gen = torch.Generator().manual_seed(7)
        draws = lab.sample_top_k(base.expand(n, -1), 3, temperature, gen)
        expect_shape(draws, (n,), "sample_top_k output")
        expect(draws.dtype == torch.int64, f"sample_top_k must return int64 ids, got {draws.dtype}")
        outside = int(((draws != 5) & (draws != 0) & (draws != 1)).sum())
        expect(outside == 0, f"sample_top_k(k=3) drew {outside} tokens outside the top 3 "
                             f"(ids 5, 0, 1)")
        freq = torch.stack([(draws == i).double().mean() for i in (5, 0, 1)])
        want = torch.softmax(torch.tensor([3.0, 2.0, 1.0], dtype=torch.float64) / temperature, 0)
        expect_close(freq, want, f"empirical top-3 frequencies at temperature {temperature}",
                     atol=0.015, rtol=0,
                     hint="softmax over the k kept logits, after dividing them by the temperature")
    x = torch.randn(64, 30)
    a = lab.sample_top_k(x, 5, 1.0, torch.Generator().manual_seed(11))
    b = lab.sample_top_k(x, 5, 1.0, torch.Generator().manual_seed(11))
    expect_equal(a, b, "two calls with identically seeded generators",
                 hint="pass generator=generator to torch.multinomial")


@step(3, "generate: cached == uncached, 1 token/step")
def _check_generate(lab):
    model = _gpt(lab)
    prompt = torch.randint(0, 64, (2, 5))
    calls = []
    orig = model.forward

    def spy(idx, *args, **kwargs):
        calls.append(idx.shape[1])
        return orig(idx, *args, **kwargs)

    model.forward = spy
    out_cache = lab.generate(model, prompt, 10, use_cache=True)
    seen_cache, calls[:] = list(calls), []
    out_full = lab.generate(model, prompt, 10, use_cache=False)
    seen_full, calls[:] = list(calls), []
    model.forward = orig
    expect_shape(out_full, (2, 15), "generate(..., use_cache=False)")
    expect_shape(out_cache, (2, 15), "generate(..., use_cache=True)")
    expect_equal(out_full[:, :5], prompt, "the output must start with the prompt")
    # Reference greedy decoding with the independent forward pass.
    ref = prompt.clone()
    with torch.no_grad():
        for _ in range(10):
            ref = torch.cat([ref, ref_gpt_forward(model, ref)[:, -1].argmax(-1, keepdim=True)], 1)
    expect_equal(out_full, ref, "greedy tokens without a cache vs reference decoding")
    expect_equal(out_cache, out_full, "greedy tokens with the KV cache vs without")
    expect(seen_full[:3] == [5, 6, 7],
           f"without a cache every step should run the whole sequence (lengths 5, 6, 7, ...); "
           f"your model calls saw lengths {seen_full}")
    expect(len(seen_cache) >= 2 and seen_cache[0] == 5 and all(n == 1 for n in seen_cache[1:]),
           f"with the cache: one prefill call of length 5, then calls with 1 new token each; "
           f"your model calls saw lengths {seen_cache}")
    a = lab.generate(model, prompt, 10, use_cache=True, top_k=5, temperature=0.8,
                     generator=torch.Generator().manual_seed(5))
    b = lab.generate(model, prompt, 10, use_cache=False, top_k=5, temperature=0.8,
                     generator=torch.Generator().manual_seed(5))
    expect_equal(a, b, "top-k sampled tokens with and without the cache (same seed)")
    return f"model calls with cache: {seen_cache[:4]}...; without: {seen_full[:4]}..."


# ===========================================================================
# Lab 04: ViT
# ===========================================================================
def ref_patchify(images, p):
    """Reference patchify with F.unfold: [B, C*p*p, N] -> [B, N, C*p*p]."""
    return F.unfold(images, kernel_size=p, stride=p).transpose(1, 2)


def ref_unpatchify(patches, H, W, p):
    return F.fold(patches.transpose(1, 2), output_size=(H, W), kernel_size=p, stride=p)


@step(4, "patchify / unpatchify vs F.unfold")
def _check_patchify(lab):
    B, C, H, W, p = 2, 3, 16, 24, 4
    img = torch.randn(B, C, H, W)
    got = lab.patchify(img, p)
    hint = None
    channels_last = img.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 3, 5, 1) \
        .reshape(B, -1, C * p * p)
    if is_close(got, channels_last):
        hint = "each patch must be flattened in (channel, row, column) order, channel first"
    elif is_close(got, img.reshape(B, C * p * p, -1).transpose(1, 2)):
        hint = "a plain reshape does not cut out p x p tiles: split H and W into (grid, p) first"
    expect_close(got, ref_patchify(img, p), "patchify(images, 4) on a 16 x 24 image",
                 atol=0, rtol=0, hint=hint)
    back = lab.unpatchify(ref_patchify(img, p), C, H, W, p)
    expect_close(back, img, "unpatchify(patches)", atol=0, rtol=0)
    return f"[2, 3, 16, 24] -> {list(got.shape)} -> back"


@step(4, "PatchEmbed: Conv2d path == matmul path")
def _check_patch_embed(lab):
    pe = lab.PatchEmbed(img_size=32, patch_size=8, in_chans=3, dim=20)
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        want = F.conv2d(x, pe.proj.weight, pe.proj.bias, stride=8).flatten(2).transpose(1, 2)
        conv = pe.forward_conv(x)
        hint = None
        if isinstance(conv, torch.Tensor) and tuple(conv.shape) == (2, 20, 16):
            hint = "you return [B, d, N]; transpose to [B, N, d] (tokens first)"
        expect_close(conv, want, "forward_conv", hint=hint)
        lin = pe.forward_linear(x)
        expect_close(lin, want, "forward_linear (patchify + matmul)",
                     hint="reshape proj.weight [d, C, p, p] to [d, C*p*p] and use x @ W.T + b")
        expect_close(lin, conv, "forward_linear vs forward_conv")
    return "the same linear map, max |diff| = %.1e" % (lin - conv).abs().max().item()


def ref_vit_forward(model, images):
    B = images.shape[0]
    pe = model.patch_embed
    x = F.conv2d(images, pe.proj.weight, pe.proj.bias, stride=pe.patch_size).flatten(2).transpose(1, 2)
    x = torch.cat([model.cls_token.expand(B, -1, -1), x], dim=1) + model.pos_embed
    for blk in model.blocks:
        x = blk(x)
    x = F.layer_norm(x, (x.shape[-1],), model.norm.weight, model.norm.bias, model.norm.eps)
    return F.linear(x[:, 0], model.head.weight, model.head.bias)


def _vit(lab):
    torch.manual_seed(4)
    model = lab.ViT(lab.ViTConfig()).eval()
    with torch.no_grad():
        model.cls_token.normal_(0, 0.5)
        model.pos_embed.normal_(0, 0.5)
    return model


def _shuffle_patches(img, p, perm):
    H, W = img.shape[-2:]
    return ref_unpatchify(ref_patchify(img, p)[:, perm], H, W, p)


@step(4, "ViT.forward: CLS, positions, logits")
def _check_vit_forward(lab):
    model = _vit(lab)
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        got = model(x)
        want = ref_vit_forward(model, x)
        hint = None
        # Common mistake: CLS appended at the end, or position embeddings forgotten.
        pe = model.patch_embed
        toks = F.conv2d(x, pe.proj.weight, pe.proj.bias, stride=8).flatten(2).transpose(1, 2)
        no_pos = torch.cat([model.cls_token.expand(3, -1, -1), toks], 1)
        for blk in model.blocks:
            no_pos = blk(no_pos)
        no_pos = model.head(model.norm(no_pos)[:, 0])
        if is_close(got, no_pos):
            hint = "add self.pos_embed after prepending the CLS token"
        expect_close(got, want, "ViT logits", hint=hint)
        # Position embeddings are what make the model see the patch order.
        perm = torch.randperm(16)
        shuffled = _shuffle_patches(x, 8, perm)
        expect(not is_close(model(shuffled), got),
               "with position embeddings, shuffling the patches should change the logits")
        model.pos_embed.zero_()
        a, b = model(x), model(shuffled)
        expect_close(b, a, "logits of shuffled patches with pos_embed = 0", atol=1e-4,
                     hint="without position embeddings a bidirectional encoder must be "
                          "permutation invariant: no causal mask in ViT")
    return "and without pos_embed, shuffling the patches changes nothing"


@step(4, "shape tracker: trace == vit_shapes()")
def _check_vit_shapes(lab):
    keys = ("input", "patch_tokens", "tokens", "encoded", "cls", "logits")

    def expected(b, img, p, c, d, k):
        n = (img // p) ** 2
        return {"input": (b, c, img, img), "patch_tokens": (b, n, d), "tokens": (b, n + 1, d),
                "encoded": (b, n + 1, d), "cls": (b, d), "logits": (b, k)}

    for args in ((2, 32, 8, 3, 48, 10), (1, 224, 16, 3, 768, 1000), (4, 448, 14, 3, 1024, 2)):
        got = lab.vit_shapes(*args)
        expect(isinstance(got, dict), f"vit_shapes must return a dict, got {type(got).__name__}")
        want = expected(*args)
        for key in keys:
            expect(key in got, f"vit_shapes{args} is missing the key {key!r}")
            expect(tuple(got[key]) == want[key],
                   f"vit_shapes{args}[{key!r}]: expected {want[key]}, got {tuple(got[key])}")
    model = _vit(lab)
    trace = {}
    with torch.no_grad():
        model(torch.randn(2, 3, 32, 32), trace=trace)
    want = expected(2, 32, 8, 3, 48, 10)
    for key in keys:
        expect(key in trace, f"forward(..., trace) did not record {key!r} (recorded: {list(trace)})")
        expect(tuple(trace[key]) == want[key],
               f"trace[{key!r}]: expected {want[key]}, got {tuple(trace[key])}")
    return "ViT-B/16 @ 224: tokens (1, 197, 768)"


# ===========================================================================
# Lab 05: VLM fusion
# ===========================================================================
@step(5, "count_image_tokens per resolution")
def _check_count_tokens(lab):
    cases = [((336, 336, 14), 1, 576), ((448, 448, 14), 1, 1024), ((448, 448, 14), 2, 256),
             ((224, 224, 16), 1, 196), ((480, 640, 16), 1, 1200), ((672, 896, 14), 2, 768),
             ((224, 224, 14), 1, 256), ((1344, 1344, 14), 2, 2304)]
    for (h, w, p), merge, want in cases:
        expect_int(lab.count_image_tokens(h, w, p, merge), want,
                   f"count_image_tokens({h}, {w}, patch={p}, merge={merge})")
    for h, w, p, merge in ((500, 500, 14, 1), (448, 450, 16, 1), (434, 434, 14, 2)):
        try:
            lab.count_image_tokens(h, w, p, merge)
        except ValueError:
            continue
        bad = h if h % (p * merge) else w
        fail(f"count_image_tokens({h}, {w}, patch={p}, merge={merge}) should raise ValueError "
             f"({bad} is not a multiple of patch * merge = {p * merge})")
    return "336px/14 -> 576, 448px/14 -> 1024, 2x2 merge -> 256"


@step(5, "pixel_shuffle_merge (2x2) vs a loop")
def _check_pixel_shuffle(lab):
    B, gh, gw, d, r = 2, 4, 6, 3, 2
    x = torch.randn(B, gh * gw, d)
    got = lab.pixel_shuffle_merge(x, gh, gw, r)
    rows = []
    for i in range(gh // r):
        for j in range(gw // r):
            parts = [x[:, (r * i + dy) * gw + (r * j + dx)] for dy in range(r) for dx in range(r)]
            rows.append(torch.cat(parts, dim=-1))
    want = torch.stack(rows, dim=1)
    hint = None
    if is_close(got, x.reshape(B, -1, r * r * d)):
        hint = ("a plain reshape merges 4 tokens of the same ROW; a 2x2 merge needs "
                "the two tokens below as well: reshape to [B, gh/r, r, gw/r, r, d]")
    expect_close(got, want, "pixel_shuffle_merge on a 4 x 6 grid", atol=0, rtol=0, hint=hint)
    n = lab.count_image_tokens(448, 448, 14, merge=2)
    big = lab.pixel_shuffle_merge(torch.zeros(1, 32 * 32, 8), 32, 32, 2)
    expect_shape(big, (1, n, 32), "merged 448px image tokens")
    return "1024 tokens x 8 ch -> 256 tokens x 32 ch"


@step(5, "MLPProjector: vision width -> LLM width")
def _check_projector(lab):
    proj = randomize_(lab.MLPProjector(24, 40))
    v = torch.randn(2, 9, 24)
    with torch.no_grad():
        want = F.linear(F.gelu(F.linear(v, proj.fc1.weight, proj.fc1.bias)), proj.fc2.weight, proj.fc2.bias)
        expect_close(proj(v), want, "MLPProjector output [2, 9, 40]")


def ref_joint_mask(n_img, n_txt, bidir):
    L = n_img + n_txt
    return torch.tensor([[j <= i or (bidir and i < n_img and j < n_img) for j in range(L)]
                         for i in range(L)])


@step(5, "build_sequence + joint_attention_mask")
def _check_joint_mask(lab):
    img, txt = torch.randn(2, 4, 8), torch.randn(2, 3, 8)
    res = lab.build_sequence(img, txt)
    expect(isinstance(res, tuple) and len(res) == 2, "build_sequence must return (embeds, is_image)")
    embeds, is_image = res
    expect_close(embeds, torch.cat([img, txt], 1), "embeds (image first, then text)", atol=0, rtol=0)
    expect_equal(is_image, torch.tensor([True] * 4 + [False] * 3), "is_image")
    for n_img, n_txt in ((4, 3), (1, 5), (6, 2)):
        for bidir in (False, True):
            hint = None
            if bidir:
                hint = "text must stay causal; only the image x image block becomes fully visible"
            expect_equal(lab.joint_attention_mask(n_img, n_txt, image_bidirectional=bidir),
                         ref_joint_mask(n_img, n_txt, bidir),
                         f"joint_attention_mask({n_img}, {n_txt}, image_bidirectional={bidir})",
                         hint=hint)


def ref_vlm_forward(model, feats, ids, bidir):
    p = model.projector
    img = F.linear(F.gelu(F.linear(feats, p.fc1.weight, p.fc1.bias)), p.fc2.weight, p.fc2.bias)
    x = torch.cat([img, model.tok_emb.weight[ids]], dim=1)
    mask = ref_joint_mask(feats.shape[1], ids.shape[1], bidir)
    return F.linear(model.decoder(x, mask), model.lm_head.weight)


@step(5, "TinyVLM.forward: logits, information flow")
def _check_vlm_forward(lab):
    torch.manual_seed(5)
    model = lab.TinyVLM(lab.VLMConfig()).eval()
    feats, ids = torch.randn(2, 6, 24), torch.randint(0, 50, (2, 5))
    n_img = 6
    with torch.no_grad():
        for bidir in (False, True):
            expect_close(model(feats, ids, image_bidirectional=bidir),
                         ref_vlm_forward(model, feats, ids, bidir),
                         f"logits [B, N_v + N_t, V] (image_bidirectional={bidir})")
        base = model(feats, ids)
        ids2 = ids.clone()
        ids2[:, 3] = (ids2[:, 3] + 1) % 50
        changed = model(feats, ids2)
        expect_close(changed[:, :n_img + 3], base[:, :n_img + 3],
                     "image outputs and earlier text after changing text token 3",
                     hint="image tokens must not see the text, and text must be causal")
        feats2 = feats.clone()
        feats2[:, 0] += 1.0
        expect(not is_close(model(feats2, ids)[:, n_img:], base[:, n_img:]),
               "text logits should change when the image changes (text attends to the image)")
        feats3 = feats.clone()
        feats3[:, -1] += 1.0
        expect_close(model(feats3, ids)[:, 0], base[:, 0],
                     "first image token after changing the last patch (causal mode)")
        bi_a, bi_b = model(feats, ids, True), model(feats3, ids, True)
        expect(not is_close(bi_a[:, 0], bi_b[:, 0]),
               "with image_bidirectional=True the first image token should see the last patch")
    return "image tokens never see text; text sees the whole image"


# ===========================================================================
# Lab 06: diffusion
# ===========================================================================
def ref_alpha_bars(T=1000):
    betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float64)
    return torch.cumprod(1 - betas, 0)


class GaussianEpsilon(nn.Module):
    """The exact epsilon-predictor when the data is N(mu, s^2 I): no training needed.

    For x_t = sqrt(ab) x0 + sqrt(1 - ab) eps with x0 ~ N(mu, s^2):
        E[eps | x_t] = sqrt(1 - ab) (x_t - sqrt(ab) mu) / (ab s^2 + 1 - ab)
    """

    def __init__(self, alpha_bars, mu, s):
        super().__init__()
        self.ab = alpha_bars
        self.mu = mu
        self.s = s

    def forward(self, x_t, t):
        ab = self.ab[t].to(x_t.dtype).view(-1, *([1] * (x_t.dim() - 1)))
        return (1 - ab).sqrt() * (x_t - ab.sqrt() * self.mu) / (ab * self.s ** 2 + 1 - ab)


def ref_ddim(model, x, alpha_bars, num_steps):
    T = len(alpha_bars)
    stride = T // num_steps
    ts = [k * stride - 1 for k in range(num_steps, 0, -1)]
    for i, t in enumerate(ts):
        eps = model(x, torch.full((x.shape[0],), t, dtype=torch.long))
        ab = alpha_bars[t].to(x.dtype)
        ab_next = alpha_bars[ts[i + 1]].to(x.dtype) if i + 1 < len(ts) \
            else torch.tensor(1.0, dtype=x.dtype)
        x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
    return x


def ring_centres(n_modes=8, radius=2.0):
    angle = 2 * math.pi * torch.arange(n_modes) / n_modes
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)


def ref_ring(n, generator, n_modes=8, radius=2.0, std=0.1):
    k = torch.randint(0, n_modes, (n,), generator=generator)
    return ring_centres(n_modes, radius)[k] + std * torch.randn(n, 2, generator=generator)


@step(6, "noise schedule and alpha_bar (guide values)")
def _check_schedule(lab):
    betas = lab.linear_beta_schedule(1000)
    expect_close(betas, torch.linspace(1e-4, 0.02, 1000, dtype=torch.float64),
                 "linear_beta_schedule(1000)", atol=1e-8, rtol=1e-5)
    expect_close(lab.linear_beta_schedule(10, 0.1, 0.2), torch.linspace(0.1, 0.2, 10),
                 "linear_beta_schedule(10, 0.1, 0.2)", atol=1e-7)
    small = torch.tensor([0.1, 0.2, 0.5])
    expect_close(lab.alpha_bars_from_betas(small), torch.tensor([0.9, 0.9 * 0.8, 0.9 * 0.8 * 0.5]),
                 "alpha_bars_from_betas([0.1, 0.2, 0.5])",
                 hint="abar_t = (1 - beta_0)(1 - beta_1)...(1 - beta_t): a cumulative product")
    ab = lab.alpha_bars_from_betas(betas)
    expect_close(ab, ref_alpha_bars(), "alpha_bars (T = 1000)", atol=1e-6, rtol=1e-4)
    s1, s2 = ab[500].sqrt().item(), (1 - ab[500]).sqrt().item()
    expect(abs(s1 - 0.2789) < 1e-4 and abs(s2 - 0.9603) < 1e-4,
           f"guide values: sqrt(abar_500) = 0.2789 and sqrt(1 - abar_500) = 0.9603; "
           f"got {s1:.4f} and {s2:.4f}")
    return f"abar_0 = {ab[0].item():.4f}, abar_500 = {ab[500].item():.4f}, abar_999 = {ab[999].item():.1e}"


@step(6, "q_sample: closed-form noising")
def _check_q_sample(lab):
    ab = ref_alpha_bars().float()
    x0 = torch.full((2, 1), 0.8)
    eps = torch.full((2, 1), -1.2)
    got = lab.q_sample(x0, torch.tensor([500, 250]), eps, ab)
    expect_close(got, torch.tensor([[-0.9292], [-0.2525]]),
                 "guide example x_500 and x_250 (x0 = 0.8, eps = -1.2)", atol=1e-4, rtol=0)
    for shape in ((6, 2), (4, 3, 5, 5)):
        x0 = torch.randn(*shape)
        eps = torch.randn(*shape)
        t = torch.randint(0, 1000, (shape[0],))
        a = ab.double()[t].view(-1, *([1] * (len(shape) - 1)))
        want = a.sqrt() * x0.double() + (1 - a).sqrt() * eps.double()
        expect_close(lab.q_sample(x0, t, eps, ab), want, f"q_sample on a {list(shape)} batch",
                     hint="index alpha_bars with t ([B]) and reshape it to [B, 1, ...] to broadcast")


@step(6, "epsilon_loss")
def _check_eps_loss(lab):
    ab = ref_alpha_bars().float()
    seen = {}

    def dummy(x_t, t):
        seen["x_t"] = x_t
        return 0.5 * x_t + t[:, None].float() / 1000.0

    x0, eps = torch.randn(8, 2), torch.randn(8, 2)
    t = torch.randint(0, 1000, (8,))
    loss = lab.epsilon_loss(dummy, x0, t, eps, ab)
    expect_tensor(loss, "epsilon_loss")
    expect(loss.dim() == 0, f"epsilon_loss must return a scalar, got shape {list(loss.shape)}")
    a = ab[t][:, None]
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * eps
    expect_close(seen.get("x_t", torch.zeros(0)), x_t, "the x_t passed to the model",
                 hint="the model sees the NOISED sample x_t, not x0")
    expect_close(loss, ((0.5 * x_t + t[:, None].float() / 1000 - eps) ** 2).mean(), "loss value")
    torch.manual_seed(6)
    net = lab.Denoiser()
    lab.epsilon_loss(net, x0, t, eps, ab).backward()
    expect(all(p.grad is not None for p in net.parameters()),
           "the loss must be differentiable w.r.t. the model parameters")


@step(6, "DDIM: timesteps, step, exact-Gaussian sampler")
def _check_ddim(lab):
    ab = ref_alpha_bars().float()
    expect_equal(lab.ddim_timesteps(1000, 50), torch.arange(999, 0, -20), "ddim_timesteps(1000, 50)")
    expect_equal(lab.ddim_timesteps(1000, 4), torch.tensor([999, 749, 499, 249]),
                 "ddim_timesteps(1000, 4)")
    # With the true noise, one DDIM step from level t1 lands exactly on q_sample(x0, t2, eps).
    x0, eps = torch.randn(5, 2), torch.randn(5, 2)
    for t1, t2 in ((500, 250), (999, 19), (300, 299)):
        x1 = ab[t1].sqrt() * x0 + (1 - ab[t1]).sqrt() * eps
        x2 = ab[t2].sqrt() * x0 + (1 - ab[t2]).sqrt() * eps
        expect_close(lab.ddim_step(x1, eps, ab[t1], ab[t2]), x2, f"ddim_step from t={t1} to t={t2}",
                     atol=1e-4, hint="x0_hat = (x_t - sqrt(1-abar_t) eps) / sqrt(abar_t); "
                                     "x_next = sqrt(abar_next) x0_hat + sqrt(1-abar_next) eps")
    x1 = ab[300].sqrt() * x0 + (1 - ab[300]).sqrt() * eps
    expect_close(lab.ddim_step(x1, eps, ab[300], 1.0), x0, "ddim_step to abar_next = 1.0 (a float)",
                 atol=1e-4)
    # Exact epsilon for N(mu, s^2): DDIM must turn N(0, I) into N(mu, s^2).
    mu, s = torch.tensor([1.0, -2.0]), 0.5
    model = GaussianEpsilon(ab, mu, s)
    x_T = torch.randn(4000, 2, generator=torch.Generator().manual_seed(1))
    got = lab.ddim_sample(model, x_T.clone(), ab, 50)
    expect_close(got, ref_ddim(model, x_T.clone(), ab, 50), "ddim_sample (50 steps) vs reference",
                 atol=1e-4, rtol=1e-4,
                 hint="visit ddim_timesteps(T, num_steps); after the last one use abar_next = 1.0")
    m, sd = got.mean(0), got.std(0)
    expect(bool(((m - mu).abs() < 0.05).all() and ((sd - s).abs() < 0.05).all()),
           f"samples should be N(mu={mu.tolist()}, std={s}); got mean {m.tolist()} std {sd.tolist()}")
    return f"exact-eps Gaussian: mean ({m[0]:.2f}, {m[1]:.2f}), std ({sd[0]:.3f}, {sd[1]:.3f})"


@step(6, "train on 8 Gaussians; samples on the ring")
def _check_diffusion_training(lab):
    ab = lab.alpha_bars_from_betas(lab.linear_beta_schedule(1000))
    data = ref_ring(20000, torch.Generator().manual_seed(0))
    torch.manual_seed(0)
    model = lab.Denoiser()
    t0 = time.perf_counter()
    losses = lab.train_denoiser(model, data, ab, steps=3000, batch_size=512, lr=2e-3, seed=0)
    train_s = time.perf_counter() - t0
    expect(isinstance(losses, (list, tuple)) and len(losses) == 3000,
           "train_denoiser must return the list of 3000 per-step losses")
    first, last = sum(losses[:50]) / 50, sum(losses[-200:]) / 200
    expect(last < 0.35 and last < first,
           f"the loss should fall below 0.35 (the reference reaches ~0.23); "
           f"first 50 steps {first:.3f}, last 200 steps {last:.3f}")
    model.eval()
    x_T = torch.randn(4000, 2, generator=torch.Generator().manual_seed(1))
    samples = lab.ddim_sample(model, x_T, ab, 50)
    expect_shape(samples, (4000, 2), "ddim_sample output")
    centres = ring_centres()
    dist, mode = torch.cdist(samples, centres).min(1)
    near = (dist < 0.5).float().mean().item()
    shares = torch.bincount(mode, minlength=8).float() / len(samples)
    radius = samples.norm(dim=1).mean().item()
    noise_near = (torch.cdist(x_T, centres).min(1).values < 0.5).float().mean().item()
    expect(near > 0.8, f"only {near:.0%} of the samples are within 0.5 of a mode centre "
                       f"(need > 80%; pure noise gives {noise_near:.0%})")
    expect(bool((shares > 0.05).all() and (shares < 0.25).all()),
           f"each of the 8 modes should get 5-25% of the samples; got "
           f"{[round(v, 3) for v in shares.tolist()]}")
    expect(abs(radius - 2.0) < 0.15, f"mean sample radius should be about 2.0, got {radius:.3f}")
    return (f"trained in {train_s:.1f} s, loss {first:.2f} -> {last:.2f}; {near:.0%} of samples "
            f"near a mode (noise: {noise_near:.0%}); mode shares {shares.min():.2f}-{shares.max():.2f}")


# ===========================================================================
# Lab 07: Mamba selective scan
# ===========================================================================
def mamba_inputs(b=2, L=9, D=3, N=4, seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(b, L, D, generator=g)
    delta = F.softplus(torch.randn(b, L, D, generator=g))
    A = -torch.exp(0.5 * torch.randn(D, N, generator=g))
    B = torch.randn(b, L, N, generator=g)
    C = torch.randn(b, L, N, generator=g)
    D_skip = torch.randn(D, generator=g)
    return x, delta, A, B, C, D_skip


def ref_scan_closed_form(x, delta, A, B, C, D_skip):
    """y_t = sum_{k<=t} C_t . (prod_{j=k+1..t} A_bar_j) B_bar_k x_k + D x_t, in float64, no recurrence."""
    x, delta, A, B, C = (t.double() for t in (x, delta, A, B, C))
    b, L, D = x.shape
    y = torch.zeros(b, L, D, dtype=torch.float64)
    for t in range(L):
        for k in range(t + 1):
            decay = torch.exp(A * delta[:, k + 1:t + 1].sum(1)[..., None])    # [b, D, N]
            contrib = decay * delta[:, k, :, None] * B[:, k, None, :] * x[:, k, :, None]
            y[:, t] += (contrib * C[:, t, None, :]).sum(-1)
    return y + D_skip.double() * x


@step(7, "discretize: simple and ZOH")
def _check_discretize(lab):
    x, delta, A, B, C, _ = mamba_inputs(b=2, L=3, D=3, N=4)
    for method in ("simple", "zoh"):
        res = lab.discretize(delta, A, B, method=method)
        expect(isinstance(res, tuple) and len(res) == 2, "discretize must return (A_bar, B_bar)")
        A_bar, B_bar = res
        want_a = torch.zeros(2, 3, 3, 4, dtype=torch.float64)
        want_b = torch.zeros_like(want_a)
        for i in range(2):
            for t in range(3):
                for d in range(3):
                    for n in range(4):
                        dA = delta[i, t, d].item() * A[d, n].item()
                        want_a[i, t, d, n] = math.exp(dA)
                        if method == "simple":
                            want_b[i, t, d, n] = delta[i, t, d].item() * B[i, t, n].item()
                        else:
                            want_b[i, t, d, n] = (math.exp(dA) - 1) / A[d, n].item() * B[i, t, n].item()
        expect_close(A_bar, want_a, f"A_bar ({method})")
        expect_close(B_bar, want_b, f"B_bar ({method})",
                     hint="B_bar[b, t, d, n] = delta[b, t, d] * B[b, t, n] (simple) - broadcast "
                          "delta[..., None] against B[:, :, None, :]")
    d = torch.tensor([0.01, 0.1, 1.0, 5.0]).view(1, 4, 1)
    A_bar, _ = lab.discretize(d, torch.tensor([[-1.0]]), torch.ones(1, 4, 1))
    expect_close(A_bar.reshape(-1), torch.tensor([0.9900, 0.9048, 0.3679, 0.0067]),
                 "guide values A_bar for A = -1, delta = 0.01, 0.1, 1, 5", atol=1e-4, rtol=0)
    tiny = torch.full((1, 1, 3), 1e-3)
    _, b_s = lab.discretize(tiny, A, torch.ones(1, 1, 4), method="simple")
    _, b_z = lab.discretize(tiny, A, torch.ones(1, 1, 4), method="zoh")
    expect_close(b_z, b_s, "ZOH vs simple B_bar at delta = 1e-3", atol=1e-6, rtol=5e-3,
                 hint="the two rules agree to first order in delta")
    return "A=-1: delta 0.1 -> 0.905, delta 5 -> 0.0067"


@step(7, "selective_scan_sequential vs closed form")
def _check_seq_scan(lab):
    x, delta, A, B, C, Ds = mamba_inputs()
    res = lab.selective_scan_sequential(x, delta, A, B, C, Ds)
    expect(isinstance(res, tuple) and len(res) == 2, "selective_scan_sequential must return (y, h_last)")
    y, h_last = res
    expect_close(y, ref_scan_closed_form(x, delta, A, B, C, Ds), "y [b, L, D]",
                 hint="h_t = A_bar_t * h_{t-1} + B_bar_t * x_t;  y_t = sum_n C_t[n] h_t[:, :, n] "
                      "+ D_skip * x_t")
    y0, _ = lab.selective_scan_sequential(x, delta, A, B, C)
    expect_close(y0, ref_scan_closed_form(x, delta, A, B, C, torch.zeros(3)), "y with D_skip=None")
    expect_shape(h_last, (2, 3, 4), "h_last")
    # Processing in two chunks with the carried state gives the same result.
    y1, h1 = lab.selective_scan_sequential(x[:, :4], delta[:, :4], A, B[:, :4], C[:, :4], Ds)
    y2, h2 = lab.selective_scan_sequential(x[:, 4:], delta[:, 4:], A, B[:, 4:], C[:, 4:], Ds, h0=h1)
    expect_close(torch.cat([y1, y2], 1), y, "two chunks with h0 = h_last of the first chunk")
    expect_close(h2, h_last, "h_last after two chunks")


@step(7, "scan_combine: correct and associative")
def _check_combine(lab):
    g = torch.Generator().manual_seed(1)
    e = [(torch.rand(5, generator=g), torch.randn(5, generator=g)) for _ in range(3)]
    h = torch.randn(5, generator=g)
    a, c = lab.scan_combine(*e[0], *e[1])
    expect_close(a * h + c, e[1][0] * (e[0][0] * h + e[0][1]) + e[1][1],
                 "combined step applied to h (step 1 first, then step 2)",
                 hint="(a2, c2) o (a1, c1) = (a2 * a1, a2 * c1 + c2)")
    left = lab.scan_combine(*lab.scan_combine(*e[0], *e[1]), *e[2])
    right = lab.scan_combine(*e[0], *lab.scan_combine(*e[1], *e[2]))
    expect_close(left[0], right[0], "associativity, a part")
    expect_close(left[1], right[1], "associativity, c part")


@step(7, "parallel_scan: log-depth, any L")
def _check_parallel_scan(lab):
    calls = {"n": 0}
    orig = lab.scan_combine

    def counting(*args):
        calls["n"] += 1
        return orig(*args)

    for L in (1, 2, 5, 8, 13, 32):
        a = torch.rand(2, L, 3, 4) * 0.9 + 0.05
        c = torch.randn(2, L, 3, 4)
        h = torch.zeros(2, 3, 4)
        want = []
        for t in range(L):
            h = a[:, t] * h + c[:, t]
            want.append(h)
        calls["n"] = 0
        lab.scan_combine = counting
        try:
            got = lab.parallel_scan(a, c)
        finally:
            lab.scan_combine = orig
        expect_close(got, torch.stack(want, 1), f"parallel_scan with L = {L}",
                     hint="each round: positions t >= offset combine with t - offset (the earlier "
                          "one goes first); positions t < offset combine with the identity (1, 0)")
        rounds = math.ceil(math.log2(L)) if L > 1 else 0
        expect(L < 8 or calls["n"] <= rounds,
               f"parallel_scan called scan_combine {calls['n']} times for L = {L}; a log-depth "
               f"scan needs only {rounds} rounds (combine whole shifted tensors, not one t at a time)")
    return "L = 32 in 5 rounds"


@step(7, "selective_scan_parallel == sequential")
def _check_parallel_selective(lab):
    for L in (9, 16):
        x, delta, A, B, C, Ds = mamba_inputs(L=L, seed=L)
        want = ref_scan_closed_form(x, delta, A, B, C, Ds)
        expect_close(lab.selective_scan_parallel(x, delta, A, B, C, Ds), want,
                     f"selective_scan_parallel y (L = {L})")
        expect_close(lab.selective_scan_parallel(x, delta, A, B, C, Ds),
                     lab.selective_scan_sequential(x, delta, A, B, C, Ds)[0],
                     f"parallel vs your sequential scan (L = {L})")


@step(7, "recurrent_step: token by token == scan")
def _check_recurrent(lab):
    x, delta, A, B, C, Ds = mamba_inputs()
    want = ref_scan_closed_form(x, delta, A, B, C, Ds)
    h = torch.zeros(2, 3, 4)
    ys = []
    for t in range(x.shape[1]):
        res = lab.recurrent_step(h, x[:, t], delta[:, t], A, B[:, t], C[:, t], Ds)
        expect(isinstance(res, tuple) and len(res) == 2, "recurrent_step must return (y_t, h_new)")
        y_t, h = res
        expect_shape(y_t, (2, 3), f"y_t at t = {t}")
        expect_shape(h, (2, 3, 4), f"state at t = {t}")
        ys.append(y_t)
    expect_close(torch.stack(ys, 1), want, "outputs of 9 recurrent steps vs the scan")
    return "state stays [b, D, N] = [2, 3, 4] whatever the length"


# ===========================================================================
# Lab 08: MoE routing
# ===========================================================================
class CountingExperts(nn.Module):
    """Wraps a ModuleList of experts and records how many times each one is called."""

    def __init__(self, experts):
        super().__init__()
        self.inner = experts
        self.calls = [0] * len(experts)

    def __len__(self):
        return len(self.inner)

    def __iter__(self):
        return iter(self._wrapped(e) for e in range(len(self.inner)))

    def __getitem__(self, e):
        return self._wrapped(e)

    def _wrapped(self, e):
        def call(x):
            self.calls[e] += 1
            return self.inner[e](x)
        return call


def ref_dispatch(x, weights, indices, experts, kept=None):
    """Brute force: every (token, slot) separately, one row at a time."""
    y = torch.zeros_like(x)
    T, k = indices.shape
    for t in range(T):
        for j in range(k):
            if kept is None or kept[t][j]:
                y[t] += weights[t, j] * experts[int(indices[t, j])](x[t:t + 1])[0]
    return y


def _moe_setup(T=12, d=8, E=5, k=2, seed=8):
    torch.manual_seed(seed)
    x = torch.randn(T, d)
    logits = torch.randn(T, E)
    probs = torch.softmax(logits, -1)
    top_p, idx = torch.topk(probs, k, -1)
    w = top_p / top_p.sum(-1, keepdim=True)
    return x, logits, w, idx


@step(8, "softmax_topk_router vs brute force")
def _check_softmax_router(lab):
    T, E, k = 10, 6, 2
    logits = torch.randn(T, E)
    res = lab.softmax_topk_router(logits, k)
    expect(isinstance(res, tuple) and len(res) == 3,
           "softmax_topk_router must return (weights, indices, probs)")
    w, idx, probs = res
    expect_close(probs, torch.softmax(logits, -1), "probs")
    want_idx, want_w = [], []
    for t in range(T):
        p = torch.softmax(logits[t].double(), 0).tolist()
        order = sorted(range(E), key=lambda e: -p[e])[:k]
        want_idx.append(order)
        s = sum(p[e] for e in order)
        want_w.append([p[e] / s for e in order])
    expect_equal(idx, torch.tensor(want_idx), "indices (descending probability)")
    hint = None
    if is_close(w, probs.gather(-1, idx)):
        hint = "renormalize the k chosen probabilities so they sum to 1"
    expect_close(w, torch.tensor(want_w), "weights", hint=hint)
    w_all, _, _ = lab.softmax_topk_router(logits, E)
    expect_close(w_all.sum(-1), torch.ones(T), "weights sum to 1 (k = E)")


@step(8, "sigmoid_topk_router: bias picks, scores weigh")
def _check_sigmoid_router(lab):
    T, E, k = 10, 6, 2
    logits = torch.randn(T, E)
    bias = torch.tensor([0.0, 0.8, 0.0, -0.8, 0.3, 0.0])
    for b in (None, bias):
        res = lab.sigmoid_topk_router(logits, k, b)
        expect(isinstance(res, tuple) and len(res) == 2, "sigmoid_topk_router must return (weights, indices)")
        w, idx = res
        g = torch.sigmoid(logits.double())
        sel = g if b is None else g + b.double()
        want_idx = torch.tensor([sorted(range(E), key=lambda e: -sel[t, e].item())[:k] for t in range(T)])
        expect_equal(idx, want_idx, f"indices (bias={'None' if b is None else 'given'})",
                     hint="choose with sigmoid(logits) + bias")
        chosen = g.gather(-1, want_idx)
        hint = None
        if b is not None:
            biased = sel.gather(-1, want_idx)
            if is_close(w, biased / biased.sum(-1, keepdim=True)):
                hint = "the bias must not enter the weights: use sigmoid(logits) of the chosen experts"
        expect_close(w, chosen / chosen.sum(-1, keepdim=True),
                     f"weights (bias={'None' if b is None else 'given'})", hint=hint)
    changed = (lab.sigmoid_topk_router(logits, k, bias)[1] != lab.sigmoid_topk_router(logits, k)[1]).any()
    expect(bool(changed), "the bias should change which experts some tokens choose")


@step(8, "load_balancing_loss: 1 uniform, E collapsed")
def _check_lb_loss(lab):
    T, E, k = 12, 4, 2
    logits = torch.randn(T, E, requires_grad=True)
    probs = torch.softmax(logits, -1)
    idx = torch.topk(probs, k, -1).indices
    loss = lab.load_balancing_loss(probs, idx)
    expect_tensor(loss, "load_balancing_loss")
    expect(loss.dim() == 0, f"load_balancing_loss must return a scalar, got shape {list(loss.shape)}")
    f = [sum(int(e) == j for e in idx.flatten()) / (T * k) for j in range(E)]
    P = probs.detach().double().mean(0)
    want = E * sum(f[j] * P[j].item() for j in range(E))
    hint = None
    if abs(loss.item() - k * want) < 1e-5:
        hint = "divide the counts by T * k (all assignments), so that the f_e sum to 1"
    expect_close(loss.detach(), torch.tensor(want), "load_balancing_loss value", hint=hint)
    loss.backward()
    expect(logits.grad is not None and logits.grad.abs().sum() > 0,
           "the loss must be differentiable through probs (P_e)")
    uni_p = torch.full((8, 4), 0.25)
    uni_i = torch.tensor([[0, 1], [2, 3]] * 4)
    expect_close(lab.load_balancing_loss(uni_p, uni_i), torch.tensor(1.0), "perfectly uniform routing")
    col_p = torch.zeros(8, 4)
    col_p[:, 0] = 1.0
    expect_close(lab.load_balancing_loss(col_p, torch.zeros(8, 1, dtype=torch.long)), torch.tensor(4.0),
                 "all tokens on expert 0 (k = 1, E = 4)")
    return f"random routing: {loss.item():.3f}; uniform 1.0; collapsed 4.0"


@step(8, "dispatch_loop vs per-token brute force")
def _check_dispatch_loop(lab):
    x, _, w, idx = _moe_setup()
    torch.manual_seed(1)
    experts = lab.make_experts(5, 8, 16)
    counting = CountingExperts(experts)
    with torch.no_grad():
        got = lab.dispatch_loop(x, w, idx, counting)
        expect_close(got, ref_dispatch(x, w, idx, experts), "dispatch_loop output [T, d]",
                     hint="y[t] = sum_j w[t, j] * expert_{idx[t, j]}(x[t])")
    expect(max(counting.calls) <= 1,
           f"each expert should run once on all its tokens; calls per expert: {counting.calls}")
    return f"expert calls: {counting.calls}"


@step(8, "dispatch_sorted == loop, 1 call per expert")
def _check_dispatch_sorted(lab):
    x, _, w, idx = _moe_setup(T=16, E=6, k=2, seed=9)
    idx[3] = torch.tensor([5, 0])   # make sure an expert can be absent / present in any order
    torch.manual_seed(2)
    experts = lab.make_experts(6, 8, 16)
    counting = CountingExperts(experts)
    with torch.no_grad():
        got = lab.dispatch_sorted(x, w, idx, counting)
        expect_close(got, ref_dispatch(x, w, idx, experts), "dispatch_sorted output [T, d]",
                     hint="after sorting, the weights and token ids must be permuted with the same order")
        idx2 = torch.zeros_like(idx)
        idx2[:, 1] = 2                 # experts 1, 3, 4, 5 get no tokens at all
        counting2 = CountingExperts(experts)
        got2 = lab.dispatch_sorted(x, w, idx2, counting2)
        expect_close(got2, ref_dispatch(x, w, idx2, experts), "dispatch_sorted with idle experts")
    expect(max(counting.calls) <= 1,
           f"each expert should run once on its contiguous slice; calls per expert: {counting.calls}")


@step(8, "capacity factor and token dropping")
def _check_capacity(lab):
    for args, want in (((8, 4, 2, 1.0), 4), ((8, 4, 2, 1.25), 5), ((10, 4, 1, 1.0), 3),
                       ((4096, 64, 8, 1.0), 512), ((4096, 64, 8, 1.25), 640)):
        expect_int(lab.expert_capacity(*args), want, f"expert_capacity{args}")
    x, _, w, _ = _moe_setup(T=12, E=4, k=2, seed=10)
    idx = torch.tensor([[0, 1], [0, 2], [1, 0], [0, 3], [2, 0], [0, 1],
                        [3, 2], [1, 0], [0, 2], [2, 1], [1, 3], [0, 1]])
    torch.manual_seed(3)
    experts = lab.make_experts(4, 8, 16)
    C = lab.expert_capacity(12, 4, 2, 1.0)                       # 6
    counts, want_kept = [0] * 4, []
    for t in range(12):
        row = []
        for j in range(2):
            e = int(idx[t, j])
            row.append(counts[e] < C)
            counts[e] += row[-1]
        want_kept.append(row)
    with torch.no_grad():
        res = lab.dispatch_with_capacity(x, w, idx, experts, C)
        expect(isinstance(res, tuple) and len(res) == 2, "dispatch_with_capacity must return (y, kept)")
        y, kept = res
        expect_equal(kept, torch.tensor(want_kept), f"kept mask with capacity {C}",
                     hint="serve assignments in token-major order; an expert keeps the first C it gets")
        expect_close(y, ref_dispatch(x, w, idx, experts, want_kept), "output with dropped assignments",
                     hint="dropped assignments must contribute nothing")
        y_big, kept_big = lab.dispatch_with_capacity(x, w, idx, experts, 24)
        expect(bool(kept_big.all()), "with capacity >= T*k nothing should be dropped")
        expect_close(y_big, ref_dispatch(x, w, idx, experts), "output with a large capacity")
    dropped = sum(not v for row in want_kept for v in row)
    return f"expert 0 is oversubscribed: {dropped} of 24 assignments dropped at C = {C}"


# ===========================================================================
# Lab 09: BEV projection
# ===========================================================================
RIG_YAWS = (0.0, 60.0, 120.0, 180.0, -120.0, -60.0)   # front, front-left, ..., front-right


def ref_ego_to_cam(yaw_deg, h):
    """Independent construction: undo the yaw (Rz^T), then swap axes to (right, down, forward)."""
    th = math.radians(yaw_deg)
    rz = torch.tensor([[math.cos(th), -math.sin(th), 0.0],
                       [math.sin(th), math.cos(th), 0.0],
                       [0.0, 0.0, 1.0]], dtype=torch.float64)
    swap = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    R = swap @ rz.T
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = R
    T[:3, 3] = -R @ torch.tensor([0.0, 0.0, h], dtype=torch.float64)
    return T


def ref_project(points, K, T):
    """Per-point loop in float64: returns uv [P, 2], depth [P]."""
    uv, depth = [], []
    K, T = K.double(), T.double()
    for p in points.double():
        pc = T[:3, :3] @ p + T[:3, 3]
        z = pc[2].item()
        zc = max(z, 1e-5)
        uv.append([(K[0, 0] * pc[0] + K[0, 1] * pc[1]).item() / zc + K[0, 2].item(),
                   (K[1, 1] * pc[1]).item() / zc + K[1, 2].item()])
        depth.append(z)
    return torch.tensor(uv, dtype=torch.float64), torch.tensor(depth, dtype=torch.float64)


def ref_pillars(bev_h, bev_w, res, zs):
    pts = []
    for i in range(bev_h):
        for j in range(bev_w):
            pts.append([[(i - (bev_h - 1) / 2) * res, (j - (bev_w - 1) / 2) * res, z] for z in zs])
    return torch.tensor(pts, dtype=torch.float64)


def ref_bilinear(feat, xn, yn):
    """Bilinear sample of feat [C, H, W] at normalized (xn, yn), align_corners=False, zero padding."""
    C, H, W = feat.shape
    ix, iy = ((xn + 1) * W - 1) / 2, ((yn + 1) * H - 1) / 2
    x0, y0 = math.floor(ix), math.floor(iy)
    out = torch.zeros(C, dtype=torch.float64)
    for yy, wy in ((y0, 1 - (iy - y0)), (y0 + 1, iy - y0)):
        for xx, wx in ((x0, 1 - (ix - x0)), (x0 + 1, ix - x0)):
            if 0 <= xx < W and 0 <= yy < H:
                out += wx * wy * feat[:, yy, xx].double()
    return out


def _rig(lab):
    Ks = torch.stack([lab.make_intrinsics(500.0, 800, 448)] * 6)
    Ts = torch.stack([lab.make_ego_to_cam(y, 1.6) for y in RIG_YAWS])
    return Ks, Ts


@step(9, "make_intrinsics / make_ego_to_cam")
def _check_camera(lab):
    K = lab.make_intrinsics(500.0, 800, 448)
    expect_close(K, torch.tensor([[500.0, 0, 400], [0, 500, 224], [0, 0, 1]]), "make_intrinsics(500, 800, 448)")
    for yaw in RIG_YAWS + (37.0,):
        T = lab.make_ego_to_cam(yaw, 1.6)
        expect_close(T, ref_ego_to_cam(yaw, 1.6), f"make_ego_to_cam(yaw={yaw}, h=1.6)",
                     hint="rows of R = camera right, down, forward in ego coords; t = -R @ C")
        R = T[:3, :3].double()
        expect_close(R @ R.T, torch.eye(3, dtype=torch.float64), f"R R^T for yaw {yaw}")
    p = torch.tensor([10.0, 2.0, 0.0, 1.0])
    expect_close(lab.make_ego_to_cam(0.0, 1.6) @ p, torch.tensor([-2.0, 1.6, 10.0, 1.0]),
                 "front camera: ego (10, 2, 0) -> camera (-2, 1.6, 10)")


@step(9, "project_points: guide example + loop")
def _check_project(lab):
    K = torch.tensor([[500.0, 0, 400], [0, 500, 224], [0, 0, 1]])
    res = lab.project_points(torch.tensor([[10.0, 2.0, 0.0]]), K, ref_ego_to_cam(0.0, 1.6).float())
    expect(isinstance(res, tuple) and len(res) == 2, "project_points must return (uv, depth)")
    uv, depth = res
    expect_close(uv, torch.tensor([[300.0, 304.0]]), "guide example: pixel of (10, 2, 0) in the front camera",
                 hint="u = f X / Z + cx, v = f Y / Z + cy with (X, Y, Z) = p_cam")
    expect_close(depth, torch.tensor([10.0]), "guide example: depth")
    K2 = torch.tensor([[450.0, 0, 390], [0, 520, 230], [0, 0, 1]])
    T2 = ref_ego_to_cam(-35.0, 1.2).float()
    pts = torch.randn(20, 3) * torch.tensor([10.0, 10.0, 1.0]) + torch.tensor([8.0, -5.0, 1.0])
    uv, depth = lab.project_points(pts, K2, T2)
    want_uv, want_d = ref_project(pts, K2, T2)
    front = want_d > 0.5
    expect_close(depth, want_d, "depth of 20 random points (yaw -35 deg, fx != fy)")
    expect_close(uv[front], want_uv[front], "pixels of the random points in front of the camera",
                 atol=1e-3, rtol=1e-5)


@step(9, "bev_pillar_points vs a loop")
def _check_pillars(lab):
    for args in ((3, 4, 0.5, (0.0, 1.0, 2.0, 3.0)), (5, 2, 1.0, (-1.0, 2.0))):
        got = lab.bev_pillar_points(*args)
        expect_close(got, ref_pillars(*args), f"bev_pillar_points{args[:3]}",
                     hint="cell (i, j): x = (i - (bev_h-1)/2) * res, y = (j - (bev_w-1)/2) * res")


@step(9, "reference_points + visibility, 6 cameras")
def _check_reference_points(lab):
    Ks, Ts = _rig(lab)
    pts = ref_pillars(10, 10, 4.0, (0.0, 1.0, 2.0, 3.0)).float()
    pts[0, 0] = torch.tensor([10.0, 2.0, 0.0])          # the guide's point
    res = lab.reference_points(pts, Ks, Ts, 448, 800)
    expect(isinstance(res, tuple) and len(res) == 2, "reference_points must return (uv_norm, valid)")
    uv, valid = res
    expect_shape(uv, (6, 100, 4, 2), "uv_norm [N_cam, Q, N_z, 2]")
    expect_shape(valid, (6, 100, 4), "valid [N_cam, Q, N_z]")
    expect_close(uv[0, 0, 0], torch.tensor([-0.25, 0.357142857]), "guide example uv_norm (front camera)",
                 atol=1e-4, hint="uv_norm = (2u / W - 1, 2v / H - 1)")
    expect(bool(valid[0, 0, 0]) and not bool(valid[3, 0, 0]),
           "the guide's point must be valid in the front camera and invalid in the rear one (Z < 0)")
    want_valid = torch.zeros(6, 400, dtype=torch.bool)
    want_uv = torch.zeros(6, 400, 2, dtype=torch.float64)
    for c in range(6):
        puv, d = ref_project(pts.reshape(-1, 3), Ks[c], ref_ego_to_cam(RIG_YAWS[c], 1.6))
        u, v = puv[:, 0], puv[:, 1]
        want_valid[c] = (d > 1e-5) & (u > 0) & (u < 800) & (v > 0) & (v < 448)
        want_uv[c] = torch.stack([2 * u / 800 - 1, 2 * v / 448 - 1], -1)
    expect_equal(valid.reshape(6, 400), want_valid, "valid mask",
                 hint="valid = depth > 1e-5 and 0 < u < W and 0 < v < H")
    m = want_valid
    expect_close(uv.reshape(6, 400, 2)[m], want_uv[m], "uv_norm of the valid points", atol=1e-4)
    hits = valid.any(dim=2).sum(0)
    return f"cells seen by 0/1/2 cameras: {int((hits == 0).sum())}/{int((hits == 1).sum())}/{int((hits == 2).sum())}"


@step(9, "sample_features: grid_sample vs bilinear")
def _check_bev_sampling(lab):
    feats = torch.randn(2, 3, 5, 7)
    uv = torch.rand(2, 6, 2, 2) * 2.4 - 1.2             # includes points outside [-1, 1]
    centre = torch.tensor([2 * (4 + 0.5) / 7 - 1, 2 * (1 + 0.5) / 5 - 1])
    uv[0, 0, 0] = centre                                 # exactly at the centre of pixel (row 1, col 4)
    got = lab.sample_features(feats, uv)
    expect_shape(got, (2, 6, 2, 3), "sampled features [N_cam, Q, N_z, C]")
    want = torch.zeros(2, 6, 2, 3, dtype=torch.float64)
    for c in range(2):
        for q in range(6):
            for z in range(2):
                want[c, q, z] = ref_bilinear(feats[c], uv[c, q, z, 0].item(), uv[c, q, z, 1].item())
    expect_close(got[0, 0, 0], feats[0, :, 1, 4], "sample at the centre of pixel (row 1, col 4)",
                 hint="use align_corners=False: -1 and +1 are the image EDGES, not pixel centres")
    expect_close(got, want, "sampled features vs a brute-force bilinear loop",
                 hint="grid_sample(features, uv_norm) returns [N_cam, C, Q, N_z]; permute to "
                      "[N_cam, Q, N_z, C]")


@step(9, "aggregate_views over cameras and heights")
def _check_aggregate(lab):
    sampled = torch.randn(3, 5, 2, 4)
    valid = torch.rand(3, 5, 2) > 0.5
    valid[:, 2] = False                                  # cell 2 is seen by nobody
    got = lab.aggregate_views(sampled, valid)
    want = torch.zeros(5, 4, dtype=torch.float64)
    for q in range(5):
        vals = [sampled[c, q, z].double() for c in range(3) for z in range(2) if valid[c, q, z]]
        if vals:
            want[q] = torch.stack(vals).mean(0)
    expect_close(got, want, "BEV features [Q, C]",
                 hint="average only over valid (camera, height) samples; unseen cells -> 0, not NaN")
    # End to end on the toy rig: a constant feature map gives that constant wherever a camera sees.
    Ks, Ts = _rig(lab)
    pts = lab.bev_pillar_points(8, 8, 1.0)
    uv, valid = lab.reference_points(pts, Ks, Ts, 448, 800)
    feats = torch.ones(6, 2, 28, 50) * torch.arange(1.0, 7.0).view(6, 1, 1, 1)
    bev = lab.aggregate_views(lab.sample_features(feats, uv), valid)
    expect_shape(bev, (64, 2), "BEV features for an 8 x 8 grid")
    seen = valid.any(dim=2).any(dim=0)
    expect(bool((bev[~seen] == 0).all()), "cells no camera sees must stay 0")
    expect(bool(((bev[seen] > 0) & (bev[seen] <= 6 + 1e-5)).all()),
           "with camera c's feature map constant = c + 1, every seen cell must average to a value "
           "in (0, 6] (below 1 only within half a feature pixel of an image edge, zero padding)")
    return (f"8 m x 8 m grid around the car: {int((~seen).sum())} of 64 cells are seen by no camera "
            f"(the ground right next to the car is below every image)")


# ===========================================================================
# Lab 10: flow matching
# ===========================================================================
def ref_mode_chunks(horizon=8):
    h = torch.arange(horizon)
    vy = torch.sin(math.pi * (h + 0.5) / horizon)
    left = torch.stack([torch.ones(horizon), vy], dim=-1)
    return torch.stack([left, left * torch.tensor([1.0, -1.0])])


@step(10, "interpolate + velocity_target")
def _check_fm_path(lab):
    noise, act = torch.randn(4, 8, 2), torch.randn(4, 8, 2)
    expect_close(lab.interpolate(noise, act, torch.zeros(4)), noise, "x_t at t = 0 (must be the noise)",
                 hint="this lab's convention: t = 0 is noise, t = 1 is data")
    expect_close(lab.interpolate(noise, act, torch.ones(4)), act, "x_t at t = 1 (must be the data)")
    t = torch.rand(4)
    want = (1 - t)[:, None, None] * noise + t[:, None, None] * act
    expect_close(lab.interpolate(noise, act, t), want, "x_t at random t (one t per sample)")
    expect_close(lab.velocity_target(noise, act), act - noise, "velocity_target")
    h = 1e-3
    fd = (lab.interpolate(noise.double(), act.double(), (t + h).double())
          - lab.interpolate(noise.double(), act.double(), t.double())) / h
    expect_close(lab.velocity_target(noise, act), fd, "velocity_target vs finite difference of x_t",
                 atol=1e-4)


@step(10, "flow_matching_loss")
def _check_fm_loss(lab):
    seen = {}

    def dummy(x_t, t):
        seen["x_t"], seen["t"] = x_t, t
        return 0.3 * x_t + t[:, None, None]

    noise, act, t = torch.randn(5, 8, 2), torch.randn(5, 8, 2), torch.rand(5)
    loss = lab.flow_matching_loss(dummy, act, noise, t)
    expect_tensor(loss, "flow_matching_loss")
    expect(loss.dim() == 0, f"flow_matching_loss must return a scalar, got shape {list(loss.shape)}")
    x_t = (1 - t)[:, None, None] * noise + t[:, None, None] * act
    expect_close(seen.get("x_t", torch.zeros(0)), x_t, "the x_t passed to the model")
    expect_close(loss, ((0.3 * x_t + t[:, None, None] - (act - noise)) ** 2).mean(), "loss value")


@step(10, "euler_sample: exact on a known field")
def _check_euler(lab):
    target = torch.randn(1, 8, 2)

    class ToTarget(nn.Module):        # exact flow to a single point: v = (a* - x) / (1 - t)
        def forward(self, x, t):
            return (target - x) / (1 - t)[:, None, None]

    noise = torch.randn(6, 8, 2)
    for n in (1, 4, 10):
        got = lab.euler_sample(ToTarget(), noise.clone(), n)
        expect_close(got, target.expand(6, -1, -1), f"Euler with {n} steps on v = (a* - x) / (1 - t)",
                     atol=1e-5, hint="evaluate at t_i = i / n for i = 0 .. n-1 (never at t = 1), "
                                     "and step by dt = 1 / n")
    torch.manual_seed(10)
    net = lab.VelocityMLP().eval()
    x = noise.clone()
    with torch.no_grad():
        for i in range(10):
            x = x + net(x, torch.full((6,), i / 10)) / 10
        expect_close(lab.euler_sample(net, noise.clone(), 10), x, "Euler with a random VelocityMLP")


@step(10, "action chunks: windows, receding horizon")
def _check_chunks(lab):
    traj = torch.arange(20.0).view(10, 2)
    for H, stride in ((4, 1), (3, 2), (10, 1)):
        got = lab.chunk_trajectory(traj, H, stride)
        want = torch.stack([traj[s:s + H] for s in range(0, 10 - H + 1, stride)])
        expect_close(got, want, f"chunk_trajectory(traj [10, 2], horizon={H}, stride={stride})",
                     atol=0, rtol=0, hint="window c is traj[c*stride : c*stride + horizon]")

    def policy(step):                 # a chunk that remembers when it was planned
        return torch.stack([torch.full((2,), 100.0 * step + h) for h in range(5)])

    res = lab.rollout_chunks(policy, 12, 3)
    expect(isinstance(res, tuple) and len(res) == 2, "rollout_chunks must return (actions, call_steps)")
    actions, calls = res
    expect(list(calls) == [0, 3, 6, 9], f"with n_exec = 3 the policy should be called at steps "
                                        f"[0, 3, 6, 9], got {list(calls)}")
    want = torch.tensor([[100.0 * s + h] * 2 for s in (0, 3, 6, 9) for h in range(3)])
    expect_close(actions, want, "executed actions (first 3 of each chunk)", atol=0, rtol=0)
    actions, calls = lab.rollout_chunks(policy, 7, 5)
    expect(list(calls) == [0, 5], f"n_actions = 7, n_exec = 5: calls at [0, 5], got {list(calls)}")
    expect_shape(actions, (7, 2), "actions for n_actions = 7")
    return "12 actions, re-plan every 3 -> 4 policy calls"


@step(10, "train on 2-mode demos; samples pick a mode")
def _check_fm_training(lab):
    gen = torch.Generator().manual_seed(0)
    modes = ref_mode_chunks()
    data = modes[torch.randint(0, 2, (10000,), generator=gen)] + 0.05 * torch.randn(10000, 8, 2, generator=gen)
    torch.manual_seed(0)
    model = lab.VelocityMLP()
    t0 = time.perf_counter()
    losses = lab.train_flow_matching(model, data, steps=2000, batch_size=256, lr=1e-3, seed=0)
    train_s = time.perf_counter() - t0
    expect(isinstance(losses, (list, tuple)) and len(losses) == 2000,
           "train_flow_matching must return the list of 2000 per-step losses")
    first, last = sum(losses[:50]) / 50, sum(losses[-200:]) / 200
    expect(last < 0.35 and last < first, f"the loss should fall below 0.35 (the reference reaches "
                                          f"~0.22); first 50 steps {first:.3f}, last 200 {last:.3f}")
    model.eval()
    noise = torch.randn(2000, 8, 2, generator=torch.Generator().manual_seed(99))
    samples = lab.euler_sample(model, noise, 10)
    expect_shape(samples, (2000, 8, 2), "sampled chunks")
    dist = ((samples[:, None] - modes[None]) ** 2).mean((2, 3)).sqrt()      # RMS to each mode
    d_min, which = dist.min(1)
    d_avg = ((samples - modes.mean(0)) ** 2).mean((1, 2)).sqrt()             # to the averaged chunk
    near = (d_min < 0.3).float().mean().item()
    left = (which == 0).float().mean().item()
    averaged = (d_avg < d_min).float().mean().item()
    expect(near > 0.9, f"only {near:.0%} of sampled chunks are within RMS 0.3 of a demonstrated "
                       f"behaviour (need > 90%)")
    expect(0.35 < left < 0.65, f"both behaviours should be sampled about equally; left = {left:.0%}")
    expect(averaged < 0.05, f"{averaged:.0%} of the samples are closer to the AVERAGE of the two "
                            f"behaviours than to either one (mode averaging)")
    return (f"trained in {train_s:.1f} s, loss {first:.2f} -> {last:.2f}; {near:.0%} near a mode, "
            f"left/right {left:.0%}/{1 - left:.0%}, averaged {averaged:.1%}")


# ===========================================================================
if __name__ == "__main__":
    sys.exit(main())
