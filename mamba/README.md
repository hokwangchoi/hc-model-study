# Mamba (Selective State Space Model)

Based on ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) and its predecessor [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396). Mamba-2, Mamba-3 and the hybrid SSM–attention LLMs of 2024–26 (Jamba, Nemotron-H / Nemotron 3, Granite 4.0) build on this pattern.

## Contents

| File | Description |
|------|-------------|
| [mamba_guide.html](./mamba_guide.html) | Interactive visual guide (open in browser): Module 5 of 9, with a step-through scan widget, calculator, self-check and exercises |
| [mamba.py](./mamba.py) | Reference implementation with annotations: sequential scan, parallel scan, recurrent `step()`, and an equivalence check |
| [../labs/lab07_mamba_scan.py](../labs/lab07_mamba_scan.py) | Build-it-yourself lab: discretization, sequential and parallel selective scans, the recurrent step (check with `python labs/check.py 7`) |

## Quick Start

```bash
# View the interactive guide
open mamba_guide.html      # macOS
xdg-open mamba_guide.html  # Linux

# Run the implementation (CPU is fine)
python mamba.py
```

## The core idea

> **Notation (Mamba paper symbols):** L = sequence length (N in Module 1); n_layer = number of layers; N = d_state = 16 state entries per channel (not the sequence length); D = channels = d_inner in shapes, and also the skip weight in y = C·h + D·x; b = batch (B is the SSM input matrix).

Mamba is an alternative to transformers whose core operation is a **linear recurrence**, run independently for every channel:

```
Ā_t = exp(Δ_t · A)                  # decay in (0, 1): A < 0, Δ_t > 0
B̄_t = Δ_t · B_t                     # input weight
h_t = Ā_t ⊙ h_{t-1} + B̄_t · x_t     # state update, N-dim (N = 16)
y_t = C_t · h_t + D · x_t           # readout, scalar per channel
```

That's a discretized state-space model (SSM). What makes Mamba "selective" is that `Δ`, `B` and `C` are **functions of the current token** (A is a learned constant, but Ā_t still varies through Δ_t). A large Δ overwrites the state with the current input; a small Δ keeps the state and ignores the input. Plain SSMs (S4) use fixed parameters, so they treat every token the same way.

Two modes of computation from the same model:

| Mode | Used for | Depth | Memory |
|------|----------|-------|--------|
| **Parallel scan** | Training / prefill | O(log L) (work O(L · D · N)) | [b, L, D, N] if done naively; the fused kernel keeps it on chip |
| **Recurrent** | Generation | L sequential steps, O(1) each | **[b, D, N] state + [b, D, 4] conv window — fixed** |

The recurrent form is the headline: no KV cache. `mamba.py` implements both (plus the plain loop) and `demo_equivalence()` shows they agree to ~1e-16 in float64.

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| In/out projections | O(L · d²) | Compute (prefill) / bandwidth (decode) | Standard GEMM, Tensor Cores, weight quantization |
| Conv1d (depthwise) | O(L · d · K) | Memory | Tiny (K=4); fuse with activation |
| Selective scan | O(L · d · N) | **Memory** | Hardware-aware kernel (SRAM-resident, like FlashAttention) |
| LM head | O(L · d · V) | Compute | Unchanged from transformers |

**N** = state dimension (16 in Mamba-1). **K** = conv kernel size (4). **d** = d_inner = 2 · d_model.

## Critical Optimizations

### 1. Parallel scan for training

Each step is the affine map `h ← a_t·h + b_t` (a_t = Ā_t, b_t = B̄_t·x_t). Applying step 1 then step 2 gives another affine map:

```
(a2, b2) ∘ (a1, b1) = (a2·a1, a2·b1 + b2)
```

This combine is associative, so all prefixes h_1..h_L can be computed with a parallel prefix scan in O(log L) depth. `selective_scan_parallel()` in `mamba.py` does this with a Hillis–Steele scan (O(L log L) work); the CUDA kernel uses a work-efficient scan in SRAM.

### 2. Recurrent inference — no KV cache

Per sequence, Mamba-130M (24 layers, d_inner = 1536, N = 16, d_conv = 4) keeps
`24 × 1536 × (16 + 4)` values = **1.41 MiB in FP16**, whatever the context length.

A transformer of about the same size (12 layers, d = 768, full multi-head K/V) keeps
`2 × 12 × L × 768` values = 36 KiB per token: **1.13 GiB at L = 32K**.

At batch-1 decode, both are bandwidth-bound. On a Jetson Orin Nano Super (102 GB/s):
Mamba-130M reads ≈ 260 MB per token (weights + state) → at most ≈ 393 tokens/s at any length;
the same-size transformer reads ≈ 1.46 GB per token at 32K context → at most ≈ 70 tokens/s.

### 3. Hardware-aware kernel (the FlashAttention trick applied to scan)

A naive selective scan materializes `[b, L, D, N]`: for Mamba-B at L = 2048 that is
`2048 × 1536 × 16 = 50.3M` values per layer per sequence (201 MB in FP32). The fused kernel:

1. Loads Δ, A, B, C and x from DRAM to SRAM once.
2. Discretizes and runs the scan in SRAM, then multiplies by C.
3. Writes only y `[b, L, D]` back (IO drops by a factor of about N), and recomputes the states in the backward pass.

Paper numbers (A100): 20–40× faster than a standard PyTorch scan, faster than FlashAttention-2 beyond 2K tokens (up to 7× at 32K), and 4–5× higher generation throughput than a similar-size Transformer, mainly because the missing KV cache allows much larger batches.

### 4. Mamba-2 and structured state-space duality (SSD)

[Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) restricts A to a scalar times identity per head, which turns the layer into a structured matrix that can be computed in chunks with matmuls (Tensor Cores):

- 2–8× faster layer than Mamba-1's selective scan (paper).
- Much larger states: `mamba_ssm`'s `Mamba2` defaults to d_state = 128, head dim 64.

[Mamba-3 (Lahoti et al., 2026)](https://arxiv.org/abs/2603.15569) adds a more expressive discretization, a complex-valued state update and a MIMO form for better decode efficiency.

### 5. Hybrid models (SSM + a few attention layers)

A fixed-size state must compress the past, so pure SSMs are weaker at exact recall. Shipping models mix in a small fraction of attention layers:

- [Jamba (AI21, 2024)](https://arxiv.org/abs/2403.19887): 1 attention : 7 Mamba layers, MoE every other layer, 52B total / 12B active.
- [Nemotron-H (NVIDIA, 2025)](https://arxiv.org/abs/2504.03624): Mamba-2 + FFN layers with about 8% self-attention layers; up to 3× faster inference than similar-size transformers.
- [Granite 4.0 (IBM, 2025)](https://www.ibm.com/granite/docs/models/granite4-0): hybrid Mamba-2 / transformer, MoE in some sizes.
- [Nemotron 3 Nano (Dec 2025)](https://arxiv.org/abs/2512.20848) and [Nemotron 3 Ultra (Jun 2026)](https://arxiv.org/abs/2606.15007): hybrid Mamba–attention MoE models (Ultra: 550B total, 55B active, 1M context).
- [Qwen3-Next (2025)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) uses Gated DeltaNet (linear attention) in a 3:1 hybrid — same idea, different recurrent layer.

Serving stacks therefore have to manage both a growing KV cache and fixed SSM states (e.g. vLLM's hybrid cache manager).

## Architecture Summary

```
Token IDs [b, L]
    ↓ Embedding
    ↓
[b, L, d_model]
    ↓
┌─────────────────────────────────────────┐
│  Mamba Layer (× n_layer)                │
│    x_in                                 │
│     ↓ RMSNorm                           │
│     ↓                                   │
│     ┌─────────────────────────────┐     │
│     │ in_proj: d_model → 2·d_i    │     │
│     │   split into (x_ssm, z)     │     │
│     │   x_ssm → conv1d → SiLU     │     │
│     │   x_ssm → [SELECTIVE SCAN]  │  ←─ the core op; data-dep Δ,B,C
│     │   y = scan_out · SiLU(z)    │  ←─ the "gated" part
│     │   out_proj: d_i → d_model   │     │
│     └─────────────────────────────┘     │
│     ↓ + residual                        │
│    x_out                                │
└─────────────────────────────────────────┘
    ↓
Final RMSNorm → LM Head (tied) → Logits [b, L, V]
```

**No positional encoding.** The causal conv and the recurrence process tokens in order, so position is implicit.

## Parameters & FLOPs

For Mamba-B (d_model=768, 24 layers, d_state=16, expand=2; same shape as `state-spaces/mamba-130m`):

| Component | Per-layer params |
|-----------|-----------------|
| in_proj (d → 2·d_i) | 2,359,296 |
| conv1d (depthwise + bias) | 7,680 |
| x_proj + dt_proj | 198,144 |
| A_log + D | 26,112 |
| out_proj (d_i → d) | 1,179,648 |
| RMSNorm | 768 |
| **Total per layer** | **3,771,648 ≈ 6.4·d²** |

A transformer layer (attention + MLP) is ≈ 12·d², so Mamba uses twice as many layers for the same size.

Model totals (`python mamba.py`, vocab 50,257, FLOPs = 2 per multiply-add, including the LM head):

| Model | d_model | n_layer | Params | GFLOPs @ L=1024 |
|-------|---------|---------|--------|-----------------|
| Mamba-S (toy) | 512 | 8 | 39.3M | 81.2 |
| Mamba-B | 768 | 24 | 129.1M | 267.8 |
| Mamba-L | 1024 | 48 | 371.5M | 769.8 |

## Key Differences from Transformers

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Core mixer | Self-attention O(L²·d) | Selective scan O(L·d·N) |
| Positional info | Added explicitly (RoPE, absolute) | Implicit (causal conv + recurrence) |
| Training | Parallel matmul | Matmuls + parallel scan |
| Inference | Autoregressive + KV cache | Recurrent, fixed state |
| Memory at long L | KV cache grows O(L) | **Fixed** |
| Best at | In-context learning, exact recall | Long sequences on a fixed memory budget |
| Kernel shape | GEMM-dominated | GEMMs + custom scan kernel |

## On Edge Hardware

- **No KV cache blow-up.** 1.41 MiB of state per sequence for Mamba-130M, versus 4.5 GiB of KV cache for a same-size transformer at 128K tokens.
- **Constant per-token cost.** Batch-1 decode is bandwidth-bound for both; Mamba's bytes per token do not grow with context.
- **The scan kernel.** The official `mamba_ssm` source build compiles for sm_87 (Jetson Orin) among other architectures; check your JetPack/CUDA/PyTorch combination. [Quamba (2024)](https://arxiv.org/abs/2410.13229) reports 8-bit Mamba 2.8B with 1.72× lower generation latency on a Jetson Orin Nano 8G and a 0.9% average accuracy drop.

Lower bounds on Orin Nano Super (17 TFLOPS FP16, 102 GB/s) for Mamba-B:
- Prefill at L = 4096: 1.07 TFLOP (incl. LM head) → ≥ 63 ms.
- Decode: ≈ 262 MFLOPs per token, but ≈ 260 MB to read → ≥ 2.5 ms per token (≤ 393 tokens/s).
- State memory: 1.41 MiB per request, regardless of L.

## Connection to the Broader Roadmap

- **Mixture-of-Experts LLMs (Module 6)**: Jamba, Granite 4.0, Nemotron 3 and Qwen3-Next combine linear-time mixers with MoE layers.
- **VLA models (Module 8)**: closed-loop control over long horizons favours a fixed-size state; RoboMamba (2024) builds a VLA on a Mamba language model.
- **World models / robot foundation models**: latent dynamics over long trajectories are naturally recurrent.

## References

- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) — the paper
- [Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) — SSD formulation, 2–8× faster
- [Mamba-3 (Lahoti et al., 2026)](https://arxiv.org/abs/2603.15569)
- [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) — the structured SSM predecessor
- [Jamba (Lieber et al., 2024)](https://arxiv.org/abs/2403.19887) — hybrid Transformer-Mamba with MoE
- [Nemotron-H (NVIDIA, 2025)](https://arxiv.org/abs/2504.03624) — hybrid Mamba-2 / Transformer
- [Quamba (Chiang et al., 2024)](https://arxiv.org/abs/2410.13229) — 8-bit SSM quantization on Jetson Orin Nano
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — same hardware-aware philosophy
- [The Annotated S4](https://srush.github.io/annotated-s4/) — accessible SSM walkthrough
- [Mamba source](https://github.com/state-spaces/mamba) — official implementation with CUDA kernels
