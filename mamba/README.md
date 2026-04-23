# Mamba (Selective State Space Model)

Based on ["Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) and its predecessor [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396). The modern state-space model family — Jamba, Mamba-2, Zamba, Samba all build on this pattern.

## Contents

| File | Description |
|------|-------------|
| [mamba_guide.html](./mamba_guide.html) | Interactive visual guide (open in browser) |
| [mamba.py](./mamba.py) | Reference implementation with annotations |

## Quick Start

```bash
# View the interactive guide
open mamba_guide.html      # macOS
xdg-open mamba_guide.html  # Linux

# Run the implementation
python mamba.py
```

## The core idea

Mamba is an alternative to transformers whose core operation is a **linear recurrence**:

```
h_t = A · h_{t-1} + B · x_t         # state update, N-dim
y_t = C · h_t + D · x_t             # readout, scalar per channel
```

That's the state-space model (SSM). What makes Mamba special is that `A`, `B`, `C`, `Δ` (the discretization step) are all **functions of the input** — "selective" state-space. Plain SSMs (S4) use fixed parameters; Mamba makes them data-dependent, closing the gap to attention on language modeling while keeping linear complexity.

Two modes of computation from the same model:

| Mode | Used for | Cost | Memory |
|------|----------|------|--------|
| **Parallel scan** | Training | O(L · log L) depth | O(B · L · D · N) activations |
| **Recurrent** | Inference | O(L) steps, O(1) per step | **O(B · D · N) — fixed** |

The recurrent form is the headline: no KV cache. State size is fixed regardless of how many tokens you've generated.

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| In/out projections | O(L · d²) | Compute | Standard GEMM, Tensor Cores |
| Conv1d (depthwise) | O(L · d · K) | Memory | Tiny (K=4); fuse with activation |
| Selective scan | O(L · d · N) | **Memory** | Hardware-aware kernel (SRAM-resident, like FlashAttention) |
| LM head | O(L · d · V) | Compute | Unchanged from transformers |

**N** = state dimension (typically 16). **K** = conv kernel size (typically 4). **d** = d_inner = 2 · d_model.

## Critical Optimizations

### 1. Parallel scan for training

The recurrence looks sequential but is actually a **prefix-sum** and so admits a parallel algorithm. Since the update `h_t = A_t · h_{t-1} + B_t · x_t` is associative under the pair operation `(A, b) · (A', b') = (A · A', A · b' + b)`, a work-efficient Blelloch scan runs in log-L depth. The real Mamba uses a custom CUDA kernel (`selective_scan_cuda`) that's SRAM-resident — the N-dimensional state never materializes in HBM.

```python
# Conceptually (not the actual implementation):
#   pair_op((A1, b1), (A2, b2)) = (A1 @ A2, A2 @ b1 + b2)
#   scan the pairs → get h_t for all t in parallel
```

### 2. Recurrent inference — no KV cache

The killer feature. At inference:

```
state h: [B, d_inner, d_state]    ← fixed size
per token: O(d_inner · d_state) compute + memory
```

Compare with a transformer's KV cache at sequence length L:

```
KV cache: [B, n_layer, 2, L, d_model]    ← grows with L
per token: O(L · d_model) compute + memory
```

At L = 32K, Mamba's per-step cost is ~100× lower than a transformer's decode step in terms of memory traffic. This is why Mamba shines at long-context inference even when per-token FLOPs are similar.

### 3. Hardware-aware kernel (the FlashAttention trick applied to scan)

Naive selective scan materializes a tensor of shape `[B, L, D, N]` — for Mamba-B at L=2048, that's `2048 × 1536 × 16 = 50M elements per batch per layer`, way too much for HBM roundtrips. The hardware-aware implementation:

1. Computes `Δ`, `B`, `C` at HBM level in parallel.
2. Performs the scan in SRAM tile-by-tile, never writing the full `[L, D, N]` tensor back to HBM.
3. Recomputes it during backward pass (activation checkpointing).

Exactly the FlashAttention pattern applied to a different O(N²)-like intermediate. This is a strong interview example of "memory-bound kernel rewritten as compute-bound-in-SRAM."

### 4. Mamba-2 and structured state-space duality (SSD)

[Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) reformulates the selective scan as a special case of **structured masked attention** — same math from two directions. Practical effects:

- 2–8× faster than Mamba-1 (reuses transformer-like matmul kernels).
- Uses larger state dimension (N = 64–128 vs 16), better quality.
- Simpler A structure (single scalar per head), easier to implement.

### 5. Hybrid models (Transformer + Mamba)

Pure Mamba struggles with some tasks attention excels at (exact recall, in-context learning). The practical winner is a hybrid: most layers Mamba, a few layers attention. [Jamba (AI21, 2024)](https://arxiv.org/abs/2403.19887) and Samba both follow this pattern — typically 7:1 Mamba:attention. You get linear compute on most tokens with attention's flexibility where it matters.

## Architecture Summary

```
Token IDs [B, L]
    ↓ Embedding
    ↓
[B, L, d_model]
    ↓
┌─────────────────────────────────────────┐
│  Mamba Layer (× L_layers)               │
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
│     ↓ residual                          │
│    x_out                                │
└─────────────────────────────────────────┘
    ↓
Final RMSNorm → LM Head → Logits [B, L, V]
```

**No positional encoding.** The recurrence is intrinsically positional — `h_t` carries the history of all previous tokens. Contrast with transformers, which are position-invariant without explicit PE.

## Parameters & FLOPs

For Mamba-B (d_model=768, 24 layers, d_state=16, expand=2):

| Component | Per-layer params |
|-----------|-----------------|
| in_proj (d → 2·d_i) | 2.36 M |
| conv1d (depthwise) | 7.7 K |
| x_proj + dt_proj | 0.20 M |
| A_log + D | 25.6 K |
| out_proj (d_i → d) | 1.18 M |
| RMSNorm | 0.8 K |
| **Total per layer** | **~3.77 M** |

Model totals:

| Model | d_model | n_layer | Params | GFLOPs @ L=1024 |
|-------|---------|---------|--------|-----------------|
| Mamba-S | 512 | 8 | ~40M | ~50 |
| Mamba-B | 768 | 24 | ~130M | ~330 |
| Mamba-L | 1024 | 48 | ~370M | ~1100 |

## Key Differences from Transformers

| Aspect | Transformer | Mamba |
|--------|-------------|-------|
| Core mixer | Self-attention O(L²·d) | Selective scan O(L·d·N) |
| Positional info | Added explicitly (RoPE, absolute) | Intrinsic to recurrence |
| Training | Parallel matmul | Parallel scan |
| Inference | Autoregressive + KV cache | Recurrent, fixed state |
| Memory at long L | O(L) per token (KV grows) | **O(1) per token** |
| Best at | In-context learning, recall | Long-horizon sequences, compressed state |
| Parallelism | Very high (all matmul) | High (scan has log-depth) |
| Kernel shape | GEMM-dominated | Scan-dominated |

## Why this matters for inference infrastructure

1. **No KV cache = much smaller memory footprint at long context.** A 130M Mamba at L=32K uses <100 MB of state; a 130M transformer's KV cache is several GB.
2. **Per-step cost is constant.** Decode latency at token 30,000 equals decode latency at token 100. Transformer decode latency grows with L.
3. **Custom kernels are a deployment requirement.** Without the hardware-aware selective scan, the model is impractically slow. A strong example of why CUDA fluency matters in production inference.
4. **Hybrid is the practical answer.** If you're building an inference stack that has to support long-context chat, you'll likely deploy both Mamba and attention layers; serving infra needs to understand both.

## On Edge Hardware

Mamba is a great fit for edge inference of long-sequence tasks:

- **No KV cache blow-up.** Orin Nano's 8 GB unified memory can host much longer contexts than a transformer of equivalent size.
- **Recurrent mode runs well on small batches.** The per-step cost is entirely in GEMMs of shape `[1, d_inner]` — well-suited to edge Tensor Cores.
- **The scan kernel needs porting.** The official mamba_ssm CUDA kernel targets sm_80+ Ampere; Jetson Orin is sm_87 but compatibility is imperfect. Production deployments usually write their own.

Rough budget on Orin Nano Super for Mamba-B at L=4096:
- Prefill: ~1.3 TFLOPs → ~150 ms (at Tensor Core rate, assuming scan kernel tuned).
- Decode: ~320 MFLOPs per token → ~10 ms per token = 100 TPS.
- State memory: ~12 MB per request, regardless of L.

## Connection to the Broader Roadmap

Mamba connects to several models ahead in the roadmap:

- **VLA models (Pi0, RoboCat)**: action chunking over long horizons. Mamba-style recurrences are a natural fit for the action expert.
- **Robot foundation models**: policy learning over long demonstration sequences. Hybrid architectures with Mamba mid-blocks are showing up.
- **World models**: trajectory prediction, where the compressed state vector is a good inductive bias.

## References

- [Mamba (Gu & Dao, 2023)](https://arxiv.org/abs/2312.00752) — the paper
- [Mamba-2 (Dao & Gu, 2024)](https://arxiv.org/abs/2405.21060) — SSD formulation, 2–8× faster
- [S4 (Gu et al., 2022)](https://arxiv.org/abs/2111.00396) — the structured SSM predecessor
- [Jamba (AI21 Labs, 2024)](https://arxiv.org/abs/2403.19887) — first major hybrid Transformer-Mamba LLM
- [FlashAttention (Dao et al., 2022)](https://arxiv.org/abs/2205.14135) — same hardware-aware philosophy
- [The Annotated S4](https://srush.github.io/annotated-s4/) — accessible SSM walkthrough
- [Mamba source](https://github.com/state-spaces/mamba) — official implementation with CUDA kernels
