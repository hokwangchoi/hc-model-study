# Vision-Language Model (VLM)

Based on the LLaVA / Qwen-VL family — the dominant "ViT + projector + LLM" pattern used by
[LLaVA](https://arxiv.org/abs/2304.08485), [Qwen-VL](https://arxiv.org/abs/2308.12966),
[Qwen2-VL](https://arxiv.org/abs/2409.12191), and NVIDIA's
[Cosmos-Reason2](https://research.nvidia.com/labs/dir/cosmos-reason2/).

## Contents

| File | Description |
|------|-------------|
| [vlm_guide.html](./vlm_guide.html) | Interactive visual guide (open in browser) |
| [vlm.py](./vlm.py) | Reference implementation with annotations |

## Quick Start

```bash
# View the interactive guide
open vlm_guide.html      # macOS
xdg-open vlm_guide.html  # Linux

# Run the implementation
python vlm.py
```

## The core idea

A VLM is not a new architecture — it's three things wired together:

1. A **vision encoder** (a ViT, with the `[CLS]` token and classification head removed)
2. A **visual projector** (a 2-layer MLP) that maps vision embeddings into the LLM's embedding space
3. An **LLM decoder** that consumes image tokens and text tokens as one concatenated sequence

```
                     ┌─────────────────────────────────┐
Image  [B,3,H,W]     │                                 │
  ↓ ViT              │   Text  "What is this?"         │
  ↓ Projector        │   ↓ Tokenize → Embed            │
Image tokens         │   ↓                             │
[B, N_v, d_t]        │  Text embeds [B, N_t, d_t]      │
  └────────┬─────────┘                                 │
           ↓ concatenate along seq dim                 │
  Combined [B, N_v + N_t, d_t]                         │
           ↓                                           │
  LLM decoder (causal) ────────────────────────────────┘
           ↓
  Next-token logits [B, N_v + N_t, vocab]
```

The image tokens form a **prefix** that the LLM attends to while generating text. In production,
the real chat template sandwiches them in control tokens:
`<s>...<im_start>[IMAGE TOKENS]<im_end>...<user>text<assistant>`.

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| ViT forward | O(N_v²·d_v) | Compute | FP16/FP8, pre-compiled engine, CUDA graph |
| Visual projector | O(N_v·d_v·d_t) | Compute | Tiny; fuse with ViT output |
| LLM prefill attention | O((N_v+N_t)²·d_t) | **Memory** | FlashAttention, chunked prefill |
| LLM decode attention | O((N_v+N_t)·d_t) per token | **Memory** | KV-cache, PagedAttention, GQA/MQA |
| LLM decode MLP | O(d_t²) per token | Memory | Weight-only INT4 quantization |

**N_v** = visual tokens (often 256–1000 per image).
**N_t** = text tokens.
**d_v, d_t** = vision / text hidden dims.

## Critical Optimizations

### 1. Image Token Reduction

A 448×448 image at patch size 14 gives **1024 visual tokens**. Left alone, a 3-image prompt
is ~3000 tokens before the user has typed a word — and KV cache memory scales linearly with
sequence length.

Four common reduction strategies:

| Method | Token count | Used by |
|--------|-------------|---------|
| Raw patches | `(H/P)²` | Plain LLaVA |
| 2×2 pixel shuffle | `/4` | Qwen2-VL, Cosmos-Reason2 |
| Perceiver resampler | 64–256 fixed | Flamingo, IDEFICS |
| Q-Former | 32 fixed | BLIP-2 |

Pixel shuffle is the current default: almost free compute, clean 4× reduction,
preserves spatial structure. Perceiver resampler decouples token count from image
resolution but adds parameters.

### 2. Vision Encoder Caching

The vision encoder is deterministic — same image always produces the same tokens.
In multi-turn chat or when reusing images across queries (dashcam frames, surveillance
feeds), cache the post-projector tokens. One full ViT forward saved per cache hit.

### 3. Prefill vs Decode Asymmetry

VLMs have a very different prefill-to-decode ratio than text LLMs:

- **Text LLM**: prefill ≈ 10–100 tokens, decode ≈ 100–1000 → **decode dominates**
- **VLM with 1 image**: prefill ≈ 300–1000 tokens (mostly visual), decode ≈ 50–200 → **prefill dominates**

This changes optimization priorities. For text LLMs, MQA/GQA and KV-cache quantization
matter most. For VLMs, chunked prefill and FlashAttention matter more — prefill compute
is what users feel as TTFT.

**My Orin Nano benchmarks showed this directly:** TTFT on image inputs was
3–6× higher than TTFT on text-only inputs across all three runtimes. That gap
is the visual-token prefill cost.

### 4. Separate Vision and LLM Engines

In production (TensorRT Edge-LLM, vLLM-multimodal), the vision encoder and LLM decoder
run as **two independent inference engines**:

```
Image → [Vision Engine] → image_embeds → [LLM Engine with KV cache] → tokens
```

Why split them:
- Vision runs **once per image**; LLM runs **once per output token**. Different cost profiles.
- Vision can batch images from multiple requests; LLM uses continuous batching separately.
- Engines can live on different devices in multi-GPU deployments.
- Quantization strategies differ — ViT tolerates INT8 well; LLM decoder usually wants INT4 weights + FP16 activations.

## Architecture Summary

```
Vision path (runs once per image):
  Image [B,3,H,W]
    ↓ PatchEmbed (Conv2d k=14 s=14)
  Patches [B, N_v, d_v]             # d_v=1024 for ViT-L
    ↓ + position embeddings
    ↓ ViT blocks × L_v              # L_v=24 for ViT-L
    ↓ LayerNorm
  Features [B, N_v, d_v]
    ↓ Projector (Linear → GELU → Linear)
  Image tokens [B, N_v, d_t]        # d_t=2048 for 1.5B LLM

Text path:
  Token IDs [B, N_t]
    ↓ Embedding lookup
  Text embeds [B, N_t, d_t]

Fusion + decode:
  Concatenate → [B, N_v + N_t, d_t]
    ↓ + position embeddings (or RoPE)
    ↓ LLM blocks × L_t (causal)     # L_t=16–28
    ↓ LayerNorm
    ↓ LM head (Linear → vocab)
  Logits [B, N_v + N_t, vocab]
    ↓ take last position
  Next-token distribution
```

## Parameters & FLOPs (VLM-Base, ~2.4B params)

Close to Cosmos-Reason2-2B in spirit (exact Cosmos is ~2B; our config is slightly larger).

| Component | Params | % of total |
|-----------|--------|------------|
| Vision encoder (ViT-L/14) | ~304M | 13% |
| Visual projector | ~6M | 0.3% |
| LLM decoder + embeddings | ~2.05B | 87% |
| **Total** | **~2.36B** | |

**FLOPs per inference** (1 image at 448×448 → 1024 visual tokens, 32 text tokens in, 100 tokens out):

| Stage | Runs | FLOPs |
|-------|------|-------|
| ViT forward | 1× per image | ~60 GFLOPs |
| Projector | 1× per image | ~0.5 GFLOPs |
| LLM prefill | 1× per request | ~4 TFLOPs (over 1056 tokens) |
| LLM decode | 100× per request | ~2 GFLOPs/token |

Decode is memory-bound on edge hardware — arithmetic intensity is ~1 FLOP/byte at batch 1,
far below Orin Nano's compute ridge (~180 FLOP/byte FP16). Decode throughput on Jetson
tracks LPDDR5 bandwidth almost perfectly.

## Key Differences from Plain LLMs

1. **Mixed-modality sequences** — image and text embeddings sit in the same tensor, pass through the same attention layers. The LLM doesn't intrinsically know which is which; it learns patterns during training.
2. **Variable visual token count** — images of different resolutions produce different `N_v`. Modern VLMs (Qwen2-VL, Cosmos-Reason2) handle dynamic resolution with runtime token counts.
3. **Prefill-dominated latency** — visual tokens inflate prefill length by 5–20×.
4. **Vision encoder is the easy part** — standard ViT, fixed shape, no KV cache, no autoregression. Compile once and forget.
5. **Training is mostly about the projector** — LLaVA-style stage 1 freezes ViT + LLM and only trains the 2-layer MLP on image-caption pairs. Stage 2 optionally fine-tunes the LLM on instruction data.

## Two Fusion Patterns

Beyond the dominant **concatenation** pattern (this implementation, LLaVA, Qwen-VL, Cosmos),
there's a second family:

- **Cross-attention** (Flamingo, BLIP-2, IDEFICS): extra cross-attention layers are inserted into the LLM so text tokens attend to image tokens through a separate path. More parameter-efficient at high image counts, but architecturally more invasive. The LLaVA/Qwen-VL pattern has won the mainstream.

When someone says "VLM" today, assume concatenation unless told otherwise.

## On Jetson — what actually matters

From my three-runtime Cosmos-Reason2-2B benchmark on Orin Nano:

- **Vision encode cost (TTFT gap):** 75–420 ms depending on runtime. Not negligible.
- **Decode TPS:** 38–60 across llama.cpp / vLLM / TRT Edge-LLM, all dominated by LPDDR5 bandwidth.
- **Peak memory:** 4.3–6.9 GB at W4A16. The KV cache for a 1024-token context is ~500 MB at FP16 — most of which is visual tokens.

Knobs that actually moved numbers:
1. Quantization scheme (W4A16 beat llama.cpp's Q4_K_M on decode throughput)
2. KV cache length cap (trades context window for memory)
3. CUDA-graph capture for the vision path (TRT Edge-LLM pre-compiled the visual engine; vLLM didn't)

## References

- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — the simplest VLM pattern
- [Qwen-VL (Bai et al., 2023)](https://arxiv.org/abs/2308.12966) — resampler variant
- [Qwen2-VL (Wang et al., 2024)](https://arxiv.org/abs/2409.12191) — dynamic resolution, M-RoPE, pixel shuffle
- [Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198) — cross-attention pattern
- [Cosmos-Reason2](https://research.nvidia.com/labs/dir/cosmos-reason2/) — NVIDIA's physical-reasoning VLM
