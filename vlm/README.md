# Vision-Language Model (VLM)

Based on the LLaVA / Qwen-VL family — the dominant "ViT + projector + LLM" pattern used by
[LLaVA](https://arxiv.org/abs/2304.08485), [Qwen-VL](https://arxiv.org/abs/2308.12966),
[Qwen2-VL](https://arxiv.org/abs/2409.12191), and NVIDIA's
[Cosmos-Reason2](https://huggingface.co/nvidia/Cosmos-Reason2-2B) (post-trained from Qwen3-VL-2B-Instruct).

## Contents

| File | Description |
|------|-------------|
| [vlm_guide.html](./vlm_guide.html) | Interactive visual guide, Module 3 of 9 (open in browser): diagrams, an attention-mask widget, a calculator that matches `vlm.py`, a self-check quiz and exercises |
| [vlm.py](./vlm.py) | Reference implementation with annotations, a FLOP counter that matches `FlopCounterMode`, and a causal-mask demo |

## Quick Start

```bash
# View the interactive guide
open vlm_guide.html      # macOS
xdg-open vlm_guide.html  # Linux

# Run the implementation (CPU is fine; ~6 GB RAM, ~15 s)
python vlm.py
```

Build it yourself: [`labs/lab05_vlm_fusion.py`](../labs/lab05_vlm_fusion.py) is a guided lab for the projector and the
fusion step (setup in [`labs/README.md`](../labs/README.md)).

## The core idea

A VLM is not a new architecture — it's three things wired together:

1. A **vision encoder** (the [ViT](../vit/) from Module 2, with the `[CLS]` token and classification head removed)
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
the chat template wraps them in special tokens, e.g. Qwen-VL style:
`<|im_start|>user <|vision_start|>[IMAGE TOKENS]<|vision_end|>What is this?<|im_end|><|im_start|>assistant`.

With a plain causal mask (LLaVA, Qwen-VL, `vlm.py`), image token *i* sees only image tokens 0..*i*;
prefix-LM models such as PaliGemma make the image + prompt block bidirectional.

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| ViT forward | O(N_v·d_v² + N_v²·d_v) | Compute | FP16/INT8, pre-compiled engine, CUDA graph |
| Visual projector | O(N_v·d_v·d_t) | Compute | Tiny; build it into the vision engine |
| LLM prefill | O(N·d_t² + N²·d_t), N = N_v+N_t | Compute (attention's N² traffic without FlashAttention) | FlashAttention, chunked prefill, token reduction |
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
| Raw patches | `(H/P)²` | Plain LLaVA, `vlm.py` |
| 2×2 merge ("pixel shuffle") | `/4` | Qwen2/2.5/3-VL, Cosmos-Reason2, InternVL |
| Perceiver resampler | 64–256 fixed | Flamingo (→ cross-attention), Idefics2 (→ concat) |
| Q-Former | 32 fixed | BLIP-2 |

The 2×2 merge is the current default: almost free compute, clean 4× reduction,
preserves spatial structure. Qwen3-VL / Cosmos-Reason2 use patch 16 + 2×2 merge, so one
LLM token covers 32×32 pixels: a 448×448 image is 196 tokens. A Perceiver resampler
decouples token count from image resolution but adds parameters (~200M in Flamingo).

### 2. Vision Encoder Caching

The vision encoder is deterministic — same image always produces the same tokens.
In multi-turn chat or when the same image appears across queries, cache the post-projector
tokens, keyed on the preprocessed tensor. One full ViT forward saved per cache hit
(~0.72 TFLOPs for VLM-Base); vLLM's V1 encoder cache does this automatically.

### 3. Prefill vs Decode Asymmetry

VLMs have a very different prefill-to-decode ratio than text LLMs:

- **Text LLM**: prefill ≈ 10–100 tokens, decode ≈ 100–1000 → **decode dominates**
- **VLM with 1 image**: prefill ≈ 200–1000 tokens (mostly visual), decode ≈ 50–200 → **prefill dominates TTFT**

This changes optimization priorities. For text LLMs, MQA/GQA and KV-cache quantization
matter most. For VLMs, token reduction, chunked prefill and FlashAttention also matter —
prefill is what users feel as TTFT. For answers of 100+ tokens, decode still dominates
end-to-end time (e.g. 75 ms TTFT vs 100 × 17 ms of decode with vLLM on Orin Nano).

**My Orin Nano benchmarks showed this directly:** TTFT on image inputs was
1.2–14.5× higher than TTFT on text-only inputs across the three runtimes. That gap
is the vision encode + visual-token prefill cost.

### 4. Separate Vision and LLM Engines

In edge deployments (e.g. TensorRT Edge-LLM, which builds a `visual.engine` plus an LLM
engine), the vision encoder and LLM decoder run as **two independent inference engines**:

```
Image → [Vision Engine] → image_embeds → [LLM Engine with KV cache] → tokens
```

Why split them:
- Vision runs **once per image**; LLM runs **once per output token**. Different cost profiles.
- Vision has a static (or bucketed) shape; the LLM needs a KV cache and, in servers, continuous batching.
- (vLLM instead runs both in one model runner but schedules and caches encoder work separately.)
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
  Image tokens [B, N_v, d_t]        # d_t=2048 (Qwen3-1.7B, inside Qwen3-VL-2B / Cosmos-Reason2-2B)

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
  Logits [B, N_v + N_t, vocab]      # generation only needs the last position
    ↓ sample, append, repeat (decode with a KV cache)
  Next token
```

## Parameters & FLOPs (VLM-Base, ~2.36B params)

VLM-Base uses the ViT and LLM widths/depths of Cosmos-Reason2-2B (Qwen3-VL-2B: ViT 1024 × 24,
LLM 2048 × 28 with 16 heads, vocab 151,936). The model card lists 2,438,696,960 params, which is
Qwen3-VL-2B's 2,127,532,032 plus exactly one 151,936 × 2048 matrix, i.e. apparently an untied LM
head, as in VLM-Base. Simplifications: patch 14 without the 2×2 merge (1024 tokens at 448² instead of 196),
MHA instead of GQA (16 q / 8 kv), GELU MLP instead of SwiGLU, LayerNorm, learned positions.

| Component | Params | % of total |
|-----------|--------|------------|
| Vision encoder (ViT-L/14 @448) | 304.0M | 12.9% |
| Visual projector | 6.3M | 0.3% |
| LLM decoder + embeddings + LM head | 2048.6M | 86.8% |
| **Total** | **2,358.9M** | |

**FLOPs per inference** (1 image at 448×448 → 1024 visual tokens, 32 text tokens in, 100 tokens out;
numbers from `compute_flops(vlm_base(), 32, 100)`, 1 MAC = 2 FLOPs):

| Stage | Runs | FLOPs |
|-------|------|-------|
| ViT forward | 1× per image | ~723 GFLOPs (ViT-L/14 over 1024 patches) |
| Projector | 1× per image | ~13 GFLOPs |
| LLM prefill | 1× per request | ~3.2 TFLOPs (over 1056 tokens) |
| LLM decode | 100× per request | ~3.7 GFLOPs/token (≈ 2 × LLM weights) |

Decode is memory-bound on edge hardware: each step reads every weight once for ~2 FLOPs per
weight, i.e. ~1 FLOP/byte at FP16 (~4 at INT4), far below the Orin Nano Super's ridge of
~170 FLOP/byte (17 FP16 TFLOPS ÷ 102 GB/s). Decode throughput on Jetson tracks LPDDR5
bandwidth: for Cosmos-Reason2-2B at W4A16 the floor is ~8–13 ms/token vs 17 ms measured.

## Key Differences from Plain LLMs

1. **Mixed-modality sequences** — image and text embeddings sit in the same tensor, pass through the same attention layers. The LLM doesn't intrinsically know which is which; it learns patterns during training.
2. **Variable visual token count** — images of different resolutions produce different `N_v`. Modern VLMs (Qwen2-VL, Cosmos-Reason2) handle dynamic resolution with runtime token counts.
3. **Prefill-dominated TTFT** — visual tokens inflate prefill length by 5–20×.
4. **Vision encoder is the easy part** — standard ViT, fixed shape, no KV cache, no autoregression. Compile once and forget.
5. **Training starts with the projector** — LLaVA-style stage 1 freezes ViT + LLM and only trains the projector on image-caption pairs. Stage 2 fine-tunes projector + LLM on visual instruction data (ViT still frozen).

## Two Fusion Patterns

Beyond the dominant **concatenation** pattern (this implementation, LLaVA, Qwen-VL, Cosmos),
there's a second family:

- **Cross-attention** (Flamingo, IDEFICS-1, Llama 3.2-Vision): extra cross-attention layers are inserted into the LLM so text tokens attend to image tokens through a separate path. Image tokens don't lengthen the LLM sequence, but the design is architecturally more invasive. The LLaVA/Qwen-VL pattern has won the mainstream. (BLIP-2's Q-Former cross-attends inside the adapter, then its 32 outputs are concatenated as an LLM prefix.)

When someone says "VLM" today, assume concatenation unless told otherwise.

## On Jetson — what actually matters

From my three-runtime Cosmos-Reason2-2B benchmark on Orin Nano:

- **Vision encode + visual prefill (TTFT gap):** 14–391 ms depending on runtime (image TTFT 75–420 ms vs text TTFT 29–61 ms). Not negligible.
- **Decode TPS:** 38–60 across llama.cpp / vLLM / TRT Edge-LLM, all dominated by LPDDR5 bandwidth.
- **Peak memory:** 4.3–6.9 GB at W4A16. The KV cache is small by comparison: 112 KiB per token at FP16 (28 layers × 8 KV heads × 128 × 2 (K,V) × 2 B), so ~112 MiB for a 1024-token context, of which each 448² image contributes 196 tokens (21 MiB).

Knobs that actually moved numbers:
1. Quantization scheme (W4A16 beat llama.cpp's Q4_K_M on decode throughput)
2. KV cache length cap (trades context window for memory)
3. CUDA-graph capture for the vision path (TRT Edge-LLM pre-compiled the visual engine; vLLM didn't)

## References

- [LLaVA (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) — the simplest VLM pattern
- [Qwen-VL (Bai et al., 2023)](https://arxiv.org/abs/2308.12966) — resampler variant
- [Qwen2-VL (Wang et al., 2024)](https://arxiv.org/abs/2409.12191) — dynamic resolution, M-RoPE, pixel shuffle
- [LLaVA-1.5 (Liu et al., 2023)](https://arxiv.org/abs/2310.03744) — 2-layer MLP projector, CLIP ViT-L/14-336
- [Qwen3-VL Technical Report (2025)](https://arxiv.org/abs/2511.21631) — base of Cosmos-Reason2
- [PaliGemma (Beyer et al., 2024)](https://arxiv.org/abs/2407.07726) — prefix-LM masking
- [Flamingo (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198) — cross-attention pattern
- [BLIP-2 (Li et al., 2023)](https://arxiv.org/abs/2301.12597) — Q-Former
- [Cosmos-Reason2-2B model card](https://huggingface.co/nvidia/Cosmos-Reason2-2B) — NVIDIA's physical-reasoning VLM

Previous module: [Vision Transformer](../vit/) · Next module: [Diffusion (DiT)](../diffusion/)
