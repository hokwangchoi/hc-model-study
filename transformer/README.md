# Transformer

Module 1 of the learning path. A decoder-only (GPT-style) transformer, building on ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762). The code differs from the 2017 encoder-decoder model in a few deliberate ways (pre-LayerNorm, GELU, learned positions, no encoder or cross-attention); the docstring at the top of `transformer.py` lists them all.

## Contents

| File | Description |
|------|-------------|
| [transformer_guide.html](./transformer_guide.html) | Interactive visual guide (open in browser) |
| [transformer.py](./transformer.py) | Reference implementation with annotations |

## What you'll learn

- Trace every tensor shape from token IDs `[B, N]` to logits `[B, N, V]`
- Compute one attention head by hand, and explain the `1/√d_k` scaling
- Why the causal mask makes a KV-cache valid, and how big that cache is
- Prefill vs decode: why single-token decode on Jetson Orin is bound by memory bandwidth, not FLOPs
- What FlashAttention changes (memory traffic) and what it does not (FLOPs)

Suggested order: guide §0–2 (with the widgets) → `python transformer.py` → guide §3–6 → self-check and exercises in §8.

## Quick Start

```bash
# View the interactive guide
open transformer_guide.html      # macOS
xdg-open transformer_guide.html  # Linux

# Run the implementation (needs torch >= 2.1)
python transformer.py
```

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| QKV Projection | O(Nd²) | Compute (prefill) / Memory (decode, small batch) | INT8/FP8, weight-only INT4 |
| Attention (QKᵀ) | O(N²d) | **Memory** (naive N×N scores) | FlashAttention |
| Softmax | O(N²) | Memory | Fused with attention |
| FFN | O(Nd·d_ff) | Compute (prefill) / Memory (decode, small batch) | Quantization, Tensor Cores |
| LayerNorm | O(Nd) | Memory | Fused kernels |

Decode at batch 1 reads every weight once per generated token (about 1 FLOP per byte in FP16), so it is weight-bandwidth bound: weight-only INT4/INT8 quantization helps by cutting bytes, not FLOPs.

## Critical Optimizations

### 1. FlashAttention
- **Problem**: Standard attention materializes an O(N²) score matrix in DRAM (HBM, or LPDDR5 on Jetson)
- **Solution**: Tiled computation in on-chip SRAM with an online softmax; the full matrix is never written out
- **Result**: 2-4× speedup, O(N) extra memory instead of O(N²); the FLOPs are unchanged

```python
# PyTorch 2.x
F.scaled_dot_product_attention(q, k, v, is_causal=True)
# dispatches to a fused kernel (cuDNN / FlashAttention-2 / memory-efficient) when the
# inputs qualify, e.g. CUDA + fp16/bf16; otherwise falls back to the math kernel
```

### 2. KV-Cache
- **Problem**: Autoregressive decoding would otherwise re-run all previous tokens at every step
- **Solution**: Cache each layer's K, V for past tokens. Each step computes Q, K, V only for the new token, appends its K, V to the cache, and attends over the whole cache
- **Result**: Over a generation, attention-score work drops from O(N³) to O(N²) and GEMM work from O(N²) to O(N)

**Memory cost**: `2 × L × N × n_kv_heads × d_k × sizeof(dtype)` per sequence (= `2 × L × N × d × sizeof(dtype)` for classic multi-head attention, where n_kv_heads = h). Modern LLMs use grouped-query attention: Llama 3 8B has 8 K/V heads for 32 query heads, so its cache is 4× smaller (128 KiB per token in FP16).

### 3. Quantization
H100 SXM, dense (NVIDIA's datasheet lists 2× these "with sparsity"):

| Precision | Bytes per weight vs FP32 | Throughput (H100 SXM, dense) |
|-----------|--------|------------------------------|
| FP32 (CUDA cores) | 1× | 67 TFLOPS |
| TF32 (Tensor Cores) | 1× | 495 TFLOPS |
| FP16/BF16 | ½ | 989 TFLOPS |
| FP8 | ¼ | 1,979 TFLOPS |
| INT8 | ¼ | 1,979 TOPS |
| FP4 | ⅛ | Blackwell only (e.g. Jetson AGX Thor) |

TensorRT 10+ uses explicit quantization (Q/DQ nodes, e.g. from NVIDIA Model Optimizer); the implicit INT8 calibrator API is deprecated since 10.1 and removed in TensorRT 11.

### 4. Kernel Fusion
Fuse consecutive memory-bound ops:
- `Add + LayerNorm` → single kernel
- `Linear + GELU` → single kernel
- `QKV projection` → one GEMM with a concatenated weight `W_qkv [d, 3d]` instead of 3 separate GEMMs

## Architecture Summary

```
Input [B, N] → Token IDs
    ↓
Embedding [B, N, d] → Learned lookup + learned positional embedding
    ↓
┌─────────────────────────────────────┐
│  Transformer Block (×L layers)      │
│  ┌─────────────────────────────────┐│
│  │ LayerNorm → Multi-Head Attn    ││ ← O(N²) scores
│  │         ↓ + residual           ││
│  │ LayerNorm → FFN                ││ ← 2/3 of block params
│  │         ↓ + residual           ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    ↓
LayerNorm → Output Projection (LM head) [B, N, V]
    ↓
Softmax → Next token probabilities
```

## Parameters & FLOPs (per layer)

| Component | Parameters | FLOPs (forward) |
|-----------|------------|-----------------|
| QKV projection | 3d² | 6BNd² |
| Output projection | d² | 2BNd² |
| FFN | 8d² | 16BNd² |
| **Total** | **12d²** | **24BNd² + 4BN²d** |

Once per model, the LM head adds 2BNdV FLOPs (about 27% of a GPT-2 Small forward pass over 1,024 tokens). `compute_flops()` in `transformer.py` includes it.

## Next

- **Hands-on follow-up (recommended next):** the tutorial [Build an LLM from scratch](../build-llm/build_llm_guide.html).
- **Next module:** [Vision Transformer](../vit/), the same block applied to image patches with bidirectional attention.

## References

- [Original Paper](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [FlashAttention-3](https://arxiv.org/abs/2407.08608)
- [GQA](https://arxiv.org/abs/2305.13245)
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)
