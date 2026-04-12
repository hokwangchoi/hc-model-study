# Transformer

Based on ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762).

## Contents

| File | Description |
|------|-------------|
| [transformer_guide.html](./transformer_guide.html) | Interactive visual guide (open in browser) |
| [transformer.py](./transformer.py) | Reference implementation with annotations |

## Quick Start

```bash
# View the interactive guide
open transformer_guide.html      # macOS
xdg-open transformer_guide.html  # Linux

# Run the implementation
python transformer.py
```

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| QKV Projection | O(Nd²) | Compute | INT8/FP8 quantization |
| Attention (QKᵀ) | O(N²d) | **Memory** | FlashAttention |
| Softmax | O(N²) | Memory | Fused with attention |
| FFN | O(Nd·d_ff) | Compute | Quantization, tensor cores |
| LayerNorm | O(Nd) | Memory | Fused kernels |

## Critical Optimizations

### 1. FlashAttention
- **Problem**: Standard attention materializes O(N²) attention matrix
- **Solution**: Tiled computation in SRAM, never write full matrix to HBM
- **Result**: 2-4× speedup, O(N) memory instead of O(N²)

```python
# PyTorch 2.0+
F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Auto uses FlashAttention
```

### 2. KV-Cache
- **Problem**: Autoregressive decoding recomputes attention for all tokens
- **Solution**: Cache K, V tensors, only compute new token's Q
- **Result**: O(N³) → O(N²) total complexity for generation

**Memory cost**: `2 × L × N × d × sizeof(dtype)` per sequence

### 3. Quantization
| Precision | Memory | Tensor Core Throughput (H100) |
|-----------|--------|------------------------------|
| FP32 | 1× | 67 TFLOPS |
| FP16/BF16 | 2× | 989 TFLOPS |
| FP8 | 4× | 1,979 TFLOPS |
| INT8 | 4× | 1,979 TOPS |

### 4. Kernel Fusion
Fuse consecutive memory-bound ops:
- `Add + LayerNorm` → single kernel
- `Linear + GELU` → single kernel
- `QKV projection` → single batched GEMM

## Architecture Summary

```
Input [B, N] → Token IDs
    ↓
Embedding [B, N, d] → Learned lookup + positional encoding
    ↓
┌─────────────────────────────────────┐
│  Transformer Block (×L layers)      │
│  ┌─────────────────────────────────┐│
│  │ LayerNorm → Multi-Head Attn    ││ ← O(N²) memory
│  │         ↓ + residual           ││
│  │ LayerNorm → FFN                ││ ← 67% of FLOPs
│  │         ↓ + residual           ││
│  └─────────────────────────────────┘│
└─────────────────────────────────────┘
    ↓
LayerNorm → Output Projection [B, N, V]
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

## References

- [Original Paper](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [PyTorch SDPA](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
