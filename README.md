# Model Study

Deep dives into ML model architectures — focused on **inference optimization** for GPU deployment.

## How to Use This Repo

Each model study follows a systematic approach for understanding inference performance:

### 1. Understand the Data Flow
Start with the visual guide (`*_guide.html`) to trace how tensors move through the model:
- Input shape → intermediate shapes → output shape
- Where does the batch dimension go?
- Which operations are sequential vs parallelizable?

### 2. Identify Computational Bottlenecks
For each operation, ask:
- **Compute-bound or memory-bound?** (FLOP/byte ratio)
- **What's the complexity?** O(N), O(N²), O(d²)?
- **Which kernel runs this?** (cuBLAS GEMM, custom CUDA, fused kernel?)

### 3. Analyze Memory Patterns
- **Activation memory**: What's kept for backward pass? What can be recomputed?
- **Weight memory**: Parameter count × dtype size
- **Intermediate buffers**: Attention scores, KV-cache, etc.

### 4. Know the Optimization Techniques
Each model has specific optimizations:
- Kernel fusion opportunities
- Quantization-friendly layers
- Caching strategies (KV-cache, activation caching)
- Sparsity patterns

### 5. Run the Code
The Python implementations let you:
```bash
python model.py  # See architecture, parameter counts, FLOPs breakdown
```

---

## Roadmap

| Model | Status | Key Optimization Focus |
|-------|--------|----------------------|
| [Transformer](./transformer/) | ✅ Complete | Attention O(N²) → FlashAttention, KV-cache |
| [Vision Transformer (ViT)](./vision-transformer/) | 🔲 Planned | Patch embedding, fixed sequence length |
| [Vision-Language Models](./vlm/) | 🔲 Planned | Cross-attention, image encoder fusion |
| [Diffusion Models](./diffusion/) | 🔲 Planned | Iterative denoising, U-Net optimization |
| [State Space Models (Mamba)](./mamba/) | 🔲 Planned | Linear attention, selective scan |

---

## Quick Reference: What to Optimize

| Bottleneck | Symptom | Solutions |
|------------|---------|-----------|
| **Memory-bound** | Low GPU utilization, high memory bandwidth | Kernel fusion, reduce memory access |
| **Compute-bound** | High GPU utilization, slow throughput | Quantization (INT8/FP8), tensor cores |
| **O(N²) attention** | OOM on long sequences | FlashAttention, sliding window, sparse attention |
| **Large KV-cache** | Limited batch size | PagedAttention, GQA/MQA, quantized cache |
| **Many small ops** | Kernel launch overhead | Operator fusion, CUDA graphs |

---

## Inference Optimization Checklist

When deploying a model, work through this checklist:

```
□ Profile baseline (torch.profiler, Nsight Systems)
□ Identify top-5 slowest operations
□ Check arithmetic intensity (compute vs memory bound)
□ Apply framework optimizations:
  □ torch.compile / TensorRT / ONNX Runtime
  □ Flash Attention enabled?
  □ Fused kernels (LayerNorm, GELU, etc.)
□ Quantization:
  □ Weight-only (INT4/INT8) for memory-bound
  □ Full quantization (INT8/FP8) for compute-bound
□ Batching strategy:
  □ Static vs dynamic batching
  □ Continuous batching for LLMs
□ Memory optimization:
  □ KV-cache quantization
  □ Activation checkpointing trade-offs
□ Hardware-specific:
  □ Tensor Core alignment (multiples of 8/16)
  □ Memory coalescing
```

---

## Structure

```
model-name/
├── README.md              # Quick overview, key concepts
├── model_guide.html       # Interactive visual guide (open in browser)
├── model.py               # Reference implementation with annotations
└── figures/               # Additional diagrams (if needed)
```

Each `model_guide.html` includes a **recommended video** section with the best visual explanation for non-ML engineers.

---

## Resources

### Profiling & Analysis
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)

### Optimization Libraries
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [vLLM](https://github.com/vllm-project/vllm)
- [NVIDIA Apex](https://github.com/NVIDIA/apex)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer
- [FlashAttention](https://arxiv.org/abs/2205.14135) — Memory-efficient attention
- [LLM.int8()](https://arxiv.org/abs/2208.07339) — Quantization
- [Mamba](https://arxiv.org/abs/2312.00752) — State space models

---

## License

MIT
