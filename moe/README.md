# Mixture-of-Experts LLMs (sparse MoE + MLA)

Based on [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066), [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) for multi-head latent attention, and the [DeepSeek-V3 Technical Report (2024)](https://arxiv.org/abs/2412.19437), with [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961) and [Mixtral (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) as the baseline. Most large open LLMs as of Oct 2026 (DeepSeek-V3/V4, Qwen3/3.5, gpt-oss, Kimi K2, GLM-5) are sparse-MoE decoders built this way.

## Contents

| File | Description |
|------|-------------|
| [moe_guide.html](./moe_guide.html) | Interactive visual guide (open in browser): router playground, KV-cache bars, batch-vs-experts plot, memory/bandwidth calculator |
| [moe.py](./moe.py) | Reference implementation with annotations and tests (CPU, runs in a few seconds) |

## Quick Start

```bash
# View the interactive guide
open moe_guide.html      # macOS
xdg-open moe_guide.html  # Linux

# Run the implementation (tests + config comparison printout)
python moe.py
```

## The core idea

Replace each layer's FFN with **E** smaller FFNs (experts) and a router that picks **k** of them per token:

```
scores = sigmoid(W_r x)            # or softmax; [E]
idx    = topk(scores + b, k)       # b = load-balancing bias, affects the CHOICE only
g      = scores[idx] / sum(scores[idx])
y      = shared(x) + Σ_{i∈idx} g_i · Expert_i(x)
```

- **Total parameters** set the memory you need.
- **Active parameters** set FLOPs per token and, at batch 1, the bytes read per decoded token.

| Model | Weights out | Total | Active | Routed (+shared) | Top-k | Attention |
|-------|-------------|-------|--------|------------------|-------|-----------|
| Mixtral 8x7B | Dec 2023 | 47B | 13B | 8 | 2 | GQA 32Q/8KV |
| DeepSeek-V3 | Dec 2024 | 671B | 37B | 256 + 1 | 8 | MLA |
| Qwen3-235B-A22B | Apr 2025 | 235B | 22B | 128 | 8 | GQA 64Q/4KV |
| Qwen3-30B-A3B | Apr 2025 | 30.5B | 3.3B | 128 | 8 | GQA 32Q/4KV |
| Kimi K2 | Jul 2025 | 1T | 32B | 384 + 1 | 8 | MLA |
| gpt-oss-120b | Aug 2025 | 116.8B | 5.1B | 128 | 4 | GQA 64Q/8KV, alternating 128-token window / full |
| gpt-oss-20b | Aug 2025 | 20.9B | 3.6B | 32 | 4 | same |
| Qwen3.5-397B-A17B | Feb 2026 | 397B | 17B | 512 + 1 | 10 | hybrid: 3 Gated DeltaNet : 1 gated attention |
| DeepSeek-V4-Flash | Apr 2026 | 284B | 13B | 256 + 1 | 6 | CSA + HCA compressed attention |
| DeepSeek-V4-Pro | Apr 2026 | 1.6T | 49B | 384 + 1 | 6 | CSA + HCA compressed attention |

Sources: papers, model cards and `config.json` files (links in the guide's References).

## Key Inference Bottlenecks

| Operation | Cost | Bound | Optimization |
|-----------|------|-------|--------------|
| Router (d × E GEMM + top-k) | tiny | Memory / latency | Fuse with top-k; keep in FP32/BF16 |
| Token permute (dispatch) | O(T·k·d) copies | Memory | Sort-by-expert + gather; DeepEP dispatch/combine across GPUs |
| Expert FFNs | 6·d·f FLOPs per token per expert | **Memory** at small batch | Grouped GEMM (contiguous for prefill, masked for decode) |
| All-to-all (expert parallel) | 2 exchanges per MoE layer | **Network** | DeepEP, node-limited routing, overlap with compute |
| MLA attention (absorbed) | 2·n_h·(2·d_c + d_R) FLOPs per cached token per layer | near compute-bound | dedicated MLA decode kernels, sparse attention (DSA in DeepSeek-V3.2) |
| Weights resident | total params | **Capacity** | FP8 / MXFP4 / FP4 experts, expert parallelism, offloading |

## Critical Optimizations

### 1. Load balancing without hurting the loss
Switch-style auxiliary loss `α · E · Σ f_i · P_i` (α = 1e-2 in Switch), or DeepSeek-V3's **auxiliary-loss-free bias**: after each step, `b_i += γ · sign(mean_load − load_i)` with γ = 0.001; the bias only changes which experts win, never the gate value. `moe.py` demo (router only, skewed inputs): max/mean expert load 5.20 → 1.30 with the bias rule.

### 2. Fine-grained + shared experts
Split experts into many small ones (DeepSeek-V3: 256 routed of 3 × 7168 × 2048 ≈ 44M params each) and add an always-on shared expert. C(256, 8) ≈ 4.1 × 10¹⁴ expert combinations vs C(8, 2) = 28 for Mixtral, at similar active FLOPs.

### 3. Multi-head latent attention (MLA)
Cache a 512-dim latent `c_KV` plus a 64-dim shared RoPE key per token per layer instead of full K/V. DeepSeek-V3 shape (61 layers, 128 heads, d_h = 128), BF16:

| Attention | KV per token | 32K context | 128K context |
|-----------|-------------|-------------|--------------|
| MHA (128 KV heads) | 3,904 KiB | 122.0 GiB | 488.0 GiB |
| GQA (8 KV heads) | 244 KiB | 7.62 GiB | 30.5 GiB |
| MQA (1 KV head) | 30.5 KiB | 0.95 GiB | 3.81 GiB |
| **MLA (512 + 64)** | **68.6 KiB** | **2.14 GiB** | **8.58 GiB** |

At decode, fold `W_UK` into the query and `W_UV` into `W_O` ("absorbed" form): attention then runs directly on the latent cache. That costs FLOPs: ≈ 557 GFLOP per token at 32K context for DeepSeek-V3's attention (vs 73 GFLOP for all its active matmuls), ≈ 242 FLOP per BF16 byte of cache read vs 16 for Qwen3-235B's GQA.

### 4. Batch size vs expert utilization
Expected distinct experts per layer for a batch of B tokens (uniform routing): `E · (1 − (1 − k/E)^B)`. DeepSeek-V3: B=1 → 8, B=8 → 57.4, B=32 → 163.3, B=128 → 251.6. Each expert is reused only B·k/E times per read, so an MoE needs E/k times the batch of a dense model to become compute-bound (≈ 9,440 tokens per step for DeepSeek-V3 on an H100).

### 5. Expert parallelism, grouped GEMM, dropless dispatch
Whole experts on different GPUs, attention data-parallel, two all-to-alls per MoE layer (DeepSeek-V3: EP32 for prefill, EP320 for decode; redundant copies of hot experts). Sort tokens by expert and run one grouped GEMM instead of E small ones. Avoid capacity-based token dropping at inference: in `moe.py`, capacity factor 1.0 changes 12 of 60 tokens' outputs depending on what else is in the batch; dropless changes none.

### 6. Quantize the experts
Routed experts hold 91–99% of the weights (DeepSeek-V3 97.4%, gpt-oss-120b 98.2%). gpt-oss ships MXFP4 experts (≈ 4.25 bits/param) with attention, router, embeddings and LM head in BF16 → 60.7 GiB computed vs 60.8 GiB checkpoint. DeepSeek-V3 uses FP8 with 128 × 128 block scales; DeepSeek-V4 uses FP4 experts.

## Architecture Summary

```
Token IDs [B, T]
    ↓ Embedding
┌───────────────────────────────────────────────┐
│ First n_dense layers: Attention + dense FFN    │  (DeepSeek-V3: 3)
├───────────────────────────────────────────────┤
│ MoE layer (× L − n_dense)                      │
│   x ← x + MLA(RMSNorm(x))      ← latent KV cache│
│   x ← x + MoE(RMSNorm(x))                      │
│         router → top-k of E routed experts     │
│         + shared expert(s), weighted sum       │
└───────────────────────────────────────────────┘
    ↓ Final RMSNorm → LM head → logits [B, T, V]
```

## Parameters & bytes (from `python moe.py`, formulas checked against the module's own parameter count)

| Model | Total | Active | Active % | KV/token BF16 | Read/token @FP8, B=1 | GFLOP/token |
|-------|-------|--------|----------|---------------|----------------------|-------------|
| Mixtral-8x7B | 46.7B | 12.88B | 27.6% | 128.0 KiB | 12.75 GB | 25.5 |
| DeepSeek-V3 | 671.0B | 37.55B | 5.6% | 68.6 KiB | 36.63 GB | 73.3 |
| Kimi-K2 | 1026.4B | 32.86B | 3.2% | 68.6 KiB | 31.69 GB | 63.4 |
| Qwen3-235B-A22B | 235.1B | 22.19B | 9.4% | 188.0 KiB | 21.57 GB | 43.1 |
| Qwen3-30B-A3B | 30.5B | 3.35B | 11.0% | 96.0 KiB | 3.04 GB | 6.1 |
| gpt-oss-120b | 116.8B | 5.71B | 4.9% | 36.1 KiB* | 5.13 GB | 10.3 |
| gpt-oss-20b | 20.9B | 4.19B | 20.0% | 24.1 KiB* | 3.61 GB | 7.2 |

Active includes the input embedding table; read/token and FLOPs exclude it (it is a row lookup). *gpt-oss KV amortised at 32K context (half its layers use a 128-token window).

## Key Differences from Dense Transformers

| Aspect | Dense | Sparse MoE |
|--------|-------|------------|
| Weights touched per token | all | attention + k of E experts |
| Memory to hold weights | total | total (4–33× the active set) |
| Decode bytes per token, batch 1 | all weights | active weights |
| Weight reuse per step at batch B | B | B·k/E per expert |
| Multi-GPU pattern | TP / PP | expert parallel + all-to-all |
| Kernel shape | large GEMMs | grouped GEMMs + permutes |

## On Edge Hardware

Batch-1 decode is the best case for MoE: only k experts per layer are read. Weight-only roofline ceilings (KV, activations and kernel efficiency lower real numbers):

| Model, format | Resident | Read/token | Jetson AGX Orin 64GB (204.8 GB/s) | Jetson AGX Thor T5000 (273 GB/s) |
|---------------|----------|-----------|-----------------------------------|----------------------------------|
| Qwen3-30B-A3B, 4-bit | 14.2 GiB | 1.52 GB | ≤ 135 tok/s | ≤ 179 tok/s |
| dense 30B, 4-bit | 14.2 GiB | 15.1 GB | ≤ 13.6 tok/s | ≤ 18.1 tok/s |
| gpt-oss-20b, MXFP4 + BF16 | 12.8 GiB | 3.71 GB | ≤ 55 tok/s | ≤ 74 tok/s |
| gpt-oss-120b, MXFP4 + BF16 | 60.7 GiB | 5.00 GB | too tight in 64 GB | ≤ 55 tok/s |

On Jetson the CPU and iGPU share the same SoC DRAM ([CUDA for Tegra app note](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html)), so "offload experts to CPU RAM" frees nothing; pick an MoE whose quantized total fits. On discrete GPUs with too little memory, [expert offloading](https://arxiv.org/abs/2312.17238) (LRU expert cache + speculative prefetch + 2–3-bit experts) ran Mixtral-8x7B at about 2–3 tokens/s on consumer hardware.

## Connection to the Broader Roadmap

- **[Transformer](../transformer/)**: MoE changes only the FFN; MLA replaces the attention block.
- **[Mamba](../mamba/)**: the other route to cheap long-context inference; Qwen3.5 combines a 512-expert MoE with Gated DeltaNet layers.
- **[VLMs](../vlm/)**: MoE language backbones are increasingly common (Qwen3.5 ships a vision encoder in the same checkpoint).

## References

- [Sparsely-Gated MoE (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
- [GShard (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668)
- [Switch Transformers (Fedus et al., 2021)](https://arxiv.org/abs/2101.03961)
- [DeepSeekMoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066)
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088)
- [DeepSeek-V2 (2024)](https://arxiv.org/abs/2405.04434) — MLA
- [Auxiliary-Loss-Free Load Balancing (Wang et al., 2024)](https://arxiv.org/abs/2408.15664)
- [DeepSeek-V3 (2024)](https://arxiv.org/abs/2412.19437)
- [Qwen3 (2025)](https://arxiv.org/abs/2505.09388)
- [Kimi K2 (2025)](https://arxiv.org/abs/2507.20534)
- [gpt-oss model card (2025)](https://arxiv.org/abs/2508.10925)
- [DeepSeek-V3.2 (2025)](https://arxiv.org/abs/2512.02556) — DeepSeek Sparse Attention
- [DeepSeek-V4 (2026)](https://arxiv.org/abs/2606.19348) — CSA/HCA attention, FP4 experts
- [MQA (Shazeer, 2019)](https://arxiv.org/abs/1911.02150) · [GQA (Ainslie et al., 2023)](https://arxiv.org/abs/2305.13245)
- [MegaBlocks (Gale et al., 2022)](https://arxiv.org/abs/2211.15841)
- [MoE offloading (Eliseev & Mazur, 2023)](https://arxiv.org/abs/2312.17238)
- [Multi-token prediction (Gloeckle et al., 2024)](https://arxiv.org/abs/2404.19737)
- [DeepEP](https://github.com/deepseek-ai/DeepEP) · [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) · [vLLM expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/) · [TensorRT-LLM expert parallelism](https://nvidia.github.io/TensorRT-LLM/advanced/expert-parallelism.html)
