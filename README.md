# Model Study

Deep dives into ML model architectures — focused on **inference optimization** for GPU and edge deployment.

**Start here: open [`index.html`](./index.html)** in a browser for the learning path. It shows the nine modules in order with their prerequisites, the labs, the efficiency cookbook, the two hands-on tutorials, how to study each module, and a suggested weekly cadence.

**Recommended order:**
1. [Transformer, step by step](./transformer/transformer_step_by_step.html), a primer that follows 3 tokens through a GPT with real numbers;
2. the full [Transformer guide](./transformer/transformer_guide.html);
3. [labs](./labs/) 01–03;
4. [Build an LLM from scratch](./build-llm/);
5. ViT → VLM → Diffusion → Mamba → MoE → BEV → VLA, each followed by its lab (04–10);
6. [Robot Foundation Models](./robot-fm/) (no lab);
7. [Build a personal agent](./build-agent/), optional: it needs only Build an LLM, so any time after step 4 works, but it is suggested at the end.

Keep the [memory & inference efficiency cookbook](./efficient-inference/efficiency_guide.html) open throughout as a reference; it recaps every optimization idea of the path.

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
- **Intermediate buffers**: Attention scores, KV cache, etc.

### 4. Know the Optimization Techniques
Each model has specific optimizations:
- Kernel fusion opportunities
- Quantization-friendly layers
- Caching strategies (KV cache, activation caching)
- Sparsity patterns

### 5. Run the Code and Check Yourself
The Python implementations let you:
```bash
cd transformer && pip install -r requirements.txt
python transformer.py  # See architecture, parameter counts, FLOPs breakdown
```
Then do the guide's exercises, take its self-check, and retake the self-check a few days later (see [`index.html`](./index.html#how)).

---

## Roadmap

### Main track (in order)

| # | Model | Status | Key Optimization Focus |
|---|-------|--------|----------------------|
| 1 | [Transformer](./transformer/) | ✅ Complete | Attention O(N²) → FlashAttention, KV cache |
| 2 | [Vision Transformer (ViT)](./vit/) | ✅ Complete | Patch embedding, [CLS] token, bidirectional attention |
| 3 | [Vision-Language Models](./vlm/) | ✅ Complete | ViT + projector + LLM, image-token prefix, image-dominated prefill (TTFT) |
| 4 | [Diffusion Models (DiT)](./diffusion/) | ✅ Complete | Iterative denoising, adaLN-Zero, DDIM/flow matching, step reduction |
| 5 | [State Space Models (Mamba)](./mamba/) | ✅ Complete | Selective scan, dual-mode (parallel/recurrent), no KV cache |
| 6 | [Mixture-of-Experts LLMs](./moe/) | ✅ New | Top-k expert routing, total vs. active parameters, expert parallelism, MLA's compressed KV cache |
| 7 | [Multi-camera Tokenizers (BEV)](./bev/) | ✅ Complete | BEVFormer-style: learnable queries + deformable cross-attention |
| 8 | [Vision-Language-Action (VLA)](./vla/) | ✅ Complete | Pi0-style: VLM + action expert, flow matching on action chunks |
| 9 | [Robot Foundation Models](./robot-fm/) | ✅ Complete | GR00T N1-style: System 1/2 + heterogeneous I/O for multi-embodiment |

### Hands-on track

| Tutorial | When | What you build |
|----------|------|----------------|
| [Transformer, step by step](./transformer/transformer_step_by_step.html) | first, before the full Transformer guide | A picture-first walkthrough of every operation in a GPT on a 3-token toy; every number is reproduced by `transformer/step_by_step.py`. |
| [Build-it-yourself labs](./labs/) | after each module: 01–03 after the full Transformer guide, 04–10 after modules 2–8 | 10 fill-in-the-blank labs (attention → flow matching) with `python labs/check.py N` giving PASS / TODO / FAIL and hints; solutions in `labs/solutions/`. |
| [Build an LLM from scratch](./build-llm/) | after labs 01–03, before module 2 | Byte-level BPE, a GPT decoder (RMSNorm, RoPE, SwiGLU, tied embeddings), pretraining, sampling, a tested KV cache and SFT. One file; trains on a laptop CPU in 1–3 minutes, no downloads. |
| [Memory & inference efficiency cookbook](./efficient-inference/) | reference throughout, from the Transformer module on; review at the end | "Will it fit?" calculator, roofline, and quantization, KV cache, FlashAttention, paged KV, speculative decoding, LoRA and checkpointing, with a runnable lab. |
| [Build a personal agent](./build-agent/) | last, optional (needs only Build an LLM, so any time after it) | A tool-use loop on the Claude Messages API with notes search (BM25), file-based memory, guardrails and scripted evals; an offline mock model runs everything without an API key. |

---

## Quick Reference: What to Optimize

| Bottleneck | Symptom | Solutions |
|------------|---------|-----------|
| **Memory-bound** (LLM decode, small batch) | Tensor cores mostly idle while DRAM bandwidth is near peak; time tracks bytes moved | Batch more requests, weight-only quantization (INT4/INT8, FP8/FP4 weights), kernel fusion, speculative decoding |
| **Compute-bound** (prefill, large batch, diffusion) | High tensor-core utilization; time tracks FLOPs | FP8 matmuls (Ada/Hopper/Blackwell), FP4 (Blackwell, incl. Jetson Thor), INT8 on Ampere (Jetson Orin has no FP8), fewer FLOPs (fewer diffusion steps, token pruning) |
| **Long-context attention** | Prefill time grows ~N² and attention dominates the profile | FlashAttention-2/3 or FlashInfer kernels, sliding-window or sparse attention, hybrid SSM / linear-attention layers, context parallelism |
| **Large KV cache** | Batch size or context length limited by memory | Paged KV cache, GQA/MQA/MLA models, FP8 KV cache, prefix sharing, offload |
| **Prefill/decode interference** | Decode latency spikes when long prompts arrive | Chunked prefill; disaggregated prefill and decode workers at scale |
| **Repeated prompt prefixes** | The same system prompt or documents are prefilled again and again | Prefix caching (vLLM automatic prefix caching, SGLang RadixAttention) |
| **Many small ops** | Kernel launch overhead, GPU idle between kernels | CUDA graphs, torch.compile / TensorRT fusion |
| **MoE routing** | All-to-all traffic, uneven expert load, all experts' weights resident | Expert parallelism, grouped GEMMs, load balancing (see [moe/](./moe/)) |

---

## Inference Optimization Checklist

When deploying a model, work through this checklist:

```
□ Profile baseline (torch.profiler, Nsight Systems; Nsight Compute for single kernels)
□ Identify top-5 slowest operations
□ Check arithmetic intensity (roofline: compute vs memory bound)
□ Apply framework optimizations:
  □ torch.compile / TensorRT / TensorRT-LLM / ONNX Runtime
  □ FlashAttention (FA2; FA3 on Hopper) or FlashInfer kernels
  □ Fused kernels (norm + residual, activations, QKV projection)
  □ CUDA graphs for small-batch decode
□ Quantization:
  □ Weight-only INT4/INT8 (AWQ, GPTQ) for memory-bound decode
  □ FP8 (Ada/Hopper/Blackwell) or FP4 (NVFP4/MXFP4, Blackwell) for compute-bound prefill
  □ INT8 on Ampere-class edge GPUs (Jetson Orin)
  □ Validate on task metrics, not only perplexity
□ Batching & scheduling (LLMs):
  □ Continuous (in-flight) batching
  □ Chunked prefill
  □ Disaggregated prefill/decode for large deployments
□ KV cache:
  □ Paged KV cache; prefix caching for shared prompts
  □ GQA/MLA-friendly models; FP8 KV cache
□ Decode speed:
  □ Speculative decoding (draft model, EAGLE-style heads, multi-token prediction)
□ Memory:
  □ Activation checkpointing trade-offs (training)
  □ Tensor / pipeline / expert parallelism, or offload, when weights do not fit
□ Hardware-specific:
  □ Tensor Core alignment (multiples of 8/16)
  □ Memory coalescing
  □ Jetson: fix the power mode before benchmarking; budget unified memory
```

Sources for the newer items: [FlashAttention-3](https://arxiv.org/abs/2407.08608), [FlashInfer](https://arxiv.org/abs/2501.01005), [FP8 formats](https://arxiv.org/abs/2209.05433), [Microscaling (MX) formats](https://arxiv.org/abs/2310.10537), [NVIDIA on 4-bit NVFP4](https://research.nvidia.com/labs/eai/blogs/pushing-intelligence-to-4-bit/), [Jetson Thor (Blackwell, FP4)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor/), [PagedAttention](https://arxiv.org/abs/2309.06180), [SGLang / RadixAttention](https://arxiv.org/abs/2312.07104), [vLLM automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html), [Sarathi-Serve (chunked prefill)](https://arxiv.org/abs/2403.02310), [DistServe](https://arxiv.org/abs/2401.09670) and [Splitwise](https://arxiv.org/abs/2311.18677) (disaggregated serving), [speculative decoding](https://arxiv.org/abs/2211.17192), [EAGLE-3](https://arxiv.org/abs/2503.01840), [DeepSeek-V3 (multi-token prediction, MoE)](https://arxiv.org/abs/2412.19437).

---

## Structure

```
index.html                  # Learning path: order, prerequisites, study method, cadence
model-name/                 # one folder per main-track module
├── README.md               # Quick overview, key concepts, bottlenecks
├── model_guide.html        # Interactive visual guide (open in browser)
├── model.py                # Reference implementation with annotations
└── requirements.txt
build-llm/                  # hands-on: build_llm_guide.html, build_llm.py
build-agent/                # hands-on: build_agent_guide.html, agent.py, mock_llm.py, evals.py, sample_notes/
labs/                       # build-it-yourself: lab01..lab10 skeletons, check.py, solutions/
efficient-inference/        # efficiency cookbook: efficiency_guide.html, efficient_inference.py
transformer/transformer_step_by_step.html  # gentle primer (start here)
```

Each guide opens with a "before you start" box and ends with key takeaways, a self-check quiz and hands-on exercises. Each guide also has a "🎬 Learn it visually" box of checked videos and visual articles, maths cards (intuition → formula → why → tiny worked example → code), and a link to its matching lab.

---

## Resources

### Profiling & Analysis
- [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)
- [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute)

### Kernels, Compilers & Runtimes
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
- [CUTLASS](https://github.com/NVIDIA/cutlass)
- [torch.compile](https://docs.pytorch.org/docs/stable/torch.compiler.html), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html), [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) (Jetson / DRIVE)
- [NVIDIA Apex](https://github.com/NVIDIA/apex)

### LLM Serving
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) and [llm-d](https://github.com/llm-d/llm-d) (distributed, disaggregated serving)

### Quantization & Low Precision
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer) (formerly TensorRT Model Optimizer: PTQ/QAT to FP8, NVFP4, INT4)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) (FP8/FP4 layers)

### Edge
- [Jetson AI Lab](https://www.jetson-ai-lab.com/)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer
- [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-3](https://arxiv.org/abs/2407.08608) — memory-efficient attention
- [LLM.int8()](https://arxiv.org/abs/2208.07339), [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978) — quantization
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — KV-cache memory management
- [Fast Inference via Speculative Decoding](https://arxiv.org/abs/2211.17192) — draft and verify
- [DistServe](https://arxiv.org/abs/2401.09670) — disaggregated prefill and decode
- [Mamba](https://arxiv.org/abs/2312.00752) — state space models
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) — MoE + MLA + multi-token prediction

---

## What to Study Next

Model families worth studying beyond this repo, as of Oct 2026, each with one or two papers to start from and why it matters for inference.

| Family | Start with | Why it matters for inference |
|--------|------------|------------------------------|
| **Reasoning models & RL post-training** (RL with verifiable rewards) | [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300), [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | Rewarding checkable answers (math, unit tests) is how current reasoning models are post-trained. The cost moves to long generated chains (decode-bound, KV-heavy), and RL training runs an inference engine inside the loop to generate rollouts. |
| **Hybrid linear-attention LLMs** | [Gated Delta Networks](https://arxiv.org/abs/2412.06464), [Kimi Linear](https://arxiv.org/abs/2510.26692) | Most layers keep a fixed-size recurrent state (like [Mamba](./mamba/)) and a few keep full attention for retrieval; [Qwen3-Next](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) uses a 3:1 Gated DeltaNet-to-attention pattern. Smaller KV cache, but serving engines need kernels and cache management for both layer types. |
| **Diffusion language models** | [LLaDA](https://arxiv.org/abs/2502.09992), [Mercury](https://arxiv.org/abs/2506.17298) | They unmask many tokens per forward pass instead of one, replacing token-by-token decode with a fixed number of bidirectional denoising passes over a block; the steps-vs-quality trade-off and caching are open problems. Speed claims are still mostly vendor-reported. |
| **Interactive / action-conditioned world models** | [Genie](https://arxiv.org/abs/2402.15391), [Dreamer 4](https://arxiv.org/abs/2509.24527) | Video world models are becoming simulators for training and evaluating robot and driving policies. Streaming generation mixes LLM-style decode (rolling KV cache) with few-step diffusion. |
| **Video diffusion transformers** | [Wan](https://arxiv.org/abs/2503.20314), [HunyuanVideo](https://arxiv.org/abs/2412.03603) | A few seconds of video is tens of thousands of tokens, so attention dominates; sparse attention, step caching and step distillation are the levers. The backbone of most generative world models. |
| **JEPA-style latent predictive models** | [V-JEPA 2](https://arxiv.org/abs/2506.09985), [LeJEPA](https://arxiv.org/abs/2511.08544) | Predict in feature space instead of pixels: no decoder and no denoising loop, so planning becomes large batched predictor passes (compute-bound GEMMs). |
| **Feed-forward 3D reconstruction** | [VGGT](https://arxiv.org/abs/2503.11651), [Depth Anything 3](https://arxiv.org/abs/2511.10647) | One transformer pass predicts cameras, depth and point maps from many images; global attention across frames makes multi-view inference an attention and memory problem, relevant to [BEV](./bev/) and robotics. |
| **Self-supervised vision backbones** | [DINOv3](https://arxiv.org/abs/2508.10104) | Frozen dense features reused by 3D, planning and VLA models; high-resolution token counts and low-precision numerics decide whether they run well on edge GPUs. |

---

## License

MIT — see [LICENSE](./LICENSE).
