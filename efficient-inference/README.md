# Memory & inference efficiency cookbook (efficiency track)

A cross-cutting module: the techniques that save memory and make inference (and your own training) faster, each with one picture, one formula and a worked number, its trade-offs, Jetson-vs-datacenter advice, and a CPU lab check. Model-specific detail stays in the module guides it links to: the [Transformer](../transformer/transformer_guide.html) (KV cache, FlashAttention), [MoE](../moe/moe_guide.html) (MLA, expert parallelism) and [VLM](../vlm/vlm_guide.html) (token reduction, measured Orin Nano numbers) guides.

## Contents

| File | Description |
|------|-------------|
| [efficiency_guide.html](./efficiency_guide.html) | Interactive guide (open in a browser): "Will it fit?" memory and decode-speed calculator, interactive roofline, 13 technique cards, decision table, self-check and exercises |
| [efficient_inference.py](./efficient_inference.py) | The lab: 12 sections, every claim asserted; torch + stdlib, CPU only, about 20 s (37 s on one thread) |

## Quick start

```bash
xdg-open efficiency_guide.html          # macOS: open efficiency_guide.html
pip install -r requirements.txt
python efficient_inference.py           # all sections, ends with a summary
python efficient_inference.py quant spec
python efficient_inference.py --list
python efficient_inference.py fusion --compile   # also try torch.compile (slow first call)
```

## The two formulas to remember

```
memory  ≈ P·b_w  +  2·L·n_kv·d_h·b_kv·T·B  +  other            (weights + KV cache + runtime)
tokens/s ≤ B · bandwidth / (P·b_w + KV bytes)                  (small-batch decode is memory-bound)
```

Example: Qwen3-8B with INT4 weights (4.25 bits incl. group scales) on a Jetson Orin Nano 8GB in Super mode: 4.05 GiB of weights + 1.125 GiB of BF16 KV cache at 8K context + 1 GiB other = 6.18 GiB (77%); decode ceiling 102 GB/s ÷ 5.56 GB = 18.3 tokens/s.

## Guide map and lab sections

| Card | Technique | Lab section | What the lab asserts |
|------|-----------|-------------|----------------------|
| a | KV cache, and its cost | `kvcache` | 160 greedy tokens identical with and without the cache; 17,840 vs 191 token-forwards (93×) |
| b | GQA / MQA / MLA | `kvsize` | formula = bytes of the real cache tensors for n_kv = 8, 4, 2, 1; MLA-style absorbed attention = explicit K/V; Llama-3.1-8B 128 KiB/token; DeepSeek-V3 MLA 68.6 KiB/token |
| c | Quantization (scale/zero-point, granularity, FP8/FP4, SmoothQuant) | `quant` | hand example q = [0, 5, 7, 15]; outlier channel: INT8 per-tensor output error 0.117 vs per-channel 0.0086; INT4 per-row 0.169 vs groups of 64 0.071 vs FP4 blocks of 16 0.058; W8A8 error 0.127 → 0.022 with SmoothQuant |
| d | FlashAttention (tiling, online softmax) | `tiled` | tiled online-softmax attention equals the reference for tiles 16/64/512 |
| e | PagedAttention, continuous batching | `paged` | 12 mixed-length requests: 83.9% of a max-length reservation wasted vs 1.8% with 16-token blocks; block tables return every sequence exactly |
| f | Prefix caching | `prefix` | 128 of 230 prompt tokens reused; logits on the cached prefix equal a full forward pass |
| g | Speculative decoding, MTP | `spec` | output identical to the target's greedy decoding; acceptance 70.7%, 2.70 tokens per target pass (formula 2.81) |
| h | Fusion, CUDA graphs, torch.compile | `fusion` | 9-kernel GELU equals the fused one; 352 MiB vs 32 MiB of traffic |
| i | 2:4 sparsity, pruning, distillation | `sparse` | 2:4 pattern holds; 56.25% of FP16 storage; distillation raises teacher agreement 35.3% → 73.9% |
| j | Training memory: mixed precision, checkpointing, accumulation, LoRA/QLoRA, ZeRO/FSDP | `ckpt`, `train` | saved activations 9.00 → 1.00 MiB with bit-identical gradients; Adam 16 B/param; ZeRO 120 / 31.4 / 16.6 / 1.9 GB; LoRA r=16 on Llama-3.1-8B = 41,943,040 params |
| k | Jetson deployment: PyTorch → ONNX → ModelOpt Q/DQ → TensorRT; TensorRT Edge-LLM; power modes | `bench` | median/p90 timing after warm-up, plus the Jetson checklist |

Numbers are deterministic for a given torch version (identical at 1 and 4 threads in our runs); wall-clock speedups vary by machine.

## Devices used in the calculator (sources in the guide)

| Device | Memory | Bandwidth | Dense FP16 TFLOPS |
|--------|--------|-----------|-------------------|
| Jetson Orin Nano 8GB, Super mode | 8 GB LPDDR5 (shared) | 102 GB/s | 17 |
| Jetson AGX Orin 64GB | 64 GB LPDDR5 (shared) | 204.8 GB/s | ≈ 42.5 (from 170 sparse INT8 TOPS) |
| Jetson AGX Thor T5000 | 128 GB LPDDR5X (shared) | 273 GB/s | ≈ 259 (derived from 2,070 sparse FP4 TFLOPS) |
| GeForce RTX 4090 | 24 GB GDDR6X | 1,008 GB/s | 165.2 |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | 989.5 |

## Tool status (as of Oct 2026)

- **NVIDIA Model Optimizer** (ModelOpt, formerly TensorRT Model Optimizer): ONNX PTQ inserts explicit Q/DQ nodes for TensorRT (int8, fp8, int4).
- **TensorRT**: implicit INT8 calibration is deprecated in 10.x and removed in 11.x together with weak-typing flags such as `--fp16`; JetPack 7.2 ships TensorRT 10.16, and TensorRT 11.4 does not support JetPack yet.
- **TensorRT Edge-LLM**: NVIDIA's C++ LLM/VLM runtime for Jetson Orin and Thor, DRIVE Thor and DGX Spark (Orin: FP16/INT8/INT4; Thor adds FP8/NVFP4). TensorRT-LLM's Jetson support stopped at the v0.12.0-jetson branch (AGX Orin, JetPack 6.1).

## References

The guide's References section lists every paper (arXiv), datasheet and doc page used; the "🎬 Learn it visually" box and each card's ▶ line list verified videos and posts.
