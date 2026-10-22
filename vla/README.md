# VLA — Vision-Language-Action

Based on [**π0 (Black et al., Physical Intelligence, 2024)**](https://arxiv.org/abs/2410.24164): a pretrained VLM (PaliGemma 3B) and a smaller **action expert** (~300M) that live in **one transformer with two sets of weights**. Image and text tokens use the VLM weights, robot state and action tokens use the expert weights, and all tokens meet in a shared, block-causal self-attention at every layer. Continuous action chunks are generated with **flow matching**.

Related: [RT-2](https://arxiv.org/abs/2307.15818) (actions as text tokens), [OpenVLA](https://arxiv.org/abs/2406.09246) and [OpenVLA-OFT](https://arxiv.org/abs/2502.19645), [Octo](https://arxiv.org/abs/2405.12213), [SmolVLA](https://arxiv.org/abs/2506.01844).

Module 8 of 9 in the learning path (previous: BEV Perception, next: Robot Foundation Models).

## Contents

| File | Description |
|------|-------------|
| [vla_guide.html](./vla_guide.html) | Interactive guide: architecture, attention mask, 1-D flow-matching step-through, roofline calculator, self-check, exercises |
| [vla.py](./vla.py) | π0-style reference implementation (shared-attention layers, prefix KV cache, flow matching) + the `pi0_cost` roofline model |
| [../labs/lab10_flow_matching.py](../labs/lab10_flow_matching.py) | Build-it-yourself lab: flow-matching loss, training, Euler sampling, action chunks (check with `python labs/check.py 10`) |

## Quick Start

```bash
open vla_guide.html       # macOS
xdg-open vla_guide.html   # Linux
python vla.py             # CPU, a few seconds
```

`python vla.py` prints parameter counts, checks that the cached-prefix velocity equals a full joint-sequence pass (max diff ~1e-6), generates a chunk, runs the 1-D flow-matching experiment (1 Euler step collapses to the mean action), and prints π0 roofline floors for Jetson devices.

## The core idea

```
prefix:  [image tokens (256 per camera) | text tokens]   -> VLM weights            (once per observation)
suffix:  [state token | a_1 ... a_50 (noisy, + flow time)] -> action-expert weights (every Euler step)

every layer:  one self-attention over [prefix ; suffix] with a 3-block causal mask
              [images+text] see only themselves, [state] sees prefix + itself,
              [actions] see everything (bidirectionally)
```

Because the prefix never attends to the suffix, its per-layer keys/values are identical for all 10 Euler steps. Compute them once (the KV cache) and run only the 51 suffix tokens per step. This is **not** "cross-attention to the VLM's final output": the expert reads the VLM's keys/values at every depth. (π0's paper contrasts this with its smaller π0-small baseline, which cross-attends to encoder outputs; GR00T N1, in the next module, uses the cross-attention style, reading one middle VLM layer (LLM layer 12) rather than the final output.)

π0 predicts H = 50 actions. On 50 Hz robots it executes 25 and re-plans every 0.5 s; on 20 Hz robots it executes 16 and re-plans every 0.8 s, open loop.

## π0 at a glance (paper + openpi)

| Part | Details |
|------|---------|
| VLM | PaliGemma: SigLIP-So400m (256 tokens per 224×224 image) + Gemma-2B (width 2048, 18 layers, 8 query heads, 1 KV head, head dim 256) |
| Action expert | width 1024, MLP 4096, same 18 layers / heads / head dim; ~300M parameters (311M in openpi), initialised from scratch |
| Total | 3.3B parameters |
| Generation | 10 Euler steps; training flow times from a Beta(1.5, 1)-based density favouring the noisy end |
| Latency (RTX 4090, 3 cameras) | 14 ms image encoders + 32 ms prefix + 27 ms for 10 action steps = 73 ms |

## Flow matching in one table

| | DDPM | Flow matching (π0) |
|-|------|---------------------|
| Path for one (noise, data) pair | set by the noise schedule ᾱ_t | straight line x_t = (1−t)·ε + t·a |
| Target | ε | v = a − ε |
| Sampling | 1000 steps (DDPM) / ~50 (DDIM) | ~10 Euler steps |

Each training path is straight, but the learned (averaged) field is not. With one Euler step every sample lands on the **mean** action, which for a two-mode task ("go left or go right") is a bad action. `vla.py` measures this exactly: 1 step → 0% of samples near a mode, 2 → ~39%, 4 → ~92%, 10 → ~99%.

**Conventions differ.** This repo (and GR00T's code): t = 0 noise, v = a − ε, integrate t: 0 → 1. openpi: x_t = t·ε + (1−t)·a, u = ε − a, integrate t: 1 → 0 with dt = −0.1. Same ODE with t_openpi = 1 − t and u = −v. The π0 paper's prose mixes the two, so port from the code.

## Roofline view (from `pi0_cost`, FP16 weights, 3 cameras)

| | FLOPs | Bytes read | Intensity |
|-|------:|-----------:|----------:|
| Prefix (SigLIP + Gemma-2B, 816 tokens), once | 3.95 T | 4.78 GB | ~826 FLOP/B |
| One Euler step (51 tokens through the expert + 15 MB KV cache) | 38.3 G | 0.64 GB | ~60 FLOP/B |

| Device (NVIDIA specs; peak FP16 from NVIDIA or derived) | Bandwidth | Floor per chunk | Without KV cache |
|-------|----------:|---------------:|-----------------:|
| Jetson Orin Nano 8GB (Super) | 102 GB/s | ≈ 295 ms | ≈ 2384 ms |
| Jetson AGX Orin 64GB | 204.8 GB/s | ≈ 124 ms | ≈ 960 ms |
| Jetson AGX Thor T5000 | 273 GB/s | ≈ 41 ms | ≈ 198 ms |

The Euler steps are memory-bound everywhere: under 10% of the FLOPs, about a quarter of the AGX Orin floor. Shrink the bytes (quantized expert, few KV heads: Gemma's single KV head makes the cache 15 MB instead of 120 MB) and the step count. These are lower bounds. Measured anchors: π0 on an RTX 4090, 73 ms; GR00T N1.7 on AGX Thor (120 W) 77.6 ms with TensorRT and 35.9 ms with mixed NVFP4, Jetson Orin (30 W) 312.2 ms with TensorRT ([Isaac-GR00T deployment README](https://github.com/NVIDIA/Isaac-GR00T)).

Peak dense FP16: Orin Nano Super 17 TFLOPS from NVIDIA's spec table; the others are derived from NVIDIA's sparse headline numbers: AGX Orin = GPU dense INT8 ÷ 2 (85 → 42.5); Thor = 2,070 sparse FP4 ÷ 2 (dense) ÷ 2 (FP4 → FP8) ÷ 2 (FP8 → FP16) ≈ 259 (our derivation, not an NVIDIA figure). Treat these as rough ceilings.

## Critical optimizations

1. **Prefix KV cache**: exact, because of the block-causal mask (the floor drops from ≈ 960 to ≈ 124 ms on AGX Orin).
2. **Train, freeze, or insulate**: π0 trains the whole model (openpi's default configs update all weights; LoRA "low_mem" variants exist). Freezing the VLM is the cheap option (GR00T N1.5 does it). [Knowledge insulation](https://arxiv.org/abs/2505.23705) trains the VLM on discrete FAST tokens and stops the expert's gradients from reaching it.
3. **Precision**: quantize the VLM (FP8/INT8, FP4 on Thor) first; keep the expert and the action I/O projections in higher precision; validate in closed loop.
4. **Fewer Euler steps**: π0 uses 10, GR00T N1 uses 4; 1–2 steps generally need distillation.
5. **Chunk execution**: open loop with re-planning (π0), temporal ensembling (ACT), or [real-time chunking](https://arxiv.org/abs/2506.07339), which freezes the actions that will execute during the inference delay and inpaints the rest, with no retraining.

## As of Oct 2026

- **π0-FAST** (Jan 2025): [FAST](https://arxiv.org/abs/2501.09747) action tokens (DCT-based compression) for autoregressive VLAs.
- **π0.5** ([Apr 2025](https://arxiv.org/abs/2504.16054)): discrete-token pre-training, flow-matching action expert in post-training, high-level subtask prediction before actions; adaRMSNorm time conditioning in openpi.
- **Knowledge insulation** (May 2025), **real-time chunking** (Jun 2025).
- **π0.6 / π\*0.6** ([Nov 2025](https://www.pi.website/blog/pistar06)): 5B-parameter VLM + action expert; Recap = RL from demonstrations, corrections and autonomous experience via advantage conditioning.
- **π0.7** ([Apr 2026](https://www.pi.website/blog/pi07)): steerable prompts (language, metadata, control-modality labels, visual subgoals). [Pi API beta](https://www.pi.website/blog/api-beta) in Oct 2026.
- Elsewhere: [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) (LIBERO 76.5% → 97.1%, 26× action throughput), [Gemini Robotics](https://arxiv.org/abs/2503.20020) / [1.5](https://arxiv.org/abs/2510.03342) and the [Gemini Robotics 2 family](https://deepmind.google/models/gemini-robotics/), NVIDIA GR00T N1.x (next module).

## Toy model sizes (`python vla.py`)

| | VLA-S | VLA-B |
|-|------:|------:|
| Layers / d_vlm / d_action | 4 / 256 / 128 | 6 / 512 / 256 |
| Total parameters | 13.9 M | 51.1 M |

## References

- [π0](https://arxiv.org/abs/2410.24164), [π0.5](https://arxiv.org/abs/2504.16054), [FAST](https://arxiv.org/abs/2501.09747), [Knowledge Insulation](https://arxiv.org/abs/2505.23705), [Real-Time Chunking](https://arxiv.org/abs/2506.07339)
- [Flow Matching (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747)
- [ACT (Zhao et al., 2023)](https://arxiv.org/abs/2304.13705)
- [PaliGemma (Beyer et al., 2024)](https://arxiv.org/abs/2407.07726)
- [RT-2](https://arxiv.org/abs/2307.15818), [OpenVLA](https://arxiv.org/abs/2406.09246), [OpenVLA-OFT](https://arxiv.org/abs/2502.19645), [Octo](https://arxiv.org/abs/2405.12213), [SmolVLA](https://arxiv.org/abs/2506.01844)
- [Open X-Embodiment (2023)](https://arxiv.org/abs/2310.08864)
- [openpi](https://github.com/Physical-Intelligence/openpi) — π0 / π0.5 reference code
