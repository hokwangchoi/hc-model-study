# VLA — Vision-Language-Action

Based on [**Pi0 (Physical Intelligence, 2024)**](https://arxiv.org/abs/2410.24164) — the current state-of-the-art open-weights VLA. Uses a VLM backbone (vision + language + proprioception) plus a parallel **action expert** transformer that cross-attends to the VLM's output and generates continuous action chunks via **flow matching**.

Related: [RT-2 (Brohan et al., 2023)](https://arxiv.org/abs/2307.15818) — the original "VLMs can output actions" demonstration, but via tokenized action vocab. [OpenVLA (Kim et al., 2024)](https://arxiv.org/abs/2406.09246) — LLaMA2-based with discrete action tokens. [Octo (Octo Team, 2024)](https://arxiv.org/abs/2405.12213) — smaller diffusion-based alternative.

## Contents

| File | Description |
|------|-------------|
| [vla_guide.html](./vla_guide.html) | Interactive visual guide (open in browser) |
| [vla.py](./vla.py) | Reference implementation |

## Quick Start

```bash
open vla_guide.html       # macOS
xdg-open vla_guide.html   # Linux
python vla.py
```

## The Core Idea

A VLA takes what a robot sees (cameras) + what it's been asked to do (language) + where it currently is (proprioception), and outputs what it should do next (joint commands). The "Action" in VLA is continuous — joint angles, end-effector velocities, gripper commands — not a discrete token.

```
Cameras ──┐
Language ──┼─→ VLM backbone ─→ context tokens
Proprio  ──┘                        │
                                    │ (cross-attention)
                                    ▼
                Noisy action   ─→  Action expert  ─→  velocity v̂
                chunk + flow      (smaller parallel         │
                time t             transformer)             ▼
                                                        Flow matching
                                                        training OR
                                                        Euler integration
                                                        to clean action chunk
```

The output is a **chunk** of H=50 future actions (1 second at 50Hz), not a single action. The robot executes the chunk open-loop, then the VLA re-predicts. This decouples prediction rate (slow, ~1Hz) from execution rate (fast, 50Hz).

## What's New vs Previous Modules

| Module | What we learned | What VLA adds |
|--------|----------------|---------------|
| Transformer | Self-attention, FFN, causal masks | (reused inside both backbones) |
| ViT | Patchify images → token sequence | (reused as vision encoder) |
| VLM | Fuse vision + language with cross-attention | VLA adds proprioception as a third modality |
| Diffusion | DDPM: curved noise schedule → clean image | **Flow matching** is the simpler cousin — straight paths, velocity prediction |
| Mamba | State-space models, linear-time sequences | — |
| BEV | 3D geometry embedded in attention | — |

The two-transformer trick is new: **a separate, smaller action expert** runs parallel to the VLM, cross-attending to its output. This lets you freeze the expensive VLM during robot fine-tuning and only train the action expert on robot data.

## Flow Matching vs Diffusion — the TL;DR

In the diffusion module we saw DDPM's noise schedule α̅_t that defines a curved path from data to noise. Flow matching simplifies this dramatically:

| | DDPM | Flow matching (Pi0) |
|-|------|---------------------|
| Path from noise to data | Curved (follows SDE) | Straight line: x_t = (1-t)·ε + t·a |
| Prediction target | ε (noise) | v = a - ε (velocity — constant along the path) |
| Training loss | MSE(ε̂, ε) | MSE(v̂, v) |
| Sampling | 50–1000 steps, DDIM magic | ~10 Euler steps |
| Code | Needs noise schedule, variance logic | Linear interp + velocity MSE |

Pi0 uses flow matching. The architecture is identical to the DiT we built — only the math around training and sampling changes.

## Action Chunks — Why Predict 50 Steps at Once?

Classical policies predict one action per observation, running at full control rate (say 50Hz). That requires the model to evaluate 50 times per second. For a 3B VLA, that's 150 TFLOPs/sec just for policy inference — infeasible on robot hardware.

Pi0 instead predicts a **chunk** of H=50 future actions every N frames:
- Prediction at 1–5Hz (amortized over the chunk)
- Execution at 50Hz (cheap — just feed the next chunk element to the controller)
- Re-predict periodically (not every frame)

Downside: the policy can't react mid-chunk to surprises. Research directions (stream-VLA, receding-horizon re-prediction) address this.

## Key Inference Bottlenecks

| Stage | Cost per prediction | Per timestep |
|-------|--------------------:|-------------:|
| Vision encoder (SigLIP) | 15–50 ms | 0.3–1 ms (amortized over H=50) |
| Language encoder (Gemma) | 5–20 ms | 0.1–0.4 ms |
| VLM cross-modal blocks | 10–30 ms | 0.2–0.6 ms |
| Action expert × 10 flow steps | 30–80 ms | 0.6–1.6 ms |
| **Total per chunk** | **60–180 ms** | **1.2–3.6 ms/step** |

The VLM backbone dominates but runs ONCE per chunk. The action expert runs 10 times (Euler steps) but is small, so it's comparable in total cost. INT8 VLM + FP16 action expert is the sweet spot.

## Critical Optimizations

### 1. Cache VLM output across Euler steps

The VLM processes the scene once per chunk. All 10 action-expert Euler steps reuse the same VLM tokens:
```python
vlm_tokens = self.encode_vlm(images, text, proprio)   # compute once
for step in range(10):
    v = self.action_expert(x, t, vlm_tokens)          # cheap per step
    x = x + v * dt
```
Without this caching, inference is 10× slower. Pi0 reference code does it; any production deployment does it.

### 2. Freeze the VLM during robot fine-tuning

The VLM backbone is pre-trained on web data (PaliGemma). During robot fine-tuning, it's often frozen, with only the action expert + small adapter layers trained. This:
- Keeps generalist VLM capabilities (language understanding, object recognition)
- Dramatically cuts training memory (no VLM grads)
- Lets you run VLM inference in INT8 (no gradients needed)

### 3. INT8 VLM, FP16 action expert

Vision + language operations quantize well to INT8 (standard LLM/ViT territory). The action expert is smaller and its flow-matching velocity predictions benefit from FP16 precision (small errors compound over Euler steps). 2× inference speedup vs full FP16 with <1% success-rate drop in the Pi0 evaluations.

### 4. Reduce Euler steps

Pi0 uses 10 steps. At deployment, 5 steps is often enough (small quality drop, 2× speedup). Research direction: distillation — train a student model to predict the full chunk in 1 step (Pi0.5 and related papers).

### 5. Chunked re-prediction strategies

- Fixed rate: re-predict every 25 timesteps (chunk overlap)
- Triggered: re-predict when action error exceeds threshold
- Streaming: predict chunks autoregressively (early exit when uncertain)

## Architecture Summary

```
                   Vision Encoder (SigLIP / ViT)
                              │
                              ▼
                        Vision tokens  ──┐
                                         │
                 Language Embedding (Gemma)
                              │
                              ▼          │
                       Language tokens ──┼──→ Concatenate ──→ VLM blocks (depth=6)
                                         │
                   Proprio MLP            │                       │
                        │                │                       ▼
                        ▼                │              Context tokens (frozen during fine-tune)
                   Proprio token ────────┘                       │
                                                                 │ cross-attn (read-only)
                                                                 │
     Noisy action chunk x_t  ──── Action Expert Block × 6 ────┐  │
     [B, H=50, action_dim]           (self-attn over H)       │◄─┘
              +                     (cross-attn to VLM)       │
     Flow time embedding             (FFN)                    │
                                                              ▼
                                                    Predicted velocity v̂
                                                    [B, H=50, action_dim]
                                                              │
                                              ┌───────────────┴──────────────┐
                                              ▼                              ▼
                                       TRAINING:                      INFERENCE:
                                       MSE(v̂, a - ε)                 Euler: x ← x + v̂·dt
                                                                      × 10 steps
                                                                              │
                                                                              ▼
                                                                    Clean action chunk
```

## Parameters & FLOPs

For a toy VLA-B-like config (d=768, vlm depth 6, action expert depth 6, H=50, action_dim=14):

| Component | Params (toy) | FLOPs/chunk |
|-----------|-------------:|------------:|
| Vision encoder | ~25M | ~6 GFLOPs (× 2 cameras) |
| Language embedding | ~25M | ~1 GFLOP |
| Proprio encoder | ~1M | ~0.01 GFLOP |
| VLM backbone | ~40M | ~15 GFLOPs |
| Action expert (× 10 Euler steps) | ~20M | ~2 GFLOPs (× 10) |
| **Total** | **~110M** | **~40 GFLOPs** |

Swap toy vision for SigLIP-400M and toy LM for Gemma-2B and you're at ~3B parameters matching the real Pi0-base.

## Differences from Previous Models

| Aspect | VLA | VLM (chat) | Diffusion (images) | BEV |
|--------|-----|-----------|-------------------|-----|
| Modalities in | Vision + Language + Proprio | Vision + Language | Noise + text | Multi-view images |
| Output | Continuous action chunk | Text tokens | Image pixels | 2D feature map |
| Generation | Flow matching (10 steps) | Autoregressive | Diffusion (50+ steps) | Single forward pass |
| Latency target | 60–180 ms / chunk (1Hz chunks) | Interactive | Offline | 50 ms (10Hz) |
| Quantization | INT8 VLM + FP16 expert | W4A16 | FP16 | INT8 mixed |
| Training loss | MSE on velocity | Next-token CE | MSE on noise | Task-specific |

## On Edge Hardware — Jetson / Drive / Humanoid Compute

VLAs are the killer app for robotic edge compute. Target hardware:

| Platform | Compute | Realistic VLA size | Rate |
|----------|--------:|-------------------:|-----:|
| Jetson Orin Nano 8GB | 67 TOPS int8 | ~300M VLA | 1–3Hz chunks |
| Jetson Orin AGX | 275 TOPS int8 | Pi0-1B | 3–5Hz chunks |
| Jetson Thor | ~2000 TFLOPS | Pi0-3B | 5–10Hz chunks |
| NVIDIA Drive Thor | ~1000 TFLOPS | Pi0-3B | 5Hz+ |

Humanoid form factors (Figure, 1X, Optimus) are the largest deployment target for VLAs. Jetson Thor is the reference hardware for open-source humanoid stacks (Berkeley HumanPlus, Stanford TeleVision). NVIDIA's Isaac GR00T project is building on exactly this premise.

## Connection to the Roadmap Ahead

- **Robot foundation models** (next module): multi-embodiment VLAs. Same Pi0 architecture, trained on cross-robot data (arms, quadrupeds, humanoids) with embodiment tokens.
- **World models**: predict the next observation given current action. The flip side of VLA — learn p(o_{t+1} | o_t, a_t) instead of p(a_t | o_t). Often shares the same VLM backbone.
- **Reinforcement learning on top**: the flow-matched action distribution can be fine-tuned with RL (REINFORCE on the velocity field, or PPO with flow as the behavior policy).

## References

- [Pi0 (Physical Intelligence, 2024)](https://arxiv.org/abs/2410.24164) — the reference VLA
- [Flow Matching (Lipman et al., 2022)](https://arxiv.org/abs/2210.02747) — the math
- [RT-2 (Brohan et al., 2023)](https://arxiv.org/abs/2307.15818) — original VLA via tokenized actions
- [OpenVLA (Kim et al., 2024)](https://arxiv.org/abs/2406.09246) — open-weights LLaMA2 VLA
- [Octo (Octo Team, 2024)](https://arxiv.org/abs/2405.12213) — smaller diffusion-based VLA
- [Action Chunking Transformer (ALOHA, 2023)](https://arxiv.org/abs/2304.13705) — chunked action prediction
- [PaliGemma (Google, 2024)](https://arxiv.org/abs/2407.07726) — the VLM backbone Pi0 uses
- [Open X-Embodiment (Padalkar et al., 2023)](https://arxiv.org/abs/2310.08864) — the cross-robot dataset
