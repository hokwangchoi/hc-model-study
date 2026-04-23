# Robot Foundation Model — Multi-Embodiment, Dual-System

Based on [**NVIDIA GR00T N1 (2025)**](https://developer.nvidia.com/isaac/gr00t) — a dual-system foundation model for multi-embodiment robotics. The architecture is the natural closure of this roadmap: it reuses the [VLM](../vlm/) backbone from VLM, the [DiT action head](../diffusion/) from diffusion, the [flow-matching training](../vla/) from VLA, and adds two distinctive properties: **explicit dual-rate execution** (System 1 / System 2) and **heterogeneous I/O routing** for cross-embodiment support.

Related: [RT-X / Open X-Embodiment (2023)](https://arxiv.org/abs/2310.08864) — the foundational cross-embodiment dataset. [RoboCat (DeepMind, 2023)](https://arxiv.org/abs/2306.11706) — multi-task, multi-embodiment agent. [Octo (Berkeley, 2024)](https://arxiv.org/abs/2405.12213) — generalist policy with diffusion head. [HPT (MIT, 2024)](https://arxiv.org/abs/2409.20537) — heterogeneous pre-trained transformers.

## Contents

| File | Description |
|------|-------------|
| [robot_fm_guide.html](./robot_fm_guide.html) | Interactive visual guide (open in browser) |
| [robot_fm.py](./robot_fm.py) | Reference implementation, 4 toy embodiments |

## Quick Start

```bash
open robot_fm_guide.html       # macOS
xdg-open robot_fm_guide.html   # Linux
python robot_fm.py
```

The demo trains one loss + generates one action chunk for each of 4 embodiments (Franka arm, xArm bimanual, Unitree H1 humanoid, Unitree Go2 quadruped) using a single shared model with heterogeneous I/O routing.

## The Two Ideas That Define It

### Idea 1: Dual-system execution (System 1 / System 2)

A human driving a car isn't constantly re-deciding strategy. "I'm going home" is a slow judgment ("System 2"); individual steering micro-corrections are fast reflexes ("System 1"). GR00T applies this split to robot control:

```
System 2 — the VLM, slow:        runs ~10 Hz  →  scene + instruction context
                                                 (updated when things change)

System 1 — the action head, fast: runs ~120 Hz →  motor commands
                                                   (cross-attends to System 2)
```

For a 1-second action chunk at 50 Hz control rate, System 2 runs once; System 1 integrates its velocity field through 10 Euler steps (same as our [VLA module](../vla/)). Separating the two lets each system be sized for its job: System 2 is massive (scene understanding) and infrequent; System 1 is smaller and fast.

### Idea 2: Heterogeneous I/O for multi-embodiment

Different robots have fundamentally different action spaces:
- Franka 7-DOF arm → 7 action dims
- Bimanual xArm → 14 action dims
- Unitree H1 humanoid → 22 action dims
- Unitree Go2 quadruped → 12 action dims

GR00T's trick: **one shared core, many per-embodiment I/O heads**.

```
          ┌─ franka_proprio_enc ──→ d_model ──┐
          ├─ xarm_proprio_enc ────→ d_model ──┤
proprio ──┤                                    ├──→ VLM ──→ System 1 ──→
          ├─ h1_proprio_enc ──────→ d_model ──┤
          └─ go2_proprio_enc ─────→ d_model ──┘

                                                        ┌─ franka_action_head ─ 7 dims
                                                        ├─ xarm_action_head ── 14 dims
         System 1 d_action ────────────────────────────┤
                                                        ├─ h1_action_head ──── 22 dims
                                                        └─ go2_action_head ─── 12 dims
```

At training time, each batch is labeled with an embodiment name. The forward pass routes through the matching proprio encoder and action head. **All four embodiments share the VLM backbone, the System 1 transformer blocks, and almost all parameters.** Only the tiny I/O projections are per-embodiment.

Why this works: most of what a robot needs to know is embodiment-agnostic (what objects are, what tasks mean, physics). Only the final step of "which specific motors do I command?" varies. Sharing 95%+ of the model across embodiments is what makes cross-embodiment transfer possible.

## What's New vs Previous Modules

| Module | Contribution reused here |
|--------|---------------------------|
| Transformer | Self-attention, FFN, pre-norm (System 2 + System 1 blocks) |
| ViT | Patchify images → vision tokens |
| VLM | Fuse vision + language + proprio into context tokens |
| Diffusion (DiT) | **AdaLN-Zero conditioning** — used by System 1 to inject flow time + embodiment into every block |
| Mamba | — |
| BEV | — (could replace vision encoder for multi-camera setups) |
| VLA | **Flow matching**, **action chunks**, **action expert** pattern |

**New in this module:**
- Explicit rate asymmetry between System 2 and System 1 (exploited by caching)
- Embodiment tokens as learnable inputs
- Heterogeneous proprio encoders and action heads (routed by name)
- Cross-embodiment training recipe

## The Data Landscape

Foundation models need foundation-scale data. Robot data is harder to collect than web text, but the landscape has grown:

| Source | Size | Embodiments | Notes |
|--------|-----:|-------------|-------|
| Open X-Embodiment (2023) | 1M+ trajectories | 22 robots | Union of 60+ existing datasets |
| DROID (2024) | 76k trajectories | Franka | Human teleop, diverse scenes |
| AgiBot World (2025) | 1M+ trajectories | 8 humanoid setups | Dedicated data collection |
| RH20T (2024) | 13k trajectories | 7 robots | Human teleop with RGB-D |
| NVIDIA Isaac simulated data | Synthetic | Any | Domain randomization |

Cross-embodiment training mixes batches from many sources. The embodiment token tells the model which calibration and action space apply.

## Inference Characteristics

For a real GR00T-N1-like model (Eagle-2 VLM + 300M DiT action head):

| Operation | Rate | Compute per call |
|-----------|-----:|-----------------:|
| System 2 (VLM) | 10 Hz | ~150 GFLOPs |
| System 1 (action head) | 120 Hz | ~5 GFLOPs |
| Full chunk: 1× S2 + 10× S1 | 1 Hz | ~200 GFLOPs |

Compared with a naive "re-run everything every step" policy (which would be ~1500 GFLOPs/chunk at 10 Hz), the dual-system split is ~7× cheaper at matched throughput.

## Critical Optimizations

### 1. Cache System 2 across System 1 steps

Foundational. Without it, each Euler step re-runs the expensive VLM. Same pattern as VLA.

### 2. Embodiment-aware batching

Training on mixed-embodiment data naïvely (random samples across embodiments) forces the model to switch between action heads every sample, causing kernel-launch overhead. Batch-by-embodiment and rotate between them:
```python
# One batch: all franka_arm
# Next batch: all xarm_bimanual
# Same proprio_enc + action_head reused across the whole batch
```
Significant speedup on GPU.

### 3. Freeze System 2 during fine-tuning

The VLM backbone can be frozen after pretraining, and only System 1 (plus per-embodiment I/O heads) trained on new robot data. This:
- Cuts training memory ~5×
- Preserves generalist VLM capabilities
- Lets you quickly adapt to new embodiments by adding I/O heads only

### 4. INT8 System 2, FP16 System 1

Same pattern as VLA. Vision + language quantize well; action-head flow matching needs FP16 precision.

### 5. Distill System 1 steps

10 Euler steps → 1–3 with distillation. Current research direction for latency-critical humanoid control.

### 6. Share attention KV across System 1 steps

Within one chunk prediction, the VLM context is fixed, so the cross-attention K and V don't change across Euler steps. Compute them once, reuse 10×. Saves ~30% of System 1 cost per chunk.

## Architecture Summary

```
                          Images      Text        Proprio (per-embodiment dim)
                            │           │           │
                            ▼           ▼           ▼
                      Vision enc    Language   Heterogeneous
                        (ViT)        embed    proprio encoder
                            │           │         │
                            └───────────┼─────────┤
                                         │         ├── Embodiment token (learnable)
                                         ▼         │
                                  Concatenate ─────┘
                                         │
                                         ▼
                         System 2: VLM Backbone (depth 4–24)
                                         │
                                         ▼
                                VLM context tokens  (cached per chunk)
                                         │   ↑
   ╔════════════════════════════════════════┷═════════════════════════════════════╗
   ║                                        │                                      ║
   ║   Noisy action x_t  ←───┐              │  (cross-attn — queried by S1 blocks) ║
   ║   [B, H, a_dim]          │             │                                      ║
   ║          │               │             │                                      ║
   ║          ▼               │             │                                      ║
   ║   Hetero action_in       │             │                                      ║
   ║   (per-embodiment)       │             │                                      ║
   ║          │               │             │                                      ║
   ║          ▼               │             │                                      ║
   ║   System 1 Block × L  ←──┴─── cond = time_emb + embodiment_summary            ║
   ║   (DiT block: AdaLN-Zero self-attn + cross-attn to VLM + FFN)                 ║
   ║          │                                                                    ║
   ║          ▼                                                                    ║
   ║   Hetero action_out                                                           ║
   ║   (per-embodiment) → predicted velocity v̂                                     ║
   ║                                                                                ║
   ║   Runs 10× per chunk (Euler integration)                                      ║
   ╚════════════════════════════════════════════════════════════════════════════════╝
                                         │
                                         ▼
                                  Clean action chunk
                                  [B, H, action_dim for this embodiment]
                                         │
                                         ▼
                                  Robot controller
                                  (executes at 50–120 Hz)
```

## Parameters & FLOPs (toy RFM-B)

| Component | Params | FLOPs per call | Rate |
|-----------|------:|---------------:|-----:|
| Vision encoder | 7.4M | 2 GFLOPs | 10 Hz |
| Language embedding | 12.3M | 0.05 GFLOPs | 10 Hz |
| VLM backbone | 7.1M | 3 GFLOPs | 10 Hz |
| Heterogeneous proprio (× 4) | 0.6M | ~0 | 10 Hz |
| Embodiment embed | 1.5K | ~0 | 10 Hz |
| DiT action head (× 10 Euler steps) | 6.5M | 1.5 GFLOPs × 10 | 120 Hz (bursty) |
| **Total per chunk** | **~34M** | **~20 GFLOPs** | — |

Real GR00T N1 is ~60× larger in every dimension.

## Differences from Previous Models

| Aspect | Robot FM | VLA (Pi0) | BEV |
|--------|----------|-----------|-----|
| Embodiments supported | Many (multi-embodiment) | Typically one per model | N/A (perception only) |
| System split | S1 / S2 explicit | Parallel VLM + action expert | Single pass |
| Action head | DiT with AdaLN | Cross-attention transformer | N/A |
| Training data | Cross-embodiment mix | Single embodiment | Multi-camera single-rig |
| Hardware target | Jetson Thor, humanoid onboard | Jetson Orin AGX, Thor | Drive |
| Scale (production) | ~2B params (GR00T N1) | ~3B (Pi0-base) | ~60M (BEVFormer-base) |

## On Edge Hardware — Humanoid / Jetson Thor

This is the target platform for robot foundation models. Jetson Thor (~2 PFLOPS FP8) is the reference compute for onboard humanoid inference.

| Platform | Compute | Realistic model | Chunk rate |
|----------|--------:|----------------:|-----------:|
| Jetson Orin Nano | 67 TOPS | GR00T-lite (~200M) | 0.5–1 Hz |
| Jetson Orin AGX | 275 TOPS | GR00T-S (~500M) | 2–3 Hz |
| Jetson Thor | ~2000 TFLOPS | GR00T-B (~2B) | 5–10 Hz |
| Future humanoid compute | ~5–10 PFLOPS | 10B+ | 20+ Hz |

The rate target for humanoid whole-body control is typically 20–50 Hz for arm motion and 100–500 Hz for balance/gait reflexes — hence the System 1 / System 2 split, letting fast balance live in System 1 while scene reasoning lives in System 2.

## What's Still Open Research

This module is intentionally an overview. Active questions:

1. **Scaling laws for robotics** — does doubling data/params predictably improve performance? Early results from GR00T and Pi0.5 suggest yes, but with different exponents than LLMs.
2. **Reward-free pretraining** — LLMs pretrain on next-token; robots currently need action labels. Can world modeling or self-supervised video substitute?
3. **Sim-to-real transfer** — simulation data is free but has distribution gap. NVIDIA Isaac, Cosmos, and similar aim to close it.
4. **Real-time RL on top of foundation policies** — fine-tune a pretrained VLA with on-robot RL. PPO on flow-matched policies is an active area.
5. **Cross-morphology transfer** — if I train on arms + quadrupeds, does a humanoid get for free? Early evidence: yes for perception, partially for high-level skills, no for low-level balance.
6. **Streaming chunks** — receding-horizon chunk prediction with continuous rollout instead of chunk-at-a-time.

## References

- [NVIDIA GR00T N1 (2025)](https://developer.nvidia.com/isaac/gr00t) — reference implementation
- [GR00T N1 Technical Report](https://d1qx31qr3h6wln.cloudfront.net/publications/GR00T_1_Whitepaper.pdf) — architecture details
- [Open X-Embodiment (Padalkar et al., 2023)](https://arxiv.org/abs/2310.08864) — the multi-robot dataset
- [RoboCat (Bousmalis et al., 2023)](https://arxiv.org/abs/2306.11706) — earlier multi-embodiment foundation
- [Pi0 (Physical Intelligence, 2024)](https://arxiv.org/abs/2410.24164) — VLA precursor
- [Octo (Octo Team, 2024)](https://arxiv.org/abs/2405.12213) — generalist with diffusion head
- [HPT (Wang et al., 2024)](https://arxiv.org/abs/2409.20537) — heterogeneous pre-trained transformers
- [RDT-1B (Liu et al., 2024)](https://arxiv.org/abs/2410.07864) — robotics diffusion transformer
- [NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) — open source code
- [Eagle-2 VLM (Chen et al., 2024)](https://arxiv.org/abs/2501.14818) — VLM backbone used by GR00T
