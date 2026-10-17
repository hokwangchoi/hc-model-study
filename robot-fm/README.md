# Robot Foundation Model — Multi-Embodiment, Dual-System

Based on [**NVIDIA GR00T N1 (2025)**](https://arxiv.org/abs/2503.14734): a dual-system foundation model for multi-embodiment robots. It reuses pieces from earlier modules — a [VLM](../vlm/) as System 2, a [DiT](../diffusion/)-style action head as System 1, [flow matching and action chunks](../vla/) — and adds two distinctive ideas: **dual-rate execution** and **embodiment-specific input/output layers** for cross-embodiment training. Code and checkpoints: [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) (N1.7 as of Oct 2026).

Related: [Open X-Embodiment](https://arxiv.org/abs/2310.08864), [RoboCat](https://arxiv.org/abs/2306.11706), [Octo](https://arxiv.org/abs/2405.12213), [HPT](https://arxiv.org/abs/2409.20537), [RDT-1B](https://arxiv.org/abs/2410.07864).

Module 9 of 9 in the learning path (previous: Vision-Language-Action).

## Contents

| File | Description |
|------|-------------|
| [robot_fm_guide.html](./robot_fm_guide.html) | Interactive guide: pipeline, embodiment routing, real-time budget timeline, calculator, self-check, exercises |
| [robot_fm.py](./robot_fm.py) | GR00T N1-style reference implementation with 4 toy embodiments + the `rfm_cost` model |

## Quick Start

```bash
open robot_fm_guide.html       # macOS
xdg-open robot_fm_guide.html   # Linux
python robot_fm.py             # CPU, ~2 s
```

The demo builds ONE mixed batch with four robots (Franka arm, xArm bimanual, Unitree H1, Unitree Go2; toy state/action sizes), computes a masked flow-matching loss and generates a 16-action chunk per robot in a single forward pass.

## GR00T N1 at a glance (paper)

| Part | GR00T N1-2B |
|------|-------------|
| System 2 | Eagle-2 VLM (SmolLM2 + SigLIP-2), 224×224 → 64 image tokens per frame, **12th LLM layer** used, 10 Hz on an L40 |
| System 1 | DiT, flow matching, alternating cross-attention (to VLM tokens) and self-attention (state + action tokens), adaptive LayerNorm on the timestep; motor actions at 120 Hz |
| Per-embodiment | state encoder, action encoder (with timestep) and action decoder MLPs |
| Chunk | H = 16, K = 4 denoising steps; flow time from a Beta(1.5, 1)-based density |
| Size / speed | 2.2B parameters (1.34B in the VLM); 16-action chunk in 63.9 ms on an L40 (bf16) |
| Freezing | language part of the VLM frozen in pre- and post-training |

## What `robot_fm.py` implements

- **System 2** (`System2VLM`): images + text only. A few transformer layers play the role of LLM layers 1…12; the output of the last one is used.
- **System 1** (`System1ActionHead`): input tokens = [state token, H action tokens]; blocks alternate cross-attention (even) and self-attention (odd); `AdaLayerNorm` = norm(x)·(1 + scale(t)) + shift(t), no gate; timestep discretised into 1000 buckets.
- **Embodiment routing** (`CategorySpecificLinear`): weights `[num_embodiments, in, out]`, gathered per sample and applied with one `torch.bmm`, as in Isaac-GR00T. States are zero-padded to 64 dims and actions to 32; `action_mask` removes padding from the loss. A batch can mix robots.
- **Flow matching**: x_t = (1−t)·noise + t·action, target action − noise, Euler from t = 0 with K = 4 steps (the same convention as Isaac-GR00T's code and the VLA module).
- **`rfm_cost`**: exact transformer parameter counts plus FLOPs, the System 2 caching speedup, the cacheable cross-attention K/V share, and roofline floors. The guide's calculator uses identical formulas.

## Numbers from `python robot_fm.py`

| | RFM-S | RFM-B |
|-|------:|------:|
| System 2 (VLM) | 12.8 M | 25.6 M |
| Shared DiT + time/pos/out | 2.3 M | 6.1 M |
| Per-robot encoders/decoder | 210,080 per robot | 362,016 per robot |
| Total (8 embodiment slots) | 16.8 M | 34.6 M |

For `rfm_base` with 2 cameras and 4 steps: System 2 = 3.50 GFLOPs once, System 1 = 0.36 GFLOPs per step. Running System 2 once instead of at every step saves 3.1× FLOPs. The K/V projections of the VL tokens are 46% of a System 1 step and can be cached across steps. With 4 robots, 92.7% of the transformer weights are shared.

## Real-time budget

With synchronous execution the robot waits for every inference call. Cycle = L + e/f, where L is latency, e the actions executed per call and f the control rate; for example π0 on a 50 Hz robot with e = 25 and L = 73 ms is idle 12.7% of the time. With asynchronous execution (start the next call while executing), the chunk must cover the re-plan period plus the latency: H/f ≥ max(e/f, L) + L. The guide's timeline widget lets you try GR00T's paper numbers and a 312 ms what-if.

## Measured GR00T N1.7 latency (Isaac-GR00T deployment README)

Full policy call incl. preprocessing, LIBERO checkpoint, batch 1, two 256×256 cameras (156 VLM tokens), horizon 40, 4 denoising steps:

| Device | PyTorch eager | torch.compile | TensorRT | TensorRT + mixed NVFP4 |
|--------|--------------:|--------------:|---------:|-----------------------:|
| H100 80GB | 101.5 ms | 53.4 ms | 22.9 ms | — |
| L40 | 123.7 ms | 66.8 ms | 35.0 ms | — |
| Jetson AGX Thor (120 W) | 111.5 ms | 100.5 ms | 77.6 ms | 35.9 ms |
| Jetson Orin (30 W) | 500.1 ms | 412.4 ms | 312.2 ms | — |

Jetson AGX Thor T5000 (NVIDIA spec page): 2070 sparse FP4 TFLOPS, 128 GB LPDDR5X, 273 GB/s, 40–130 W.

## As of Oct 2026: N1 → N1.7

- **N1.5**: frozen VLM (Eagle 2.5), FLARE objective for learning from human egocentric video; the README reports language following on GR-1 tasks rising from 46.6% to 93.3%.
- **N1.6**: Cosmos-Reason-2B VLM variant, a 32-layer DiT (vs 16), top 4 VLM layers unfrozen in pre-training, state-relative action chunks.
- **N1.7** (GA, Apache 2.0): Cosmos-Reason2-2B (Qwen3-VL architecture), layer 12, relative end-effector actions shared by robots and humans, 20K hours of EgoScale human video in pre-training, state/action padded to 132 dims, horizon 40, full TensorRT export.

## References

- [GR00T N1 (NVIDIA, 2025)](https://arxiv.org/abs/2503.14734), [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), [project page](https://developer.nvidia.com/isaac/gr00t)
- [Eagle 2 (Li et al., 2025)](https://arxiv.org/abs/2501.14818)
- [π0 (Black et al., 2024)](https://arxiv.org/abs/2410.24164), [Real-Time Chunking (Black et al., 2025)](https://arxiv.org/abs/2506.07339)
- Datasets: [Open X-Embodiment](https://arxiv.org/abs/2310.08864), [DROID](https://arxiv.org/abs/2403.12945), [AgiBot World](https://arxiv.org/abs/2503.06669), [RH20T](https://arxiv.org/abs/2307.00595)
- [RoboCat](https://arxiv.org/abs/2306.11706), [Octo](https://arxiv.org/abs/2405.12213), [HPT](https://arxiv.org/abs/2409.20537), [RDT-1B](https://arxiv.org/abs/2410.07864)
