# Diffusion Model (DiT)

Based on the **Diffusion Transformer** architecture —
[Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)](https://arxiv.org/abs/2212.09748).
The transformer backbone used by Stable Diffusion 3, FLUX, Sora, and Pi0's
action expert. For the diffusion math itself:
[Ho et al., DDPM (2020)](https://arxiv.org/abs/2006.11239) and
[Song et al., DDIM (2020)](https://arxiv.org/abs/2010.02502).

## Contents

| File | Description |
|------|-------------|
| [diffusion_guide.html](./diffusion_guide.html) | Interactive visual guide (open in browser) |
| [diffusion.py](./diffusion.py) | Reference implementation with annotations |

## Quick Start

```bash
# View the interactive guide
open diffusion_guide.html      # macOS
xdg-open diffusion_guide.html  # Linux

# Run the implementation
python diffusion.py
```

## The core idea

A diffusion model generates samples by **iteratively removing noise** from a
random starting point. Training teaches a neural network to predict the noise
that was added to data at some timestep; sampling runs that network in reverse,
step by step.

```
Training:   clean data x_0 ──add noise ε──→ noisy x_t ──→ model predicts ε̂
                                                           └─ loss: MSE(ε̂, ε)

Sampling:   pure noise x_T ──model──→ denoised x_{T-1} ──model──→ ... ──→ x_0
                       ↑─────────── N steps ──────────────↑
```

The key insight: if the model can predict noise accurately, we can subtract a
little noise at each step and slowly walk from pure Gaussian noise to a real
sample. Training is MSE regression — stable and well-understood.

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| One DiT forward | O(N·d² + N²·d) per layer | **Compute** | FP16/BF16, Tensor Cores |
| Sampling loop | N_steps × forward | **Iterative** | DDIM, DPM-Solver, consistency distillation |
| CFG | 2 × forward per step | Compute | Can be distilled away |
| VAE decode | O(H·W·C) | Memory | Optional; only for latent diffusion |

**Critical:** unlike LLMs where decode is memory-bound at batch 1, diffusion
steps are **compute-bound** — each step does dense batched matmuls with no
KV cache to read from HBM. Tensor Core utilization matters; memory bandwidth
usually does not.

## Critical Optimizations

### 1. Step Reduction

The dominant cost of diffusion is `N_steps × per_step_cost`. Cutting steps is
the biggest lever.

| Method | Typical steps | Quality loss |
|--------|---------------|--------------|
| DDPM (vanilla) | 1000 | none (reference) |
| DDIM | 50 | negligible |
| DPM-Solver++ | 20 | small |
| Flow matching (Euler) | 20 | depends on model |
| LCM / distilled | 4–8 | noticeable but usable |
| Turbo / Lightning | 1–4 | significant at 1 step |

DDIM lets you subsample timesteps and use a deterministic update — you trade
none of training for 20× fewer steps at inference. This is the default.

For production:
- **Design for DDIM/DPM-Solver from the start** — always.
- **Consider distillation** (LCM, Turbo) if you need sub-second latency at batch 1.

### 2. Classifier-Free Guidance (CFG)

To bias generation toward the conditional class/prompt, run the model twice
per step — once with the condition, once with a null/empty condition:

```
ε̂ = ε̂_uncond + w · (ε̂_cond - ε̂_uncond)     w = guidance scale, e.g. 4.0
```

Cost: **2× the per-step compute**. Quality impact: usually +5–15% on human
preference at `w=3..7`. Common optimizations:

- **CFG distillation**: train a student to match the CFG output in one forward pass
- **Skip CFG at final steps**: most of CFG's benefit is in early steps
- **Batch CFG**: put conditional and unconditional into the same batch of 2 (which this code does)

### 3. Latent Diffusion

Run diffusion in a compressed latent space (from a pre-trained VAE) instead of
pixel space. Stable Diffusion: 512×512 pixels → 64×64 latent (8× compression)
→ 64× reduction in per-step FLOPs. Essentially free quality.

```
Image [B, 3, 512, 512]
  ↓ VAE encoder (once, offline during training)
Latent [B, 4, 64, 64]
  ↓ diffusion (here's where the iteration happens)
Denoised latent [B, 4, 64, 64]
  ↓ VAE decoder (once per sample)
Image [B, 3, 512, 512]
```

This implementation operates on 32×32 latents (4-channel) to match DiT-XL/2's
configuration. Pair with a pretrained VAE (SD's `stabilityai/sd-vae-ft-ema`)
for pixel output.

### 4. FlashAttention

DiT blocks are pure transformer blocks, so FlashAttention applies directly.
For DiT-XL/2 at 256 tokens, the attention matrix is modest (256² × 16 heads ×
2 bytes = 2 MB), but the HBM traffic savings still matter at batch 32+. PyTorch
SDPA picks FlashAttention automatically when available.

### 5. Flow Matching

A different framing of the same generative modeling problem. Instead of
training a denoiser, train a velocity predictor `v_θ(x, t)` that transports
noise to data along straight-line paths:

```
x_t = (1 - t) · ε + t · x_0            # linear interpolation
target = x_0 - ε                        # constant velocity
```

Sampling becomes Euler integration: 10-30 steps instead of 50, with straighter
trajectories that tolerate larger step sizes. Adopted by FLUX, Stable Diffusion
3, and Pi0 (the VLA in our roadmap). The same DiT backbone works for both —
only the training and sampling loops change.

## Architecture Summary (DiT)

```
Input latent x_t [B, 4, 32, 32] ──┐
Timestep t [B]            ────────┤
Class label y [B]         ────────┤
                                  ▼
                          ┌────────────────────┐
                          │  Condition encoders │
                          │  t → sinusoidal+MLP │
                          │  y → learned embed  │
                          │  c = t_emb + y_emb  │  [B, d]
                          └──────────┬─────────┘
                                     │
  [B, 4, 32, 32]                     │
        │                            │
        ▼ Patchify (Conv2d k=2 s=2)  │
  [B, 256, d]                        │
        │                            │
        ▼ + position embed           │
        │                            │
  ┌─────▼─────────────────────────┐  │
  │ DiT Block × L                 │◄─┤
  │  ┌──────────────────────────┐ │  │
  │  │ adaLN (shift, scale ← c) │ │  │
  │  │ Attention                │ │  │
  │  │ residual × gate          │ │  │
  │  └──────────────────────────┘ │  │
  │  ┌──────────────────────────┐ │  │
  │  │ adaLN → MLP → gate       │ │  │
  │  └──────────────────────────┘ │  │
  └─────┬─────────────────────────┘  │
        │                            │
        ▼ Final adaLN + Linear       │
  [B, 256, P²·C_out]                 │
        │                            │
        ▼ Unpatchify                 │
  Predicted noise ε̂ [B, 4, 32, 32]  │
```

**adaLN-Zero conditioning** (Peebles & Xie's key contribution): the condition
vector `c` predicts 6 modulation params per block (shift/scale/gate for both
attention and MLP). Zero-initialized so the model starts as an identity function
— training is stable from step 1.

## Parameters & FLOPs

Per the DiT paper (MAC=1 convention):

| Model | Params | GFLOPs/step | Use case |
|-------|--------|-------------|----------|
| DiT-S/2 | 33M | 6 | Experimentation |
| DiT-B/2 | 130M | 24 | Consumer deployment |
| DiT-L/2 | 458M | 80 | Research |
| DiT-XL/2 | 675M | 119 | ImageNet SOTA |

**Total sampling cost** = `GFLOPs/step × num_steps × CFG_mult`.

For DiT-XL/2 at 50 DDIM steps with CFG:
```
119 GFLOPs × 50 steps × 2 (CFG) = 11,900 GFLOPs ≈ 12 TFLOPs per image
```

On an RTX 4090 (83 TFLOPs FP16): ~150 ms. On Orin Nano Super (10 TFLOPs FP16):
~1.2 s per image. Flow matching at 20 steps cuts this ~2.5×.

## Key Differences from LLMs

1. **Compute-bound, not memory-bound.** No KV cache to stream per step — each step does dense GEMMs from scratch. Tensor Core utilization is the main lever, not LPDDR5 bandwidth. (Opposite of LLM decode.)
2. **Sampling cost scales with steps, not sequence length.** There's no autoregressive chain. 50 full forward passes > 1 LLM prefill.
3. **No KV cache, so no PagedAttention.** Memory is dominated by model weights and intermediate activations, both are fixed per step.
4. **Conditioning goes through adaLN, not cross-attention** (in DiT). Text conditioning in SD3 uses cross-attention; class/timestep conditioning uses adaLN.
5. **No tokenizer, no vocab.** Output is continuous (noise or velocity), regressed via MSE. Closer to regression than classification.

## On Edge Hardware

Diffusion on Jetson-class hardware is tight but doable with the right recipe:

- **Always use latent diffusion.** Pixel-space diffusion at 512² is 64× more expensive.
- **Use 10–20 flow-matching steps**, not 50 DDIM steps.
- **Distill CFG** if possible, otherwise accept the 2× overhead.
- **Compile the DiT to TensorRT** — fixed shapes, perfect for static optimization.
- **INT8 weight quantization** works well on DiT (transformer blocks tolerate it).

Rough budget on Orin Nano Super for DiT-B/2 at 32×32 latent, 20 flow-matching steps:
```
24 GFLOPs × 20 steps × 1 (no CFG) = 480 GFLOPs  ≈ 50 ms at 10 TFLOPs FP16
```

Plus VAE decode (another ~20 ms). Total ~70 ms per image. Feasible for real-time
applications at low resolution.

## Connection to VLA (Pi0)

Flow matching on a DiT-style backbone is the **action head** of modern
vision-language-action models:

- Input: state vector (from the VLM) + current action
- Model: small DiT predicting velocity over action dimensions
- Sampling: 10 Euler steps → action chunk

The pattern is: VLM produces a latent representation of "what to do", then a
flow-matching DiT expert generates a chunk of continuous actions at high
frequency. Will be covered in the VLA study later in this roadmap.

## References

- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) — the architecture
- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — foundational diffusion
- [DDIM (Song et al., 2020)](https://arxiv.org/abs/2010.02502) — deterministic fast sampling
- [DPM-Solver (Lu et al., 2022)](https://arxiv.org/abs/2206.00927) — better ODE solvers
- [Flow Matching (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747) — straight-line alternative
- [LCM (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) — 4-step distilled diffusion
- [Latent Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) — the Stable Diffusion paper
- [Pi0 (Physical Intelligence, 2024)](https://www.physicalintelligence.company/blog/pi0) — flow matching VLA for robotics
