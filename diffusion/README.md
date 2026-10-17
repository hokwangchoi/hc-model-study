# Diffusion Model (DiT)

Based on the **Diffusion Transformer** architecture —
[Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)](https://arxiv.org/abs/2212.09748).
DiT-style transformers are the backbone of today's image and video generators
(Stable Diffusion 3, FLUX, Sora-style video models), and the same flow-matching
recipe drives the action expert of VLAs such as π0. For the diffusion math itself:
[Ho et al., DDPM (2020)](https://arxiv.org/abs/2006.11239) and
[Song et al., DDIM (2020)](https://arxiv.org/abs/2010.02502).

## Contents

| File | Description |
|------|-------------|
| [diffusion_guide.html](./diffusion_guide.html) | Interactive visual guide (open in browser): Module 4 of 9, with a noise-level explorer, calculator, self-check and exercises |
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
| VAE decode | ≈0.62 TFLOP at 256×256 (SD f=8 decoder) | Compute (large convs) | Once per image; dominates few-step pipelines |

**Critical:** unlike LLM decode, which is memory-bound at batch 1, a diffusion
step behaves like a prefill: every weight is reused by all N tokens, so it is
mostly **compute-bound** (small DiTs at 256×256 sit near Jetson Orin's
compute/bandwidth ridge point). Tensor Core utilization and precision matter most.

## Critical Optimizations

### 1. Step Reduction

The dominant cost of diffusion is `N_steps × per_step_cost`. Cutting steps is
the biggest lever.

| Method | Typical steps | Quality loss |
|--------|---------------|--------------|
| DDPM (vanilla) | 1000 | none (reference) |
| DDIM | 50 | small (CIFAR-10 FID 4.67 vs 3.17 for 1000-step DDPM, DDIM paper Table 1) |
| DPM-Solver++ / UniPC | 10–20 | small |
| Flow matching (Euler-type) | 28–50 (FLUX [dev] examples) | needs a flow-matching model |
| LCM / consistency | 2–4 | needs distillation |
| ADD (Turbo), FLUX.1 [schnell], DMD2 | 1–4 | needs (adversarial) distillation |

DDIM lets you subsample timesteps and use a deterministic update: no retraining,
20× fewer steps than DDPM's 1000. With fewer than ~10 steps you need a network
trained or distilled for it.

For production:
- **Use a deterministic ODE sampler** (DDIM / DPM-Solver++ / Euler for flow models).
- **Consider a distilled few-step model** (LCM, Turbo, schnell) for interactive latency at batch 1.

### 2. Classifier-Free Guidance (CFG)

To bias generation toward the conditional class/prompt, run the model twice
per step — once with the condition, once with a null/empty condition:

```
ε̂ = ε̂_uncond + w · (ε̂_cond - ε̂_uncond)     w = guidance scale, e.g. 4.0
```

Cost: **2× the per-step compute**. Benefit: DiT-XL/2 goes from FID 9.62 to 2.27
with w = 1.5 (DiT paper, Table 2). Common optimizations:

- **Guidance distillation** (Meng et al., 2023): a student takes w as an input and
  reproduces the guided output in one forward pass. FLUX.1/FLUX.2 [dev] are
  released guidance-distilled.
- **Guide only part of the trajectory**: Kynkäänniemi et al. (2024) found guidance
  harmful at high noise and largely unnecessary at low noise.
- **Batch CFG**: put conditional and unconditional into one batch of 2 (which this
  code does). Fewer launches and one weight read, but the FLOPs still double.

### 3. Latent Diffusion

Run diffusion in a compressed latent space (from a pre-trained VAE) instead of
pixel space. Stable Diffusion: 512×512×3 pixels → 64×64×4 latent: 64× fewer
spatial positions, so 64× fewer tokens at the same patch size (and 4096× less
attention-score work). The decoder is not free, though: the SD/LDM f=8 decoder
costs about 0.62 TFLOP for a 256×256 image (2.5 TFLOP at 512×512), as much as
13.5 DiT-B/2 forward passes.

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

### 3b. Noise schedule

`NoiseSchedule` defaults to DiT's linear β schedule (1e-4 → 0.02, T = 1000);
`schedule="cosine"` gives Nichol & Dhariwal's cosine schedule. The DDIM sampler
divides by √ᾱ_t, which multiplies ε̂ errors by √((1−ᾱ)/ᾱ) = 157 at t = 999 for
linear but ≈ 20,000 for cosine (ᾱ ≈ 2.4e-9), so a sampler starting at t = 999 is
far more fragile with cosine.

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

Sampling becomes Euler integration of dx/dt = v. Training paths are straight;
the learned field is only roughly straight, so practical samplers still use tens
of steps (FLUX.1 [dev]'s example: 50; π0: 10 for robot actions). Stable Diffusion 3
and FLUX are rectified-flow transformers (note: they write x_t = (1−t)·x_0 + t·ε,
so their t = 1 is noise). The same DiT backbone works for both — only the
training and sampling loops change.

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
        ▼ + fixed 2D sin-cos pos.    │
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

Per the DiT paper (Table 4, MAC=1 "Gflops"; `diffusion.py` prints 2× these):

| Model | Params | Gflops/step (paper) | GFLOPs/step (`diffusion.py`) |
|-------|--------|---------------------|------------------------------|
| DiT-S/2 | 33M | 6.06 | 12.1 |
| DiT-B/2 | 130M | 23.01 | 46.0 |
| DiT-L/2 | 458M | 80.71 | 161.4 |
| DiT-XL/2 | 675M | 118.64 | 237.2 |

DiT-XL/2 reached FID 2.27 on ImageNet 256×256 with CFG 1.5 (state of the art in
Dec 2022).

**Total sampling cost** = `FLOPs/step × num_steps × CFG_mult`.

For DiT-XL/2 at 50 DDIM steps with CFG (2 FLOPs per multiply-add):
```
237 GFLOPs × 50 steps × 2 (CFG) ≈ 23.7 TFLOP per image
```

Lower bounds at 100% of peak: RTX 4090 (165 TFLOPS dense FP16 Tensor Core,
FP32 accumulate) ≥ 144 ms; Jetson Orin Nano Super (17 TFLOPS FP16) ≥ 1.4 s.
A 28-step guidance-distilled model needs 28 instead of 100 forward passes.

## Key Differences from LLMs

1. **Mostly compute-bound, like an LLM prefill.** Each weight is reused by all N tokens, so FP16 arithmetic intensity ≈ N FLOP/byte (DiT-B/2 at 256 tokens: ≈176 FLOP/B, just above Orin Nano Super's ridge of 17 TFLOPS / 102 GB/s ≈ 167). CFG and higher resolutions push it further into compute-bound territory. (Opposite of LLM decode, ≈1 FLOP/B.)
2. **Sampling cost scales with steps, not sequence length.** There's no autoregressive chain. 50 full forward passes > 1 LLM prefill.
3. **No KV cache, so no PagedAttention.** Memory is dominated by model weights and intermediate activations, both are fixed per step.
4. **Timestep/class conditioning goes through adaLN** (in DiT). Text needs a token-level path: cross-attention (PixArt-α, Wan) or joint text-image attention (SD3's MMDiT).
5. **No tokenizer, no vocab.** Output is continuous (noise or velocity), regressed via MSE. Closer to regression than classification.

## On Edge Hardware

Diffusion on Jetson-class hardware is tight but doable with the right recipe:

- **Always use latent diffusion.** Pixel space at the same patch size has 64× more tokens at 512².
- **Use few steps**: a distilled model (1–4) or a flow-matching model (~28), not 1000 DDPM steps.
- **Avoid the 2× CFG cost** with a guidance-distilled model if possible.
- **Compile the DiT (and the VAE decoder) to TensorRT** — fixed shapes, good for static optimization.
- **INT8**: Orin's dense INT8 peak (33 TOPS) is 2× FP16 (17 TFLOPS), and INT8 weights halve the bytes to stream.

Lower-bound budget on Orin Nano Super for DiT-B/2 at a 32×32 latent, 28 steps, guidance-distilled:
```
46 GFLOPs × 28 steps × 1 = 1.29 TFLOP  ≥ 76 ms at 17 TFLOPS FP16
SD-style VAE decode ≈ 0.62 TFLOP       ≥ 37 ms
```
With a 4-step distilled model the DiT needs ≥ 11 ms and the decoder becomes ~3/4 of the
time. Real latency is higher; measure with `trtexec` and compare.

## Connection to VLA (Pi0)

Flow matching on a transformer is the **action expert** of π0 (Black et al., 2024):

- Input: a noisy action chunk (H = 50 future steps × action dims) plus the robot state
- Model: a ~300M-parameter "action expert" whose tokens attend to the VLM's tokens in
  every layer (shared self-attention, separate weights); the flow-matching time is
  mixed in with an MLP, not adaLN
- Sampling: 10 Euler steps → action chunk

Covered in Module 8 (VLA).

## References

- [DiT (Peebles & Xie, 2023)](https://arxiv.org/abs/2212.09748) — the architecture
- [DDPM (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — foundational diffusion
- [DDIM (Song et al., 2020)](https://arxiv.org/abs/2010.02502) — deterministic fast sampling
- [DPM-Solver (Lu et al., 2022)](https://arxiv.org/abs/2206.00927) — better ODE solvers
- [Flow Matching (Lipman et al., 2023)](https://arxiv.org/abs/2210.02747) — straight-line alternative
- [LCM (Luo et al., 2023)](https://arxiv.org/abs/2310.04378) — 4-step distilled diffusion
- [Latent Diffusion (Rombach et al., 2022)](https://arxiv.org/abs/2112.10752) — the Stable Diffusion paper
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) — cosine schedule
- [Rectified Flow (Liu et al., 2022)](https://arxiv.org/abs/2209.03003) and [SD3 / MMDiT (Esser et al., 2024)](https://arxiv.org/abs/2403.03206)
- [On Distillation of Guided Diffusion Models (Meng et al., 2023)](https://arxiv.org/abs/2210.03142)
- [Applying Guidance in a Limited Interval (Kynkäänniemi et al., 2024)](https://arxiv.org/abs/2404.07724)
- [π0 (Black et al., 2024)](https://arxiv.org/abs/2410.24164) and [blog post](https://www.pi.website/blog/pi0) — flow matching VLA for robotics
