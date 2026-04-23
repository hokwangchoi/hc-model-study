"""
Diffusion Model (DiT: Diffusion Transformer) Reference Implementation
Peebles & Xie, "Scalable Diffusion Models with Transformers" (2023)

Usage:
    python diffusion.py

Architecture:
    Noisy latent x_t ─┐
                      ├─→ DiT blocks → predicted noise ε̂
    Timestep t    ────┤
    Class label y ────┘

Model Configurations:
    - DiT-S/2:  33M params    (small, for experimentation)
    - DiT-B/2:  130M params
    - DiT-L/2:  458M params
    - DiT-XL/2: 675M params   (SOTA on ImageNet 256×256)

Note on FLOPs: this file counts every multiply-add as 2 FLOPs (MAC=2
convention, consistent with transformer.py / vit.py / vlm.py). The DiT paper
uses MAC=1, so its numbers are ~half of what's printed here.

This implementation focuses on:
- The DiT block with adaLN-Zero conditioning (the modern way to condition)
- DDIM sampling (deterministic, fast)
- Flow matching variant (what Pi0 / FLUX use — simpler math, straighter trajectories)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# ============================================================================
# Noise Schedule — the heart of diffusion
# ============================================================================

class NoiseSchedule:
    """
    Cosine beta schedule (Nichol & Dhariwal 2021) — smoother than linear.

    The forward process is q(x_t | x_0) = N(√α̅_t · x_0, (1 - α̅_t) · I).
    At t=0, x_0 is clean data. At t=T, x_T is (almost) pure Gaussian noise.

    Reparameterized: x_t = √α̅_t · x_0 + √(1 - α̅_t) · ε, where ε ~ N(0, I).
    """
    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        self.num_timesteps = num_timesteps
        # Cosine schedule: α̅_t = cos²((t/T + s) / (1 + s) · π/2)
        t = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]                    # α̅_0 = 1 (clean data)
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        self.betas = torch.clamp(betas, 0.0001, 0.9999)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        # Pre-compute what we need at sampling time
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor,
                 noise: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean data at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        # Broadcast schedule values to match x_0 shape
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = sqrt_a * x_0 + sqrt_1ma * noise
        return x_t, noise

# ============================================================================
# Timestep Embedding — turn scalar t into a d-dim vector
# ============================================================================

class TimestepEmbedding(nn.Module):
    """
    Sinusoidal position embedding of the timestep, followed by a 2-layer MLP.
    Same pattern as position embeddings in the original Transformer —
    just indexing by timestep instead of position.
    """
    def __init__(self, hidden_dim: int, frequency_embedding_dim: int = 256):
        super().__init__()
        self.frequency_embedding_dim = frequency_embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int, max_period: int = 10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] integer timesteps
        emb = self.sinusoidal_embedding(t, self.frequency_embedding_dim)
        return self.mlp(emb)                      # [B, hidden_dim]

# ============================================================================
# Class Embedding — simple learnable lookup
# ============================================================================

class LabelEmbedding(nn.Module):
    """
    Turn a class label y into a hidden_dim vector. Supports classifier-free
    guidance via a "null" class used at a dropout probability during training.
    """
    def __init__(self, num_classes: int, hidden_dim: int, dropout_prob: float = 0.1):
        super().__init__()
        # +1 for the null class used for CFG
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_dim)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def forward(self, labels: torch.Tensor, train: bool = False) -> torch.Tensor:
        if train and self.dropout_prob > 0:
            drop_mask = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            labels = torch.where(drop_mask, self.num_classes, labels)  # null class
        return self.embedding_table(labels)        # [B, hidden_dim]

# ============================================================================
# Patchify — image to token sequence, same as ViT
# ============================================================================

class PatchEmbed(nn.Module):
    """
    Split the input (latent or image) into patches and project to hidden_dim.

    For DiT-XL/2 operating on 32×32 latents with patch_size=2:
        input [B, 4, 32, 32] → [B, 256, 1152] (16×16=256 patches)
    """
    def __init__(self, input_size: int = 32, patch_size: int = 2,
                 in_channels: int = 4, hidden_dim: int = 1152):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = (input_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, hidden_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, hidden_dim, H/P, W/P] → [B, N, hidden_dim]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

# ============================================================================
# DiT Block — transformer block with adaLN-Zero conditioning
# ============================================================================

def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply per-sample affine modulation to a LayerNormed tensor."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    """
    DiT block: standard transformer (attn + MLP) with adaptive LayerNorm.

    adaLN-Zero is the key contribution. Each LayerNorm's (γ, β) plus a
    residual gate α are predicted from the condition c = t_emb + y_emb.
    At init, α=0 so residuals are zeroed — the block starts as an identity,
    stabilizing early training.
    """
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        # Predict 6 modulation params per block from the condition vector:
        #   (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        # Zero-init the final linear → adaLN-Zero: α starts at 0
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: [B, hidden_dim] condition vector (timestep + class)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # Attention path with modulated LayerNorm + gated residual
        x_mod = modulate(self.norm1(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(x_mod, x_mod, x_mod, need_weights=False)
        x = x + gate_attn.unsqueeze(1) * attn_out

        # MLP path
        x_mod = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_mod)
        return x

# ============================================================================
# DiT — the full model
# ============================================================================

class FinalLayer(nn.Module):
    """Final adaLN + linear. Maps hidden_dim back to patch content (for unpatchify)."""
    def __init__(self, hidden_dim: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

class DiT(nn.Module):
    """
    Diffusion Transformer (Peebles & Xie 2023).

    Inputs:
        x: [B, C, H, W]    — noisy image or latent
        t: [B]             — timestep (integer)
        y: [B]             — class label (optional, None for unconditional)

    Outputs:
        [B, C, H, W]       — predicted noise (or velocity for flow matching)
    """
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_dim: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        num_classes: int = 1000,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels     # predict noise with same shape as input
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(input_size, patch_size, in_channels, hidden_dim)
        self.t_embedder = TimestepEmbedding(hidden_dim)
        self.y_embedder = LabelEmbedding(num_classes, hidden_dim, class_dropout_prob) \
            if num_classes > 0 else None

        # Learned position embeddings (simple; DiT-XL uses fixed sincos — similar)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_dim, patch_size, self.out_channels)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, P²·C_out] → [B, C_out, H, W]"""
        B = x.shape[0]
        P = self.patch_size
        C = self.out_channels
        H = W = self.input_size // P
        x = x.reshape(B, H, W, P, P, C)
        x = x.permute(0, 5, 1, 3, 2, 4)                   # [B, C, H, P, W, P]
        x = x.reshape(B, C, H * P, W * P)
        return x

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                y: Optional[torch.Tensor] = None,
                train: bool = False) -> torch.Tensor:
        x = self.patch_embed(x) + self.pos_embed          # [B, N, hidden_dim]
        c = self.t_embedder(t)                            # [B, hidden_dim]
        if y is not None and self.y_embedder is not None:
            c = c + self.y_embedder(y, train=train)       # add class cond
        for block in self.blocks:
            x = block(x, c)                                # [B, N, hidden_dim]
        x = self.final_layer(x, c)                         # [B, N, P²·C_out]
        x = self.unpatchify(x)                             # [B, C_out, H, W]
        return x

# ============================================================================
# DDIM Sampler — deterministic sampling in N_steps << T
# ============================================================================

class DDIMSampler:
    """
    Denoising Diffusion Implicit Models (Song et al. 2020).

    Training uses T=1000 timesteps. DDIM lets you sample with, say, 50 steps
    by taking a subsequence of [0, T) and using the deterministic DDIM update.

    CFG (classifier-free guidance): run the model twice per step, once with
    the real class and once with the null class, then extrapolate:
        ε̂ = ε̂_uncond + w · (ε̂_cond - ε̂_uncond)
    where w is the guidance scale (typical: 4.0 for ImageNet).
    """
    def __init__(self, model: DiT, schedule: NoiseSchedule, num_steps: int = 50):
        self.model = model
        self.schedule = schedule
        self.num_steps = num_steps
        # Evenly-spaced timestep subsequence from [0, T)
        self.timesteps = torch.linspace(
            schedule.num_timesteps - 1, 0, num_steps + 1
        ).long()

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        y: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples by iterative denoising.

        Args:
            shape: (B, C, H, W) of output
            y: class labels [B], or None for unconditional
            guidance_scale: CFG strength; 1.0 = no CFG (single forward),
                            >1.0 = CFG (double forward per step)
        """
        B = shape[0]
        x = torch.randn(shape, device=device)
        use_cfg = guidance_scale > 1.0 and y is not None

        for i in range(self.num_steps):
            t = self.timesteps[i]
            t_next = self.timesteps[i + 1]
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)

            if use_cfg:
                # Double the batch: one copy conditional, one unconditional (null class)
                x_cat = torch.cat([x, x], dim=0)
                y_null = torch.full_like(y, self.model.y_embedder.num_classes)
                y_cat = torch.cat([y, y_null], dim=0)
                t_cat = torch.cat([t_batch, t_batch], dim=0)
                eps = self.model(x_cat, t_cat, y_cat)
                eps_cond, eps_uncond = eps.chunk(2, dim=0)
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                eps = self.model(x, t_batch, y)

            # DDIM update
            alpha_t = self.schedule.alphas_cumprod[t].to(device)
            alpha_next = self.schedule.alphas_cumprod[t_next].to(device) if t_next > 0 \
                else torch.tensor(1.0, device=device)
            # Predict x_0 from ε
            x_0_pred = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
            # Deterministic DDIM reverse step (σ=0)
            x = alpha_next.sqrt() * x_0_pred + (1 - alpha_next).sqrt() * eps

        return x

# ============================================================================
# Flow Matching — the modern alternative
# ============================================================================

class FlowMatchingTrainer:
    """
    Flow matching (Lipman et al. 2023) — reframes generative modeling as
    learning a vector field v_θ(x, t) that transports noise to data along
    straight lines.

    Training: sample t ∈ [0, 1], sample noise ε, construct
        x_t = (1 - t) · ε + t · x_0
    and train the model to predict the velocity
        v = x_0 - ε

    Sampling: Euler integration of dx/dt = v_θ(x, t) from t=0 to t=1.
    Typically 10-30 steps (much fewer than DDIM's 50+) because paths are straighter.

    Used by: FLUX, Stable Diffusion 3, Pi0 (the VLA we'll cover later).
    """
    @staticmethod
    def training_step(model: DiT, x_0: torch.Tensor,
                      y: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x_0.shape[0]
        t = torch.rand(B, device=x_0.device)               # t ∈ [0, 1]
        noise = torch.randn_like(x_0)
        # Linear interpolation: at t=0 we have noise, at t=1 we have data
        x_t = (1 - t.view(-1, 1, 1, 1)) * noise + t.view(-1, 1, 1, 1) * x_0
        target_v = x_0 - noise                              # straight-line velocity
        # Scale t to match the model's integer timestep interface
        t_int = (t * 999).long()
        pred_v = model(x_t, t_int, y, train=True)
        return F.mse_loss(pred_v, target_v)

    @staticmethod
    @torch.no_grad()
    def sample(model: DiT, shape: Tuple[int, int, int, int],
               num_steps: int = 20, y: Optional[torch.Tensor] = None,
               device: str = "cpu") -> torch.Tensor:
        x = torch.randn(shape, device=device)
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = i * dt
            t_batch = torch.full((shape[0],), int(t * 999), device=device, dtype=torch.long)
            v = model(x, t_batch, y)
            x = x + v * dt                                   # Euler step
        return x

# ============================================================================
# Model Configurations
# ============================================================================

def dit_s_2() -> DiT:
    """DiT-S/2: 33M params."""
    return DiT(input_size=32, patch_size=2, hidden_dim=384, depth=12, num_heads=6)

def dit_b_2() -> DiT:
    """DiT-B/2: 130M params."""
    return DiT(input_size=32, patch_size=2, hidden_dim=768, depth=12, num_heads=12)

def dit_l_2() -> DiT:
    """DiT-L/2: 458M params."""
    return DiT(input_size=32, patch_size=2, hidden_dim=1024, depth=24, num_heads=16)

def dit_xl_2() -> DiT:
    """DiT-XL/2: 675M params. The SOTA ImageNet 256×256 config."""
    return DiT(input_size=32, patch_size=2, hidden_dim=1152, depth=28, num_heads=16)

# ============================================================================
# Analysis Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: DiT):
    print("=" * 60)
    print("DiT Model Summary")
    print("=" * 60)
    patch_params = count_parameters(model.patch_embed)
    t_params = count_parameters(model.t_embedder)
    y_params = count_parameters(model.y_embedder) if model.y_embedder else 0
    block_params = sum(count_parameters(b) for b in model.blocks)
    final_params = count_parameters(model.final_layer)
    total = count_parameters(model)

    print(f"Patch Embedding:     {patch_params:>12,} params")
    print(f"Timestep Embedding:  {t_params:>12,} params")
    print(f"Label Embedding:     {y_params:>12,} params")
    print(f"DiT Blocks ({len(model.blocks):2d}):     {block_params:>12,} params")
    print(f"Final Layer:         {final_params:>12,} params")
    print("-" * 60)
    print(f"Total:               {total:>12,} params ({total / 1e6:.2f}M)")
    print("=" * 60)

def compute_flops_per_step(model: DiT) -> dict:
    """FLOPs for one forward pass (one denoising step)."""
    N = model.patch_embed.num_patches
    d = model.hidden_dim
    L = len(model.blocks)

    # Patch embed: equivalent GEMM over N tokens of P² * in_channels features
    P = model.patch_embed.patch_size
    C_in = model.patch_embed.proj.in_channels
    patch_flops = 2 * N * (P * P * C_in) * d

    # Per DiT block: attention + MLP + adaLN_modulation
    qkv_flops = 2 * N * 3 * d * d
    attn_flops = 2 * 2 * N * N * d                  # Q·Kᵀ and softmax·V
    proj_flops = 2 * N * d * d
    mlp_flops = 2 * (2 * N * d * (4 * d))           # fc1 + fc2
    adaLN_flops = 2 * d * (6 * d)                   # modulation params
    block_flops = qkv_flops + attn_flops + proj_flops + mlp_flops + adaLN_flops

    total = patch_flops + L * block_flops
    return {
        "patch_embed": patch_flops,
        "per_block": block_flops,
        "total_gflops": total / 1e9,
        "num_patches": N,
    }

def compute_total_sampling_flops(model: DiT, num_steps: int = 50,
                                  use_cfg: bool = True) -> float:
    """Total FLOPs to generate one sample (accounting for CFG doubling)."""
    per_step = compute_flops_per_step(model)["total_gflops"]
    cfg_mult = 2 if use_cfg else 1
    return per_step * num_steps * cfg_mult

def demo_forward_pass():
    """One forward pass with shape annotations."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (DiT-B/2)")
    print("=" * 60)

    model = dit_b_2()
    model.eval()

    B = 2
    x = torch.randn(B, 4, 32, 32)                   # noisy latent
    t = torch.randint(0, 1000, (B,))                # random timestep
    y = torch.randint(0, 1000, (B,))                # random class

    print(f"\n1. Input latent x:     {list(x.shape)}")
    print(f"2. Timesteps t:        {list(t.shape)}")
    print(f"3. Class labels y:     {list(y.shape)}")

    with torch.no_grad():
        patches = model.patch_embed(x) + model.pos_embed
        print(f"4. After patchify:     {list(patches.shape)}  (256 patches)")

        t_emb = model.t_embedder(t)
        y_emb = model.y_embedder(y)
        c = t_emb + y_emb
        print(f"5. Condition c:        {list(c.shape)}  (t_emb + y_emb)")

        h = patches
        for i, block in enumerate(model.blocks):
            h = block(h, c)
            if i == 0 or i == len(model.blocks) - 1:
                print(f"6. After block {i:2d}:     {list(h.shape)}")
            elif i == 1:
                print("   ...")

        out = model.final_layer(h, c)
        out = model.unpatchify(out)
        print(f"7. Output (predicted ε): {list(out.shape)}  (same shape as input)")

    print("\n" + "=" * 60)

def demo_sampling():
    """Run a tiny DDIM sampling loop to show the iterative structure."""
    print("\n" + "=" * 60)
    print("DDIM Sampling Demo (DiT-S/2, 10 steps for speed)")
    print("=" * 60)

    model = dit_s_2()
    model.eval()
    schedule = NoiseSchedule(num_timesteps=1000)
    sampler = DDIMSampler(model, schedule, num_steps=10)

    y = torch.tensor([0, 1])                         # classes 0 and 1
    samples = sampler.sample(
        shape=(2, 4, 32, 32), y=y, guidance_scale=4.0, device="cpu"
    )
    print(f"Generated samples:     {list(samples.shape)}")
    print("(Would be decoded through VAE → [B, 3, 256, 256] RGB image)")
    print("\n" + "=" * 60)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Diffusion Transformer (DiT) Reference Implementation")
    print("=" * 60)

    model = dit_b_2()
    print_model_summary(model)

    flops = compute_flops_per_step(model)
    print(f"\nFLOPs Analysis (DiT-B/2):")
    print(f"  Num patches:          {flops['num_patches']}")
    print(f"  Per-block FLOPs:      {flops['per_block'] / 1e9:.2f} GFLOPs")
    print(f"  One forward pass:     {flops['total_gflops']:.2f} GFLOPs")
    print(f"  50 steps, no CFG:     "
          f"{compute_total_sampling_flops(model, 50, False):.1f} GFLOPs")
    print(f"  50 steps, with CFG:   "
          f"{compute_total_sampling_flops(model, 50, True):.1f} GFLOPs")
    print(f"  Flow match 20 steps:  "
          f"{compute_total_sampling_flops(model, 20, True):.1f} GFLOPs")

    demo_forward_pass()
    demo_sampling()

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    configs = [
        ("DiT-S/2",  dit_s_2),
        ("DiT-B/2",  dit_b_2),
        ("DiT-L/2",  dit_l_2),
        ("DiT-XL/2", dit_xl_2),
    ]
    print(f"{'Model':<12} {'Params':>12} {'GFLOPs/step':>14} {'50-step CFG':>14}")
    print("-" * 60)
    for name, fn in configs:
        m = fn()
        p = count_parameters(m)
        fs = compute_flops_per_step(m)["total_gflops"]
        tot = compute_total_sampling_flops(m, 50, True)
        print(f"{name:<12} {p/1e6:>10.1f}M {fs:>12.2f} {tot:>12.1f}")
