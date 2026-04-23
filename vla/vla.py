"""
VLA (Vision-Language-Action) Reference Implementation — Pi0-style

A robot foundation model that takes:
    - Multi-view camera images
    - Language instruction ("put the block in the red bowl")
    - Proprioception (current joint states)

And produces:
    - An action chunk: a sequence of H future actions (joint commands),
      trained and sampled via flow matching.

Usage:
    python vla.py

Architecture — two transformers, running in parallel:

    Images   ──┐
    Language ──┼──→ VLM backbone (frozen-ish) ──→ conditioning tokens
    Proprio  ──┘                                      │
                                                      │ (cross-attn)
                                                      ▼
                  Noisy action chunk    ─→  Action expert  ─→  predicted velocity v̂
                  + flow time t        (smaller transformer)      [B, H, action_dim]
                                                                      │
                                                                      ▼
                                                   Flow matching loss OR
                                                   Euler ODE integration to get clean action

Model configs (roughly):
    - VLA-S:  ~400M params   (small vision + small LLM + small action expert)
    - VLA-B: ~3.3B params    (Pi0-base: PaliGemma 3B + 300M action expert)

KEY INSIGHTS this file teaches:

1. TWO TRANSFORMERS, NOT ONE. The VLM backbone processes images + text + proprio
   into conditioning tokens. A SEPARATE action expert cross-attends to those
   tokens and generates actions. This decouples "understand the scene" from
   "decide what to do" and lets you freeze the VLM during robot fine-tuning.

2. ACTION CHUNK, NOT ONE-SHOT. Output is H=50 future actions (1 second at 50Hz)
   predicted at once. The robot executes them open-loop, then re-predicts.
   Chunking gives smoother motion than per-step prediction and cuts inference
   rate requirements 50× (predict at 1Hz, execute at 50Hz).

3. FLOW MATCHING, NOT DIFFUSION. Straight-line paths from noise to data
   instead of the curved SDE paths of DDPM. Simpler math (velocity = x_1 - x_0),
   same architecture. Converges in ~10 ODE steps vs ~50-1000 diffusion steps.

4. CONTINUOUS ACTIONS. Unlike RT-2 (which discretizes actions into vocab tokens)
   or diffusion policy (which uses DDPM), flow matching works natively in
   continuous action space. No tokenization, no discretization error.

Simplifications in this file:
- Vision encoder is a stub ViT-style transformer, not real SigLIP
- Language model is just an embedding table (not a real LLM)
- VLM backbone is bidirectional for simplicity (real Pi0 uses causal attn)
- Skipping the "action-to-VLM-token" projection variants
- No actual robot kinematics
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Vision encoder — stub SigLIP / ViT
# ============================================================================

class VisionEncoder(nn.Module):
    """
    Toy ViT-style vision encoder. In production, this is SigLIP-ViT-L
    (400M params, 14×14 patches) pretrained on web image-text pairs.
    Its weights are typically frozen during robot fine-tuning.

    Input:  images [B, N_cam, 3, H, W]
    Output: vision tokens [B, N_cam * N_patches, d_model]
    """
    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 d_model: int = 768, depth: int = 6, n_heads: int = 12):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.d_model = d_model

        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, causal=False) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cam, C, H, W = images.shape
        x = self.patch_embed(images.reshape(B * N_cam, C, H, W))        # [B·N_cam, d, h', w']
        x = x.flatten(2).transpose(1, 2)                                # [B·N_cam, N_patches, d]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.reshape(B, N_cam * self.n_patches, self.d_model)


# ============================================================================
# 2. Language embedding — stub
# ============================================================================

class LanguageEmbedding(nn.Module):
    """
    Stub language side. In Pi0 this is Gemma-2B (a real LLM).
    Here we just embed token IDs and let the VLM attention do the work.
    """
    def __init__(self, vocab_size: int = 32000, d_model: int = 768, max_len: int = 256):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:   # [B, T] → [B, T, d]
        return self.tok_embed(token_ids) + self.pos_embed[:, :token_ids.size(1)]


# ============================================================================
# 3. Proprioception encoder — joint state → conditioning token
# ============================================================================

class ProprioEncoder(nn.Module):
    """
    Map the robot's current joint state (positions, velocities, gripper) to
    one or more conditioning tokens.

    For a 7-DOF arm: proprio_dim ≈ 14  (7 joints × position + 7 × velocity)
    For a humanoid:   proprio_dim ≈ 50+ (many more DOFs)
    """
    def __init__(self, proprio_dim: int = 14, d_model: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(proprio_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:     # [B, proprio_dim] → [B, 1, d]
        return self.mlp(proprio).unsqueeze(1)


# ============================================================================
# 4. Transformer block (shared by VLM and action expert)
# ============================================================================

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block. Self-attention + FFN + residuals."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0,
                 causal: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.causal = causal

    def forward(self, x: torch.Tensor, kv: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention (or cross if kv provided)
        q = self.norm1(x)
        k = v = self.norm1(kv) if kv is not None else q
        mask = None
        if self.causal and kv is None:
            T = x.size(1)
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(q, k, v, attn_mask=mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# 5. Flow matching — schedule + timestep embedding
# ============================================================================

def sinusoidal_time_embedding(t: torch.Tensor, d_model: int) -> torch.Tensor:
    """Sinusoidal embedding for scalar flow-time t ∈ [0, 1]."""
    half = d_model // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class TimestepEmbedding(nn.Module):
    """Scalar flow-time t → d-dim embedding."""
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_time_embedding(t, self.d_model))


# ============================================================================
# 6. Action expert — the parallel transformer that generates actions
# ============================================================================

class ActionExpertBlock(nn.Module):
    """
    One block of the action expert. Three sub-layers:
      1. Self-attention over action-chunk tokens (they share info across timesteps)
      2. Cross-attention to VLM tokens (the scene/instruction context)
      3. FFN
    """
    def __init__(self, d_action: int, d_vlm: int, n_heads: int = 8,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_sa = nn.LayerNorm(d_action)
        self.self_attn = nn.MultiheadAttention(d_action, n_heads, batch_first=True)

        # Cross-attn: queries are action tokens (d_action), keys/values are VLM tokens (d_vlm)
        self.norm_ca_q = nn.LayerNorm(d_action)
        self.norm_ca_kv = nn.LayerNorm(d_vlm)
        self.ca_kv_proj = nn.Linear(d_vlm, d_action, bias=False)   # bring VLM dim into action dim
        self.cross_attn = nn.MultiheadAttention(d_action, n_heads, batch_first=True)

        self.norm_ffn = nn.LayerNorm(d_action)
        hidden = int(d_action * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_action, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_action),
        )

    def forward(self, action_tokens: torch.Tensor, vlm_tokens: torch.Tensor) -> torch.Tensor:
        # 1. Self-attention across action-chunk timesteps (no causal — all-to-all is fine
        #    for action chunks; they're co-generated, not autoregressive)
        q = self.norm_sa(action_tokens)
        sa_out, _ = self.self_attn(q, q, q)
        action_tokens = action_tokens + sa_out

        # 2. Cross-attention to VLM scene/instruction tokens
        q = self.norm_ca_q(action_tokens)
        kv = self.ca_kv_proj(self.norm_ca_kv(vlm_tokens))
        ca_out, _ = self.cross_attn(q, kv, kv)
        action_tokens = action_tokens + ca_out

        # 3. FFN
        action_tokens = action_tokens + self.ffn(self.norm_ffn(action_tokens))
        return action_tokens


class ActionExpert(nn.Module):
    """
    Pi0's action expert: a small transformer that takes the (noisy) action chunk
    + flow time + VLM context, and predicts the velocity v̂ = ∂x/∂t for flow
    matching.

    Shape flow:
      action_t:   [B, H, action_dim]    (noisy action chunk, where H = horizon)
      t:          [B]                    (flow time in [0, 1])
      vlm_tokens: [B, N_vlm, d_vlm]     (from the VLM backbone)
    Returns:
      v̂:          [B, H, action_dim]    (predicted velocity)
    """
    def __init__(
        self,
        action_dim: int = 14,
        horizon: int = 50,
        d_action: int = 384,
        d_vlm: int = 768,
        depth: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.d_action = d_action

        # Embed the raw action dim → action-expert hidden dim
        self.action_in = nn.Linear(action_dim, d_action)
        # Learnable positional embedding over the action chunk's H timesteps
        self.action_pos_embed = nn.Parameter(torch.zeros(1, horizon, d_action))
        nn.init.trunc_normal_(self.action_pos_embed, std=0.02)

        # Flow time embedding
        self.time_embed = TimestepEmbedding(d_action)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ActionExpertBlock(d_action, d_vlm, n_heads)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(d_action)
        # Project back to the raw action dim (produces velocity)
        self.action_out = nn.Linear(d_action, action_dim)

    def forward(
        self,
        action_t: torch.Tensor,       # [B, H, action_dim]
        t: torch.Tensor,              # [B]
        vlm_tokens: torch.Tensor,     # [B, N_vlm, d_vlm]
    ) -> torch.Tensor:
        B, H, _ = action_t.shape
        assert H == self.horizon

        # Embed action + position + time
        x = self.action_in(action_t) + self.action_pos_embed        # [B, H, d_action]
        t_emb = self.time_embed(t).unsqueeze(1)                     # [B, 1, d_action]
        x = x + t_emb                                                # broadcast across horizon

        # Transformer blocks (each does self-attn + cross-attn to VLM)
        for blk in self.blocks:
            x = blk(x, vlm_tokens)

        x = self.norm_out(x)
        return self.action_out(x)                                    # [B, H, action_dim]


# ============================================================================
# 7. VLM backbone — processes vision + language + proprio into context tokens
# ============================================================================

class VLMBackbone(nn.Module):
    """
    Lightweight VLM. Concatenates [vision, language, proprio] tokens and
    runs transformer blocks over them. Produces a single tokenized
    representation of the scene + instruction.

    Real Pi0: PaliGemma (SigLIP vision + Gemma 2B LLM) with a specific
    attention pattern. We use bidirectional attention here for simplicity.
    """
    def __init__(self, d_model: int = 768, depth: int = 6, n_heads: int = 12):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, causal=False) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        vision_tokens: torch.Tensor,    # [B, N_v, d]
        language_tokens: torch.Tensor,  # [B, N_l, d]
        proprio_tokens: torch.Tensor,   # [B, 1, d]
    ) -> torch.Tensor:
        # Concatenate into one sequence (modality segment embeddings would help here,
        # but skipped for simplicity)
        x = torch.cat([vision_tokens, language_tokens, proprio_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ============================================================================
# 8. The full VLA model — ties it all together
# ============================================================================

class Pi0(nn.Module):
    """
    Pi0-style Vision-Language-Action model.

    Two things it does:
      - compute_loss(batch): training loss via flow matching
      - generate_action(...): inference via Euler ODE integration
    """
    def __init__(
        self,
        # Vision
        img_size: int = 224, patch_size: int = 14, d_model: int = 768,
        vision_depth: int = 6, vision_heads: int = 12,
        # Language
        vocab_size: int = 32000, max_text_len: int = 256,
        # VLM backbone
        vlm_depth: int = 6,
        # Proprioception
        proprio_dim: int = 14,
        # Action expert
        action_dim: int = 14, horizon: int = 50,
        d_action: int = 384, action_depth: int = 6, action_heads: int = 8,
    ):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        self.vision = VisionEncoder(img_size=img_size, patch_size=patch_size,
                                    d_model=d_model, depth=vision_depth,
                                    n_heads=vision_heads)
        self.language = LanguageEmbedding(vocab_size=vocab_size, d_model=d_model,
                                          max_len=max_text_len)
        self.proprio = ProprioEncoder(proprio_dim=proprio_dim, d_model=d_model)
        self.vlm = VLMBackbone(d_model=d_model, depth=vlm_depth, n_heads=vision_heads)

        self.action_expert = ActionExpert(
            action_dim=action_dim, horizon=horizon,
            d_action=d_action, d_vlm=d_model,
            depth=action_depth, n_heads=action_heads,
        )

    def encode_vlm(
        self,
        images: torch.Tensor,         # [B, N_cam, 3, H, W]
        token_ids: torch.Tensor,      # [B, T]
        proprio: torch.Tensor,        # [B, proprio_dim]
    ) -> torch.Tensor:
        """Run the VLM backbone to get conditioning tokens. Shape [B, N_vlm, d]."""
        vision = self.vision(images)
        language = self.language(token_ids)
        proprio_tok = self.proprio(proprio)
        return self.vlm(vision, language, proprio_tok)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Flow matching training loss.

        batch fields:
            images:    [B, N_cam, 3, H, W]
            token_ids: [B, T]
            proprio:   [B, proprio_dim]
            action:    [B, horizon, action_dim]  (target action chunk)
        """
        images, token_ids = batch["images"], batch["token_ids"]
        proprio, action = batch["proprio"], batch["action"]
        B = action.size(0)

        # 1. Run VLM once
        vlm_tokens = self.encode_vlm(images, token_ids, proprio)

        # 2. Sample a random flow time t ∈ [0, 1] and noise ε
        t = torch.rand(B, device=action.device)
        eps = torch.randn_like(action)

        # 3. Linear interpolation x_t = (1-t)·ε + t·a
        #    This is the STRAIGHT-LINE path between noise and data,
        #    the defining feature of flow matching (vs curved SDE paths of DDPM).
        t_exp = t.view(B, 1, 1)
        x_t = (1 - t_exp) * eps + t_exp * action

        # 4. The true velocity along this straight path is constant: v = a - ε
        v_target = action - eps

        # 5. Predict velocity with the action expert
        v_pred = self.action_expert(x_t, t, vlm_tokens)

        # 6. MSE
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def generate_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Inference: integrate the learned velocity field from noise to a clean action chunk.

        Pi0 uses 10 Euler steps. Each step:
            x ← x + v̂(x, t) · dt
        where dt = 1/n_steps. Start at t=0 with pure noise, end at t=1 with action.
        """
        B = images.size(0)
        device = images.device

        # Encode scene ONCE (expensive) — reused across all integration steps
        vlm_tokens = self.encode_vlm(images, token_ids, proprio)

        # Start at noise
        x = torch.randn(B, self.horizon, self.action_dim, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.action_expert(x, t, vlm_tokens)
            x = x + v * dt                                      # Euler step

        return x    # [B, horizon, action_dim] — the action chunk


# ============================================================================
# Model configs
# ============================================================================

def vla_small() -> Pi0:
    """VLA-S: ~80M params. Stub vision + tiny LLM + small action expert."""
    return Pi0(
        img_size=224, patch_size=14, d_model=384, vision_depth=4, vision_heads=6,
        vocab_size=32000, vlm_depth=4, proprio_dim=14,
        action_dim=14, horizon=50, d_action=256, action_depth=4, action_heads=4,
    )

def vla_base() -> Pi0:
    """VLA-B: ~300M params (toy). Pi0-base is ~3B with PaliGemma backbone."""
    return Pi0(
        img_size=224, patch_size=14, d_model=768, vision_depth=6, vision_heads=12,
        vocab_size=32000, vlm_depth=6, proprio_dim=14,
        action_dim=14, horizon=50, d_action=384, action_depth=6, action_heads=8,
    )


# ============================================================================
# Analysis
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: Pi0):
    """Parameter breakdown by component."""
    print("=" * 60)
    print("VLA (Pi0-style) Model Summary")
    print("=" * 60)

    vision = count_parameters(model.vision)
    language = count_parameters(model.language)
    proprio = count_parameters(model.proprio)
    vlm = count_parameters(model.vlm)
    action_exp = count_parameters(model.action_expert)
    total = count_parameters(model)

    print(f"Vision encoder:        {vision:>12,}  ({vision / 1e6:.2f}M)")
    print(f"Language embedding:    {language:>12,}  ({language / 1e6:.2f}M)")
    print(f"Proprio encoder:       {proprio:>12,}  ({proprio / 1e6:.2f}M)")
    print(f"VLM backbone:          {vlm:>12,}  ({vlm / 1e6:.2f}M)")
    print(f"Action expert:         {action_exp:>12,}  ({action_exp / 1e6:.2f}M)")
    print("-" * 60)
    print(f"Total:                 {total:>12,}  ({total / 1e6:.2f}M)")
    print("=" * 60)
    print(f"  horizon H = {model.horizon} timesteps")
    print(f"  action_dim = {model.action_dim}")


def demo_forward():
    """One training loss + one inference pass with shape trace."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (VLA-S)")
    print("=" * 60)

    model = vla_small()
    model.eval()

    B = 2
    N_cam = 2
    H_img = W_img = 224
    T_text = 16

    batch = {
        "images": torch.randn(B, N_cam, 3, H_img, W_img),
        "token_ids": torch.randint(0, 32000, (B, T_text)),
        "proprio": torch.randn(B, 14),
        "action": torch.randn(B, model.horizon, model.action_dim),
    }

    print(f"\nInputs:")
    print(f"  Images:    {list(batch['images'].shape)}")
    print(f"  Tokens:    {list(batch['token_ids'].shape)}")
    print(f"  Proprio:   {list(batch['proprio'].shape)}")
    print(f"  Action:    {list(batch['action'].shape)}  (training target)")

    # 1. VLM encoding
    with torch.no_grad():
        vlm_tokens = model.encode_vlm(batch["images"], batch["token_ids"], batch["proprio"])
        print(f"\n1. VLM context tokens:  {list(vlm_tokens.shape)}")

    # 2. Training loss
    loss = model.compute_loss(batch)
    print(f"\n2. Flow matching loss: {loss.item():.4f}")

    # 3. Inference (generate an action chunk from noise)
    with torch.no_grad():
        action_chunk = model.generate_action(
            batch["images"], batch["token_ids"], batch["proprio"], n_steps=10
        )
    print(f"\n3. Generated action chunk: {list(action_chunk.shape)}  (10 Euler steps)")

    print("\n" + "=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VLA (Pi0-style) Reference Implementation")
    print("=" * 60)

    print("\n--- VLA-S ---")
    print_model_summary(vla_small())

    print("\n--- VLA-B ---")
    print_model_summary(vla_base())

    try:
        demo_forward()
    except Exception as e:
        print(f"\n(Skipping demo: {e})")

    # Quick comparison
    print("\n" + "=" * 60)
    print("Config comparison")
    print("=" * 60)
    for name, fn in [("VLA-S", vla_small), ("VLA-B", vla_base)]:
        m = fn()
        print(f"{name:<8} d_model={m.vision.d_model:<4}  "
              f"action_dim={m.action_dim:<3}  H={m.horizon:<3}  "
              f"params={count_parameters(m)/1e6:>6.1f}M")
