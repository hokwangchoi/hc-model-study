"""
Robot Foundation Model Reference Implementation (GR00T N1-style)

A dual-system architecture for multi-embodiment robot control:
    - System 2 (slow, ~10 Hz): VLM backbone — scene understanding + language
    - System 1 (fast, ~120 Hz): DiT-style diffusion action head with
      AdaLN conditioning, cross-attending to System 2's output

Plus heterogeneous I/O: per-embodiment proprio encoders and action heads so the
same core model can drive Franka arms, bimanual setups, quadrupeds, and
humanoids, transferring features across embodiments.

Usage:
    python robot_fm.py

Model configs:
    - RFM-S: ~16M params (toy backbones, small action head)
    - RFM-B: ~34M params (toy backbones scaled up)

Real GR00T N1 is ~2B total (Eagle-2 VLM backbone ~1.5B + DiT action head ~300M).
The toy numbers here are dominated by the 32k × d language-embedding table
(~12M at d=384); the transformer blocks themselves are small in this reference.

KEY INSIGHTS this file teaches:

1. DUAL-RATE DESIGN. Scene understanding changes slowly; motor commands
   change fast. GR00T explicitly decouples these — VLM runs at ~10 Hz
   ("big tick"), action head runs at ~120 Hz ("small tick"). Amortizes
   expensive VLM compute across many fast action steps.

2. HETEROGENEOUS I/O. Different robots have different proprio dims (14
   for a 7-DOF arm, 50+ for a humanoid) and different action dims. Each
   embodiment gets its OWN proprio encoder and action head, but they
   all share the same VLM backbone and diffusion core. This lets us
   train on cross-embodiment data while respecting each robot's physical
   action space.

3. EMBODIMENT TOKENS. A learnable "who am I?" embedding per embodiment
   is injected into the input stream. Lets the same VLM produce
   embodiment-conditional context ("if this is a humanoid, reason about
   bipedal balance; if this is a quadruped, gait coordination").

4. DiT-STYLE ACTION HEAD. We reuse the AdaLN-Zero conditioning pattern
   from our diffusion module. The "condition" here is (flow time,
   VLM context summary, embodiment token). This is the architectural
   bridge from "image diffusion" to "action diffusion".

Simplifications in this file:
- Vision encoder is a stub ViT, not Eagle-2
- Language model is just a token embedding table
- We use flow matching (from VLA) rather than full DDPM
- Only 4 toy embodiments (Franka, xArm bimanual, H1 humanoid, Go2 quadruped)
- No real robot kinematics / URDFs
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Embodiment registry
# ============================================================================

EMBODIMENTS: Dict[str, Dict[str, int]] = {
    # 7-DOF robot arm + gripper. Proprio = joint pos (7) + joint vel (7).
    "franka_arm":    {"proprio_dim": 14, "action_dim": 7},
    # Two 7-DOF arms side-by-side.
    "xarm_bimanual": {"proprio_dim": 28, "action_dim": 14},
    # Humanoid — many more DOFs.
    "unitree_h1":    {"proprio_dim": 50, "action_dim": 22},
    # Quadruped — 3 DOF per leg × 4 legs = 12 action dims.
    "unitree_go2":   {"proprio_dim": 36, "action_dim": 12},
}


# ============================================================================
# 2. Heterogeneous I/O — per-embodiment routing
# ============================================================================

class HeterogeneousProprioEncoder(nn.Module):
    """
    One MLP per embodiment. Routes the forward call by embodiment name
    to project variable-size proprio vectors into the shared d_model space.

    At training time each batch contains samples from one embodiment (or
    you can also mix embodiments within a batch — then you'd route per-sample).
    """
    def __init__(self, embodiments: Dict[str, Dict], d_model: int):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(spec["proprio_dim"], d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for name, spec in embodiments.items()
        })

    def forward(self, proprio: torch.Tensor, embodiment: str) -> torch.Tensor:
        """proprio: [B, proprio_dim_for_this_embodiment] → [B, 1, d_model]"""
        return self.encoders[embodiment](proprio).unsqueeze(1)


class HeterogeneousActionHead(nn.Module):
    """
    Per-embodiment projections between the shared action-expert dim
    (d_action) and each embodiment's actual action_dim.

    - in_proj:  action_dim → d_action (embeds noisy actions during training)
    - out_proj: d_action → action_dim (emits predicted velocity)
    """
    def __init__(self, embodiments: Dict[str, Dict], d_action: int):
        super().__init__()
        self.in_proj = nn.ModuleDict({
            name: nn.Linear(spec["action_dim"], d_action)
            for name, spec in embodiments.items()
        })
        self.out_proj = nn.ModuleDict({
            name: nn.Linear(d_action, spec["action_dim"])
            for name, spec in embodiments.items()
        })

    def encode(self, action: torch.Tensor, embodiment: str) -> torch.Tensor:
        return self.in_proj[embodiment](action)

    def decode(self, x: torch.Tensor, embodiment: str) -> torch.Tensor:
        return self.out_proj[embodiment](x)


class EmbodimentEmbedding(nn.Module):
    """Learnable [d_model] embedding per embodiment — a 'who am I?' token
    added to the VLM input stream. Conditions downstream reasoning on
    morphology."""
    def __init__(self, embodiments: Dict[str, Dict], d_model: int):
        super().__init__()
        self._names = list(embodiments.keys())
        self._index = {n: i for i, n in enumerate(self._names)}
        self.embed = nn.Embedding(len(embodiments), d_model)

    def forward(self, embodiment: str, batch_size: int,
                device: torch.device) -> torch.Tensor:
        idx = torch.full((batch_size,), self._index[embodiment],
                         device=device, dtype=torch.long)
        return self.embed(idx).unsqueeze(1)                         # [B, 1, d]


# ============================================================================
# 3. Shared building blocks
# ============================================================================

class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block (same as VLA). Used in VLM backbone
    and vision encoder."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm1(x)
        attn_out, _ = self.attn(q, q, q)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    """Stub ViT-style encoder. Real GR00T uses Eagle-2 VLM's vision tower (~400M)."""
    def __init__(self, img_size: int = 224, patch_size: int = 14,
                 d_model: int = 384, depth: int = 4, n_heads: int = 6):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, N_cam, C, H, W = images.shape
        x = self.patch_embed(images.reshape(B * N_cam, C, H, W))
        x = x.flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).reshape(B, N_cam * self.n_patches, -1)


class LanguageEmbedding(nn.Module):
    """Stub language side — embedding table only. Real GR00T uses Eagle-2's LM."""
    def __init__(self, vocab: int = 32000, d_model: int = 384, max_len: int = 128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok(ids) + self.pos[:, :ids.size(1)]


# ============================================================================
# 4. System 2 — VLM backbone (slow, runs once per "big tick")
# ============================================================================

class System2_VLM(nn.Module):
    """
    Processes vision + language + proprio + embodiment token into shared
    context. Slow and expensive; the rest of the model cross-attends to
    its output without re-running it on every action step.
    """
    def __init__(self, d_model: int = 384, depth: int = 4, n_heads: int = 6):
        super().__init__()
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        vision: torch.Tensor,       # [B, N_v, d]
        language: torch.Tensor,     # [B, N_l, d]
        proprio: torch.Tensor,      # [B, 1, d]
        embodiment: torch.Tensor,   # [B, 1, d]
    ) -> torch.Tensor:
        # Concatenate all modality tokens. Embodiment token goes first so
        # every subsequent attention block can easily route on it.
        x = torch.cat([embodiment, vision, language, proprio], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ============================================================================
# 5. System 1 — DiT-style action head (fast, runs many times per big tick)
# ============================================================================

def sinusoidal_time_embedding(t: torch.Tensor, d_model: int) -> torch.Tensor:
    """Scalar flow-time t ∈ [0, 1] → sinusoidal [d_model] embedding."""
    half = d_model // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class AdaLNModulation(nn.Module):
    """
    AdaLN-Zero modulation — same pattern as our DiT diffusion model.
    Given condition c, predict 6 modulation scalars per token:
        (shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp)
    Zero-init → identity function at step 0 → stable training.
    """
    def __init__(self, d_model: int, d_cond: int):
        super().__init__()
        self.proj = nn.Linear(d_cond, 6 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor):
        return self.proj(cond).chunk(6, dim=-1)             # each: [B, d_model]


class System1Block(nn.Module):
    """
    One DiT-style block inside the action head.

    Three operations, each gated by AdaLN-Zero using (flow time + VLM summary):
        1. Self-attention over the H-step action chunk
        2. Cross-attention to VLM context tokens
        3. FFN
    """
    def __init__(self, d_action: int, d_vlm: int, d_cond: int, n_heads: int = 6):
        super().__init__()
        # AdaLN-Zero modulation from condition vector (flow time + VLM summary)
        self.mod = AdaLNModulation(d_action, d_cond)

        self.norm_sa = nn.LayerNorm(d_action, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(d_action, n_heads, batch_first=True)

        self.norm_ca = nn.LayerNorm(d_action, elementwise_affine=False)
        self.kv_proj = nn.Linear(d_vlm, d_action, bias=False)
        self.cross_attn = nn.MultiheadAttention(d_action, n_heads, batch_first=True)

        self.norm_mlp = nn.LayerNorm(d_action, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_action, 4 * d_action),
            nn.GELU(),
            nn.Linear(4 * d_action, d_action),
        )

    def forward(
        self,
        x: torch.Tensor,            # [B, H, d_action] — action tokens
        cond: torch.Tensor,         # [B, d_cond]
        vlm_tokens: torch.Tensor,   # [B, N_vlm, d_vlm]
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.mod(cond)

        def modulate(y, shift, scale):
            return y * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # 1. Self-attention (modulated)
        y = modulate(self.norm_sa(x), shift_a, scale_a)
        attn_out, _ = self.self_attn(y, y, y)
        x = x + gate_a.unsqueeze(1) * attn_out

        # 2. Cross-attention to VLM (no modulation here — keep simple)
        y = self.norm_ca(x)
        kv = self.kv_proj(vlm_tokens)
        ca_out, _ = self.cross_attn(y, kv, kv)
        x = x + ca_out

        # 3. FFN (modulated, gated)
        y = modulate(self.norm_mlp(x), shift_m, scale_m)
        x = x + gate_m.unsqueeze(1) * self.mlp(y)
        return x


class System1ActionHead(nn.Module):
    """
    Fast action head. Takes a noisy action chunk, flow time, and VLM
    context tokens. Predicts velocity (flow matching target).

    Heterogeneous in/out projections route the action_dim per embodiment.
    """
    def __init__(
        self,
        d_action: int = 256,
        d_vlm: int = 384,
        d_cond: int = 256,
        horizon: int = 50,
        depth: int = 4,
        n_heads: int = 4,
        embodiments: Dict[str, Dict] = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.d_action = d_action

        # Heterogeneous action projections (per embodiment)
        self.action_heads = HeterogeneousActionHead(embodiments, d_action)

        # Learnable positional embedding over the H-step action chunk
        self.action_pos = nn.Parameter(torch.zeros(1, horizon, d_action))
        nn.init.trunc_normal_(self.action_pos, std=0.02)

        # Flow time → conditioning vector
        self.time_embed = nn.Sequential(
            nn.Linear(d_cond, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond),
        )

        # VLM summary → conditioning vector (we use the embodiment token only)
        self.vlm_summary = nn.Linear(d_vlm, d_cond)

        self.blocks = nn.ModuleList([
            System1Block(d_action, d_vlm, d_cond, n_heads) for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(d_action)

    def forward(
        self,
        action_t: torch.Tensor,         # [B, H, action_dim_of_this_embodiment]
        flow_time: torch.Tensor,        # [B]
        vlm_tokens: torch.Tensor,       # [B, N_vlm, d_vlm]
        embodiment: str,
    ) -> torch.Tensor:
        B = action_t.size(0)

        # 1. Encode action into shared d_action (per-embodiment projection)
        x = self.action_heads.encode(action_t, embodiment)          # [B, H, d_action]
        x = x + self.action_pos

        # 2. Build conditioning vector: flow time + embodiment-token summary
        t_emb = self.time_embed(
            sinusoidal_time_embedding(flow_time, self.d_action)
        )                                                           # [B, d_cond]
        # The embodiment token is vlm_tokens[:, 0] (we put it first in System2)
        emb_summary = self.vlm_summary(vlm_tokens[:, 0])            # [B, d_cond]
        cond = t_emb + emb_summary

        # 3. DiT blocks
        for blk in self.blocks:
            x = blk(x, cond, vlm_tokens)
        x = self.final_norm(x)

        # 4. Decode back to this embodiment's action dim
        return self.action_heads.decode(x, embodiment)              # [B, H, action_dim]


# ============================================================================
# 6. Full model — GR00T-style robot foundation model
# ============================================================================

class RobotFoundationModel(nn.Module):
    """
    GR00T N1-style multi-embodiment robot foundation model.

    Two systems:
      - System 2: VLM backbone (vision + language + proprio + embodiment)
      - System 1: DiT-style action head (flow matching over action chunks)

    Per-embodiment routing:
      - Proprio encoders (one per embodiment)
      - Action heads (one per embodiment, in+out)
      - Embodiment tokens (learnable, one per embodiment)
    """
    def __init__(
        self,
        embodiments: Dict[str, Dict] = None,
        horizon: int = 50,
        # System 2 (VLM)
        img_size: int = 224, patch_size: int = 14,
        d_model: int = 384, vision_depth: int = 4, vision_heads: int = 6,
        vocab_size: int = 32000, vlm_depth: int = 4,
        # System 1 (action head)
        d_action: int = 256, action_depth: int = 4, action_heads: int = 4,
    ):
        super().__init__()
        if embodiments is None:
            embodiments = EMBODIMENTS
        self.embodiments = embodiments
        self.horizon = horizon

        # System 2 components
        self.vision = VisionEncoder(img_size, patch_size, d_model, vision_depth, vision_heads)
        self.language = LanguageEmbedding(vocab_size, d_model)
        self.proprio = HeterogeneousProprioEncoder(embodiments, d_model)
        self.embodiment_embed = EmbodimentEmbedding(embodiments, d_model)
        self.system2 = System2_VLM(d_model=d_model, depth=vlm_depth, n_heads=vision_heads)

        # System 1 component
        self.system1 = System1ActionHead(
            d_action=d_action, d_vlm=d_model, d_cond=d_action,
            horizon=horizon, depth=action_depth, n_heads=action_heads,
            embodiments=embodiments,
        )

    def encode_system2(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        embodiment: str,
    ) -> torch.Tensor:
        """Run the slow System 2 to get VLM context tokens."""
        B = images.size(0)
        vision = self.vision(images)
        language = self.language(token_ids)
        proprio_tok = self.proprio(proprio, embodiment)
        embodiment_tok = self.embodiment_embed(embodiment, B, images.device)
        return self.system2(vision, language, proprio_tok, embodiment_tok)

    def compute_loss(self, batch: dict) -> torch.Tensor:
        """
        Flow matching training loss. Batch must declare which embodiment it
        comes from.

        batch fields:
            images:     [B, N_cam, 3, H, W]
            token_ids:  [B, T]
            proprio:    [B, proprio_dim_for_this_embodiment]
            action:     [B, horizon, action_dim_for_this_embodiment]
            embodiment: str (same for all samples in this batch)
        """
        embodiment = batch["embodiment"]
        action = batch["action"]
        B = action.size(0)

        # 1. System 2 encoding (embodiment-aware)
        vlm_tokens = self.encode_system2(
            batch["images"], batch["token_ids"], batch["proprio"], embodiment
        )

        # 2. Flow matching — sample t, interpolate, predict velocity
        t = torch.rand(B, device=action.device)
        eps = torch.randn_like(action)
        x_t = (1 - t.view(B, 1, 1)) * eps + t.view(B, 1, 1) * action
        v_target = action - eps

        v_pred = self.system1(x_t, t, vlm_tokens, embodiment)
        return F.mse_loss(v_pred, v_target)

    @torch.no_grad()
    def generate_action(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        proprio: torch.Tensor,
        embodiment: str,
        n_steps: int = 10,
    ) -> torch.Tensor:
        """
        Inference. Encodes System 2 ONCE, then integrates the flow ODE through
        System 1 for n_steps to produce a clean action chunk.
        """
        B = images.size(0)
        action_dim = self.embodiments[embodiment]["action_dim"]

        vlm_tokens = self.encode_system2(images, token_ids, proprio, embodiment)

        x = torch.randn(B, self.horizon, action_dim, device=images.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=images.device)
            v = self.system1(x, t, vlm_tokens, embodiment)
            x = x + v * dt
        return x


# ============================================================================
# Model configs
# ============================================================================

def rfm_small() -> RobotFoundationModel:
    """RFM-S: ~16M params, tiny backbones."""
    return RobotFoundationModel(
        horizon=50,
        d_model=256, vision_depth=3, vlm_depth=3, vision_heads=4,
        d_action=192, action_depth=3, action_heads=4,
    )


def rfm_base() -> RobotFoundationModel:
    """RFM-B: ~34M params. Real GR00T N1 is ~2B with Eagle-2 backbone."""
    return RobotFoundationModel(
        horizon=50,
        d_model=384, vision_depth=4, vlm_depth=4, vision_heads=6,
        d_action=256, action_depth=4, action_heads=4,
    )


# ============================================================================
# Analysis
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: RobotFoundationModel):
    print("=" * 70)
    print("Robot Foundation Model Summary")
    print("=" * 70)

    vision = count_parameters(model.vision)
    language = count_parameters(model.language)
    proprio = count_parameters(model.proprio)
    embodiment_emb = count_parameters(model.embodiment_embed)
    system2 = count_parameters(model.system2)
    system1 = count_parameters(model.system1)
    total = count_parameters(model)

    print(f"System 2 (slow, runs 1× per big tick):")
    print(f"  Vision encoder:         {vision:>12,}  ({vision / 1e6:.2f}M)")
    print(f"  Language embedding:     {language:>12,}  ({language / 1e6:.2f}M)")
    print(f"  Proprio encoders (× {len(model.embodiments)}):"
          f"  {proprio:>12,}  ({proprio / 1e6:.2f}M)")
    print(f"  Embodiment embeddings:  {embodiment_emb:>12,}  "
          f"({embodiment_emb / 1e6:.2f}M)")
    print(f"  VLM backbone:           {system2:>12,}  ({system2 / 1e6:.2f}M)")
    print(f"System 1 (fast, runs 10× per big tick):")
    print(f"  DiT action head:        {system1:>12,}  ({system1 / 1e6:.2f}M)")
    print("-" * 70)
    print(f"Total:                    {total:>12,}  ({total / 1e6:.2f}M)")
    print("=" * 70)
    print(f"Embodiments supported: {list(model.embodiments.keys())}")
    print(f"Horizon: H = {model.horizon} timesteps")


def demo_multi_embodiment():
    """Demonstrate the same model driving different embodiments."""
    print("\n" + "=" * 70)
    print("Multi-Embodiment Forward Pass Demo (RFM-S)")
    print("=" * 70)

    model = rfm_small()
    model.eval()

    B, N_cam, H_img = 1, 2, 224
    T_text = 12

    for embodiment, spec in EMBODIMENTS.items():
        batch = {
            "images":     torch.randn(B, N_cam, 3, H_img, H_img),
            "token_ids":  torch.randint(0, 32000, (B, T_text)),
            "proprio":    torch.randn(B, spec["proprio_dim"]),
            "action":     torch.randn(B, model.horizon, spec["action_dim"]),
            "embodiment": embodiment,
        }

        loss = model.compute_loss(batch)

        with torch.no_grad():
            action_chunk = model.generate_action(
                batch["images"], batch["token_ids"], batch["proprio"],
                embodiment, n_steps=10,
            )

        print(f"\n  Embodiment: {embodiment:<18} "
              f"proprio [{spec['proprio_dim']:>2}]  "
              f"action [{spec['action_dim']:>2}]")
        print(f"    Training loss: {loss.item():.4f}")
        print(f"    Generated action chunk shape: {list(action_chunk.shape)}")

    print("\n" + "=" * 70)
    print("  Same model, same weights — driving 4 different robot embodiments.")
    print("  Only the per-embodiment proprio encoder / action head change.")
    print("=" * 70)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Robot Foundation Model (GR00T N1 style) Reference Implementation")
    print("=" * 70)

    print("\n--- RFM-S ---")
    print_model_summary(rfm_small())

    print("\n--- RFM-B ---")
    print_model_summary(rfm_base())

    try:
        demo_multi_embodiment()
    except Exception as e:
        print(f"\n(Skipping demo: {e})")

    print("\n" + "=" * 70)
    print("Config comparison")
    print("=" * 70)
    for name, fn in [("RFM-S", rfm_small), ("RFM-B", rfm_base)]:
        m = fn()
        p = count_parameters(m)
        print(f"  {name}  params = {p / 1e6:>6.1f}M  "
              f"embodiments = {len(m.embodiments)}  H = {m.horizon}")
