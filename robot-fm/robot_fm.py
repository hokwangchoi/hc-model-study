"""
Robot Foundation Model Reference Implementation (GR00T N1-style)

A dual-system, multi-embodiment robot policy (GR00T N1, arXiv 2503.14734):
    - System 2 (slow, ~10 Hz in the paper): a VLM reads camera images + the
      instruction. GR00T takes the hidden states of a MIDDLE LLM layer (the 12th)
      instead of the last one; the paper reports this is both faster and better.
    - System 1 (fast, motor actions at ~120 Hz in the paper): a DiT action head
      trained with flow matching. Its input tokens are [state, noisy action chunk];
      its blocks ALTERNATE cross-attention to the VLM tokens and self-attention
      over the state/action tokens, with adaptive LayerNorm conditioned on the
      flow timestep. 4 denoising steps produce a 16-action chunk (N1).
    - Heterogeneous I/O: the state encoder, action encoder and action decoder are
      EMBODIMENT-SPECIFIC MLPs. Isaac-GR00T implements them as CategorySpecificLinear:
      one weight tensor [num_embodiments, in, out], gathered per sample and applied
      with a batched matmul. States/actions are zero-padded to a max dimension and the
      loss is masked, so one batch can mix robots.

Usage:
    python robot_fm.py

Flow matching convention (same as the VLA module and Isaac-GR00T's code):
    x_t = (1 - t) * noise + t * action,  target v = action - noise,
    sample: start from noise at t = 0, x += v * dt for K steps.
    t ~ 0.999 * (1 - Beta(1.5, 1)); the DiT sees t discretised into 1000 buckets.

Model configs (toy, CPU friendly):
    - RFM-S: d_vlm=256, d_action=192
    - RFM-B: d_vlm=384, d_action=256
Real GR00T N1-2B: 2.2B parameters, 1.34B of them in the VLM (Eagle-2: SigLIP-2 +
SmolLM2); 64 image tokens per 224x224 frame; a 16-action chunk takes 63.9 ms on an
L40 in bf16 (paper). Later releases (N1.5, N1.6, N1.7) changed the backbone and
horizon; see the guide's "As of Oct 2026" box.

Simplifications:
- Toy ViT + embedding-table "VLM"; a few transformer layers stand in for the first
  12 layers of the LLM.
- 4 toy embodiments with made-up state/action sizes (GR00T N1 itself was evaluated
  on the GR-1 humanoid, Franka arms and bimanual arms).
- No action-chunk normalisation, no state dropout, no FLARE / future tokens.
"""
import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Embodiment registry (toy numbers) and padding
# ============================================================================

EMBODIMENTS: Dict[str, Dict[str, int]] = {
    "franka_arm":    {"id": 0, "state_dim": 14, "action_dim": 7},    # 7 joints (pos + vel) -> 7 commands
    "xarm_bimanual": {"id": 1, "state_dim": 28, "action_dim": 14},   # two arms
    "unitree_h1":    {"id": 2, "state_dim": 50, "action_dim": 22},   # humanoid upper + lower body (toy)
    "unitree_go2":   {"id": 3, "state_dim": 36, "action_dim": 12},   # quadruped, 3 joints x 4 legs (toy)
}
MAX_STATE_DIM = 64
MAX_ACTION_DIM = 32
NUM_EMBODIMENT_SLOTS = 8          # weight slots; GR00T N1.7 reserves 32


def pad_to(x: torch.Tensor, size: int) -> torch.Tensor:
    """Zero-pad the last dimension to `size`."""
    return F.pad(x, (0, size - x.shape[-1]))


def action_mask(embodiment_ids: torch.Tensor, horizon: int,
                embodiments: Dict[str, Dict[str, int]] = EMBODIMENTS) -> torch.Tensor:
    """[B, H, MAX_ACTION_DIM] mask: 1 for this robot's real action dims, 0 for padding."""
    dims = {spec["id"]: spec["action_dim"] for spec in embodiments.values()}
    lens = torch.tensor([dims[int(i)] for i in embodiment_ids])
    m = (torch.arange(MAX_ACTION_DIM)[None, :] < lens[:, None]).float()
    return m[:, None, :].expand(-1, horizon, -1)


# ============================================================================
# 2. Embodiment-specific layers (Isaac-GR00T's CategorySpecificLinear / MLP)
# ============================================================================

class CategorySpecificLinear(nn.Module):
    """Linear layer with one weight matrix PER EMBODIMENT, stored as a single tensor.
    forward gathers each sample's matrix and uses one batched matmul, so a batch can
    mix embodiments without a Python loop or a ModuleDict lookup per robot."""
    def __init__(self, num_categories: int, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, in_dim, out_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, out_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x [B, T, in], cat_ids [B] -> [B, T, out]
        return torch.bmm(x, self.W[cat_ids]) + self.b[cat_ids].unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, in_dim, hidden, out_dim):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, in_dim, hidden)
        self.layer2 = CategorySpecificLinear(num_categories, hidden, out_dim)

    def forward(self, x, cat_ids):
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


def sinusoidal(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sin-cos features of a (bucketed) timestep, any leading shape -> [..., dim]."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[..., None] * freqs
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class MultiEmbodimentActionEncoder(nn.Module):
    """GR00T's action encoder: W1(noisy action) concatenated with a sinusoidal
    embedding of the flow timestep, then W2 + swish, then W3 — all embodiment-specific."""
    def __init__(self, action_dim: int, hidden: int, num_categories: int):
        super().__init__()
        self.hidden = hidden
        self.W1 = CategorySpecificLinear(num_categories, action_dim, hidden)
        self.W2 = CategorySpecificLinear(num_categories, 2 * hidden, hidden)
        self.W3 = CategorySpecificLinear(num_categories, hidden, hidden)

    def forward(self, actions, t_bucket, cat_ids):
        B, T, _ = actions.shape
        tau = sinusoidal(t_bucket[:, None].expand(B, T), self.hidden)
        x = torch.cat([self.W1(actions, cat_ids), tau], dim=-1)
        return self.W3(F.silu(self.W2(x, cat_ids)), cat_ids)


# ============================================================================
# 3. System 2 — the VLM (images + language only)
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, int(d * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(d * mlp_ratio), d))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class VisionEncoder(nn.Module):
    """Stub vision tower: 224x224 image -> 64 tokens (GR00T N1 also uses 64 tokens per
    frame, via pixel shuffle on SigLIP-2 features; we use 28x28 patches)."""
    def __init__(self, img_size=224, patch_size=28, d=384, depth=2, n_heads=6):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, d, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)

    def forward(self, images):
        B, N_cam = images.shape[:2]
        x = self.patch_embed(images.flatten(0, 1)).flatten(2).transpose(1, 2) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x).reshape(B, N_cam * self.n_patches, -1)


class System2VLM(nn.Module):
    """Images + instruction -> VL tokens. `select_layer` LLM layers are built and the
    output of the last one is used (GR00T: the 12th layer; layers above it are never
    run). No robot state goes in here: in GR00T the state enters System 1."""
    def __init__(self, d=384, n_heads=6, vision_depth=2, select_layer=4, vocab=32000):
        super().__init__()
        self.vision = VisionEncoder(d=d, depth=vision_depth, n_heads=n_heads)
        self.tok = nn.Embedding(vocab, d)
        self.llm_layers = nn.ModuleList([TransformerBlock(d, n_heads) for _ in range(select_layer)])
        self.norm = nn.LayerNorm(d)                 # GR00T applies a LayerNorm ("vlln") to these features

    def forward(self, images, token_ids):
        x = torch.cat([self.vision(images), self.tok(token_ids)], dim=1)
        for blk in self.llm_layers:
            x = blk(x)
        return self.norm(x)


# ============================================================================
# 4. System 1 — DiT action head (alternating cross/self attention, AdaLN on t)
# ============================================================================

class AdaLayerNorm(nn.Module):
    """LayerNorm whose scale/shift come from the timestep embedding (no gate):
    norm(x) * (1 + scale) + shift — the 'ada_norm' variant GR00T's DiT uses."""
    def __init__(self, d):
        super().__init__()
        self.linear = nn.Linear(d, 2 * d)
        self.norm = nn.LayerNorm(d, elementwise_affine=False)

    def forward(self, x, temb):
        scale, shift = self.linear(F.silu(temb)).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    """One attention (cross-attn to VL tokens OR self-attn) + one FFN."""
    def __init__(self, d, n_heads, d_ctx: Optional[int]):
        super().__init__()
        self.is_cross = d_ctx is not None
        self.norm1 = AdaLayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True,
                                          kdim=d_ctx, vdim=d_ctx)
        self.norm3 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"),
                                nn.Linear(4 * d, d))

    def forward(self, x, temb, ctx=None):
        h = self.norm1(x, temb)
        kv = ctx if self.is_cross else h
        x = x + self.attn(h, kv, kv, need_weights=False)[0]
        return x + self.ff(self.norm3(x))


class System1ActionHead(nn.Module):
    def __init__(self, d_action=256, d_vlm=384, n_heads=4, depth=4, horizon=16,
                 num_embodiments=NUM_EMBODIMENT_SLOTS, n_buckets=1000):
        super().__init__()
        self.horizon, self.d, self.n_buckets = horizon, d_action, n_buckets
        self.state_encoder = CategorySpecificMLP(num_embodiments, MAX_STATE_DIM, d_action, d_action)
        self.action_encoder = MultiEmbodimentActionEncoder(MAX_ACTION_DIM, d_action, num_embodiments)
        self.pos_embed = nn.Embedding(horizon, d_action)
        self.time_mlp = nn.Sequential(nn.Linear(d_action, d_action), nn.SiLU(),
                                      nn.Linear(d_action, d_action))
        # even blocks: cross-attention to the VLM tokens, odd blocks: self-attention
        self.blocks = nn.ModuleList([DiTBlock(d_action, n_heads, d_vlm if i % 2 == 0 else None)
                                     for i in range(depth)])
        self.norm_out = nn.LayerNorm(d_action, elementwise_affine=False)
        self.proj_out_1 = nn.Linear(d_action, 2 * d_action)
        self.proj_out_2 = nn.Linear(d_action, d_action)
        self.action_decoder = CategorySpecificMLP(num_embodiments, d_action, d_action, MAX_ACTION_DIM)

    def forward(self, x_t, t, state, vl_tokens, emb_ids):
        """x_t [B, H, MAX_ACTION_DIM] (padded), t [B] in [0, 1), state [B, MAX_STATE_DIM],
        vl_tokens [B, N_vl, d_vlm], emb_ids [B]  ->  velocity [B, H, MAX_ACTION_DIM]."""
        t_bucket = (t * self.n_buckets).long()
        s = self.state_encoder(state[:, None], emb_ids)                      # [B, 1, d]
        a = self.action_encoder(x_t, t_bucket, emb_ids) + self.pos_embed.weight[None]
        h = torch.cat([s, a], dim=1)                                         # [B, 1 + H, d]
        temb = self.time_mlp(sinusoidal(t_bucket, self.d))
        for blk in self.blocks:
            h = blk(h, temb, vl_tokens)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
        h = self.proj_out_2(self.norm_out(h) * (1 + scale[:, None]) + shift[:, None])
        return self.action_decoder(h, emb_ids)[:, -self.horizon:]           # action tokens only


# ============================================================================
# 5. The full model
# ============================================================================

def sample_flow_time(batch: int, s: float = 0.999) -> torch.Tensor:
    """t = s * (1 - u), u ~ Beta(1.5, 1): favours the noisy end (t near 0)."""
    return s * (1.0 - torch.distributions.Beta(1.5, 1.0).sample((batch,)))


class RobotFoundationModel(nn.Module):
    def __init__(self, embodiments=EMBODIMENTS, horizon=16, d_vlm=384, vlm_heads=6,
                 vision_depth=2, select_layer=4, d_action=256, action_heads=4,
                 action_depth=4):
        super().__init__()
        self.embodiments, self.horizon = embodiments, horizon
        self.system2 = System2VLM(d_vlm, vlm_heads, vision_depth, select_layer)
        self.system1 = System1ActionHead(d_action, d_vlm, action_heads, action_depth, horizon)

    def encode_system2(self, images, token_ids):
        return self.system2(images, token_ids)                               # [B, N_vl, d_vlm]

    def compute_loss(self, batch):
        """batch: images, token_ids, state [B, MAX_STATE_DIM] (padded),
        action [B, H, MAX_ACTION_DIM] (padded), embodiment_id [B] — may mix robots."""
        a, ids = batch["action"], batch["embodiment_id"]
        B = a.shape[0]
        vl = self.encode_system2(batch["images"], batch["token_ids"])
        t = sample_flow_time(B).to(a.device)
        noise = torch.randn_like(a)
        tt = t.view(B, 1, 1)
        x_t = (1 - tt) * noise + tt * a
        v_target = a - noise
        v_pred = self.system1(x_t, t, batch["state"], vl, ids)
        m = action_mask(ids, self.horizon, self.embodiments).to(a.device)
        return ((v_pred - v_target) ** 2 * m).sum() / m.sum()                 # padded dims ignored

    @torch.no_grad()
    def generate_action(self, images, token_ids, state, embodiment_ids, n_steps: int = 4,
                        vl_tokens: Optional[torch.Tensor] = None):
        """System 2 once (or pass cached vl_tokens), then n_steps Euler steps of System 1.
        Returns PADDED actions [B, H, MAX_ACTION_DIM]; use unpad() per robot."""
        B = state.shape[0]
        if vl_tokens is None:
            vl_tokens = self.encode_system2(images, token_ids)
        x = torch.randn(B, self.horizon, MAX_ACTION_DIM, device=state.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=state.device)
            x = x + self.system1(x, t, state, vl_tokens, embodiment_ids) * dt
        return x

    def unpad(self, actions: torch.Tensor, name: str) -> torch.Tensor:
        return actions[..., :self.embodiments[name]["action_dim"]]


# ============================================================================
# Configs
# ============================================================================

def rfm_small() -> RobotFoundationModel:
    return RobotFoundationModel(d_vlm=256, vlm_heads=4, vision_depth=2, select_layer=3,
                                d_action=192, action_heads=4, action_depth=4)


def rfm_base() -> RobotFoundationModel:
    return RobotFoundationModel(d_vlm=384, vlm_heads=6, vision_depth=3, select_layer=4,
                                d_action=256, action_heads=4, action_depth=6)


# ============================================================================
# Analysis — the guide's calculator implements exactly rfm_cost()
# ============================================================================

def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def per_embodiment_params(d_action: int, max_state: int = MAX_STATE_DIM,
                          max_action: int = MAX_ACTION_DIM) -> int:
    """Parameters owned by ONE embodiment slot: state encoder MLP, action encoder
    (W1, W2, W3) and action decoder MLP (weights + biases)."""
    d = d_action
    state_enc = (max_state * d + d) + (d * d + d)
    action_enc = (max_action * d + d) + (2 * d * d + d) + (d * d + d)
    action_dec = (d * d + d) + (d * max_action + max_action)
    return state_enc + action_enc + action_dec


def rfm_cost(n_cams: int = 2, img_tokens: int = 64, text_tokens: int = 12,
             d_vlm: int = 384, vlm_layers: int = 4, vision_layers: int = 3,
             d_action: int = 256, dit_depth: int = 6, horizon: int = 16,
             n_steps: int = 4, n_embodiments: int = 4, bytes_per_param: float = 2.0,
             bw_gbs: float = 204.8, tflops: float = 42.5) -> Dict[str, float]:
    """Analytic cost of one action chunk for this file's architecture (defaults =
    rfm_base with 2 cameras). Parameter counts are exact for the transformer layers
    (embedding tables and the patch embedding are excluded). FLOPs = 2 x weight-matrix
    params x tokens + attention (4 x Tq x Tk x d per layer)."""
    T_img, T_vl, T_s = n_cams * img_tokens, n_cams * img_tokens + text_tokens, 1 + horizon
    d, dv = d_action, d_vlm
    p_tblock = 12 * dv * dv + 13 * dv                                 # System 2 transformer layer
    p_vis_layers, p_llm_layers = vision_layers * p_tblock, vlm_layers * p_tblock
    n_cross, n_self = (dit_depth + 1) // 2, dit_depth // 2            # blocks alternate cross / self
    p_self_block = 14 * d * d + 13 * d                                # AdaLN + self-attn + FFN
    p_cross_block = 12 * d * d + 2 * d * dv + 13 * d                  # AdaLN + cross-attn + FFN
    p_dit = n_cross * p_cross_block + n_self * p_self_block + horizon * d + 5 * d * d + 5 * d
    p_emb = per_embodiment_params(d)

    s2_flops = (2 * 12 * dv * dv * (vision_layers * T_img + vlm_layers * T_vl)
                + vision_layers * n_cams * 4 * img_tokens * img_tokens * dv
                + vlm_layers * 4 * T_vl * T_vl * dv)
    kv_proj_flops = n_cross * 2 * (2 * T_vl * dv * d)                 # K and V of the VL tokens
    mat_s1 = n_cross * 12 * d * d + n_self * 14 * d * d               # matrices that see the 1+H tokens
    step_flops = (2 * mat_s1 * T_s + kv_proj_flops
                  + n_cross * 4 * T_s * T_vl * d + n_self * 4 * T_s * T_s * d
                  + 2 * p_emb * T_s)                                   # this robot's encoders/decoder
    s2_bytes = (p_vis_layers + p_llm_layers) * bytes_per_param
    step_bytes = (p_dit + p_emb) * bytes_per_param
    peak, bw = tflops * 1e12, bw_gbs * 1e9
    t_s2 = max(s2_flops / peak, s2_bytes / bw)
    t_step = max(step_flops / peak, step_bytes / bw)
    total = s2_flops + n_steps * step_flops
    shared = p_vis_layers + p_llm_layers + p_dit
    return {
        "T_vl": T_vl, "T_s1": T_s,
        "params_s2_layers": p_vis_layers + p_llm_layers, "params_dit": p_dit,
        "params_per_embodiment": p_emb, "params_all_embodiments": n_embodiments * p_emb,
        "shared_fraction": shared / (shared + n_embodiments * p_emb),
        "s2_flops": s2_flops, "step_flops": step_flops, "s1_flops": n_steps * step_flops,
        "total_flops": total,
        "rerun_s2_every_step_flops": n_steps * (s2_flops + step_flops),
        "caching_speedup": n_steps * (s2_flops + step_flops) / total,
        "kv_proj_fraction_of_step": kv_proj_flops / step_flops,
        "t_s2_ms": 1e3 * t_s2, "t_step_ms": 1e3 * t_step,
        "t_chunk_ms": 1e3 * (t_s2 + n_steps * t_step),
    }


def print_model_summary(model: RobotFoundationModel, name: str):
    s2 = count_parameters(model.system2)
    s1 = model.system1
    per_emb = sum(count_parameters(m) for m in (s1.state_encoder, s1.action_encoder, s1.action_decoder))
    shared_s1 = count_parameters(s1) - per_emb
    slots = NUM_EMBODIMENT_SLOTS
    total = count_parameters(model)
    print("=" * 70)
    print(f"{name}: GR00T N1-style robot foundation model")
    print("=" * 70)
    print(f"System 2 (VLM, runs once per chunk):          {s2:>12,}")
    print(f"System 1 shared DiT + time/pos/out layers:    {shared_s1:>12,}")
    print(f"Embodiment-specific encoders/decoder ({slots} slots): {per_emb:>10,}"
          f"  = {per_emb // slots:,} per robot")
    print("-" * 70)
    print(f"Total:                                        {total:>12,}  ({total / 1e6:.2f}M)")
    print(f"Horizon H = {model.horizon}, padded dims: state {MAX_STATE_DIM}, action {MAX_ACTION_DIM}")


def make_batch(names: List[str], horizon: int, n_cams: int = 2, text_len: int = 12):
    """A mixed-embodiment batch with padded state/action and embodiment ids."""
    B = len(names)
    ids = torch.tensor([EMBODIMENTS[n]["id"] for n in names])
    state = torch.stack([pad_to(torch.randn(EMBODIMENTS[n]["state_dim"]), MAX_STATE_DIM) for n in names])
    action = torch.stack([pad_to(torch.randn(horizon, EMBODIMENTS[n]["action_dim"]), MAX_ACTION_DIM)
                          for n in names])
    return {"images": torch.randn(B, n_cams, 3, 224, 224),
            "token_ids": torch.randint(0, 32000, (B, text_len)),
            "state": state, "action": action, "embodiment_id": ids}


def demo():
    print("\n" + "=" * 70)
    print("Mixed-embodiment demo (RFM-S): ONE batch, four different robots")
    print("=" * 70)
    torch.manual_seed(0)
    model = rfm_small().eval()
    names = list(EMBODIMENTS)
    batch = make_batch(names, model.horizon)
    loss = model.compute_loss(batch)
    print(f"Training loss on a mixed batch (padding masked): {loss.item():.3f}")
    with torch.no_grad():
        vl = model.encode_system2(batch["images"], batch["token_ids"])
        print(f"System 2 tokens: {list(vl.shape)}  (2 cameras x 64 + 12 text)")
        acts = model.generate_action(None, None, batch["state"], batch["embodiment_id"],
                                     n_steps=4, vl_tokens=vl)
    for i, n in enumerate(names):
        a = model.unpad(acts[i:i + 1], n)
        print(f"  {n:<14} -> action chunk {list(a.shape)}")
    print("Same weights, one forward pass, four robots: only the gathered per-robot")
    print("encoder/decoder matrices differ (CategorySpecificLinear).")


if __name__ == "__main__":
    for nm, fn in [("RFM-S", rfm_small), ("RFM-B", rfm_base)]:
        print_model_summary(fn(), nm)
    demo()
    print("\n" + "=" * 70)
    print("Cost model (rfm_base, 2 cameras, 4 denoising steps) — the guide's calculator")
    print("=" * 70)
    c = rfm_cost()
    print(f"System 2: {c['T_vl']} tokens, {c['s2_flops'] / 1e9:.2f} GFLOPs, once per chunk")
    print(f"System 1: {c['T_s1']} tokens, {c['step_flops'] / 1e9:.3f} GFLOPs per step x 4")
    print(f"Caching System 2 instead of re-running it every step: {c['caching_speedup']:.2f}x fewer FLOPs")
    print(f"K/V projection of VL tokens = {100 * c['kv_proj_fraction_of_step']:.0f}% of a System 1 step "
          f"(cacheable across steps)")
    print(f"Per-embodiment parameters: {c['params_per_embodiment']:,}; shared fraction with 4 robots: "
          f"{100 * c['shared_fraction']:.1f}%")
