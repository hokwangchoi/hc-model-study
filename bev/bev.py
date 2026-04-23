"""
BEV (Bird's-Eye-View) Multi-View Tokenizer Reference Implementation
BEVFormer-style: learnable BEV queries cross-attend to multi-view camera
features via deformable attention, producing a dense BEV feature map.

Usage:
    python bev.py

Architecture:
    6 cameras [B, N_cam, 3, H, W]
      ↓ per-camera backbone (ResNet)
    Image features [B, N_cam, C, H', W']
                                    ↓
    Learnable BEV queries ─── spatial cross-attention ──→ BEV feature map
    [H_bev × W_bev, d]          (deformable, guided by              [B, H_bev, W_bev, d]
                                 camera projection)                          ↓
                                                                   Detection / segmentation
                                                                   heads (not shown)

Model configs (roughly):
    - BEV-S: ~6M params   (stub backbone + 100×100 BEV grid + 4 layers)
    - BEV-B: ~28M params  (stub backbone + 200×200 BEV grid + 6 layers)

    Note: ~70% of these counts are the learnable BEV query + positional
    embedding grid (huge: 200² × 256 × 2 = 20M for BEV-B). The image backbone
    here is a TOY CNN (~1M params). Real BEVFormer-base swaps in a ResNet-50
    or ResNet-101 backbone (+25–45M) for ~60–120M total params.

KEY INSIGHTS this file teaches:

1. BEV QUERIES ARE LEARNABLE. One trainable d-dim vector per BEV cell.
   A 200×200 grid at d=256 is 10.2M params just for queries. The queries
   themselves encode "what should I look for in this physical location"
   after training.

2. CAMERA PROJECTION IS PART OF THE MODEL. For each BEV cell (which has a
   known 3D position in ego frame), we project it through each camera's
   intrinsics/extrinsics to find WHERE in each image the BEV cell "shows
   up". These projected 2D points become the REFERENCE POINTS for
   deformable cross-attention.

3. DEFORMABLE ATTENTION MAKES IT TRACTABLE. Full cross-attention from
   every BEV query to every image pixel would be O(N_bev · N_cam · H · W).
   Deformable attention samples only K points per query per camera,
   reducing to O(N_bev · N_cam · K). Production uses K=4 to K=8.

4. OUTPUT IS A DENSE FEATURE MAP, not a token sequence or bounding boxes.
   Heads (detection, segmentation, motion forecasting) run on top.

Simplifications in this file:
- Image backbone is a stub CNN, not a real ResNet
- Deformable sampling uses F.grid_sample with predicted offsets (correct math,
  but the production version uses a fused CUDA kernel for speed)
- Temporal self-attention (which attends to previous BEV frame) is omitted
- Actual detection heads are omitted
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Image backbone — stub per-camera feature extractor
# ============================================================================

class ImageBackbone(nn.Module):
    """
    Toy CNN feature extractor. In production this is a ResNet-50 or ViT-Small
    with FPN-style multi-scale outputs.

    Input:  [B*N_cam, 3, H_img, W_img]
    Output: [B*N_cam, out_channels, H_img/stride, W_img/stride]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 256, stride: int = 16):
        super().__init__()
        # Downsample by factor `stride` through repeated conv+pool
        assert stride in (8, 16, 32), "stride must be 8, 16, or 32"
        n_downsample = int(math.log2(stride))
        chs = [in_channels, 64, 128, 256, out_channels]
        layers = []
        c_prev = in_channels
        for i in range(n_downsample):
            c_next = chs[min(i + 1, len(chs) - 1)]
            layers += [
                nn.Conv2d(c_prev, c_next, 3, stride=2, padding=1),
                nn.GroupNorm(32 if c_next >= 32 else 8, c_next),
                nn.ReLU(inplace=True),
            ]
            c_prev = c_next
        # Final projection to out_channels
        layers += [nn.Conv2d(c_prev, out_channels, 1), nn.GroupNorm(32, out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================================
# 2. Camera projection — 3D ego points → 2D image coords
# ============================================================================

def project_ego_to_image(
    points_3d: torch.Tensor,      # [N_pts, 3] in ego frame (car = origin)
    intrinsics: torch.Tensor,     # [B, N_cam, 3, 3]
    ego_to_cam: torch.Tensor,     # [B, N_cam, 4, 4] — ego-frame → camera-frame
    img_h: int,
    img_w: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points (in the car's ego frame) into each camera's 2D image plane.

    Returns:
        uv_norm:  [B, N_cam, N_pts, 2] — pixel coords normalized to [-1, 1] (for grid_sample)
        valid:    [B, N_cam, N_pts]    — bool mask: point is in front of camera AND in image

    The math:
      1. Ego → camera frame: p_cam = R · p_ego + t, via ego_to_cam
      2. Camera → pixel: u = fx·X/Z + cx, v = fy·Y/Z + cy, via intrinsics
      3. Filter points with Z <= 0 (behind camera) or outside image bounds
    """
    B, N_cam, _, _ = intrinsics.shape
    N_pts = points_3d.shape[0]

    # Homogeneous 3D points: [N_pts, 4]
    ones = torch.ones(N_pts, 1, device=points_3d.device, dtype=points_3d.dtype)
    p_ego_h = torch.cat([points_3d, ones], dim=-1)              # [N_pts, 4]

    # Ego → camera (batched matmul): [B, N_cam, 4, 4] @ [4, N_pts]
    p_ego_h_exp = p_ego_h.T.unsqueeze(0).unsqueeze(0)           # [1, 1, 4, N_pts]
    p_cam_h = ego_to_cam @ p_ego_h_exp                          # [B, N_cam, 4, N_pts]
    p_cam = p_cam_h[..., :3, :]                                 # [B, N_cam, 3, N_pts]

    # Project: x_pixel = K · p_cam / z_cam
    pixels_h = intrinsics @ p_cam                               # [B, N_cam, 3, N_pts]
    z = pixels_h[..., 2:3, :].clamp(min=1e-5)                   # [B, N_cam, 1, N_pts]
    uv = pixels_h[..., :2, :] / z                               # [B, N_cam, 2, N_pts]
    uv = uv.transpose(-1, -2)                                   # [B, N_cam, N_pts, 2]

    # Validity mask: positive depth AND in image bounds
    z_pos = p_cam_h[..., 2, :] > 0                              # [B, N_cam, N_pts]
    u, v = uv[..., 0], uv[..., 1]
    u_ok = (u >= 0) & (u < img_w)
    v_ok = (v >= 0) & (v < img_h)
    valid = z_pos & u_ok & v_ok                                 # [B, N_cam, N_pts]

    # Normalize to [-1, 1] for grid_sample
    uv_norm = torch.stack([
        2.0 * u / max(img_w - 1, 1) - 1.0,
        2.0 * v / max(img_h - 1, 1) - 1.0,
    ], dim=-1)                                                  # [B, N_cam, N_pts, 2]

    return uv_norm, valid


def make_bev_grid_points(
    bev_h: int, bev_w: int, resolution: float = 0.5, z_height: float = 0.0
) -> torch.Tensor:
    """
    Generate 3D points (in ego frame) for the center of every BEV cell.

    Typical convention: x forward, y left, z up. BEV is the xy plane.

    Returns: [bev_h * bev_w, 3]
    """
    # x_range covers the "in front of" and "behind" the car symmetrically
    x = (torch.arange(bev_h) - (bev_h - 1) / 2) * resolution      # forward axis
    y = (torch.arange(bev_w) - (bev_w - 1) / 2) * resolution      # lateral axis
    xx, yy = torch.meshgrid(x, y, indexing="ij")                  # [H, W]
    zz = torch.full_like(xx, z_height)
    return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)       # [H*W, 3]


# ============================================================================
# 3. Spatial cross-attention (simplified deformable)
# ============================================================================

class SpatialCrossAttention(nn.Module):
    """
    For each BEV query, sample K points from each camera's feature map
    at (reference_point + predicted_offset), weight-sum with predicted
    attention weights.

    This is the core operation of BEVFormer. Real implementation uses a
    custom CUDA kernel (ms_deform_attn). Here we use F.grid_sample with
    bilinear interpolation — same math, slower in Python.

    For each query q and each camera c:
        offsets_k = Linear(q) → K 2D offsets (normalized to feature size)
        weights_k = softmax(Linear(q))
        sampled_k = grid_sample(img_feat[c], ref_point[c] + offsets_k)
        contrib_c = sum_k (weights_k * sampled_k)
    Final output = (1 / n_valid_cams) · sum_c contrib_c
    """
    def __init__(
        self,
        d_model: int = 256,
        n_cams: int = 6,
        n_points: int = 4,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_cams = n_cams
        self.n_points = n_points
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Per-query: predict (x, y) offset for each (cam, head, point)
        self.offset_proj = nn.Linear(d_model, n_cams * n_heads * n_points * 2)
        # Per-query: predict attention weight for each (cam, head, point)
        self.weight_proj = nn.Linear(d_model, n_cams * n_heads * n_points)
        # Standard value projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        # Small initialization for offsets — start near the reference points
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)

    def forward(
        self,
        queries: torch.Tensor,          # [B, N_q, d]
        img_features: torch.Tensor,     # [B, N_cam, d, H_feat, W_feat]
        ref_points_norm: torch.Tensor,  # [B, N_cam, N_q, 2] — normalized [-1, 1]
        valid_mask: torch.Tensor,       # [B, N_cam, N_q] — bool
    ) -> torch.Tensor:
        B, N_q, d = queries.shape
        N_cam = self.n_cams
        H, W = self.n_heads, self.n_points

        # Value projection on image features
        # img_features: [B, N_cam, d, H_feat, W_feat] — we apply value_proj per cam
        Bf, Nf, Df, Hf, Wf = img_features.shape
        img_flat = img_features.permute(0, 1, 3, 4, 2).reshape(Bf * Nf, Hf * Wf, Df)
        values = self.value_proj(img_flat).reshape(Bf, Nf, Hf, Wf, Df)
        values = values.permute(0, 1, 4, 2, 3)                          # [B, N_cam, d, Hf, Wf]

        # Predict offsets: [B, N_q, N_cam, H, W_pts, 2]
        offsets = self.offset_proj(queries).reshape(B, N_q, N_cam, H, W, 2)

        # Predict attention weights: [B, N_q, N_cam, H, W_pts] → softmax over (N_cam, H, W_pts)
        # We softmax over the full (cams × heads × points) so attention is shared across cams.
        attn_logits = self.weight_proj(queries).reshape(B, N_q, N_cam * H * W)
        # Mask out invalid cams before softmax (−∞ logits for bins in invalid cams)
        # Expand valid_mask from [B, N_cam, N_q] to [B, N_q, N_cam*H*W]
        mask_expanded = valid_mask.permute(0, 2, 1).unsqueeze(-1).expand(B, N_q, N_cam, H * W)
        mask_expanded = mask_expanded.reshape(B, N_q, N_cam * H * W)
        attn_logits = attn_logits.masked_fill(~mask_expanded, float("-inf"))
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)                     # if all cams invalid
        attn_weights = attn_weights.reshape(B, N_q, N_cam, H, W)

        # Per-camera: sample values at (ref_points + offsets) using grid_sample.
        # We accumulate contribution from each camera.
        output = torch.zeros(B, N_q, d, device=queries.device, dtype=queries.dtype)

        for c in range(N_cam):
            # Reference point (normalized) for this cam: [B, N_q, 2]
            ref_c = ref_points_norm[:, c]                               # [B, N_q, 2]
            # Offsets in normalized coords, scaled to feature-map size.
            # Convert offset from "feature pixels" to "normalized [-1,1]" scale.
            offsets_c = offsets[:, :, c]                                # [B, N_q, H, W, 2]
            scale = torch.tensor([2.0 / max(Wf - 1, 1), 2.0 / max(Hf - 1, 1)],
                                 device=queries.device, dtype=queries.dtype)
            # Sampling points = ref + offset · scale. Shape [B, N_q, H, W, 2]
            sample_pts = ref_c.view(B, N_q, 1, 1, 2) + offsets_c * scale

            # grid_sample expects [B, d, H_out, W_out, 2]-style grid
            # We reshape queries as a "grid" of shape [N_q, H*W] points (flat)
            grid = sample_pts.reshape(B, N_q * H * W, 1, 2)             # [B, N_q·H·W, 1, 2]
            val_c = values[:, c]                                        # [B, d, Hf, Wf]
            sampled = F.grid_sample(val_c, grid, mode="bilinear",
                                    align_corners=True, padding_mode="zeros")
            sampled = sampled.squeeze(-1).transpose(1, 2)               # [B, N_q·H·W, d]
            sampled = sampled.reshape(B, N_q, H, W, d)                  # [B, N_q, H, W_pts, d]

            # Weight and accumulate: w_{b,q,c,h,p} · sampled_{b,q,h,p,d}
            w_c = attn_weights[:, :, c]                                 # [B, N_q, H, W_pts]
            contrib = (w_c.unsqueeze(-1) * sampled).sum(dim=(2, 3))     # [B, N_q, d]
            output = output + contrib

        return self.output_proj(output)


# ============================================================================
# 4. BEV Transformer layer
# ============================================================================

class BEVFormerLayer(nn.Module):
    """
    One transformer layer for BEV processing:
        BEV queries ─ self-attention ─→ spatial cross-attention (to cameras) ─→ FFN

    In the full BEVFormer, there's also a temporal self-attention that attends
    to the previous BEV frame. We omit that for brevity.
    """
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_cams: int = 6,
        n_points: int = 4,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = SpatialCrossAttention(d_model, n_cams, n_points, n_heads)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        bev_queries: torch.Tensor,      # [B, N_q, d]
        img_features: torch.Tensor,     # [B, N_cam, d, H, W]
        ref_points: torch.Tensor,       # [B, N_cam, N_q, 2]
        valid_mask: torch.Tensor,       # [B, N_cam, N_q]
    ) -> torch.Tensor:
        # Self-attention among BEV queries (lets adjacent cells share info)
        x = self.norm1(bev_queries)
        attn_out, _ = self.self_attn(x, x, x)
        bev_queries = bev_queries + attn_out

        # Spatial cross-attention to camera features
        x = self.norm2(bev_queries)
        ca_out = self.cross_attn(x, img_features, ref_points, valid_mask)
        bev_queries = bev_queries + ca_out

        # Feed-forward
        bev_queries = bev_queries + self.ffn(self.norm3(bev_queries))
        return bev_queries


# ============================================================================
# 5. Full BEVFormer model
# ============================================================================

class BEVFormer(nn.Module):
    """
    Minimal BEVFormer-style model.

    Args:
        bev_h, bev_w: BEV grid size (number of cells along x / y)
        bev_resolution: meters per cell
        bev_z_height: height above ground to sample (meters; 0 = ground plane)
        img_h, img_w: camera image size (after resize)
        n_cams: number of cameras
        d_model: transformer hidden dim (also image feature dim)
        n_layers: number of BEVFormer layers
        n_heads: attention heads
        n_points: sample points per query per camera in deformable CA
    """
    def __init__(
        self,
        bev_h: int = 200,
        bev_w: int = 200,
        bev_resolution: float = 0.5,
        bev_z_height: float = 0.0,
        img_h: int = 928,
        img_w: int = 1600,
        n_cams: int = 6,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        n_points: int = 4,
        backbone_stride: int = 16,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.img_h = img_h
        self.img_w = img_w
        self.n_cams = n_cams
        self.d_model = d_model

        # Per-camera image backbone (shared weights across all cameras).
        self.backbone = ImageBackbone(in_channels=3, out_channels=d_model,
                                      stride=backbone_stride)
        self.feat_h = img_h // backbone_stride
        self.feat_w = img_w // backbone_stride

        # Learnable BEV queries — one vector per BEV cell
        self.bev_queries = nn.Parameter(torch.zeros(bev_h * bev_w, d_model))
        nn.init.trunc_normal_(self.bev_queries, std=0.02)

        # Learnable positional embedding for BEV cells (could also be 2D sin-cos)
        self.bev_pos_embed = nn.Parameter(torch.zeros(bev_h * bev_w, d_model))
        nn.init.trunc_normal_(self.bev_pos_embed, std=0.02)

        # 3D positions of BEV cell centers (fixed, derived from geometry)
        bev_points_3d = make_bev_grid_points(bev_h, bev_w, bev_resolution, bev_z_height)
        self.register_buffer("bev_points_3d", bev_points_3d)   # [H*W, 3]

        # Transformer layers
        self.layers = nn.ModuleList([
            BEVFormerLayer(d_model=d_model, n_heads=n_heads,
                           n_cams=n_cams, n_points=n_points)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        images: torch.Tensor,       # [B, N_cam, 3, H_img, W_img]
        intrinsics: torch.Tensor,   # [B, N_cam, 3, 3]
        ego_to_cam: torch.Tensor,   # [B, N_cam, 4, 4]
    ) -> torch.Tensor:
        """
        Returns: BEV feature map [B, bev_h, bev_w, d_model]
        """
        B, N_cam, _, H_img, W_img = images.shape
        assert N_cam == self.n_cams

        # 1. Per-camera backbone (shared weights)
        img_flat = images.reshape(B * N_cam, 3, H_img, W_img)
        feats = self.backbone(img_flat)                                 # [B*N, d, H', W']
        feats = feats.reshape(B, N_cam, self.d_model, self.feat_h, self.feat_w)

        # 2. Project each BEV cell's 3D point to every camera's pixel coords
        #    (reference points for the deformable cross-attention)
        ref_points_norm, valid_mask = project_ego_to_image(
            self.bev_points_3d, intrinsics, ego_to_cam, H_img, W_img
        )                                                               # [B, N_cam, N_q, 2], [..]

        # 3. Initialize BEV queries
        bev_q = self.bev_queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        bev_q = bev_q + self.bev_pos_embed.unsqueeze(0)

        # 4. Transformer layers
        for layer in self.layers:
            bev_q = layer(bev_q, feats, ref_points_norm, valid_mask)

        # 5. Reshape into a 2D feature map
        bev_map = bev_q.reshape(B, self.bev_h, self.bev_w, self.d_model)
        return bev_map


# ============================================================================
# Model configs
# ============================================================================

def bev_small() -> BEVFormer:
    """BEV-S: ~6M params with stub backbone. 100×100 BEV grid, 4 layers."""
    return BEVFormer(
        bev_h=100, bev_w=100, bev_resolution=1.0,
        img_h=448, img_w=800,
        d_model=192, n_layers=4, n_heads=6, n_points=4,
        backbone_stride=16,
    )

def bev_base() -> BEVFormer:
    """BEV-B: ~28M params with stub backbone. 200×200 BEV grid, 6 layers
    matching BEVFormer-base geometry (add a real ResNet-50 for ~60M)."""
    return BEVFormer(
        bev_h=200, bev_w=200, bev_resolution=0.5,
        img_h=928, img_w=1600,
        d_model=256, n_layers=6, n_heads=8, n_points=4,
        backbone_stride=16,
    )


# ============================================================================
# Analysis functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: BEVFormer):
    """Parameter breakdown by component."""
    print("=" * 60)
    print("BEVFormer Model Summary")
    print("=" * 60)

    backbone = count_parameters(model.backbone)
    queries = model.bev_queries.numel()
    pos_emb = model.bev_pos_embed.numel()
    per_layer = sum(p.numel() for p in model.layers[0].parameters())
    all_layers = per_layer * len(model.layers)
    total = count_parameters(model)

    print(f"Image backbone (shared):     {backbone:>12,} params  ({backbone / 1e6:.2f}M)")
    print(f"BEV queries ({model.bev_h}×{model.bev_w}×{model.d_model}): "
          f"{queries:>12,} params  ({queries / 1e6:.2f}M)")
    print(f"BEV position embedding:      {pos_emb:>12,} params  ({pos_emb / 1e6:.2f}M)")
    print(f"Transformer layers × {len(model.layers)}:"
          f"       {all_layers:>12,} params  ({all_layers / 1e6:.2f}M)")
    print(f"  (per layer: {per_layer:,})")
    print("-" * 60)
    print(f"Total:                       {total:>12,} params  ({total / 1e6:.2f}M)")
    print("=" * 60)


def compute_flops(model: BEVFormer) -> dict:
    """
    Approximate FLOPs for ONE forward pass.

    Four components:
      - Image backbone (dominant): N_cam × CNN forward
      - Self-attention over BEV queries (O(N_q²·d) — big!)
      - Spatial cross-attention (O(N_q · N_cam · K · d²))
      - FFNs
    """
    d = model.d_model
    N_q = model.bev_h * model.bev_w
    N_cam = model.n_cams
    L = len(model.layers)
    K = model.layers[0].cross_attn.n_points
    H_f, W_f = model.feat_h, model.feat_w

    # Backbone: count as ~2·H·W·C_out·K² per layer, sum across the conv stack.
    # Rough estimate: for resnet18-like with downsample 16, about
    # backbone_flops ≈ 10% of a ResNet-50 which is 4 GFLOPs at 224×224.
    # Scale to our image size: FLOPs ≈ 4e9 · (H·W)/(224·224)
    # per camera, × N_cam.
    per_cam_flops = 4e9 * (model.img_h * model.img_w) / (224 * 224) * 0.3  # small backbone
    backbone_flops = N_cam * per_cam_flops

    # Self-attention among BEV queries: O(N_q² · d)
    self_attn_flops = L * (
        2 * N_q * d * d * 3                # QKV projection
        + 2 * N_q * N_q * d                # scores
        + 2 * N_q * N_q * d                # attn · V
        + 2 * N_q * d * d                  # output projection
    )

    # Spatial cross-attention: O(N_q · N_cam · K · d²) for projections + sampling
    cross_attn_flops = L * (
        2 * N_q * d * (N_cam * K * 8)           # offset prediction (H·K·2 output)
        + 2 * N_q * d * (N_cam * K)             # attention-weight prediction
        + 2 * N_q * d * d                       # value projection
        + 2 * N_q * d * d                       # output projection
        + N_q * N_cam * K * d * 4               # grid_sample (bilinear = ~4 mults)
    )

    # FFN: O(N_q · d · 4d)
    ffn_flops = L * 2 * N_q * d * (4 * d) * 2   # 2 linear layers each

    total = backbone_flops + self_attn_flops + cross_attn_flops + ffn_flops

    return {
        "backbone_gflops": backbone_flops / 1e9,
        "self_attn_gflops": self_attn_flops / 1e9,
        "cross_attn_gflops": cross_attn_flops / 1e9,
        "ffn_gflops": ffn_flops / 1e9,
        "total_gflops": total / 1e9,
        "n_bev_queries": N_q,
    }


def demo_forward_pass():
    """One forward pass with shape trace."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (BEV-S, 6 cameras, small resolution)")
    print("=" * 60)

    model = bev_small()
    model.eval()

    B, N_cam = 1, 6
    H_img, W_img = model.img_h, model.img_w
    images = torch.randn(B, N_cam, 3, H_img, W_img)

    # Fake intrinsics: 800 focal length, principal point at image center
    fx = fy = 400.0
    cx, cy = W_img / 2, H_img / 2
    K_single = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    intrinsics = K_single.unsqueeze(0).unsqueeze(0).expand(B, N_cam, 3, 3).contiguous()

    # Fake extrinsics: 6 cameras pointing at 60° intervals around the car.
    ego_to_cam = torch.zeros(B, N_cam, 4, 4)
    for c in range(N_cam):
        angle = 2 * math.pi * c / N_cam
        R = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle),  math.cos(angle), 0.0],
            [0.0,              0.0,             1.0],
        ])
        # Camera frame convention: x right, y down, z forward.
        # Here we just use identity-ish; not geometrically correct for a real rig
        # but enough to demonstrate the pipeline.
        ego_to_cam[:, c, :3, :3] = R
        ego_to_cam[:, c, 3, 3] = 1.0
        ego_to_cam[:, c, 2, 3] = -1.5                           # cameras ~1.5m above ground

    print(f"\n1. Images:            {list(images.shape)}")
    print(f"   Intrinsics:        {list(intrinsics.shape)}")
    print(f"   ego_to_cam:        {list(ego_to_cam.shape)}")

    with torch.no_grad():
        # 1. Backbone
        feats = model.backbone(images.reshape(B * N_cam, 3, H_img, W_img))
        feats = feats.reshape(B, N_cam, model.d_model, model.feat_h, model.feat_w)
        print(f"\n2. Image features:    {list(feats.shape)}  (per-camera)")

        # 2. Camera projection
        ref_pts, valid = project_ego_to_image(
            model.bev_points_3d, intrinsics, ego_to_cam, H_img, W_img
        )
        print(f"3. BEV→image refs:   {list(ref_pts.shape)}  "
              f"(per cam per BEV cell)")
        print(f"   Valid mask:       "
              f"{valid.sum().item() / valid.numel() * 100:.1f}% of (cam, cell) pairs valid")

        # 3. Run full model
        bev_map = model(images, intrinsics, ego_to_cam)
        print(f"\n4. BEV feature map:   {list(bev_map.shape)}")
        print(f"   → Heads (detection, segmentation, forecasting) run on this map.")

    print("\n" + "=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BEVFormer (BEV Tokenizer) Reference Implementation")
    print("=" * 60)

    model = bev_base()
    print_model_summary(model)

    flops = compute_flops(model)
    print(f"\nFLOPs (per forward pass):")
    print(f"  Image backbone (× {model.n_cams} cameras): "
          f"{flops['backbone_gflops']:>8.1f} GFLOPs  (dominant)")
    print(f"  BEV self-attention:            "
          f"{flops['self_attn_gflops']:>8.1f} GFLOPs")
    print(f"  Spatial cross-attention:       "
          f"{flops['cross_attn_gflops']:>8.1f} GFLOPs")
    print(f"  FFN:                           "
          f"{flops['ffn_gflops']:>8.1f} GFLOPs")
    print(f"  ─────────────────────────────  ─────")
    print(f"  Total:                         "
          f"{flops['total_gflops']:>8.1f} GFLOPs")
    print(f"  BEV queries (N_q = {model.bev_h} × {model.bev_w} = "
          f"{flops['n_bev_queries']:,})")

    try:
        demo_forward_pass()
    except Exception as e:
        print(f"\n(Skipping demo — torch not available here: {e})")

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    configs = [("BEV-S", bev_small), ("BEV-B", bev_base)]
    print(f"{'Model':<10} {'Params':>10} {'GFLOPs':>12} {'BEV grid':>12}")
    print("-" * 48)
    for name, fn in configs:
        m = fn()
        p = count_parameters(m)
        g = compute_flops(m)["total_gflops"]
        print(f"{name:<10} {p/1e6:>8.1f}M {g:>10.1f}   {m.bev_h}×{m.bev_w}")
