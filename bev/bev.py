"""
BEV (Bird's-Eye-View) Multi-View Tokenizer Reference Implementation
BEVFormer-style: learnable BEV queries look up multi-view camera features
through deformable attention, producing a dense BEV feature map.

Usage:
    python bev.py

Architecture (one BEVFormer encoder layer = TSA + SCA + FFN, stacked L times):

    6 cameras [B, N_cam, 3, H, W]
      | per-camera backbone (shared weights)
      v
    Image features [B, N_cam, d, H', W']
                                    |
    Learnable BEV queries --> temporal self-attention (deformable, over BEV)
    [H_bev * W_bev, d]          --> spatial cross-attention (deformable, over cameras,
                                    reference points from camera projection)
                                --> FFN
                                    |  x L layers
                                    v
                          BEV feature map [B, H_bev, W_bev, d]
                                    |
                          detection / segmentation heads (not shown)

What is faithful to the BEVFormer paper (arXiv 2203.17270) and its official code:
  * each BEV cell is lifted to a PILLAR of N_z height anchors (BEVFormer: 4 heights)
    and every anchor is projected into every camera;
  * a camera is "hit" if any anchor lands inside its image; spatial cross-attention
    samples only around those reference points and AVERAGES over hit cameras
    (SCA = 1/|V_hit| * sum_i sum_j DeformAttn(...));
  * offsets / attention weights are predicted from the query and are SHARED by all
    cameras; attention is multi-head (each head has its own channels, its own
    softmax over sampling points);
  * the BEV "self-attention" is temporal self-attention: deformable attention over
    {previous BEV, current queries}; with no history it degenerates to {Q, Q};
  * learned positional embedding is factorised (row table + column table);
  * FFN hidden size is 2 * d (BEVFormer config: _ffn_dim_ = _dim_ * 2).

Simplifications (all clearly marked in the code):
  - the image backbone is a toy CNN (BEVFormer-base uses ResNet-101-DCN + FPN,
    multi-scale features); we use a single feature level;
  - previous-frame BEV is NOT re-aligned by ego motion (the paper rotates/shifts it);
  - every camera is sampled for every query and masked afterwards; the official code
    gathers only the hit queries per camera (each query hits ~1-2 of 6 views);
  - pre-norm residual blocks (the official config is post-norm);
  - no detection / segmentation heads.

Model configs (toy backbone, see print_model_summary for exact counts):
    - BEV-S: 100x100 grid, d=192, 3 layers  (runs a forward pass on CPU in seconds)
    - BEV-B: 200x200 grid, d=256, 6 layers  (BEVFormer-base geometry; parameter
             count and FLOP estimate only)
    Note that half of BEV-B's parameters are the 200*200*256 = 10.24M query table.
"""
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. Image backbone — stub per-camera feature extractor
# ============================================================================

class ImageBackbone(nn.Module):
    """
    Toy CNN feature extractor (a few strided 3x3 convs). In BEVFormer-base this is
    a ResNet-101-DCN with an FPN producing several feature scales.

    Input:  [B*N_cam, 3, H_img, W_img]
    Output: [B*N_cam, out_channels, H_img/stride, W_img/stride]
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 256, stride: int = 16):
        super().__init__()
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
        layers += [nn.Conv2d(c_prev, out_channels, 1), nn.GroupNorm(32, out_channels)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def stub_backbone_macs(img_h: int, img_w: int, d: int, stride: int) -> int:
    """Exact multiply-accumulate count of ImageBackbone's convolutions for ONE image.
    (Same formula as the guide's calculator, 'toy stub' option.)"""
    n_down = int(math.log2(stride))
    chs = [3, 64, 128, 256, d]
    h, w, c_prev, macs = img_h, img_w, 3, 0
    for i in range(n_down):
        c_next = chs[min(i + 1, 4)]
        h, w = (h + 1) // 2, (w + 1) // 2           # 3x3, stride 2, pad 1 -> ceil(x/2)
        macs += h * w * c_next * c_prev * 9
        c_prev = c_next
    macs += h * w * d * c_prev                      # final 1x1 projection
    return macs


def stub_backbone_params(d: int, stride: int) -> int:
    """Exact parameter count of ImageBackbone (conv weights + biases + GroupNorm affine)."""
    n_down = int(math.log2(stride))
    chs = [3, 64, 128, 256, d]
    c_prev, n = 3, 0
    for i in range(n_down):
        c_next = chs[min(i + 1, 4)]
        n += c_prev * c_next * 9 + c_next + 2 * c_next          # conv3x3 + bias + GN
        c_prev = c_next
    return n + c_prev * d + d + 2 * d                            # 1x1 conv + bias + GN


# Backbone parameter counts without the classifier, counted from torchvision's
# resnet50() / resnet101() (BEV models drop the ImageNet FC layer).
RESNET_PARAMS = {"resnet50": 23_508_032, "resnet101": 42_500_160}

# ResNet multiply-adds at 224x224 from He et al. 2015, Table 1 ("FLOPs" there are
# multiply-adds). Conv cost scales with pixel count, so we scale by H*W / 224^2.
RESNET_MACS_224 = {"resnet50": 3.8e9, "resnet101": 7.6e9}


# ============================================================================
# 2. Camera rig + projection — 3D ego points -> 2D image coords
# ============================================================================

# A simple 6-camera ring (nuScenes also has 6 cameras covering 360 degrees).
# Yaw is measured from the ego +x (forward) axis, counter-clockwise (towards +y = left).
CAMERA_NAMES = ("front", "front_left", "back_left", "back", "back_right", "front_right")
CAMERA_YAWS_DEG = (0.0, 60.0, 120.0, 180.0, -120.0, -60.0)


def make_camera_rig(
    img_h: int, img_w: int, batch: int = 1, focal_frac: float = 0.625,
    cam_height: float = 1.6, yaws_deg: Sequence[float] = CAMERA_YAWS_DEG,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build intrinsics K [B, N_cam, 3, 3] and extrinsics ego_to_cam [B, N_cam, 4, 4]
    for cameras mounted at (0, 0, cam_height) in the ego frame.

    Frames:  ego    x forward, y left,  z up      (origin on the ground)
             camera x right,   y down,  z forward (OpenCV convention)
    Focal length f = focal_frac * img_w (0.625 -> horizontal FOV = 2*atan(0.8) = 77.3 deg,
    so neighbouring cameras 60 deg apart overlap a little).
    """
    f = focal_frac * img_w
    K = torch.tensor([[f, 0.0, img_w / 2], [0.0, f, img_h / 2], [0.0, 0.0, 1.0]])
    Ks, Ts = [], []
    for yaw in yaws_deg:
        th = math.radians(yaw)
        z_c = torch.tensor([math.cos(th), math.sin(th), 0.0])     # optical axis
        x_c = torch.tensor([math.sin(th), -math.cos(th), 0.0])    # image right
        y_c = torch.tensor([0.0, 0.0, -1.0])                      # image down
        R = torch.stack([x_c, y_c, z_c])                          # rows: camera axes in ego coords
        C = torch.tensor([0.0, 0.0, cam_height])                  # camera centre in ego frame
        T = torch.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ C                                         # p_cam = R (p_ego - C)
        Ks.append(K)
        Ts.append(T)
    intrinsics = torch.stack(Ks).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    ego_to_cam = torch.stack(Ts).unsqueeze(0).expand(batch, -1, -1, -1).contiguous()
    return intrinsics, ego_to_cam


def project_ego_to_image(
    points_3d: torch.Tensor,      # [..., 3] in ego frame (any leading shape, shared by the batch)
    intrinsics: torch.Tensor,     # [B, N_cam, 3, 3]
    ego_to_cam: torch.Tensor,     # [B, N_cam, 4, 4]
    img_h: int,
    img_w: int,
    return_pixels: bool = False,
):
    """
    Project 3D ego-frame points into every camera.

    Returns:
        uv_norm: [B, N_cam, ..., 2] — pixel coords mapped to [-1, 1] for
                 F.grid_sample(align_corners=False): x_norm = 2*u/W - 1, y_norm = 2*v/H - 1
                 (the same image-size normalisation BEVFormer's point_sampling uses)
        valid:   [B, N_cam, ...]    — in front of the camera AND strictly inside the image
        (uv_pix: [B, N_cam, ..., 2] raw pixels, if return_pixels)

    The math (per camera):
      1. ego -> camera:  p_cam = R p_ego + t          (ego_to_cam)
      2. camera -> pixel: u = fx X/Z + cx, v = fy Y/Z + cy   (intrinsics)
      3. keep points with Z > eps and 0 < u < W, 0 < v < H
    """
    B, N_cam = intrinsics.shape[:2]
    lead = points_3d.shape[:-1]
    pts = points_3d.reshape(-1, 3)
    N_pts = pts.shape[0]

    ones = torch.ones(N_pts, 1, device=pts.device, dtype=pts.dtype)
    p_ego_h = torch.cat([pts, ones], dim=-1).T                    # [4, N_pts]
    p_cam = (ego_to_cam @ p_ego_h)[..., :3, :]                    # [B, N_cam, 3, N_pts]
    pix_h = intrinsics @ p_cam                                    # [B, N_cam, 3, N_pts]

    eps = 1e-5
    z = pix_h[..., 2, :]
    uv = pix_h[..., :2, :] / z.clamp(min=eps).unsqueeze(-2)       # [B, N_cam, 2, N_pts]
    uv = uv.transpose(-1, -2)                                     # [B, N_cam, N_pts, 2]
    u, v = uv[..., 0], uv[..., 1]
    valid = (z > eps) & (u > 0) & (u < img_w) & (v > 0) & (v < img_h)

    uv_norm = torch.stack([2.0 * u / img_w - 1.0, 2.0 * v / img_h - 1.0], dim=-1)
    uv_norm = uv_norm.reshape(B, N_cam, *lead, 2)
    valid = valid.reshape(B, N_cam, *lead)
    if return_pixels:
        return uv_norm, valid, uv.reshape(B, N_cam, *lead, 2)
    return uv_norm, valid


def make_bev_pillar_points(
    bev_h: int, bev_w: int, resolution: float = 0.5,
    z_anchors: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
) -> torch.Tensor:
    """
    3D points (ego frame) for every BEV cell centre, lifted into a pillar of
    len(z_anchors) heights. BEVFormer samples 4 anchors uniformly in [-5 m, 3 m]
    of its LiDAR-centred frame; we use heights above our ground plane.

    Cell (i, j): x = (i - (bev_h-1)/2) * res  (forward),  y = (j - (bev_w-1)/2) * res  (left).
    Query index q = i * bev_w + j.

    Returns: [bev_h * bev_w, N_z, 3]
    """
    x = (torch.arange(bev_h) - (bev_h - 1) / 2) * resolution
    y = (torch.arange(bev_w) - (bev_w - 1) / 2) * resolution
    xx, yy = torch.meshgrid(x, y, indexing="ij")                  # [H, W]
    zs = torch.tensor(z_anchors, dtype=xx.dtype)
    N_z = len(z_anchors)
    pts = torch.stack([
        xx[..., None].expand(-1, -1, N_z),
        yy[..., None].expand(-1, -1, N_z),
        zs.view(1, 1, N_z).expand(bev_h, bev_w, N_z),
    ], dim=-1)                                                    # [H, W, N_z, 3]
    return pts.reshape(bev_h * bev_w, N_z, 3)


# ============================================================================
# 3. Deformable attention helpers
# ============================================================================

def _grid_init_offsets(linear: nn.Linear, n_heads: int, n_groups: int, n_points: int):
    """Deformable-DETR-style init: zero weights, and a bias that points head h in
    direction 2*pi*h/n_heads with radius 1, 2, ..., n_points (in feature pixels).
    So at step 0 every head already looks at a different small neighbourhood."""
    nn.init.zeros_(linear.weight)
    thetas = torch.arange(n_heads, dtype=torch.float32) * (2.0 * math.pi / n_heads)
    grid = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid = grid / grid.abs().max(-1, keepdim=True)[0]             # [n_heads, 2] on the unit square
    grid = grid.view(n_heads, 1, 1, 2).repeat(1, n_groups, n_points, 1)
    for p in range(n_points):
        grid[:, :, p, :] *= p + 1
    with torch.no_grad():
        linear.bias.copy_(grid.reshape(-1))


# ============================================================================
# 4. Spatial cross-attention (BEV queries -> camera features)
# ============================================================================

class SpatialCrossAttention(nn.Module):
    """
    BEVFormer spatial cross-attention, single feature level.

    For BEV query q with pillar anchors j = 1..N_z and camera i:
        ref[i, j]   = project(pillar_j)                           (fixed geometry)
        loc[h,j,p]  = ref[i, j] + offset[h, j, p]                 (offsets SHARED by cameras)
        out_i       = concat_h  sum_{j,p} w[h, j, p] * bilinear(V_i^h, loc[h, j, p])
        SCA(q)      = W_o ( 1/|V_hit| * sum_{i in V_hit} out_i )

    w is a softmax per head over all (anchor, point) pairs. V^h is the head's
    d/n_heads-channel slice of the value-projected features.
    Official config (BEVFormer-base): 8 heads, 8 points per head split over 4 anchors.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_z: int = 4,
                 points_per_z: int = 2):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads, self.n_z, self.n_pts = d_model, n_heads, n_z, points_per_z
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_z * points_per_z * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_z * points_per_z)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        _grid_init_offsets(self.sampling_offsets, n_heads, n_z, points_per_z)
        nn.init.zeros_(self.attention_weights.weight)             # uniform weights at init
        nn.init.zeros_(self.attention_weights.bias)

    def forward(
        self,
        query: torch.Tensor,         # [B, N_q, d]
        img_feats: torch.Tensor,     # [B, N_cam, d, H_f, W_f]
        ref_uv: torch.Tensor,        # [B, N_cam, N_q, N_z, 2]  in [-1, 1]
        valid: torch.Tensor,         # [B, N_cam, N_q, N_z]     bool
    ) -> torch.Tensor:
        B, N_q, d = query.shape
        _, N_cam, _, H_f, W_f = img_feats.shape
        nH, nZ, P, hd = self.n_heads, self.n_z, self.n_pts, self.head_dim

        # Value projection, then split channels into heads: [B, N_cam, nH, hd, H_f, W_f]
        v = self.value_proj(img_feats.permute(0, 1, 3, 4, 2))     # [B, N_cam, H_f, W_f, d]
        v = v.view(B, N_cam, H_f, W_f, nH, hd).permute(0, 1, 4, 5, 2, 3)

        # Offsets are in feature pixels -> convert to normalised [-1, 1] units (2/W_f per pixel)
        off = self.sampling_offsets(query).view(B, N_q, nH, nZ, P, 2)
        off = off * off.new_tensor([2.0 / W_f, 2.0 / H_f])
        w = self.attention_weights(query).view(B, N_q, nH, nZ * P).softmax(-1)
        w = w.permute(0, 2, 1, 3).reshape(B * nH, 1, N_q, nZ * P)

        hit = valid.any(dim=-1)                                   # [B, N_cam, N_q]
        out = query.new_zeros(B, N_q, d)
        for c in range(N_cam):                                    # official code: gather hit queries only
            loc = ref_uv[:, c][:, :, None, :, None, :] + off      # [B, N_q, nH, nZ, P, 2]
            grid = loc.permute(0, 2, 1, 3, 4, 5).reshape(B * nH, N_q, nZ * P, 2)
            vc = v[:, c].reshape(B * nH, hd, H_f, W_f)
            s = F.grid_sample(vc, grid, mode="bilinear", padding_mode="zeros",
                              align_corners=False)                # [B*nH, hd, N_q, nZ*P]
            o = (s * w).sum(-1).view(B, nH, hd, N_q)              # weighted sum per head
            o = o.permute(0, 3, 1, 2).reshape(B, N_q, d)          # concat heads
            out = out + o * hit[:, c].unsqueeze(-1).to(o.dtype)

        count = hit.sum(dim=1).clamp(min=1).to(out.dtype)         # |V_hit| per query
        return self.output_proj(out / count.unsqueeze(-1))


# ============================================================================
# 5. Temporal self-attention (BEV queries -> {previous BEV, current queries})
# ============================================================================

def bev_reference_grid(bev_h: int, bev_w: int, device=None) -> torch.Tensor:
    """Normalised [-1, 1] location of every BEV cell centre in the BEV feature map
    (grid_sample order: x = column j, y = row i). Shape [bev_h * bev_w, 2]."""
    jj = (torch.arange(bev_w, device=device) + 0.5) / bev_w * 2 - 1
    ii = (torch.arange(bev_h, device=device) + 0.5) / bev_h * 2 - 1
    gi, gj = torch.meshgrid(ii, jj, indexing="ij")
    return torch.stack([gj, gi], dim=-1).reshape(-1, 2)


class TemporalSelfAttention(nn.Module):
    """
    BEVFormer temporal self-attention: deformable attention where each BEV query
    samples P points around ITS OWN cell in two value maps {B'_{t-1}, Q}:
        TSA(Q_p) = W_o( mean_{V in {B'_{t-1}, Q}} DeformAttn(Q_p, p, V) )
    Offsets / weights are predicted from concat(B'_{t-1}, Q). With no history
    (first frame) B'_{t-1} := Q, i.e. plain deformable self-attention.
    Cost is linear in the number of BEV cells — no N^2 attention matrix.
    """
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model, self.n_heads, self.n_pts = d_model, n_heads, n_points
        self.head_dim = d_model // n_heads
        self.sampling_offsets = nn.Linear(2 * d_model, 2 * n_heads * n_points * 2)
        self.attention_weights = nn.Linear(2 * d_model, 2 * n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        _grid_init_offsets(self.sampling_offsets, 2 * n_heads, 1, n_points)
        nn.init.zeros_(self.attention_weights.weight)
        nn.init.zeros_(self.attention_weights.bias)

    def forward(self, query: torch.Tensor, prev_bev: Optional[torch.Tensor],
                bev_h: int, bev_w: int) -> torch.Tensor:
        B, N, d = query.shape
        nH, P, hd = self.n_heads, self.n_pts, self.head_dim
        if prev_bev is None:
            prev_bev = query                                      # first frame: {Q, Q}
        # (The paper first warps prev_bev by ego motion; omitted here.)
        values = torch.stack([prev_bev, query], dim=1)            # [B, 2, N, d]
        q_cat = torch.cat([prev_bev, query], dim=-1)              # [B, N, 2d]

        off = self.sampling_offsets(q_cat).view(B, N, 2, nH, P, 2)
        off = off * off.new_tensor([2.0 / bev_w, 2.0 / bev_h])    # BEV cells -> [-1, 1] units
        w = self.attention_weights(q_cat).view(B, N, 2, nH, P).softmax(-1)

        ref = bev_reference_grid(bev_h, bev_w, query.device).to(query.dtype)    # [N, 2]
        loc = ref.view(1, N, 1, 1, 1, 2) + off                    # [B, N, 2, nH, P, 2]
        grid = loc.permute(0, 2, 3, 1, 4, 5).reshape(B * 2 * nH, N, P, 2)

        v = self.value_proj(values).view(B, 2, bev_h, bev_w, nH, hd)
        v = v.permute(0, 1, 4, 5, 2, 3).reshape(B * 2 * nH, hd, bev_h, bev_w)
        s = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False)                    # [B*2*nH, hd, N, P]
        wq = w.permute(0, 2, 3, 1, 4).reshape(B * 2 * nH, 1, N, P)
        o = (s * wq).sum(-1).view(B, 2, nH, hd, N).mean(dim=1)    # average the two value maps
        o = o.permute(0, 3, 1, 2).reshape(B, N, d)
        return self.output_proj(o)


# ============================================================================
# 6. BEVFormer encoder layer and model
# ============================================================================

class BEVFormerLayer(nn.Module):
    """One encoder layer: temporal self-attn -> spatial cross-attn -> FFN.
    (Pre-norm residuals here; BEVFormer's config is post-norm: attn, norm, attn, norm, ffn, norm.)"""
    def __init__(self, d_model: int = 256, n_heads: int = 8, n_z: int = 4,
                 sca_points_per_z: int = 2, tsa_points: int = 4, ffn_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.tsa = TemporalSelfAttention(d_model, n_heads, tsa_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.sca = SpatialCrossAttention(d_model, n_heads, n_z, sca_points_per_z)
        self.norm3 = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_ratio)
        self.ffn = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(inplace=True),
                                 nn.Linear(hidden, d_model))

    def forward(self, bev_q, prev_bev, img_feats, ref_uv, valid, bev_h, bev_w):
        bev_q = bev_q + self.tsa(self.norm1(bev_q), prev_bev, bev_h, bev_w)
        bev_q = bev_q + self.sca(self.norm2(bev_q), img_feats, ref_uv, valid)
        return bev_q + self.ffn(self.norm3(bev_q))


class LearnedPositionalEncoding2D(nn.Module):
    """BEVFormer's LearnedPositionalEncoding: pos[i, j] = concat(col_embed[j], row_embed[i]).
    Only (bev_h + bev_w) * d/2 parameters instead of bev_h * bev_w * d."""
    def __init__(self, bev_h: int, bev_w: int, d_model: int):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.row_embed = nn.Embedding(bev_h, d_model // 2)
        self.col_embed = nn.Embedding(bev_w, d_model // 2)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self) -> torch.Tensor:
        H, W = self.bev_h, self.bev_w
        row = self.row_embed.weight[:, None, :].expand(H, W, -1)
        col = self.col_embed.weight[None, :, :].expand(H, W, -1)
        return torch.cat([col, row], dim=-1).reshape(H * W, -1)  # [H*W, d]


class BEVFormer(nn.Module):
    """
    Minimal BEVFormer-style encoder.

    Args:
        bev_h, bev_w:      BEV grid size (cells along x / y)
        bev_resolution:    metres per cell (BEVFormer-base: 0.512 m, 200x200 -> +-51.2 m)
        z_anchors:         pillar heights (m) projected for every cell
        img_h, img_w:      camera image size after resize/pad
        d_model:           hidden dim (also image-feature dim)
        n_layers, n_heads: encoder depth / heads
        sca_points_per_z:  sampling points per head per height anchor (official: 8 per head / 4 anchors)
        tsa_points:        sampling points per head in temporal self-attention
    """
    def __init__(
        self, bev_h: int = 200, bev_w: int = 200, bev_resolution: float = 0.512,
        z_anchors: Sequence[float] = (0.0, 1.0, 2.0, 3.0),
        img_h: int = 928, img_w: int = 1600, n_cams: int = 6, d_model: int = 256,
        n_layers: int = 6, n_heads: int = 8, sca_points_per_z: int = 2,
        tsa_points: int = 4, backbone_stride: int = 16, ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.bev_h, self.bev_w = bev_h, bev_w
        self.img_h, self.img_w = img_h, img_w
        self.n_cams, self.d_model = n_cams, d_model
        self.backbone_stride = backbone_stride
        self.z_anchors = tuple(z_anchors)

        self.backbone = ImageBackbone(3, d_model, backbone_stride)
        self.feat_h = math.ceil(img_h / backbone_stride)
        self.feat_w = math.ceil(img_w / backbone_stride)

        # One learnable query per BEV cell + factorised learned positional embedding
        self.bev_queries = nn.Parameter(torch.zeros(bev_h * bev_w, d_model))
        nn.init.trunc_normal_(self.bev_queries, std=0.02)
        self.bev_pos = LearnedPositionalEncoding2D(bev_h, bev_w, d_model)

        # Fixed geometry: pillar points for every cell
        self.register_buffer("bev_pillars",
                             make_bev_pillar_points(bev_h, bev_w, bev_resolution, z_anchors))

        self.layers = nn.ModuleList([
            BEVFormerLayer(d_model, n_heads, len(z_anchors), sca_points_per_z,
                           tsa_points, ffn_ratio)
            for _ in range(n_layers)
        ])

    def forward(self, images, intrinsics, ego_to_cam, prev_bev=None):
        """
        images [B, N_cam, 3, H, W], intrinsics [B, N_cam, 3, 3], ego_to_cam [B, N_cam, 4, 4],
        prev_bev [B, bev_h*bev_w, d] or None.   Returns BEV map [B, bev_h, bev_w, d].
        """
        B, N_cam, _, H_img, W_img = images.shape
        assert N_cam == self.n_cams

        # 1. Shared backbone, cameras folded into the batch
        feats = self.backbone(images.reshape(B * N_cam, 3, H_img, W_img))
        feats = feats.reshape(B, N_cam, self.d_model, feats.shape[-2], feats.shape[-1])

        # 2. Geometry: project every pillar anchor into every camera (no learning involved)
        ref_uv, valid = project_ego_to_image(self.bev_pillars, intrinsics, ego_to_cam,
                                             H_img, W_img)        # [B, N_cam, N_q, N_z, (2)]

        # 3. Queries + positional embedding
        bev_q = self.bev_queries.unsqueeze(0).expand(B, -1, -1) + self.bev_pos().unsqueeze(0)

        # 4. Encoder layers (prev_bev is the same for every layer, as in BEVFormer)
        for layer in self.layers:
            bev_q = layer(bev_q, prev_bev, feats, ref_uv, valid, self.bev_h, self.bev_w)

        return bev_q.reshape(B, self.bev_h, self.bev_w, self.d_model)


# ============================================================================
# Model configs
# ============================================================================

def bev_small() -> BEVFormer:
    """BEV-S: 100x100 grid at 1.0 m (+-50 m), d=192, 3 layers, 448x800 images."""
    return BEVFormer(bev_h=100, bev_w=100, bev_resolution=1.0, img_h=448, img_w=800,
                     d_model=192, n_layers=3, n_heads=6, backbone_stride=16)


def bev_base() -> BEVFormer:
    """BEV-B: BEVFormer-base geometry — 200x200 at 0.512 m, d=256, 6 layers,
    928x1600 images (900x1600 padded to a multiple of 32). Toy backbone."""
    return BEVFormer(bev_h=200, bev_w=200, bev_resolution=0.512, img_h=928, img_w=1600,
                     d_model=256, n_layers=6, n_heads=8, backbone_stride=16)


# ============================================================================
# Analysis — the guide's calculator implements exactly this cost model
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def bev_cost(bev: int = 200, d: int = 256, img_h: int = 928, img_w: int = 1600,
             n_cams: int = 6, stride: int = 16, n_layers: int = 6, n_heads: int = 8,
             n_z: int = 4, sca_pts_per_z: int = 2, tsa_pts: int = 4,
             ffn_ratio: float = 2.0, backbone: str = "stub", dtype_bytes: int = 2) -> Dict:
    """
    Analytic FLOP / parameter / memory model (1 FLOP = 1 multiply or add, so a
    multiply-accumulate = 2 FLOPs). Bilinear sampling + weighted accumulation is
    counted as 10 FLOPs per sampled channel (4 taps x mul-add + 1 weight mul-add).
    Every camera is sampled for every query (as this file does); BEVFormer gathers only
    the ~1-2 hit cameras per query, which shrinks the SCA sampling term.
    """
    N = bev * bev
    Hf, Wf = math.ceil(img_h / stride), math.ceil(img_w / stride)
    n_feat = n_cams * Hf * Wf
    fh = int(d * ffn_ratio)
    S = 10                                                        # FLOPs per sampled channel

    if backbone == "stub":
        bb_macs = stub_backbone_macs(img_h, img_w, d, stride)
        bb_params = stub_backbone_params(d, stride)
    else:
        bb_macs = RESNET_MACS_224[backbone] * img_h * img_w / (224 * 224)
        bb_params = RESNET_PARAMS[backbone]
    backbone_flops = 2 * bb_macs * n_cams

    tsa = (2 * N * (2 * d) * (2 * n_heads * tsa_pts * 2)          # offsets from concat(prev, q)
           + 2 * N * (2 * d) * (2 * n_heads * tsa_pts)            # attention weights
           + 2 * (2 * N) * d * d                                  # value proj on {prev, q}
           + 2 * N * d * d                                        # output proj
           + N * 2 * tsa_pts * d * S)                             # sample 2 maps x P points
    sca = (2 * N * d * (n_heads * n_z * sca_pts_per_z * 2)
           + 2 * N * d * (n_heads * n_z * sca_pts_per_z)
           + 2 * n_feat * d * d                                   # value proj on image features
           + 2 * N * d * d
           + N * n_cams * n_z * sca_pts_per_z * d * S)
    ffn = 2 * N * d * fh * 2
    full_sa = 2 * N * d * d * 4 + 4 * N * N * d                   # QKV+O projections + QK^T and AV

    p_tsa = (2 * d + 1) * (2 * n_heads * tsa_pts * 3) + 2 * (d * d + d)
    p_sca = (d + 1) * (n_heads * n_z * sca_pts_per_z * 3) + 2 * (d * d + d)
    p_ffn = d * fh + fh + fh * d + d
    per_layer_params = p_tsa + p_sca + p_ffn + 3 * 2 * d
    return {
        "N_bev": N, "H_feat": Hf, "W_feat": Wf,
        "params_queries": N * d, "params_pos": (bev + bev) * (d // 2),
        "params_per_layer": per_layer_params, "params_layers": n_layers * per_layer_params,
        "params_backbone": bb_params,
        "params_total": bb_params + N * d + (bev + bev) * (d // 2) + n_layers * per_layer_params,
        "backbone_flops": backbone_flops,
        "tsa_flops": n_layers * tsa, "sca_flops": n_layers * sca, "ffn_flops": n_layers * ffn,
        "encoder_flops": n_layers * (tsa + sca + ffn),
        "total_flops": backbone_flops + n_layers * (tsa + sca + ffn),
        "full_self_attn_flops": n_layers * full_sa,
        "full_attn_scores_bytes": N * N * n_heads * dtype_bytes,  # one layer, if materialised
        "bev_map_bytes": N * d * dtype_bytes,
        "img_feat_bytes": n_feat * d * dtype_bytes,
        "ref_points_bytes": n_cams * N * n_z * 2 * 4,             # fp32 lookup table
    }


def compute_flops(model: BEVFormer, backbone: str = "stub") -> Dict:
    sca = model.layers[0].sca
    return bev_cost(bev=model.bev_h, d=model.d_model, img_h=model.img_h, img_w=model.img_w,
                    n_cams=model.n_cams, stride=model.backbone_stride,
                    n_layers=len(model.layers), n_heads=sca.n_heads, n_z=sca.n_z,
                    sca_pts_per_z=sca.n_pts, tsa_pts=model.layers[0].tsa.n_pts,
                    backbone=backbone)


def print_model_summary(model: BEVFormer):
    print("=" * 64)
    print("BEVFormer Model Summary")
    print("=" * 64)
    backbone = count_parameters(model.backbone)
    queries = model.bev_queries.numel()
    pos = count_parameters(model.bev_pos)
    per_layer = count_parameters(model.layers[0])
    total = count_parameters(model)
    print(f"Image backbone (toy, shared):  {backbone:>12,}  ({backbone / 1e6:.2f}M)")
    print(f"BEV queries ({model.bev_h}x{model.bev_w}x{model.d_model}):  {queries:>12,}  ({queries / 1e6:.2f}M)")
    print(f"BEV pos. embedding (row+col):  {pos:>12,}  ({pos / 1e6:.2f}M)")
    print(f"Encoder layers x {len(model.layers)}:          {per_layer * len(model.layers):>12,}  "
          f"({per_layer * len(model.layers) / 1e6:.2f}M, {per_layer:,} per layer)")
    print("-" * 64)
    print(f"Total:                         {total:>12,}  ({total / 1e6:.2f}M)")
    print("=" * 64)


def probe_point(x: float, y: float, z: float, img_h: int = 448, img_w: int = 800):
    """Project one ego point with the demo rig and print it per camera — the guide's
    interactive projection widget uses the identical rig, so the numbers match."""
    K, T = make_camera_rig(img_h, img_w)
    _, valid, uv = project_ego_to_image(torch.tensor([[x, y, z]]), K, T, img_h, img_w,
                                        return_pixels=True)
    p_cam = (T[0] @ torch.tensor([x, y, z, 1.0]))[:, :3]          # [N_cam, 3]
    print(f"\nProbe: ego point (x={x}, y={y}, z={z}) with a {img_w}x{img_h} rig, "
          f"f = {0.625 * img_w:.0f} px, cameras at 1.6 m")
    for c, name in enumerate(CAMERA_NAMES):
        X, Y, Z = p_cam[c].tolist()
        u, v = uv[0, c, 0].tolist()
        tag = "VISIBLE" if valid[0, c, 0] else ("behind" if Z <= 0 else "outside image")
        uv_txt = f"u={u:7.1f}  v={v:7.1f}" if Z > 0 else "u=   -     v=   -   "
        print(f"  {name:<12} X={X:6.2f} Y={Y:6.2f} Z={Z:6.2f}  {uv_txt}  {tag}")


def demo_forward_pass():
    print("\n" + "=" * 64)
    print("Forward Pass Demo (BEV-S, 6 cameras, 2 consecutive frames)")
    print("=" * 64)
    torch.manual_seed(0)
    model = bev_small().eval()
    B = 1
    H, W = model.img_h, model.img_w
    intrinsics, ego_to_cam = make_camera_rig(H, W, batch=B)
    images = torch.randn(B, model.n_cams, 3, H, W)
    print(f"\n1. Images:           {list(images.shape)}")
    print(f"   intrinsics:       {list(intrinsics.shape)}   ego_to_cam: {list(ego_to_cam.shape)}")

    with torch.no_grad():
        feats = model.backbone(images.reshape(-1, 3, H, W))
        print(f"2. Image features:   {list(feats.shape)}  (B*N_cam, d, H/16, W/16)")
        ref_uv, valid = project_ego_to_image(model.bev_pillars, intrinsics, ego_to_cam, H, W)
        hits = valid.any(-1).sum(1).float()                       # cameras hit per query
        print(f"3. Pillar refs:      {list(ref_uv.shape)}  (B, N_cam, N_q, N_z, 2)")
        print(f"   valid anchors:    {valid.float().mean().item() * 100:.1f}% of (cam, cell, height)")
        print(f"   cameras hit/cell: mean {hits.mean().item():.2f}, "
              f"min {hits.min().item():.0f}, max {hits.max().item():.0f}")
        bev_t0 = model(images, intrinsics, ego_to_cam)
        print(f"\n4. BEV map, frame t: {list(bev_t0.shape)}")
        prev = bev_t0.reshape(B, -1, model.d_model)
        bev_t1 = model(torch.randn_like(images), intrinsics, ego_to_cam, prev_bev=prev)
        print(f"   BEV map, frame t+1 (uses frame t via temporal self-attn): {list(bev_t1.shape)}")
    print("=" * 64)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 64)
    print("BEVFormer (BEV Tokenizer) Reference Implementation")
    print("=" * 64)

    model = bev_base()
    print_model_summary(model)

    c = compute_flops(model)
    r101 = compute_flops(model, backbone="resnet101")
    G = 1e9
    print(f"\nFLOPs per frame (BEV-B geometry, analytic model in bev_cost):")
    print(f"  Toy backbone x {model.n_cams} cameras:          {c['backbone_flops'] / G:>9.1f} GFLOPs")
    print(f"  ResNet-101 x {model.n_cams} cameras (estimate): {r101['backbone_flops'] / G:>9.1f} GFLOPs  <- real models")
    print(f"  Temporal self-attn (deformable):   {c['tsa_flops'] / G:>9.1f} GFLOPs")
    print(f"  Spatial cross-attn (deformable):   {c['sca_flops'] / G:>9.1f} GFLOPs")
    print(f"  FFN:                               {c['ffn_flops'] / G:>9.1f} GFLOPs")
    print(f"  BEV encoder total:                 {c['encoder_flops'] / G:>9.1f} GFLOPs")
    print(f"  (for contrast) FULL self-attn over {c['N_bev']:,} queries: "
          f"{c['full_self_attn_flops'] / G:,.0f} GFLOPs, "
          f"{c['full_attn_scores_bytes'] / 1e9:.1f} GB of FP16 scores per layer")

    probe_point(10.0, 2.0, 0.0)

    try:
        demo_forward_pass()
    except Exception as e:                                        # pragma: no cover
        print(f"\n(Skipping demo: {e})")

    print("\n" + "=" * 64)
    print("Model Comparison")
    print("=" * 64)
    print(f"{'Model':<8} {'Params':>9} {'Encoder GFLOPs':>15} {'BEV grid':>10}")
    print("-" * 46)
    for name, fn in [("BEV-S", bev_small), ("BEV-B", bev_base)]:
        m = fn()
        print(f"{name:<8} {count_parameters(m) / 1e6:>8.1f}M "
              f"{compute_flops(m)['encoder_flops'] / 1e9:>15.1f} {m.bev_h:>6}x{m.bev_w}")
