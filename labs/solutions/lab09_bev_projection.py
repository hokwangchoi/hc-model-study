"""Lab 09: from a bird's-eye-view grid to camera pixels (BEVFormer geometry) (reference solution)

Goal
----
A BEV model builds one top-down feature map around the car from several
cameras. The geometric core is plain linear algebra: lift every BEV cell to
a few 3-D points, project them into each camera, keep the ones that land
inside an image, and bilinearly sample the image features there. You build
camera intrinsics and extrinsics, the projection, the BEV pillar points,
the reference points with their visibility mask, and the feature sampling
with F.grid_sample.

The maths, step by step
-----------------------
Frames:  ego     x forward, y left, z up (origin on the ground under the car)
         camera  x right,   y down, z forward = depth (OpenCV convention)

  1. Intrinsics K (pinhole):   [[f, 0, cx], [0, f, cy], [0, 0, 1]],  cx = W/2, cy = H/2
  2. Extrinsics ego -> camera for a camera at height h, turned by yaw psi
     (counter-clockwise from +x, towards +y = left):
       forward  z_c = ( cos psi,  sin psi,  0)     (optical axis, in ego coords)
       right    x_c = ( sin psi, -cos psi,  0)
       down     y_c = ( 0,        0,       -1)
       R = [x_c; y_c; z_c]  (rows),  C = (0, 0, h)  camera centre
       p_cam = R (p_ego - C) = R p_ego + t,  t = -R C
     stored as a 4 x 4 matrix [[R, t], [0, 0, 0, 1]].
  3. Projection:  (X, Y, Z) = p_cam;   u = f X / Z + cx,   v = f Y / Z + cy
     Z is the depth; a point with Z <= 0 is behind the camera.
  4. BEV pillars: cell (i, j) of a bev_h x bev_w grid with resolution r metres
       x = (i - (bev_h - 1) / 2) * r     (forward)
       y = (j - (bev_w - 1) / 2) * r     (left)
     lifted to N_z heights z_1..z_Nz. Query index q = i * bev_w + j.
       points: [Q = bev_h * bev_w, N_z, 3]
  5. Reference points and mask, per camera:
       uv_norm = (2u / W - 1, 2v / H - 1)          in [-1, 1] inside the image
       valid   = (Z > eps) and (0 < u < W) and (0 < v < H)
     ([-1, 1] coordinates are resolution-free: the same uv_norm samples the
     image or a stride-16 feature map of it.)
  6. Sampling: F.grid_sample(features [N_cam, C, Hf, Wf], uv_norm as grid
     [N_cam, Q, N_z, 2], bilinear, align_corners=False) -> [N_cam, C, Q, N_z].
     BEVFormer's spatial cross-attention then averages over the cameras that
     see the cell (and mixes the heights with learned attention weights);
     here a plain average over the valid cameras and heights stands in for it.

  Worked example (guide, front camera, W x H = 800 x 448, f = 500, h = 1.6 m):
     p = (10, 2, 0) -> p_cam = (-2, 1.6, 10) -> (u, v) = (300, 304)
     -> uv_norm = (-0.25, 0.357). The rear camera sees Z = -10: invalid.

Read alongside
--------------
  ../bev/bev_guide.html   "The Core Problem: 6 Cameras -> One Top-Down Map",
                          3. Camera Projection (worked example, #proj-widget),
                          4. Deformable Spatial Cross-Attention
  ../bev/bev.py           make_camera_rig, project_ego_to_image, make_bev_pillar_points

Run:  python labs/lab09_bev_projection.py   (or: python labs/check.py 9)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab09_bev_projection.py
"""

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: intrinsics and extrinsics
# ---------------------------------------------------------------------------
def make_intrinsics(f: float, img_w: int, img_h: int) -> torch.Tensor:
    """K [3, 3] float32 with focal length f (pixels) and principal point at the image centre."""
    return torch.tensor([[f, 0.0, img_w / 2], [0.0, f, img_h / 2], [0.0, 0.0, 1.0]])


def make_ego_to_cam(yaw_deg: float, cam_height: float) -> torch.Tensor:
    """[4, 4] float32 ego -> camera transform for a camera at (0, 0, cam_height) facing yaw_deg."""
    th = math.radians(yaw_deg)
    z_c = torch.tensor([math.cos(th), math.sin(th), 0.0])       # optical axis (forward)
    x_c = torch.tensor([math.sin(th), -math.cos(th), 0.0])      # image right
    y_c = torch.tensor([0.0, 0.0, -1.0])                        # image down
    R = torch.stack([x_c, y_c, z_c])                            # rows = camera axes
    C = torch.tensor([0.0, 0.0, cam_height])
    T = torch.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ C                                           # p_cam = R (p_ego - C)
    return T


# ---------------------------------------------------------------------------
# Step 2: project 3-D points into one camera
# ---------------------------------------------------------------------------
def project_points(points: torch.Tensor, K: torch.Tensor, ego_to_cam: torch.Tensor):
    """points [P, 3] in the ego frame -> (uv [P, 2] pixels, depth [P]).

    Divide by the depth clamped to at least 1e-5 so points behind the camera do
    not produce inf (they are marked invalid later anyway).
    """
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    p_h = torch.cat([points, ones], dim=1)                      # [P, 4]
    p_cam = (p_h @ ego_to_cam.T)[:, :3]                         # [P, 3] camera frame
    pix = p_cam @ K.T                                           # [P, 3] = (fX + cx Z, fY + cy Z, Z)
    depth = pix[:, 2]
    uv = pix[:, :2] / depth.clamp(min=1e-5)[:, None]
    return uv, depth


# ---------------------------------------------------------------------------
# Step 3: BEV pillar points
# ---------------------------------------------------------------------------
def bev_pillar_points(bev_h: int, bev_w: int, resolution: float,
                      z_anchors=(0.0, 1.0, 2.0, 3.0)) -> torch.Tensor:
    """[bev_h * bev_w, N_z, 3] ego-frame points; cell (i, j) -> query q = i * bev_w + j."""
    x = (torch.arange(bev_h) - (bev_h - 1) / 2) * resolution   # forward
    y = (torch.arange(bev_w) - (bev_w - 1) / 2) * resolution   # left
    xx, yy = torch.meshgrid(x, y, indexing="ij")               # [bev_h, bev_w]
    n_z = len(z_anchors)
    zs = torch.tensor(z_anchors, dtype=xx.dtype)
    pts = torch.stack([xx[..., None].expand(-1, -1, n_z),
                       yy[..., None].expand(-1, -1, n_z),
                       zs.expand(bev_h, bev_w, n_z)], dim=-1)   # [bev_h, bev_w, N_z, 3]
    return pts.reshape(bev_h * bev_w, n_z, 3)


# ---------------------------------------------------------------------------
# Step 4: reference points and visibility for every camera
# ---------------------------------------------------------------------------
def reference_points(points: torch.Tensor, Ks: torch.Tensor, ego_to_cams: torch.Tensor,
                     img_h: int, img_w: int):
    """Project pillar points into every camera.

    Args:
        points: [Q, N_z, 3]   Ks: [N_cam, 3, 3]   ego_to_cams: [N_cam, 4, 4]
    Returns:
        uv_norm: [N_cam, Q, N_z, 2]  (2u/W - 1, 2v/H - 1)
        valid:   [N_cam, Q, N_z] bool, depth > 1e-5 and strictly inside the image
    """
    Q, n_z, _ = points.shape
    flat = points.reshape(-1, 3)
    uvs, valids = [], []
    for K, T in zip(Ks, ego_to_cams):
        uv, depth = project_points(flat, K, T)
        u, v = uv[:, 0], uv[:, 1]
        valids.append((depth > 1e-5) & (u > 0) & (u < img_w) & (v > 0) & (v < img_h))
        uvs.append(torch.stack([2 * u / img_w - 1, 2 * v / img_h - 1], dim=-1))
    n_cam = len(Ks)
    return (torch.stack(uvs).reshape(n_cam, Q, n_z, 2),
            torch.stack(valids).reshape(n_cam, Q, n_z))


# ---------------------------------------------------------------------------
# Step 5: sample image features and fuse them into BEV features
# ---------------------------------------------------------------------------
def sample_features(features: torch.Tensor, uv_norm: torch.Tensor) -> torch.Tensor:
    """Bilinear sampling at the reference points.

    features: [N_cam, C, Hf, Wf]   uv_norm: [N_cam, Q, N_z, 2]
    Returns [N_cam, Q, N_z, C]. Use F.grid_sample(mode="bilinear",
    padding_mode="zeros", align_corners=False): uv_norm treats the image edges
    as -1 and +1, which is the align_corners=False convention.
    """
    out = F.grid_sample(features, uv_norm, mode="bilinear", padding_mode="zeros",
                        align_corners=False)                    # [N_cam, C, Q, N_z]
    return out.permute(0, 2, 3, 1)                              # [N_cam, Q, N_z, C]


def aggregate_views(sampled: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average the valid samples of each BEV cell over cameras and heights.

    sampled [N_cam, Q, N_z, C], valid [N_cam, Q, N_z] -> bev [Q, C].
    Cells that no camera sees get zeros.
    """
    m = valid[..., None].to(sampled.dtype)                      # [N_cam, Q, N_z, 1]
    total = (sampled * m).sum(dim=(0, 2))                       # [Q, C]
    count = m.sum(dim=(0, 2)).clamp(min=1.0)                    # [Q, 1], avoid 0 / 0
    return total / count


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
