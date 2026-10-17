"""Lab 09: from a bird's-eye-view grid to camera pixels (BEVFormer geometry)

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

import torch
# You will need two more imports: math (cos, sin, radians) and
# torch.nn.functional as F (F.grid_sample).


# ---------------------------------------------------------------------------
# Step 1: intrinsics and extrinsics
# ---------------------------------------------------------------------------
def make_intrinsics(f: float, img_w: int, img_h: int) -> torch.Tensor:
    """K [3, 3] float32 with focal length f (pixels) and principal point at the image centre."""
    # Hint 1: a 3 x 3 torch.tensor; write 0.0 rather than 0 so the dtype is float32.
    raise NotImplementedError  # TODO


def make_ego_to_cam(yaw_deg: float, cam_height: float) -> torch.Tensor:
    """[4, 4] float32 ego -> camera transform for a camera at (0, 0, cam_height) facing yaw_deg."""
    # Hint 1: build the three camera axes in ego coordinates (see the docstring), stack
    #         them as ROWS of R: then R @ p expresses p in camera axes.
    # Hint 2: t = -R @ C with C = (0, 0, cam_height).
    # Hint 3: T = torch.eye(4); T[:3, :3] = R; T[:3, 3] = t
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 2: project 3-D points into one camera
# ---------------------------------------------------------------------------
def project_points(points: torch.Tensor, K: torch.Tensor, ego_to_cam: torch.Tensor):
    """points [P, 3] in the ego frame -> (uv [P, 2] pixels, depth [P]).

    Divide by the depth clamped to at least 1e-5 so points behind the camera do
    not produce inf (they are marked invalid later anyway).
    """
    # Hint 1: homogeneous coordinates: append a column of ones -> [P, 4], then
    #         p_cam = (ego_to_cam @ p_h.T).T[:, :3]   (or p_h @ ego_to_cam.T).
    # Hint 2: pix = p_cam @ K.T gives (f X + cx Z, f Y + cy Z, Z) per row.
    # Hint 3: depth = pix[:, 2]; uv = pix[:, :2] / depth.clamp(min=1e-5)[:, None]
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 3: BEV pillar points
# ---------------------------------------------------------------------------
def bev_pillar_points(bev_h: int, bev_w: int, resolution: float,
                      z_anchors=(0.0, 1.0, 2.0, 3.0)) -> torch.Tensor:
    """[bev_h * bev_w, N_z, 3] ego-frame points; cell (i, j) -> query q = i * bev_w + j."""
    # Hint 1: x = (torch.arange(bev_h) - (bev_h - 1) / 2) * resolution; same for y with bev_w.
    # Hint 2: torch.meshgrid(x, y, indexing="ij") gives [bev_h, bev_w] grids xx, yy.
    # Hint 3: stack (xx, yy, z) for every z into [bev_h, bev_w, N_z, 3], then reshape.
    raise NotImplementedError  # TODO


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
    # Hint 1: loop over cameras and reuse project_points on points.reshape(-1, 3).
    # Hint 2: valid = (depth > 1e-5) & (u > 0) & (u < img_w) & (v > 0) & (v < img_h)
    # Hint 3: torch.stack the per-camera results and reshape to [N_cam, Q, N_z, ...].
    raise NotImplementedError  # TODO


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
    # Hint 1: grid_sample takes input [N, C, H_in, W_in] and grid [N, H_out, W_out, 2]
    #         (x first, then y) and returns [N, C, H_out, W_out]; here H_out = Q, W_out = N_z.
    # Hint 2: out = F.grid_sample(features, uv_norm, mode="bilinear", padding_mode="zeros",
    #                             align_corners=False)
    # Hint 3: return out.permute(0, 2, 3, 1)
    raise NotImplementedError  # TODO


def aggregate_views(sampled: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Average the valid samples of each BEV cell over cameras and heights.

    sampled [N_cam, Q, N_z, C], valid [N_cam, Q, N_z] -> bev [Q, C].
    Cells that no camera sees get zeros.
    """
    # Hint 1: zero the invalid samples, sum over cameras and heights, divide by the count.
    # Hint 2: m = valid[..., None].to(sampled.dtype); total = (sampled * m).sum(dim=(0, 2))
    # Hint 3: return total / m.sum(dim=(0, 2)).clamp(min=1)
    raise NotImplementedError  # TODO


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
