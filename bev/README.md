# BEV — Multi-Camera Tokenizer

Based on [**BEVFormer** (Li et al., 2022)](https://arxiv.org/abs/2203.17270) — learnable BEV queries + deformable spatial cross-attention to multi-view camera features. The dominant perception pattern in modern autonomous driving stacks (Tesla's HydraNet-style production systems, Waymo, NVIDIA Drive, Wayve). Related approaches: [LSS (Lift-Splat-Shoot, Philion & Fidler 2020)](https://arxiv.org/abs/2008.05711) for the depth-prediction path, [Occupancy networks](https://arxiv.org/abs/2302.07817) for the 3D voxel-grid generalization.

## Contents

| File | Description |
|------|-------------|
| [bev_guide.html](./bev_guide.html) | Interactive visual guide (open in browser) |
| [bev.py](./bev.py) | Reference implementation with annotations |

## Quick Start

```bash
open bev_guide.html       # macOS
xdg-open bev_guide.html   # Linux
python bev.py
```

## The core problem

A self-driving car has 6 (or 8, or 11) cameras mounted around it. Each camera sees a different slice of the world. The downstream planning stack needs ONE coherent top-down view showing every object, lane, and obstacle in the car's vicinity, in a fixed ego-centric coordinate frame.

The "BEV tokenizer" takes multi-view camera images in and produces a dense 2D feature map in bird's-eye view as output. Everything downstream (3D object detection, map segmentation, motion forecasting, planning) runs on this feature map.

```
    6 cameras [B, 6, 3, H, W]
         ↓  (per-camera CNN backbone)
    Image features [B, 6, C, H', W']
         ↓  (cross-attention with BEV queries)
    BEV feature map [B, H_bev, W_bev, d]
         ↓
    Detection / segmentation / forecasting heads
```

## Two approaches to the same problem

### 1. Lift-Splat-Shoot (LSS) — the OG

For each pixel in each camera, predict a **depth distribution** over a discrete set of depth bins. Multiply pixel features by the depth distribution, "lift" each pixel to a frustum of 3D points weighted by depth probability. Then **splat** those points into BEV grid cells (bilinearly or nearest-neighbor).

- Pros: simple, parallelizable, no iterative attention
- Cons: depth prediction from a single monocular image is ill-posed; quality capped by how well the per-pixel depth network can predict

### 2. BEVFormer (this module's focus)

Start with **learnable BEV queries** — one trainable vector per BEV cell. Each query cross-attends to the camera features, using the cell's known 3D location to compute **reference points** (where this physical location projects to in each camera's image plane). Deformable attention samples only a few pixels per query per camera.

- Pros: no per-pixel depth network needed; the geometry is baked in via projection; scales to more cameras cleanly
- Cons: attention is more complex; needs a custom CUDA kernel to be fast

Modern production systems often combine ideas from both, but BEVFormer-style dominates research benchmarks (nuScenes, Waymo Open Dataset).

## Key Inference Bottlenecks

| Operation | Complexity | Bound | Optimization |
|-----------|------------|-------|--------------|
| Image backbones | O(N_cam · H · W · d²) | Compute | Shared weights across cameras, FP16, TensorRT |
| Camera projection | O(N_bev · N_cam) | Compute | Done once per frame; static for fixed cameras |
| Spatial cross-attention | O(N_bev · N_cam · K · d²) | Mixed | Deformable attention kernel (K=4 instead of H·W) |
| BEV self-attention | O(N_bev²·d) | **Memory** | Can be windowed or deformable too |
| FFN | O(N_bev · d²) | Compute | Standard GEMM |

**N_bev** = number of BEV cells (40,000 for a 200×200 grid).
**N_cam** = cameras (6 on nuScenes, 8 on Waymo).
**K** = deformable sample points per query per camera (typically 4).

Image backbones are the single biggest cost on real resolution (1600×900 per camera × 6 cameras).

## Critical Optimizations

### 1. Deformable spatial cross-attention

Full cross-attention from every BEV query to every image pixel is `N_bev × N_cam × H_feat × W_feat` — ~2B attention scores per layer for BEVFormer-base. Intractable.

Deformable attention samples only **K points per query per camera**:

```
For each BEV query q in cell (i, j):
  3D point p = bev_grid[i, j]  (in ego frame)
  For each camera c:
    ref_point_c = project(p, K_c, T_c)         # 2D reference pixel
    offsets_{c,k} = Linear(q)                   # K predicted offsets
    weights_{c,k} = softmax(Linear(q))          # K attention weights
    sampled_{c,k} = grid_sample(feat_c, ref_point_c + offsets_{c,k})
  output(q) = sum_c,k  weights_{c,k} · sampled_{c,k}
```

Complexity drops from O(N_bev · N_cam · H · W) to O(N_bev · N_cam · K). For BEVFormer-base at K=4, that's ~256× fewer memory accesses. Production uses a custom CUDA kernel (`MultiScaleDeformableAttention`) — installed with MMDetection3D, adapted in NVIDIA driveOS.

### 2. Shared image backbones

All 6 cameras share a single CNN backbone's weights. This is parameter-efficient AND lets you batch the cameras:

```python
# Stack the cameras into the batch dimension
imgs = images.reshape(B * N_cam, 3, H, W)
feats = backbone(imgs)                         # one GPU kernel, N_cam × more work
feats = feats.reshape(B, N_cam, C, H', W')
```

Concretely: instead of 6 separate backbone calls, one call with 6× batch size. TensorRT can further fuse the whole backbone into a single engine for each camera stream.

### 3. Static camera geometry precomputation

For a fixed camera rig (as on production vehicles), the `bev_grid → image` projection is **deterministic given intrinsics/extrinsics**. You can precompute:
- Reference points (2D pixel coords in each camera for each BEV cell)
- Validity masks (which BEV cells are visible to which cameras)

At inference, these become lookup tables. Only the deformable offsets (predicted from queries) are computed per-frame.

### 4. Temporal aggregation for free

BEVFormer's temporal self-attention attends to previous-frame BEV features, warped by ego-motion. You get:
- Better velocity estimation (moving objects are obvious across frames)
- Occlusion resolution (partially seen objects accumulate info over time)
- ~1 ms per layer on top of per-frame cost

The cost is modest because you only carry ONE previous BEV feature map (~100 MB at 200×200×256 FP16), not a growing history. Analogous to Mamba's fixed-state story: per-frame cost stays constant.

### 5. INT8 quantization works well on BEV

- Backbone: INT8 PTQ is mature (same as any CNN)
- Cross-attention projections: INT8 is fine
- Deformable kernel: usually kept in FP16 (grid_sample precision matters)
- BEV queries + pos embeddings: FP16 (small, always loaded)

Gives ~2× throughput on Orin with <1% mAP drop.

## Architecture Summary

```
Input:  images [B, 6, 3, H_img, W_img]
        intrinsics [B, 6, 3, 3]
        ego_to_cam [B, 6, 4, 4]
    ↓
Per-camera backbone (shared weights)
    ↓
Image features [B, 6, d, H_img/s, W_img/s]
    ↓
                        ← learnable BEV queries [H_bev·W_bev, d]
                        ← learnable BEV pos embed
                        ← static BEV→image reference points (from projection)
BEVFormer layers × L
  (each: self-attn over BEV queries
        + deformable spatial cross-attn to cameras
        + FFN)
    ↓
BEV feature map [B, H_bev, W_bev, d]
    ↓
Detection / segmentation / motion / planning heads  (not in this file)
```

## Parameters & FLOPs

For a BEV-B-like config (200×200 grid, d=256, 6 layers, 6 cameras, 1600×928 images):

| Component | Params (stub backbone) | FLOPs/frame (est.) |
|-----------|-----------------------:|--------------------:|
| Image backbone (× 6 cameras) | ~1M | ~500 GFLOPs |
| BEV queries | 10.2M | 0 (static) |
| BEV pos embedding | 10.2M | 0 (static) |
| Self-attention (L=6) | ~1.6M | ~300 GFLOPs (N² term) |
| Cross-attention (L=6) | ~1.7M | ~25 GFLOPs (K=4) |
| FFN (L=6) | ~3.1M | ~50 GFLOPs |
| **Total** | **~28M** | **~875 GFLOPs** |

Swap the stub backbone for a real ResNet-50 (~25M, ~500 GFLOPs for 6 cameras at 1600×928) and you're at ~55M, ~1.4 TFLOPs, roughly matching published BEVFormer-base numbers.

The self-attention N² term is the non-obvious cost: for 40,000 BEV queries, a full O(N²) self-attention is 1.6 billion attention scores per layer. In practice this is windowed or replaced with sparse alternatives.

## Key Differences from LLMs / VLMs / Diffusion

| Aspect | BEV | Transformer LLM | VLM |
|--------|-----|-----------------|-----|
| Input | Multi-view images + camera matrices | Token IDs | Image + text |
| Output | Dense 2D feature map | Token logits | Token logits |
| Geometry awareness | **Explicit** (camera projection in the model) | None | None |
| Autoregressive? | No — single forward pass | Yes | Yes |
| Runtime target | 10–20 Hz hard real-time | Interactive/streaming | Interactive |
| Typical batch size | 1 (per car) | Many | Moderate |
| Quantization sweet spot | INT8 mixed | W4A16 | W4A16 |

## On Edge Hardware — Orin / Drive

- Production targets: 10 Hz (100 ms budget) minimum, 20 Hz (50 ms budget) for highway speeds
- Orin AGX (275 TOPS int8) can run a real BEVFormer-base at ~15–20 Hz with INT8 backbone + deformable CUDA kernel + TensorRT fusion
- **Memory budget is tight**: BEV feature map alone at 200×200×256 FP16 is ~20 MB; × batch × layers of intermediates can blow past the unified memory budget
- **Latency is dominated by backbones**: a ResNet-50 at 1600×900 is ~2 GFLOPs/frame/camera, × 6 cameras = 12 GFLOPs for the backbone path alone. Backbone replacement (smaller ResNet, ViT-Tiny) is the first knob to turn.

Common tricks for tight budgets:
- Reduce image resolution (1600×900 → 800×448 → 4× FLOPs saving at modest mAP cost)
- Reduce BEV grid (200×200 → 128×128 → ~2.5× N² saving in self-attention)
- Skip temporal attention (save 1 layer-worth of ops)
- Use a frozen backbone from Waymo Foundation Model / similar for features

## Connection to the Roadmap

BEV connects to the models ahead:
- **VLA models (Pi0, OpenVLA, RoboCat)**: the perception stack feeds the VLA's vision encoder. A robot VLM that consumes 6-camera input will reuse a BEV-style front end.
- **Occupancy networks**: a direct generalization — replace 2D BEV grid with 3D voxel grid. Same cross-attention mechanics, extra Z dimension.
- **World models / trajectory diffusion**: the BEV feature map is the conditioning input for a diffusion-based planner predicting future trajectories.

## References

- [BEVFormer (Li et al., 2022)](https://arxiv.org/abs/2203.17270) — the main paper
- [BEVFormer v2 (Yang et al., 2022)](https://arxiv.org/abs/2211.10439) — two-stage, stronger backbones
- [LSS (Philion & Fidler, 2020)](https://arxiv.org/abs/2008.05711) — the depth-prediction alternative
- [Deformable DETR (Zhu et al., 2020)](https://arxiv.org/abs/2010.04159) — deformable attention foundation
- [Occupancy Networks (Tian et al., 2023)](https://arxiv.org/abs/2302.07817) — 3D voxel generalization
- [nuScenes](https://www.nuscenes.org/) — the standard BEV benchmark dataset
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) — reference implementations
