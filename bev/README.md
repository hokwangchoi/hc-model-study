# BEV — Multi-Camera Tokenizer

Based on [**BEVFormer** (Li et al., 2022)](https://arxiv.org/abs/2203.17270): learnable BEV queries, camera-projected pillar reference points, deformable spatial cross-attention to multi-view camera features, and deformable temporal self-attention to the previous frame's BEV map. Related approaches: [LSS (Lift-Splat-Shoot, Philion & Fidler 2020)](https://arxiv.org/abs/2008.05711) for the depth-prediction path, and 3D occupancy ([TPVFormer](https://arxiv.org/abs/2302.07817), [Occ3D](https://arxiv.org/abs/2304.14365)) for the voxel-grid generalization.

Module 7 of 9 in the learning path (previous: Mixture-of-Experts LLMs, next: Vision-Language-Action).

## Contents

| File | Description |
|------|-------------|
| [bev_guide.html](./bev_guide.html) | Interactive visual guide (open in a browser): projection explorer, calculator, self-check, exercises |
| [bev.py](./bev.py) | Reference implementation with annotations and an analytic cost model (`bev_cost`) |

## Quick Start

```bash
open bev_guide.html       # macOS
xdg-open bev_guide.html   # Linux
python bev.py             # CPU is fine: prints params, FLOPs, a projection probe, a 2-frame forward pass
```

## The core problem

A self-driving car has several cameras mounted around it (nuScenes: 6 cameras covering 360°; the Waymo Open Dataset: 5 cameras covering about 252°). Each camera sees a different slice of the world. Downstream detection, mapping, forecasting and planning want ONE coherent top-down view in a fixed ego-centric frame.

The "BEV tokenizer" turns multi-view images into a dense 2D feature map in bird's-eye view:

```
    6 cameras [B, 6, 3, H, W]
         ↓  (shared per-camera backbone)
    Image features [B, 6, C, H', W']
         ↓  (BEV queries: temporal self-attn → spatial cross-attn → FFN, x L)
    BEV feature map [B, H_bev, W_bev, d]
         ↓
    Detection / segmentation / planning heads
```

## Two approaches to the same problem

### 1. Lift-Splat-Shoot (LSS): "push"

For each pixel, predict a depth distribution over discrete bins, lift the pixel to a frustum of 3D points weighted by those probabilities, then **splat** (sum-pool) them into BEV cells.

- Pros: simple, parallel, no iterative attention
- Cons: monocular depth is ill-posed, so quality is capped by the depth network

### 2. BEVFormer (this module): "pull"

Start with **learnable BEV queries**, one vector per cell. Each cell is lifted into a **pillar of 4 heights**, every height is projected into every camera, and **deformable attention** samples a few points around the projected **reference points** in the cameras that are hit (usually 1–2 of 6), then averages over those cameras.

- Pros: no explicit depth network; geometry is built in; temporal context is cheap
- Cons: more complex attention; needs a fused sampling kernel to be fast

## BEVFormer-base at a glance (paper + official config)

| Setting | Value |
|---------|-------|
| Input | 6 views, 900×1600 (padded to 928×1600) |
| Backbone | ResNet-101-DCN + FPN, C = 256 |
| BEV grid | 200×200, 0.512 m per cell (±51.2 m) |
| Encoder | 6 layers of temporal self-attn + spatial cross-attn + FFN (FFN width 512) |
| Pillar | 4 height anchors in −5 m … 3 m (LiDAR-centred frame) |
| Deformable attention | 8 heads; SCA 8 points/head (2 per height), TSA 4 points/head |

## What `bev.py` implements

- `make_camera_rig`: a 6-camera ring (yaws 0, ±60°, ±120°, 180°), f = 0.625·W, cameras 1.6 m up. `probe_point(10, 2, 0)` reproduces the guide's worked example (front camera: u = 300, v = 304).
- `make_bev_pillar_points` + `project_ego_to_image`: pillar points, pinhole projection, validity mask, `grid_sample(align_corners=False)` normalisation (`2u/W − 1`), the same convention as BEVFormer's `point_sampling`.
- `SpatialCrossAttention`: camera-independent offsets / weights, real per-head channels, softmax per head over (height × point), average over hit cameras (paper Eq. 2).
- `TemporalSelfAttention`: deformable attention over {previous BEV, current queries}, offsets predicted from their concatenation, mean over the two maps; `{Q, Q}` on the first frame.
- `LearnedPositionalEncoding2D`: factorised row + column embedding (51,200 parameters at 200×200, d = 256).
- `bev_cost`: the analytic FLOP / parameter / memory model the guide's calculator uses (it matches the real parameter count exactly).

Simplifications: a toy CNN backbone with a single feature level, no ego-motion warp of the previous BEV, every camera sampled then masked (the official code gathers only hit queries per camera), pre-norm residuals, no heads.

## Key inference facts

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Image backbone | O(N_cam · H · W) | The dominant cost: about 87% of FLOPs for BEVFormer-base geometry with a ResNet-101 estimate |
| Camera projection | O(N_bev · N_z · N_cam) | Static for a fixed rig: precompute (7.68 MB of FP32 reference points at 200×200, 4 heights, 6 cameras) |
| Spatial cross-attention | O(N_bev · hit cams · heads · points) | Deformable: 64 samples per query per hit camera instead of 34,800 feature pixels |
| Temporal self-attention | O(N_bev · heads · points) | Linear in cells; carries one previous BEV map (≈ 20 MB in FP16) |
| FFN | O(N_bev · d²) | Plain GEMMs |

**Why there is no dense BEV self-attention:** 40,000 queries → 1.6 billion scores per head per layer, 25.6 GB of FP16 scores for 8 heads, and 1.64 TFLOPs per layer for QKᵀ and AV alone.

### Measured latency (BEVFormer paper, Table 6, V100, R101-DCN, 900×1600)

| Config | Backbone | BEV encoder | Head | FPS | NDS |
|--------|---------:|------------:|-----:|----:|----:|
| BEVFormer (multi-scale, 200×200, 6 layers) | 391 ms | 130 ms | 19 ms | 1.7 | 0.517 |
| 1 encoder layer | 391 ms | 25 ms | 19 ms | 2.1 | 0.501 |
| single-scale, 100×100, 1 layer | 387 ms | 7 ms | 18 ms | 2.3 | 0.478 |

The backbone is the bottleneck. Optimize resolution, backbone size and precision first.

### Quantization (NVIDIA DL4AGX BEVFormer-tiny sample, A40, TensorRT 10.14)

FP32 18.68 ms → FP16 9.27 ms → INT8 (ModelOpt explicit quantization) 5.76 ms, NDS 0.355 → 0.352. The sample leaves the attention MatMuls in FP16. Source: [NVIDIA/DL4AGX bevformer-int8-eq](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/bevformer-int8-eq).

### Temporal self-attention pays off (paper Table 2, nuScenes val, R101-DCN)

| Model | NDS | mAP | mAVE (m/s) |
|-------|----:|----:|-----------:|
| BEVFormer-S (no temporal) | 0.448 | 0.375 | 0.802 |
| BEVFormer | 0.517 | 0.416 | 0.394 |

## Parameters and FLOPs of `bev.py` (from `python bev.py`)

| | BEV-S (100×100, d=192, 3 layers) | BEV-B (200×200, d=256, 6 layers) |
|---|---:|---:|
| Toy backbone | 0.85 M | 1.03 M |
| BEV query table | 1.92 M | 10.24 M |
| Positional embedding | 0.02 M | 0.05 M |
| Encoder layers | 1.14 M | 4.05 M |
| **Total** | **3.9 M** | **15.4 M** |
| Encoder FLOPs / frame | 27.8 G | 384.2 G |

For BEV-B geometry, six ResNet-101 passes at 928×1600 are ≈ 2.7 TFLOPs (scaling the paper's 7.6 G multiply-adds at 224×224 by pixel count), about 7× the whole BEV encoder.

## Edge hardware reference (NVIDIA spec pages)

- Jetson AGX Orin 64GB: 275 sparse INT8 TOPS total (GPU: 85 dense INT8 TOPS), 204.8 GB/s
- Jetson AGX Thor T5000: 2070 sparse FP4 TFLOPS, 128 GB LPDDR5X, 273 GB/s

Headline numbers are sparse, low-precision peaks; measure with `trtexec` on the device.

## As of Oct 2026

BEVFormer remains the clearest way to learn camera-to-BEV lifting. Since then: sparse query-based temporal detectors ([StreamPETR](https://arxiv.org/abs/2303.11926)); end-to-end planners on BEV features ([UniAD](https://arxiv.org/abs/2212.10156), [VAD](https://arxiv.org/abs/2303.12077)); and driving VLAs that consume camera tokens directly with no BEV grid ([EMMA](https://arxiv.org/abs/2410.23262); NVIDIA [Alpamayo-R1](https://arxiv.org/abs/2511.00088) and the 34B [Alpamayo 2 Super](https://huggingface.co/nvidia/Alpamayo2-Super), released August 2026).

## References

- [BEVFormer (Li et al., 2022)](https://arxiv.org/abs/2203.17270) and [official code](https://github.com/fundamentalvision/BEVFormer)
- [BEVFormer v2 (Yang et al., 2022)](https://arxiv.org/abs/2211.10439)
- [LSS (Philion & Fidler, 2020)](https://arxiv.org/abs/2008.05711)
- [Deformable DETR (Zhu et al., 2020)](https://arxiv.org/abs/2010.04159)
- [ResNet (He et al., 2015)](https://arxiv.org/abs/1512.03385)
- [StreamPETR (Wang et al., 2023)](https://arxiv.org/abs/2303.11926)
- [UniAD (Hu et al., 2023)](https://arxiv.org/abs/2212.10156), [VAD (Jiang et al., 2023)](https://arxiv.org/abs/2303.12077)
- [TPVFormer (Huang et al., 2023)](https://arxiv.org/abs/2302.07817), [Occ3D (Tian et al., 2023)](https://arxiv.org/abs/2304.14365)
- [EMMA (Waymo, 2024)](https://arxiv.org/abs/2410.23262), [Alpamayo-R1 (NVIDIA, 2025)](https://arxiv.org/abs/2511.00088)
- [nuScenes](https://www.nuscenes.org/), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
