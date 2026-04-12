# Vision Transformer (ViT)

GPU inference study guide for Vision Transformer (ViT) architecture.

## Contents

- **vit_guide.html** — Interactive visual guide with diagrams, complexity analysis, and GPU optimization techniques
- **vit.py** — Reference PyTorch implementation with detailed comments

## Quick Start

```bash
# Run the reference implementation
python vit.py

# Open the visual guide
open vit_guide.html  # macOS
xdg-open vit_guide.html  # Linux
```

## Key Concepts

### Image → Patches → Tokens
ViT treats an image as a sequence of patches, each becoming a "token":
- 224×224 image with P=16 → 14×14 = 196 patches
- Each patch: 16×16×3 = 768 values → Linear projection → 768-dim embedding

### Architecture
```
Image [B,3,224,224]
  ↓ Patch Embed (Conv2d k=16 s=16)
Patches [B,196,768]
  ↓ Prepend [CLS] token
Tokens [B,197,768]
  ↓ + Position Embeddings
  ↓ Transformer Encoder × L
  ↓ Extract [CLS] token
Features [B,768]
  ↓ Linear Head
Logits [B,1000]
```

### Model Variants

| Model | Params | GFLOPs | Patch Size | Layers | Heads |
|-------|--------|--------|------------|--------|-------|
| ViT-B/16 | 86M | 17.6 | 16 | 12 | 12 |
| ViT-L/16 | 307M | 61.6 | 16 | 24 | 16 |
| ViT-H/14 | 632M | 167 | 14 | 32 | 16 |

## Key Differences from LLM Transformers

1. **No causal mask** — Bidirectional attention (all patches see all patches)
2. **Pre-norm** — LayerNorm before attention/FFN (not after)
3. **[CLS] token** — Learnable token prepended for classification
4. **Fixed sequence length** — N patches determined by image/patch size
5. **Small N** — 197 patches vs 4096+ tokens in LLMs (attention is NOT the bottleneck)

## Recommended Learning

1. Watch: [Yannic Kilcher's ViT explanation](https://www.youtube.com/watch?v=TrdevFK_am4) (35 min)
2. Read: [Original ViT paper](https://arxiv.org/abs/2010.11929)
3. Code: Study `vit.py` with the visual guide open
