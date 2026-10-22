# Vision Transformer (ViT)

GPU inference study guide for Vision Transformer (ViT) architecture.

## Contents

- **vit_guide.html** — Interactive visual guide (Module 2 of 9): diagrams, a position-embedding widget, a FLOPs/params calculator that matches `vit.py`, GPU optimization notes, a self-check quiz and exercises
- **vit.py** — Reference PyTorch implementation with detailed comments, a MAC/FLOP counter, and two sanity checks (Conv2d ≡ patchify + Linear; patch shuffle without position embeddings)

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
- Each patch: 16×16×3 = 768 pixel values → Linear projection → d-dim token (d = 768 for ViT-B, 1024 for ViT-L; the 768 = 768 match in ViT-B/16 is a coincidence)

### Architecture
```
Image [B,3,224,224]
  ↓ Patch Embed (Conv2d k=16 s=16)
Patches [B,196,768]
  ↓ Prepend [CLS] token
Tokens [B,197,768]
  ↓ + Position Embeddings
  ↓ Transformer Encoder × L
  ↓ Final LayerNorm
  ↓ Extract [CLS] token
Features [B,768]
  ↓ Linear Head
Logits [B,1000]
```

### Model Variants

| Model | Params | GMACs (papers call these "GFLOPs") | GFLOPs (= 2 × MACs) | Patch Size | Layers | Heads |
|-------|--------|-------|--------|------------|--------|-------|
| ViT-B/16 | 86.6M | 17.6 | 35.1 | 16 | 12 | 12 |
| ViT-L/16 | 304.3M (paper Table 1: 307M) | 61.6 | 123.1 | 16 | 24 | 16 |
| ViT-H/14 | 632.0M | 167.3 | 334.6 | 14 | 32 | 16 |

224×224 input, 1000-class head; numbers printed by `python vit.py`. Use the GFLOPs column
when dividing by a GPU's peak TFLOPS.

## Key Differences from (Decoder-Only) LLM Transformers

1. **No causal mask** — Bidirectional attention (all patches see all patches)
2. **No tokenizer** — a Conv2d (k = s = P) projects real-valued pixel patches instead of an embedding lookup
3. **[CLS] token** — Learnable token prepended for classification; read out through one Linear head
4. **Fixed sequence length** — T = (H/P)² + 1 tokens for a given resolution (Module 1 calls the token count N; here N = (H/P)² counts patches) → static shapes
5. **No KV cache, no decode loop** — one forward pass per image
6. **Small T at 224 px** — 197 tokens, so attention's T² part is ~4% of compute and the FFN (~64%) dominates. At detection/VLM resolutions (e.g. SAM at 1024² → 4096 patches) attention cost matters again

The block itself is the same pre-norm block as modern LLMs (GPT-2, LLaMA); only the
original 2017 Transformer and BERT were post-norm.

## Recommended Learning

1. Watch: [Yannic Kilcher's ViT explanation](https://www.youtube.com/watch?v=TrdevFK_am4) (30 min)
2. Read: [Original ViT paper](https://arxiv.org/abs/2010.11929)
3. Code: Study `vit.py` with the visual guide open, then do the exercises at the end of the guide
4. Build it yourself: [`labs/lab04_vit.py`](../labs/lab04_vit.py), a guided from-scratch ViT lab (setup in [`labs/README.md`](../labs/README.md))

Previous module: [Transformer](../transformer/) · Next module: [Vision-Language Models](../vlm/)
