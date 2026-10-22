"""Lab 04: the Vision Transformer front end and forward pass (reference solution)

Goal
----
Turn an image into a sequence of tokens and classify it with a transformer
encoder. You implement patchify twice (pure reshapes, and a strided Conv2d)
and show they are the same linear map, then the CLS token, the position
embeddings, the forward pass, and a shape tracker. The transformer blocks
themselves are torch's (you built one in lab 02).

The maths, step by step
-----------------------
B batch, C channels, H x W image, P patch size (argument p in the code),
d width, K classes (as in the ViT guide). N counts patches; N+1 tokens with CLS.

  1. Patchify      [B, C, H, W] -> [B, N, C*P*P],   N = (H/P) * (W/P)
                   cut the image into non-overlapping P x P tiles, row by
                   row; flatten each tile in (channel, row, column) order.
  2. Patch embed   tokens = patches @ W_e^T + b       W_e: [d, C*P*P]  -> [B, N, d]
                   Identical to Conv2d(C, d, kernel_size=P, stride=P): the
                   kernel visits each tile once and takes a dot product with
                   it. conv.weight [d, C, P, P] reshaped to [d, C*P*P] is W_e.
  3. CLS token     prepend one learned vector:  [B, N, d] -> [B, N+1, d]
                   It owns no patch; through attention it gathers a summary
                   of the whole image, and the classifier reads only it.
  4. Positions     x = x + pos_embed                  pos_embed: [1, N+1, d]
                   Attention is permutation-equivariant: without this,
                   shuffling the patches would not change the CLS output.
  5. Encoder       L pre-norm blocks, bidirectional (no causal mask)  [B, N+1, d]
  6. Head          LayerNorm, take token 0 (CLS), Linear(d, K)        [B, K]

  ViT-B/16 at 224 x 224: N = 14 * 14 = 196 patches, 197 tokens, d = 768.

Read alongside
--------------
  ../vit/vit_guide.html   section 2.1 High-Level Inference Pipeline (shape flow),
                          2.2 Patch Embedding, 2.3 Position Embeddings,
                          2.5 Full Model: Where the Parameters and Compute Go,
                          and #python-code (condensed reference implementation)

Run:  python labs/lab04_vit.py   (or: python labs/check.py 4)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab04_vit.py
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Step 1: patchify with reshapes (and its inverse)
# ---------------------------------------------------------------------------
def patchify(images: torch.Tensor, p: int) -> torch.Tensor:
    """[B, C, H, W] -> [B, N, C*p*p].

    Patch n = (row r, column c) of the patch grid, n = r * (W/p) + c.
    Its vector is images[b, :, r*p:(r+1)*p, c*p:(c+1)*p].reshape(-1),
    i.e. ordered (channel, row inside the patch, column inside the patch).
    """
    B, C, H, W = images.shape
    gh, gw = H // p, W // p
    x = images.reshape(B, C, gh, p, gw, p)          # split rows and columns into (grid, inside)
    x = x.permute(0, 2, 4, 1, 3, 5)                 # [B, gh, gw, C, p, p]
    return x.reshape(B, gh * gw, C * p * p)         # [B, N, C*p*p]


def unpatchify(patches: torch.Tensor, C: int, H: int, W: int, p: int) -> torch.Tensor:
    """Inverse of patchify: [B, N, C*p*p] -> [B, C, H, W].

    (DiT, the diffusion transformer, ends with exactly this step.)
    """
    B = patches.shape[0]
    gh, gw = H // p, W // p
    x = patches.reshape(B, gh, gw, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)                 # [B, C, gh, p, gw, p]
    return x.reshape(B, C, H, W)


# ---------------------------------------------------------------------------
# Step 2: patch embedding, two ways
# ---------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 8, in_chans: int = 3, dim: int = 48):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N, d] using the strided convolution self.proj."""
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, d, gh, gw] -> [B, N, d]

    def forward_linear(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] -> [B, N, d] using patchify() and a matmul with the SAME weights."""
        w = self.proj.weight.reshape(self.proj.out_channels, -1)     # [d, C*p*p]
        return patchify(x, self.patch_size) @ w.T + self.proj.bias   # [B, N, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_conv(x)


# ---------------------------------------------------------------------------
# Step 3: the ViT forward pass (CLS token, position embeddings, encoder, head)
# ---------------------------------------------------------------------------
@dataclass
class ViTConfig:
    img_size: int = 32
    patch_size: int = 8
    in_chans: int = 3
    dim: int = 48
    depth: int = 2
    n_heads: int = 4
    n_classes: int = 10


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.img_size, cfg.patch_size, cfg.in_chans, cfg.dim)
        n = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(0.02 * torch.randn(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, n + 1, cfg.dim))
        # Pre-norm encoder blocks (LayerNorm -> MHA -> add; LayerNorm -> MLP -> add), as in lab 02.
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(cfg.dim, cfg.n_heads, 4 * cfg.dim, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
            for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.n_classes)

    def forward(self, images: torch.Tensor, trace: dict = None) -> torch.Tensor:
        """images [B, C, H, W] -> logits [B, n_classes].

        If `trace` is a dict, also record tuple(shape) under these keys:
            "input"         the images                       [B, C, H, W]
            "patch_tokens"  after patch embedding            [B, N, d]
            "tokens"        after CLS + position embeddings  [B, N+1, d]
            "encoded"       after the blocks and self.norm   [B, N+1, d]
            "cls"           the CLS feature                  [B, d]
            "logits"        the output                       [B, n_classes]
        """
        def record(name, t):
            if trace is not None:
                trace[name] = tuple(t.shape)

        B = images.shape[0]
        record("input", images)
        x = self.patch_embed(images)                       # [B, N, d]
        record("patch_tokens", x)
        cls = self.cls_token.expand(B, -1, -1)             # [B, 1, d]
        x = torch.cat([cls, x], dim=1) + self.pos_embed    # [B, N+1, d]
        record("tokens", x)
        for block in self.blocks:
            x = block(x)                                   # bidirectional attention
        x = self.norm(x)
        record("encoded", x)
        cls_out = x[:, 0]                                  # [B, d]
        record("cls", cls_out)
        logits = self.head(cls_out)                        # [B, n_classes]
        record("logits", logits)
        return logits


# ---------------------------------------------------------------------------
# Step 4: predict the shapes without running anything
# ---------------------------------------------------------------------------
def vit_shapes(batch: int, img_size: int, patch_size: int, in_chans: int, dim: int,
               n_classes: int) -> dict:
    """Return the shapes forward() records in `trace`, computed by hand (tuples of ints).

    Example: vit_shapes(1, 224, 16, 3, 768, 1000)["tokens"] == (1, 197, 768).
    """
    n = (img_size // patch_size) ** 2
    return {
        "input": (batch, in_chans, img_size, img_size),
        "patch_tokens": (batch, n, dim),
        "tokens": (batch, n + 1, dim),
        "encoded": (batch, n + 1, dim),
        "cls": (batch, dim),
        "logits": (batch, n_classes),
    }


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
