"""Lab 05: fusing image tokens into a language model (LLaVA style)

Goal
----
A vision-language model (VLM) is a ViT + a small projector + an LLM. The
ViT turns the image into patch features; the projector maps each feature
into the LLM's embedding space; the LLM then reads image tokens and text
tokens as one sequence. You build the fusion: token counting, the 2 x 2
"pixel shuffle" merge, the MLP projector, the joint sequence, and its
attention mask.

The maths, step by step
-----------------------
B batch, P patch size, d_v vision width, d_t LLM width (d in the decoder
code), V vocab, N_v image tokens, N_t text tokens (as in the VLM guide).

  1. How many image tokens?  N_v = (H / P) * (W / P) / merge^2
       336 x 336, P = 14         -> 24 * 24 = 576  (LLaVA-1.5)
       448 x 448, P = 14         -> 1024; with a 2 x 2 merge -> 256
     Every image token costs prefill compute and KV-cache memory, like text.
  2. 2 x 2 merge (pixel shuffle, Qwen-VL / InternVL): on the patch grid
     gh x gw, stack each 2 x 2 neighbourhood along channels:
       [B, gh*gw, d_v] -> [B, (gh/2)*(gw/2), 4*d_v]      4x fewer tokens
  3. Projector (LLaVA-1.5): z = W2 GELU(W1 v + b1) + b2
       [B, N_v, d_v] -> [B, N_v, d_t]
  4. Joint sequence: image tokens first, then the text embeddings:
       x = cat([z, E[text_ids]], dim=1)                 [B, N_v + N_t, d_t]
  5. Attention mask over the joint sequence, N = N_v + N_t
     (True = may attend):
       causal (LLaVA):     allowed[i, j] = j <= i
       image-bidirectional option (a partial prefix-LM): additionally,
                           image token i sees ALL image tokens (bidirectional
                           inside the image block). A full prefix-LM
                           (PaliGemma, pi0) makes the whole prefix (image
                           AND prompt) bidirectional; here only the image.
     Text always sees every image token (they come first), and the image
     tokens never see the text.
  6. LLM decoder on x with that mask -> logits [B, N, V]
     (the loss/next-token prediction uses only the text positions).

Read alongside
--------------
  ../vlm/vlm_guide.html   "The Key Idea: VLM = ViT + Projector + LLM",
                          1.1 Shape flow, 2. The Fusion Step (2.1 causal attention
                          across modalities, #mask-widget), 4. Image Token Reduction
  ../vit/vit_guide.html   the vision encoder (lab 04)

Run:  python labs/lab05_vlm_fusion.py   (or: python labs/check.py 5)
Hints are graded (Hint 1 first). Solution: labs/solutions/lab05_vlm_fusion.py
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Step 1: count image tokens
# ---------------------------------------------------------------------------
def count_image_tokens(height: int, width: int, patch_size: int, merge: int = 1) -> int:
    """Number of image tokens the LLM sees for one height x width image.

    The image is cut into patch_size x patch_size patches; then every
    merge x merge block of patches becomes one token. Raise ValueError if
    height or width is not a multiple of patch_size * merge.
    """
    # Hint 1: patches per row = width // patch_size, per column = height // patch_size.
    # Hint 2: check divisibility by patch_size * merge first, then divide the
    #         patch count by merge * merge.
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 2: the 2 x 2 merge ("pixel shuffle")
# ---------------------------------------------------------------------------
def pixel_shuffle_merge(x: torch.Tensor, grid_h: int, grid_w: int, r: int = 2) -> torch.Tensor:
    """Merge every r x r block of patch tokens into one token.

    x: [B, grid_h * grid_w, d], tokens in row-major grid order.
    Returns [B, (grid_h // r) * (grid_w // r), r * r * d].

    Merged token (i, j) (row-major over the coarse grid) is the
    concatenation, in row-major (dy, dx) order, of the r*r fine tokens at
    grid positions (r*i + dy, r*j + dx).
    """
    # Hint 1: same idea as patchify in lab 04, but on a grid of tokens instead
    #         of pixels: reshape to [B, grid_h // r, r, grid_w // r, r, d].
    # Hint 2: you want [B, gh/r, gw/r, r(dy), r(dx), d] before the final reshape.
    # Hint 3: x.reshape(B, gh // r, r, gw // r, r, d).permute(0, 1, 3, 2, 4, 5)
    #             .reshape(B, (gh // r) * (gw // r), r * r * d)
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 3: the projector
# ---------------------------------------------------------------------------
class MLPProjector(nn.Module):
    """LLaVA-1.5 projector: Linear(d_in, d_out) -> GELU -> Linear(d_out, d_out)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, d_in] -> [B, N, d_out], applied to every image token independently."""
        # Hint 1: one line; use the exact GELU (F.gelu default).
        raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Step 4: the joint sequence and its attention mask
# ---------------------------------------------------------------------------
def build_sequence(image_embeds: torch.Tensor, text_embeds: torch.Tensor):
    """Image tokens first, then text.

    Args:
        image_embeds: [B, N_v, d_t]  (already projected to the LLM width)
        text_embeds:  [B, N_t, d_t]
    Returns:
        embeds:   [B, N_v + N_t, d_t]
        is_image: [N_v + N_t] bool, True at the image positions
    """
    # Hint 1: torch.cat along the sequence dim.
    # Hint 2: is_image = torch.arange(n_img + n_txt, device=...) < n_img
    raise NotImplementedError  # TODO


def joint_attention_mask(n_img: int, n_txt: int, image_bidirectional: bool = False,
                         device=None) -> torch.Tensor:
    """[N, N] bool mask over the joint sequence, N = n_img + n_txt; True = may attend.

    image_bidirectional=False: plain causal mask (LLaVA).
    image_bidirectional=True:  causal, except that image tokens also see every
                               other image token (image-bidirectional: a partial
                               prefix-LM that covers the image only).
    """
    # Hint 1: start from the causal mask (lab 01): key j <= query i.
    # Hint 2: for the image-bidirectional variant, also allow the [:n_img, :n_img] block.
    # Hint 3: allowed = torch.ones(N, N, dtype=torch.bool, device=device).tril()
    #         if image_bidirectional: allowed[:n_img, :n_img] = True
    raise NotImplementedError  # TODO


# ---------------------------------------------------------------------------
# Given: a tiny LLM decoder that accepts an explicit attention mask
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, N, d = x.shape
        q, k, v = (t.view(B, N, self.n_heads, -1).transpose(1, 2)
                   for t in self.qkv(self.ln1(x)).split(d, dim=-1))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)   # mask: True = may attend
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, d))
        return x + self.mlp(self.ln2(x))


class TinyDecoder(nn.Module):
    def __init__(self, d: int, n_heads: int, n_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList(DecoderBlock(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x [B, N, d], mask [N, N] bool -> hidden states [B, N, d]"""
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)


@dataclass
class VLMConfig:
    d_vision: int = 24
    d_model: int = 32
    vocab_size: int = 50
    n_heads: int = 4
    n_layers: int = 2


# ---------------------------------------------------------------------------
# Step 5: the fused forward pass
# ---------------------------------------------------------------------------
class TinyVLM(nn.Module):
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.projector = MLPProjector(cfg.d_vision, cfg.d_model)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.decoder = TinyDecoder(cfg.d_model, cfg.n_heads, cfg.n_layers)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(self, vision_feats: torch.Tensor, text_ids: torch.Tensor,
                image_bidirectional: bool = False) -> torch.Tensor:
        """
        Args:
            vision_feats: [B, N_v, d_vision]  patch features from a ViT (lab 04)
            text_ids:     [B, N_t]            token ids
        Returns:
            logits:       [B, N_v + N_t, vocab_size]
        """
        # Hint 1: project the image, embed the text, build_sequence, joint_attention_mask,
        #         self.decoder(x, mask), self.lm_head.
        # Hint 2: create the mask on vision_feats.device.
        raise NotImplementedError  # TODO


if __name__ == "__main__":
    import pathlib
    import sys

    _here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(_here if (_here / "check.py").exists() else _here.parent))
    from check import run_file

    sys.exit(run_file(__file__))
