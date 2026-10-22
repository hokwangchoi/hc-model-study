"""
Vision Transformer (ViT) Reference Implementation
Optimized for clarity and inference understanding

Usage:
    python vit.py

Model Configurations (224x224 input, 1000-class head):
    - ViT-B/16:  86.6M params, 17.6 GMACs (= 35.1 GFLOPs)
    - ViT-L/16: 304.3M params, 61.6 GMACs (= 123.1 GFLOPs)   (paper Table 1 lists 307M)
    - ViT-H/14: 632.0M params, 167.3 GMACs (= 334.6 GFLOPs)

A note on units: 1 multiply-accumulate (MAC) = 2 FLOPs. Paper and timm tables
quote MACs but call them "FLOPs", so "ViT-B/16 = 17.6 GFLOPs" means 17.6 GMACs.
compute_flops() below reports both.
"""
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# The original JAX ViT (flax default), timm and torchvision all use LayerNorm
# eps=1e-6. PyTorch's default is 1e-5, which makes ported weights drift slightly.
LayerNorm = partial(nn.LayerNorm, eps=1e-6)

class PatchEmbed(nn.Module):
    """
    Convert image to patch embeddings.

    Uses Conv2d which is equivalent to:
    1. Split image into non-overlapping patches
    2. Flatten each patch
    3. Linear projection

    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_chans: Number of input channels
        embed_dim: Embedding dimension
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input image
        Returns:
            [B, num_patches, embed_dim] patch embeddings
        """
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}x{W}) doesn't match expected ({self.img_size}x{self.img_size})"

        # Conv2d: [B, 3, 224, 224] -> [B, 768, 14, 14]
        x = self.proj(x)
        # Flatten spatial dims and transpose: [B, 768, 14, 14] -> [B, 196, 768]
        x = x.flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    """
    Multi-head self-attention.

    Key differences from decoder-only LLM attention:
    - No causal mask (bidirectional attention)
    - All tokens can attend to all other tokens

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        qkv_bias: Whether to add bias to QKV projection
        attn_drop: Attention dropout rate
        proj_drop: Output projection dropout rate
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # 1/sqrt(d_k). Not passed below: SDPA applies exactly this scale by default.
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d] input tokens (T = patches + 1, d = width)
        Returns:
            [B, T, d] output tokens
        """
        B, N, C = x.shape  # batch, tokens T, width d (timm names them N, C)

        # Compute Q, K, V in single matmul
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, T, d_k]
        q, k, v = qkv.unbind(0)  # Each: [B, h, T, d_k]

        # Fused attention: computes softmax(q @ k^T * self.scale) @ v in one kernel
        # (FlashAttention / memory-efficient backend when available). No mask: bidirectional.
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        # Reshape back: [B, h, T, d_k] -> [B, T, d]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    """
    Feed-forward network with GELU activation.

    Structure: Linear -> GELU -> Dropout -> Linear -> Dropout

    Args:
        in_features: Input dimension
        hidden_features: Hidden layer dimension (typically 4x input)
        out_features: Output dimension (typically same as input)
        drop: Dropout rate
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Block(nn.Module):
    """
    Transformer encoder block.

    Uses pre-norm architecture (LayerNorm before attention/MLP), like GPT-2 and
    LLaMA. Only the original 2017 Transformer (and BERT) used post-norm.

    Args:
        dim: Input dimension
        num_heads: Number of attention heads
        mlp_ratio: Ratio of MLP hidden dim to input dim
        drop: Dropout rate
        attn_drop: Attention dropout rate
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = LayerNorm(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    """
    Vision Transformer for image classification.

    Architecture:
    1. Patch embedding (Conv2d)
    2. Add [CLS] token + position embeddings
    3. L transformer encoder blocks
    4. LayerNorm on output
    5. Linear classification head on [CLS] token

    Args:
        img_size: Input image size
        patch_size: Size of each patch
        in_chans: Number of input channels
        num_classes: Number of classification classes
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer encoder blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])

        # Output head. The paper pre-trains with an MLP head (one tanh hidden layer)
        # and fine-tunes with a single Linear; released fine-tuned checkpoints use Linear.
        self.norm = LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """timm/DeiT-style init: normal(std=0.02) for Linear weights, pos_embed and cls_token.

        The original JAX ViT differs (xavier_uniform Linear weights, zero-init CLS token,
        zero-init head). Init only matters for training from scratch, not for loading
        pretrained weights. Note: trunc_normal_'s default bounds a=-2, b=2 are absolute,
        so with std=0.02 nothing is actually truncated (timm behaves the same way).
        """
        # Position and class token
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Apply to all modules
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        B = x.shape[0]

        # Patch embedding: [B, 3, 224, 224] -> [B, 196, 768]
        x = self.patch_embed(x)

        # Prepend [CLS] token: [B, 196, 768] -> [B, 197, 768]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer encoder blocks
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] input images
        Returns:
            [B, num_classes] classification logits
        """
        x = self.forward_features(x)
        # Extract [CLS] token and classify
        cls_token = x[:, 0]
        return self.head(cls_token)

# ============================================================================
# Model Configurations
# ============================================================================

def vit_tiny_patch16(**kwargs) -> ViT:
    """ViT-Tiny/16 (DeiT-Ti config, not in the ViT paper): 5.7M params, 1.25 GMACs"""
    return ViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)

def vit_small_patch16(**kwargs) -> ViT:
    """ViT-Small/16 (DeiT-S config, not in the ViT paper): 22.1M params, 4.6 GMACs"""
    return ViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)

def vit_base_patch16(**kwargs) -> ViT:
    """ViT-Base/16: 86.6M params, 17.6 GMACs (= 35.1 GFLOPs)"""
    return ViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_base_patch32(**kwargs) -> ViT:
    """ViT-Base/32: 88.2M params, 4.4 GMACs (= 8.8 GFLOPs); 49 patches instead of 196"""
    return ViT(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_large_patch16(**kwargs) -> ViT:
    """ViT-Large/16: 304.3M params with a 1000-class head (paper Table 1: 307M), 61.6 GMACs"""
    return ViT(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)

def vit_huge_patch14(**kwargs) -> ViT:
    """ViT-Huge/14: 632.0M params, 167.3 GMACs (= 334.6 GFLOPs)"""
    return ViT(patch_size=14, embed_dim=1280, depth=32, num_heads=16, **kwargs)

# ============================================================================
# Analysis Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_model_summary(model: ViT):
    """Print detailed parameter breakdown."""
    print("=" * 60)
    print("ViT Model Summary")
    print("=" * 60)

    # Patch embedding
    patch_params = sum(p.numel() for p in model.patch_embed.parameters())
    print(f"Patch Embedding:     {patch_params:>12,} params")

    # CLS token and position embeddings
    cls_params = model.cls_token.numel()
    pos_params = model.pos_embed.numel()
    print(f"CLS Token:           {cls_params:>12,} params")
    print(f"Position Embedding:  {pos_params:>12,} params")

    # Transformer blocks
    block_params = sum(p.numel() for p in model.blocks.parameters())
    per_block = block_params // len(model.blocks)
    print(f"Transformer Blocks:  {block_params:>12,} params ({len(model.blocks)} × {per_block:,})")

    # Output head
    norm_params = sum(p.numel() for p in model.norm.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Final LayerNorm:     {norm_params:>12,} params")
    print(f"Classification Head: {head_params:>12,} params")

    print("-" * 60)
    total = count_parameters(model)
    print(f"Total Parameters:    {total:>12,}")
    print(f"                     {total / 1e6:>12.2f}M")
    print("=" * 60)

def compute_flops(model: ViT, img_size: Optional[int] = None) -> dict:
    """
    Matmul/conv cost of ONE forward pass for ONE image, counted in MACs
    (multiply-accumulates) and FLOPs (= 2 x MACs).

    Ignores LayerNorm, softmax, GELU and bias adds (< 1% here). Matches
    torch.utils.flop_counter.FlopCounterMode exactly (it reports FLOPs).

    T = number of tokens = patches + 1 (CLS). Per layer (MACs):
        QKV 3*T*d^2 + out-proj T*d^2 + (QK^T and A@V) 2*T^2*d + FFN 2*T*d*d_ff
    """
    pe = model.patch_embed
    img_size = img_size or pe.img_size
    P, C = pe.patch_size, pe.proj.in_channels
    n_patches = (img_size // P) ** 2
    T = n_patches + 1                                # tokens = patches + CLS
    d = model.embed_dim
    d_ff = model.blocks[0].mlp.fc1.out_features      # respects mlp_ratio
    L = len(model.blocks)

    # Patch embedding: Conv2d(k=s=P) == GEMM [n_patches, C*P*P] @ [C*P*P, d]
    patch_macs = n_patches * (C * P * P) * d

    # Per-layer attention
    qkv_macs = 3 * T * d * d                         # fused Q, K, V projection
    score_macs = 2 * T * T * d                       # Q@K^T and attn@V: T*T*d each (all heads)
    proj_macs = T * d * d                            # output projection W_o
    attn_macs = qkv_macs + score_macs + proj_macs

    # Per-layer FFN: fc1 (d -> d_ff) + fc2 (d_ff -> d)
    ffn_macs = 2 * T * d * d_ff

    head_macs = d * model.num_classes                # classifier on the CLS token only
    total_macs = patch_macs + L * (attn_macs + ffn_macs) + head_macs

    return {
        "tokens": T,
        "patch_embed_macs": patch_macs,
        "attention_macs_per_layer": attn_macs,
        "score_macs_per_layer": score_macs,          # the part that grows with T^2
        "ffn_macs_per_layer": ffn_macs,
        "total_macs": total_macs,
        "total_gmacs": total_macs / 1e9,
        "total_gflops": 2 * total_macs / 1e9,
    }

def demo_forward_pass():
    """Demonstrate forward pass with shape annotations."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (ViT-B/16)")
    print("=" * 60)

    model = vit_base_patch16()
    model.eval()

    # Create dummy input
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    print(f"\n1. Input:              {list(x.shape)}")
    with torch.no_grad():
        logits = _trace_forward(model, x)
        # The step-by-step trace must reproduce model(x) exactly.
        assert torch.allclose(logits, model(x), atol=1e-5)
    print("   check: step-by-step logits == model(x)  ✓")
    print("\n" + "=" * 60)


def _trace_forward(model: ViT, x: torch.Tensor) -> torch.Tensor:
    """forward() unrolled, printing the shape after every stage."""

    # Patch embedding
    patches = model.patch_embed(x)
    print(f"2. Patch embeddings:   {list(patches.shape)}")

    # Add CLS token
    B = x.shape[0]
    cls_tokens = model.cls_token.expand(B, -1, -1)
    x_with_cls = torch.cat([cls_tokens, patches], dim=1)
    print(f"3. + CLS token:        {list(x_with_cls.shape)}")

    # Add position embeddings
    x_pos = x_with_cls + model.pos_embed
    print(f"4. + Position embed:   {list(x_pos.shape)}")

    # Through transformer blocks
    x_out = x_pos
    for i, block in enumerate(model.blocks):
        x_out = block(x_out)
        if i == 0 or i == len(model.blocks) - 1:
            print(f"5. After block {i:2d}:     {list(x_out.shape)}")
        elif i == 1:
            print("   ...")

    # Final norm and head
    x_norm = model.norm(x_out)
    cls_output = x_norm[:, 0]
    print(f"6. Extract [CLS]:      {list(cls_output.shape)}")

    logits = model.head(cls_output)
    print(f"7. Classification:     {list(logits.shape)}")
    return logits


def demo_patchify_equivalence():
    """Conv2d(k=s=P) == cut into patches + flatten + one Linear, checked numerically."""
    torch.manual_seed(0)
    P, d = 16, 8
    pe = PatchEmbed(img_size=32, patch_size=P, in_chans=3, embed_dim=d)
    x = torch.randn(2, 3, 32, 32)                                   # 2x2 grid = 4 patches
    with torch.no_grad():
        conv_out = pe(x)                                            # [2, 4, 8]
        patches = x.unfold(2, P, P).unfold(3, P, P)                 # [B, C, 2, 2, P, P]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, 4, 3 * P * P)  # row-major patches
        W = pe.proj.weight.reshape(d, -1)                           # conv kernel IS the Linear weight
        lin_out = patches @ W.T + pe.proj.bias
    err = (conv_out - lin_out).abs().max().item()
    print(f"Conv2d(k=s=16) vs unfold + flatten + Linear: max |diff| = {err:.1e}")
    assert err < 1e-5


def demo_permutation():
    """Without pos_embed, shuffling the patches only shuffles the outputs (attention is a set op)."""
    torch.manual_seed(0)
    m = vit_tiny_patch16(num_classes=10).eval()
    with torch.no_grad():
        tok = m.patch_embed(torch.randn(1, 3, 224, 224))            # [1, 196, 192]
        perm = torch.randperm(tok.shape[1])

        def run(t, use_pos):
            t = torch.cat([m.cls_token.expand(1, -1, -1), t], 1)
            if use_pos:
                t = t + m.pos_embed
            for blk in m.blocks:
                t = blk(t)
            return m.norm(t)

        for use_pos in (False, True):
            a, b = run(tok, use_pos), run(tok[:, perm], use_pos)
            # compare patch outputs after undoing the shuffle
            err = ((a[:, 1:][:, perm] - b[:, 1:]).norm() / a[:, 1:].norm()).item()
            print(f"pos_embed {'on ' if use_pos else 'off'}: patch outputs vs shuffled input, rel. error = {err:.1e}")

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Vision Transformer (ViT) Reference Implementation")
    print("=" * 60)

    # Create model
    model = vit_base_patch16()

    # Print summary
    print_model_summary(model)

    # Compute FLOPs
    f = compute_flops(model)
    print(f"\nCompute per image (T = {f['tokens']} tokens; 1 MAC = 2 FLOPs):")
    print(f"  Patch embedding:  {f['patch_embed_macs'] / 1e9:.3f} GMACs")
    print(f"  Attention/layer:  {f['attention_macs_per_layer'] / 1e9:.3f} GMACs "
          f"(of which QK^T + A@V: {f['score_macs_per_layer'] / 1e9:.3f})")
    print(f"  FFN/layer:        {f['ffn_macs_per_layer'] / 1e9:.3f} GMACs")
    print(f"  Total:            {f['total_gmacs']:.2f} GMACs = {f['total_gflops']:.2f} GFLOPs "
          f"(papers quote the GMAC number as 'GFLOPs')")

    # Demo forward pass
    demo_forward_pass()

    # Two small checks behind the guide's claims
    print("\nSanity checks")
    print("-" * 60)
    demo_patchify_equivalence()
    demo_permutation()

    # Compare model sizes
    print("\n" + "=" * 60)
    print("Model Comparison (224x224)")
    print("=" * 60)
    configs = [
        ("ViT-Tiny/16", vit_tiny_patch16),
        ("ViT-Small/16", vit_small_patch16),
        ("ViT-Base/16", vit_base_patch16),
        ("ViT-Base/32", vit_base_patch32),
        ("ViT-Large/16", vit_large_patch16),
        ("ViT-Huge/14", vit_huge_patch14),
    ]

    print(f"{'Model':<15} {'Params':>12} {'GMACs':>8} {'GFLOPs':>8}")
    print("-" * 46)
    for name, fn in configs:
        with torch.device("meta"):   # shapes only: no RAM, no init time
            m = fn()
        params = count_parameters(m)
        f = compute_flops(m)
        print(f"{name:<15} {params:>12,} {f['total_gmacs']:>8.1f} {f['total_gflops']:>8.1f}")
