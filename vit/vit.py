"""
Vision Transformer (ViT) Reference Implementation
Optimized for clarity and inference understanding

Usage:
    python vit.py

Model Configurations:
    - ViT-B/16: 86M params, 17.6 GFLOPs
    - ViT-L/16: 307M params, 61.6 GFLOPs
    - ViT-H/14: 632M params, 167 GFLOPs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

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
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Combined QKV projection for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] input tokens
        Returns:
            [B, N, C] output tokens
        """
        B, N, C = x.shape

        # Compute Q, K, V in single matmul
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, d_k]
        q, k, v = qkv.unbind(0)  # Each: [B, h, N, d_k]

        # Use PyTorch 2.0+ efficient attention (FlashAttention when available)
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )

        # Reshape back: [B, h, N, d_k] -> [B, N, C]
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

    Uses pre-norm architecture (LayerNorm before attention/MLP).
    This is different from original Transformer which uses post-norm.

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
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
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

        # Output head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following original ViT paper."""
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
    """ViT-Tiny/16: 5.7M params"""
    return ViT(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)

def vit_small_patch16(**kwargs) -> ViT:
    """ViT-Small/16: 22M params"""
    return ViT(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)

def vit_base_patch16(**kwargs) -> ViT:
    """ViT-Base/16: 86M params, 17.6 GFLOPs"""
    return ViT(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_base_patch32(**kwargs) -> ViT:
    """ViT-Base/32: 88M params, 4.4 GFLOPs (fewer patches)"""
    return ViT(patch_size=32, embed_dim=768, depth=12, num_heads=12, **kwargs)

def vit_large_patch16(**kwargs) -> ViT:
    """ViT-Large/16: 307M params, 61.6 GFLOPs"""
    return ViT(patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)

def vit_huge_patch14(**kwargs) -> ViT:
    """ViT-Huge/14: 632M params, 167 GFLOPs"""
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

def compute_flops(model: ViT, img_size: int = 224) -> dict:
    """Compute FLOPs breakdown for forward pass."""
    P = model.patch_embed.patch_size
    N = (img_size // P) ** 2 + 1  # +1 for CLS token
    d = model.embed_dim
    h = model.blocks[0].attn.num_heads
    L = len(model.blocks)
    d_ff = int(d * 4)  # MLP hidden dim

    # Patch embedding: Conv2d equivalent
    patch_flops = img_size * img_size * 3 * d

    # Per-layer attention FLOPs
    qkv_flops = 2 * N * d * 3 * d  # QKV projection
    attn_flops = 2 * N * N * d     # Q@K and attn@V
    proj_flops = 2 * N * d * d     # Output projection
    attn_total = qkv_flops + attn_flops + proj_flops

    # Per-layer FFN FLOPs
    ffn_flops = 2 * N * d * d_ff + 2 * N * d_ff * d

    # Total
    total = patch_flops + L * (attn_total + ffn_flops) + 2 * d * model.num_classes

    return {
        "patch_embed": patch_flops,
        "attention_per_layer": attn_total,
        "ffn_per_layer": ffn_flops,
        "total": total,
        "total_gflops": total / 1e9,
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
            print(f"   ...")

    # Final norm and head
    x_norm = model.norm(x_out)
    cls_output = x_norm[:, 0]
    print(f"6. Extract [CLS]:      {list(cls_output.shape)}")

    logits = model.head(cls_output)
    print(f"7. Classification:     {list(logits.shape)}")

    print("\n" + "=" * 60)

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
    flops = compute_flops(model)
    print(f"\nFLOPs Analysis:")
    print(f"  Patch Embedding: {flops['patch_embed'] / 1e9:.3f} GFLOPs")
    print(f"  Attention/layer: {flops['attention_per_layer'] / 1e9:.3f} GFLOPs")
    print(f"  FFN/layer:       {flops['ffn_per_layer'] / 1e9:.3f} GFLOPs")
    print(f"  Total:           {flops['total_gflops']:.2f} GFLOPs")

    # Demo forward pass
    demo_forward_pass()

    # Compare model sizes
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    configs = [
        ("ViT-Tiny/16", vit_tiny_patch16),
        ("ViT-Small/16", vit_small_patch16),
        ("ViT-Base/16", vit_base_patch16),
        ("ViT-Base/32", vit_base_patch32),
        ("ViT-Large/16", vit_large_patch16),
        ("ViT-Huge/14", vit_huge_patch14),
    ]

    print(f"{'Model':<15} {'Params':>12} {'GFLOPs':>10}")
    print("-" * 40)
    for name, fn in configs:
        m = fn()
        params = count_parameters(m)
        flops = compute_flops(m)
        print(f"{name:<15} {params:>12,} {flops['total_gflops']:>10.1f}")
