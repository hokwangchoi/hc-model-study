"""
Vision-Language Model (VLM) Reference Implementation
LLaVA / Qwen-VL style: ViT encoder + MLP projector + LLM decoder

Usage:
    python vlm.py

Architecture:
    Image → ViT → projector → image tokens ──┐
                                             ├─→ LLM → next-token logits
    Text  → tokenizer → embedding  ──────────┘

Model Configurations (approximate):
    - VLM-Tiny:  ~150M params  (ViT-S/16 + 100M LLM)
    - VLM-Small: ~1.25B params (ViT-L/14 + 950M LLM)  similar to LLaVA-1B
    - VLM-Base:  ~2.36B params (ViT-L/14 + 2B LLM)    similar spirit to Cosmos-Reason2-2B

This implementation focuses on clarity of the FUSION PATH (how image tokens
get stitched into the LLM's input sequence), not on training or generation
loops. KV-cache, beam search, and sampling are omitted.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# ============================================================================
# Vision Encoder — ViT body (no CLS token, no classification head)
# ============================================================================

class PatchEmbed(nn.Module):
    """Image → sequence of patch embeddings. Same as standalone ViT."""
    def __init__(self, img_size=448, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2  # (448/14)² = 1024
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 448, 448] → Conv2d → [B, 1024, 32, 32] → flatten → [B, 1024, 1024]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionAttention(nn.Module):
    """Bidirectional self-attention (no causal mask) — same as ViT."""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # No causal mask — image patches are bidirectional
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class VisionBlock(nn.Module):
    """Pre-norm transformer block for ViT. Same pattern as vit/vit.py."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = VisionAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class VisionEncoder(nn.Module):
    """
    Stripped-down ViT used as a feature extractor inside a VLM.

    Two differences from standalone ViT:
    1. No [CLS] token — we keep all patch tokens for the LLM to attend over.
    2. No classification head — output is the sequence of patch embeddings.

    Args:
        img_size: Input image size (square)
        patch_size: Patch side length
        embed_dim: Vision hidden dim (often 1024 for ViT-L)
        depth: Number of transformer blocks
        num_heads: Attention heads per block
    """
    def __init__(
        self,
        img_size: int = 448,
        patch_size: int = 14,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Position embeddings: one per patch (no extra slot for CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList([
            VisionBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            [B, num_patches, embed_dim] — patch features for the projector
        """
        x = self.patch_embed(images)          # [B, N_v, d_v]
        x = x + self.pos_embed                # [B, N_v, d_v]
        for block in self.blocks:
            x = block(x)                      # [B, N_v, d_v] — shape unchanged
        x = self.norm(x)
        return x

# ============================================================================
# Visual Projector — bridges vision hidden dim to LLM hidden dim
# ============================================================================

class VisualProjector(nn.Module):
    """
    2-layer MLP that maps [B, N_v, d_v] → [B, N_v, d_t].

    This is usually the ONLY component trained from scratch in stage-1 VLM
    training (LLaVA-style). The ViT and LLM are frozen; the projector learns
    to align the two embedding spaces on image-caption pairs.

    Args:
        vision_dim: Vision encoder output dim (d_v)
        text_dim: LLM embedding dim (d_t)
    """
    def __init__(self, vision_dim: int, text_dim: int):
        super().__init__()
        # Standard LLaVA-style: linear → GELU → linear
        # Some variants add LayerNorm; some use a single Linear (cheaper, works worse)
        self.fc1 = nn.Linear(vision_dim, text_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(text_dim, text_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N_v, d_v] → [B, N_v, d_t]
        return self.fc2(self.act(self.fc1(x)))

# ============================================================================
# LLM Decoder — causal transformer (the text-generation half)
# ============================================================================

class CausalAttention(nn.Module):
    """
    Multi-head causal self-attention — same math as vit/vit.py's Attention,
    but with `is_causal=True` so future tokens are masked out.
    """
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)   # LLM convention: no QKV bias
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Causal mask: position i can only attend to positions 0..i
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class LLMBlock(nn.Module):
    """
    Pre-norm transformer block, LLM-style.

    Differences from VisionBlock:
    - Causal attention (not bidirectional)
    - No bias in linear layers (modern LLM convention)
    - SwiGLU activation in MLP (used by Llama, Qwen) — simplified to GELU here for clarity
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class LLMDecoder(nn.Module):
    """
    Causal LLM decoder body. Consumes a sequence of embeddings (text OR a mix
    of image + text embeddings) and produces next-token logits.

    Note: this receives EMBEDDINGS, not token IDs. The embedding lookup for
    text happens in the outer VLM class so we can concatenate with image
    tokens before feeding to this module.

    Args:
        vocab_size: Token vocabulary size
        hidden_dim: LLM hidden dim (d_t)
        depth: Number of transformer blocks
        num_heads: Attention heads per block
        max_seq_len: Maximum sequence length (for position embeddings)
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        depth: int = 24,
        num_heads: int = 16,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Token embedding (used by outer VLM for text; not by this forward pass)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
        # Absolute position embeddings — real LLMs use RoPE, simplified here
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))

        self.blocks = nn.ModuleList([
            LLMBlock(hidden_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        # LM head — often tied to the embedding matrix to save params
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: [B, N, hidden_dim] — pre-embedded sequence (image + text)
        Returns:
            [B, N, vocab_size] — logits for every position
        """
        B, N, _ = embeds.shape
        x = embeds + self.pos_embed[:, :N]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

# ============================================================================
# VLM — the whole thing
# ============================================================================

class VLM(nn.Module):
    """
    Vision-Language Model: ViT encoder + MLP projector + LLM decoder.

    The forward pass:
    1. Encode the image → [B, N_v, d_v]
    2. Project to LLM space → [B, N_v, d_t]
    3. Embed text tokens → [B, N_t, d_t]
    4. Concatenate: [image_tokens; text_embeds] → [B, N_v + N_t, d_t]
    5. Run LLM on the combined sequence → next-token logits

    The image tokens form a PREFIX that the text generation attends to.
    """
    def __init__(
        self,
        vision_encoder: VisionEncoder,
        llm: LLMDecoder,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.projector = VisualProjector(
            vision_dim=vision_encoder.embed_dim,
            text_dim=llm.hidden_dim,
        )
        self.llm = llm

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run the vision path: image → projected tokens in LLM space.

        This is the step that's cached in production — the same image always
        produces the same output, so it's worth running once per image, not
        once per output token.
        """
        vision_feats = self.vision_encoder(images)        # [B, N_v, d_v]
        image_tokens = self.projector(vision_feats)       # [B, N_v, d_t]
        return image_tokens

    def forward(
        self,
        images: torch.Tensor,
        text_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W]
            text_token_ids: [B, N_t] — tokenized text prompt
        Returns:
            [B, N_v + N_t, vocab_size] — logits over combined sequence
        """
        # Vision path (runs once per image in production)
        image_tokens = self.encode_image(images)          # [B, N_v, d_t]

        # Text embedding lookup — notice we use the LLM's embed_tokens table
        text_embeds = self.llm.embed_tokens(text_token_ids)  # [B, N_t, d_t]

        # Fusion: concatenate along sequence dim. Image tokens go FIRST as a prefix,
        # though real VLMs often sandwich them in a chat template like:
        #   <system>...<im_start>[IMAGE TOKENS]<im_end>...<user>text<assistant>
        combined = torch.cat([image_tokens, text_embeds], dim=1)  # [B, N_v + N_t, d_t]

        # LLM forward — produces logits over the combined sequence
        logits = self.llm(combined)                       # [B, N_v + N_t, vocab]
        return logits

# ============================================================================
# Model Configurations
# ============================================================================

def vlm_tiny() -> VLM:
    """~250M params. Toy config for debugging."""
    vision = VisionEncoder(
        img_size=224, patch_size=16,
        embed_dim=384, depth=12, num_heads=6,   # ViT-Small
    )
    llm = LLMDecoder(
        vocab_size=32000, hidden_dim=768,
        depth=12, num_heads=12, max_seq_len=2048,
    )
    return VLM(vision, llm)

def vlm_small() -> VLM:
    """~1.2B params. Close to LLaVA-1B."""
    vision = VisionEncoder(
        img_size=336, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,   # ViT-Large/14
    )
    llm = LLMDecoder(
        vocab_size=32000, hidden_dim=2048,
        depth=16, num_heads=16, max_seq_len=4096,
    )
    return VLM(vision, llm)

def vlm_base() -> VLM:
    """~2.2B params. Close to Cosmos-Reason2-2B."""
    vision = VisionEncoder(
        img_size=448, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,   # ViT-Large/14
    )
    llm = LLMDecoder(
        vocab_size=151936, hidden_dim=2048,       # Qwen2.5-1.5B-ish
        depth=28, num_heads=16, max_seq_len=8192,
    )
    return VLM(vision, llm)

# ============================================================================
# Analysis Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_vlm_summary(model: VLM):
    """Print parameter breakdown by component."""
    print("=" * 60)
    print("VLM Model Summary")
    print("=" * 60)

    vision_params = count_parameters(model.vision_encoder)
    proj_params = count_parameters(model.projector)
    llm_params = count_parameters(model.llm)
    total = vision_params + proj_params + llm_params

    print(f"Vision Encoder:      {vision_params:>12,} params "
          f"({100 * vision_params / total:5.1f}%)")
    print(f"Visual Projector:    {proj_params:>12,} params "
          f"({100 * proj_params / total:5.1f}%)")
    print(f"LLM Decoder:         {llm_params:>12,} params "
          f"({100 * llm_params / total:5.1f}%)")
    print("-" * 60)
    print(f"Total:               {total:>12,} params")
    print(f"                     {total / 1e6:>12.2f}M")
    print("=" * 60)

def compute_flops(model: VLM, n_text_tokens: int = 32, n_output_tokens: int = 100) -> dict:
    """
    Compute approximate FLOPs for ONE inference call.

    Stages:
    - Vision: runs ONCE per image
    - Prefill: LLM sees [N_v + N_t] tokens, computes hidden states (one pass)
    - Decode: LLM generates N_output tokens, one at a time
    """
    N_v = model.vision_encoder.patch_embed.num_patches
    d_v = model.vision_encoder.embed_dim
    d_t = model.llm.hidden_dim
    L_v = len(model.vision_encoder.blocks)
    L_t = len(model.llm.blocks)
    vocab = model.llm.vocab_size

    # -------- Vision encoder (runs once per image) --------
    # Patch embed: conv equivalent to N_v × (P² × C) × d_v GEMM
    patch_flops = 2 * N_v * (14 * 14 * 3) * d_v
    # Per-block: QKV (6N_v·d_v²) + attention (4N_v²·d_v) + FFN (16N_v·d_v²)
    vision_block_flops = 2 * (3 * N_v * d_v * d_v +        # QKV projection
                              2 * N_v * N_v * d_v +        # attention scores + attn@V
                              2 * N_v * d_v * (4 * d_v))   # FFN (4x ratio)
    vision_flops = patch_flops + L_v * vision_block_flops

    # -------- Projector (2-layer MLP on N_v tokens) --------
    proj_flops = 2 * N_v * d_v * d_t + 2 * N_v * d_t * d_t

    # -------- LLM prefill (N_v + N_t tokens in one pass) --------
    N_prefill = N_v + n_text_tokens
    prefill_block = 2 * (3 * N_prefill * d_t * d_t +
                         2 * N_prefill * N_prefill * d_t +
                         2 * N_prefill * d_t * (4 * d_t))
    prefill_flops = L_t * prefill_block + 2 * N_prefill * d_t * vocab  # + LM head

    # -------- LLM decode (one token at a time, but attending to growing KV) --------
    # Assume N_output steps, average sequence length ≈ N_prefill + N_output/2
    avg_context = N_prefill + n_output_tokens // 2
    decode_per_step = 2 * L_t * (3 * d_t * d_t +                # QKV for 1 token
                                 2 * avg_context * d_t +         # attention
                                 2 * d_t * (4 * d_t))            # FFN for 1 token
    decode_flops = n_output_tokens * (decode_per_step + 2 * d_t * vocab)

    return {
        "vision_gflops": vision_flops / 1e9,
        "projector_gflops": proj_flops / 1e9,
        "prefill_gflops": prefill_flops / 1e9,
        "decode_gflops": decode_flops / 1e9,
        "total_gflops": (vision_flops + proj_flops + prefill_flops + decode_flops) / 1e9,
        "n_visual_tokens": N_v,
        "n_prefill_tokens": N_prefill,
    }

def demo_forward_pass():
    """Run one forward pass with shape annotations at each stage."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (VLM-Small)")
    print("=" * 60)

    model = vlm_small()
    model.eval()

    # Dummy inputs
    batch = 1
    n_text = 20
    image = torch.randn(batch, 3, 336, 336)
    text_ids = torch.randint(0, 32000, (batch, n_text))

    print(f"\n1. Image input:        {list(image.shape)}")
    print(f"2. Text token IDs:     {list(text_ids.shape)}")

    with torch.no_grad():
        # Vision path
        vision_feats = model.vision_encoder(image)
        print(f"3. Vision features:    {list(vision_feats.shape)}  "
              f"← ViT output, still in vision space")

        image_tokens = model.projector(vision_feats)
        print(f"4. Image tokens:       {list(image_tokens.shape)}  "
              f"← projected to LLM space")

        # Text path
        text_embeds = model.llm.embed_tokens(text_ids)
        print(f"5. Text embeddings:    {list(text_embeds.shape)}")

        # Fusion
        combined = torch.cat([image_tokens, text_embeds], dim=1)
        print(f"6. Combined sequence:  {list(combined.shape)}  "
              f"← image prefix + text")

        # LLM
        logits = model.llm(combined)
        print(f"7. Output logits:      {list(logits.shape)}  "
              f"← next-token probs for every position")

    print("\n" + "=" * 60)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Vision-Language Model (VLM) Reference Implementation")
    print("=" * 60)

    # Use VLM-Small as the default — fits on one modest GPU
    model = vlm_small()
    print_vlm_summary(model)

    # FLOPs breakdown
    flops = compute_flops(model, n_text_tokens=32, n_output_tokens=100)
    print(f"\nFLOPs Analysis (1 image, 32 text tokens in, 100 tokens out):")
    print(f"  Visual tokens:    {flops['n_visual_tokens']} "
          f"(prefill total: {flops['n_prefill_tokens']})")
    print(f"  Vision encoder:   {flops['vision_gflops']:>7.2f} GFLOPs  (once per image)")
    print(f"  Projector:        {flops['projector_gflops']:>7.2f} GFLOPs")
    print(f"  LLM prefill:      {flops['prefill_gflops']:>7.2f} GFLOPs  (once per request)")
    print(f"  LLM decode:       {flops['decode_gflops']:>7.2f} GFLOPs  (100 tokens × ~decode cost)")
    print(f"  Total:            {flops['total_gflops']:>7.2f} GFLOPs")

    # Forward pass demo
    demo_forward_pass()

    # Compare configurations
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    configs = [
        ("VLM-Tiny",  vlm_tiny),
        ("VLM-Small", vlm_small),
        ("VLM-Base",  vlm_base),
    ]
    print(f"{'Model':<12} {'Vision':>10} {'Projector':>12} {'LLM':>12} {'Total':>10}")
    print("-" * 60)
    for name, fn in configs:
        m = fn()
        v = count_parameters(m.vision_encoder)
        p = count_parameters(m.projector)
        l = count_parameters(m.llm)
        total = v + p + l
        print(f"{name:<12} {v/1e6:>8.1f}M {p/1e6:>10.1f}M {l/1e6:>10.1f}M {total/1e6:>8.1f}M")
