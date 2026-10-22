"""
Vision-Language Model (VLM) Reference Implementation
LLaVA / Qwen-VL style: ViT encoder + MLP projector + LLM decoder

Usage:
    python vlm.py

Architecture:
    Image → ViT → projector → image tokens ──┐
                                             ├─→ LLM → next-token logits
    Text  → tokenizer → embedding  ──────────┘

Model Configurations (exact counts printed by `python vlm.py`):
    - VLM-Tiny:  ~158M params  (ViT-S/16 @224 + 136M LLM)          toy config for debugging
    - VLM-Small: ~1.25B params (ViT-L/14 @336 + 945M LLM)           LLaVA-1.5-style vision tower
                                                                     (CLIP ViT-L/14-336, 576 tokens)
    - VLM-Base:  ~2.36B params (ViT-L/14 @448 + 2.05B LLM)          sized like Cosmos-Reason2-2B

VLM-Base vs the real Cosmos-Reason2-2B (post-trained from Qwen3-VL-2B-Instruct):
    same:  ViT width/depth (1024 x 24), LLM width/depth/heads (2048 x 28, 16 heads), vocab 151936,
           and (apparently) an untied LM head: the model card's 2,438,696,960 params equal
           Qwen3-VL-2B's 2,127,532,032 plus exactly one 151936 x 2048 matrix
    real:  patch 16 + 2x2 token merge (448x448 -> 784 patches -> 196 LLM tokens; here 1024),
           GQA with 8 KV heads (KV cache half of this MHA), SwiGLU MLP (6144), RMSNorm,
           interleaved M-RoPE instead of learned absolute positions, DeepStack feature fusion

This implementation focuses on clarity of the FUSION PATH (how image tokens
get stitched into the LLM's input sequence), not on training or generation
loops. KV-cache generation, beam search, and sampling are omitted (see the
exercises at the end of vlm_guide.html).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Vision Encoder — ViT body (no CLS token, no classification head)
# ============================================================================

class PatchEmbed(nn.Module):
    """Image → sequence of patch embeddings. Same as standalone ViT."""
    def __init__(self, img_size=448, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
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

    Two differences from standalone ViT (vit/vit.py):
    1. No [CLS] token — we keep all patch tokens for the LLM to attend over
       (SigLIP-style). LLaVA's CLIP tower does have a CLS token; LLaVA drops it
       after encoding.
    2. No classification head — output is the sequence of patch embeddings.

    Real towers differ in one more detail: LLaVA takes features from the
    second-to-last ViT block (vision_feature_layer=-2), before any final norm.
    Here we use the last block + final LayerNorm for simplicity.

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
        S = self.patch_embed.img_size
        assert images.shape[-2:] == (S, S), (
            f"expected {S}x{S} images, got {tuple(images.shape[-2:])} "
            "(learned pos_embed has a fixed grid; real VLMs resize or use 2D RoPE)")
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
        self.qkv = nn.Linear(dim, dim * 3, bias=False)   # Llama/Qwen3: no QKV bias (Qwen2/2.5 keep one)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Causal mask: position i can only attend to positions 0..i — this applies to
        # the image tokens too (image token 3 cannot see image token 7). Prefix-LM
        # models such as PaliGemma use a bidirectional block over image + prompt instead.
        # (Careful when adding a KV cache: with is_causal=True and a single query row,
        # SDPA aligns the mask top-left, so the new token would only see key 0.)
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
        tie_embeddings: share one [vocab, d] matrix between embed_tokens and lm_head
            (Qwen3-1.7B / Qwen3-VL-2B do this; LLaVA-1.5's Vicuna does not)
    """
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 2048,
        depth: int = 24,
        num_heads: int = 16,
        max_seq_len: int = 4096,
        tie_embeddings: bool = False,
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
        # LM head: hidden state -> one logit per vocab entry
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
        if tie_embeddings:
            # Same [vocab, d] tensor for lookup and output; parameters() counts it once.
            self.lm_head.weight = self.embed_tokens.weight

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeds: [B, N, hidden_dim] — pre-embedded sequence (image + text)
        Returns:
            [B, N, vocab_size] — logits for every position
        """
        B, N, _ = embeds.shape
        assert N <= self.pos_embed.shape[1], f"sequence length {N} > max_seq_len {self.pos_embed.shape[1]}"
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
    """~158M params. Toy config for debugging."""
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
    """~1.25B params. LLaVA-1.5-style vision tower (ViT-L/14 @336 -> 576 tokens) on a ~1B LLM."""
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
    """~2.36B params. ViT and LLM dims of Cosmos-Reason2-2B / Qwen3-VL-2B (see module docstring)."""
    vision = VisionEncoder(
        img_size=448, patch_size=14,
        embed_dim=1024, depth=24, num_heads=16,   # ViT-Large/14
    )
    llm = LLMDecoder(
        vocab_size=151936, hidden_dim=2048,       # Qwen3-1.7B dims (text model of Qwen3-VL-2B)
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
    Matmul FLOPs (1 MAC = 2 FLOPs) for ONE inference call: 1 image, n_text_tokens
    prompt tokens, n_output_tokens generated with a KV cache. Ignores norms,
    softmax and activations. Vision, projector and prefill match
    torch.utils.flop_counter.FlopCounterMode exactly.

    Per transformer block over N tokens of width d (MLP 4d):
        QKV 6*N*d^2 + out-proj 2*N*d^2 + (QK^T, A@V) 4*N^2*d + MLP 16*N*d^2
      = 24*N*d^2 + 4*N^2*d

    Stages:
    - Vision: runs ONCE per image
    - Prefill: LLM sees [N_v + N_t] tokens in one pass; logits only for the last
      position (that is all generation needs; vLLM/TensorRT do the same)
    - Decode: n_output_tokens - 1 steps, each 1 new token attending to the KV cache
      (the prefill's LM head already produced output token 1, so latency is
      TTFT + (N_out - 1) x TPOT)
    """
    ve = model.vision_encoder
    N_v = ve.patch_embed.num_patches
    P = ve.patch_embed.patch_size
    d_v = ve.embed_dim
    d_t = model.llm.hidden_dim
    L_v = len(ve.blocks)
    L_t = len(model.llm.blocks)
    vocab = model.llm.vocab_size

    # -------- Vision encoder (runs once per image) --------
    # Patch embed: conv equivalent to an [N_v, P*P*C] @ [P*P*C, d_v] GEMM
    patch_flops = 2 * N_v * (P * P * 3) * d_v
    vision_block_flops = 2 * (3 * N_v * d_v * d_v +        # QKV projection
                              N_v * d_v * d_v +            # output projection W_o
                              2 * N_v * N_v * d_v +        # attention scores + attn@V
                              2 * N_v * d_v * (4 * d_v))   # FFN (4x ratio)
    vision_flops = patch_flops + L_v * vision_block_flops

    # -------- Projector (2-layer MLP on N_v tokens) --------
    proj_flops = 2 * N_v * d_v * d_t + 2 * N_v * d_t * d_t

    # -------- LLM prefill (N_v + N_t tokens in one pass) --------
    N_prefill = N_v + n_text_tokens
    prefill_block = 2 * (4 * N_prefill * d_t * d_t +             # QKV + out-proj
                         2 * N_prefill * N_prefill * d_t +        # attention
                         2 * N_prefill * d_t * (4 * d_t))         # FFN
    prefill_flops = L_t * prefill_block + 2 * d_t * vocab        # + LM head, last position only

    # -------- LLM decode (one token at a time, attending to a growing KV cache) --------
    # Average context length ≈ N_prefill + n_output_tokens / 2
    avg_context = N_prefill + n_output_tokens // 2
    decode_per_step = 2 * L_t * (4 * d_t * d_t +                # QKV + out-proj for 1 token
                                 2 * avg_context * d_t +         # attention over the cache
                                 2 * d_t * (4 * d_t))            # FFN for 1 token
    decode_per_step += 2 * d_t * vocab                           # LM head for 1 token
    decode_flops = max(n_output_tokens - 1, 0) * decode_per_step    # token 1 came from prefill

    return {
        "vision_gflops": vision_flops / 1e9,
        "projector_gflops": proj_flops / 1e9,
        "prefill_gflops": prefill_flops / 1e9,
        "decode_gflops": decode_flops / 1e9,
        "decode_gflops_per_token": decode_per_step / 1e9,
        "total_gflops": (vision_flops + proj_flops + prefill_flops + decode_flops) / 1e9,
        "n_visual_tokens": N_v,
        "n_prefill_tokens": N_prefill,
    }

def demo_forward_pass(model: VLM = None):
    """Run one forward pass with shape annotations at each stage."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (VLM-Small)")
    print("=" * 60)

    model = (model if model is not None else vlm_small()).eval()

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
              f"← pre-softmax logits; generation reads only [:, -1]")

    print("\n" + "=" * 60)


def demo_causal_image_tokens():
    """
    With a plain causal mask, image token i only sees image tokens 0..i.
    Perturb the LAST image token and check which outputs change.
    """
    torch.manual_seed(0)
    model = vlm_tiny().eval()
    image = torch.randn(1, 3, 224, 224)
    text_ids = torch.randint(0, 32000, (1, 8))
    with torch.no_grad():
        img_tok = model.encode_image(image)                       # [1, 196, 768]
        txt = model.llm.embed_tokens(text_ids)
        base = model.llm(torch.cat([img_tok, txt], 1))
        img_tok2 = img_tok.clone()
        img_tok2[:, -1] += torch.randn_like(img_tok2[:, -1])      # change the last image token only
        pert = model.llm(torch.cat([img_tok2, txt], 1))
    n_v = img_tok.shape[1]
    d_img = (pert[:, :n_v - 1] - base[:, :n_v - 1]).abs().max().item()
    d_txt = (pert[:, n_v:] - base[:, n_v:]).abs().max().item()
    print(f"Change last image token -> earlier image positions change by {d_img:.1e}, "
          f"text positions by {d_txt:.1e}")
    print("  (image tokens are causally masked among themselves; text tokens see the whole image)")

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
    print("\nFLOPs Analysis (1 image, 32 text tokens in, 100 tokens out):")
    print(f"  Visual tokens:    {flops['n_visual_tokens']} "
          f"(prefill total: {flops['n_prefill_tokens']})")
    print(f"  Vision encoder:   {flops['vision_gflops']:>7.2f} GFLOPs  (once per image)")
    print(f"  Projector:        {flops['projector_gflops']:>7.2f} GFLOPs")
    print(f"  LLM prefill:      {flops['prefill_gflops']:>7.2f} GFLOPs  (once per request)")
    print(f"  LLM decode:       {flops['decode_gflops']:>7.2f} GFLOPs  "
          f"(99 steps × {flops['decode_gflops_per_token']:.2f}; token 1 comes from prefill)")
    print(f"  Total:            {flops['total_gflops']:>7.2f} GFLOPs")

    # Forward pass demo (reuse the model instead of building a second copy)
    demo_forward_pass(model)
    del model

    print("\nCausal mask over the image prefix")
    print("-" * 60)
    demo_causal_image_tokens()

    # Compare configurations (meta device: shapes only, no RAM, no init time)
    print("\n" + "=" * 60)
    print("Model Comparison (1 image, 32 text tokens in, 100 out)")
    print("=" * 60)
    configs = [
        ("VLM-Tiny",  vlm_tiny),
        ("VLM-Small", vlm_small),
        ("VLM-Base",  vlm_base),
    ]
    print(f"{'Model':<10} {'Vision':>8} {'Proj':>6} {'LLM':>8} {'Total':>8} "
          f"{'N_v':>5} {'ViT GF':>7} {'Prefill GF':>10} {'GF/token':>8}")
    print("-" * 80)
    for name, fn in configs:
        with torch.device("meta"):
            m = fn()
        v = count_parameters(m.vision_encoder)
        p = count_parameters(m.projector)
        l = count_parameters(m.llm)
        f = compute_flops(m)
        print(f"{name:<10} {v/1e6:>7.1f}M {p/1e6:>5.1f}M {l/1e6:>7.1f}M {(v+p+l)/1e6:>7.1f}M "
              f"{f['n_visual_tokens']:>5} {f['vision_gflops']:>7.1f} {f['prefill_gflops']:>10.1f} "
              f"{f['decode_gflops_per_token']:>8.2f}")
