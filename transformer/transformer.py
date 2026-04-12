"""
Minimal Transformer Implementation
Run: python transformer.py

This implementation prioritizes clarity over efficiency.
For production, use torch.nn.TransformerEncoder or HuggingFace.

References:
- Paper: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
- PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Hidden dimension (embedding size)
        n_heads: Number of attention heads
        dropout: Dropout probability

    Shapes:
        Input:  [B, N, d_model]
        Output: [B, N, d_model]

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension per head
        self.scale = self.d_k ** -0.5   # 1/sqrt(d_k)

        # Linear projections for Q, K, V
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, _ = x.shape

        # Project to Q, K, V
        # Each: [B, N, d_model] @ [d_model, d_model] -> [B, N, d_model]
        # GPU: cuBLAS GEMM kernel
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        # [B, N, d_model] -> [B, N, n_heads, d_k] -> [B, n_heads, N, d_k]
        Q = Q.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        # [B, h, N, d_k] @ [B, h, d_k, N] -> [B, h, N, N]
        # This is O(N²) memory - the bottleneck!
        # GPU: cuBLAS batched GEMM
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [B, h, N, N] @ [B, h, N, d_k] -> [B, h, N, d_k]
        context = torch.matmul(attn_weights, V)

        # Reshape back
        # [B, h, N, d_k] -> [B, N, h, d_k] -> [B, N, d_model]
        context = context.transpose(1, 2).contiguous().view(B, N, self.d_model)

        # Output projection
        return self.W_o(context)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    FFN(x) = Linear2(GELU(Linear1(x)))

    This is where ~67% of the parameters live!

    Shapes:
        Input:  [B, N, d_model]
        Output: [B, N, d_model]

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model  # Default: 4x expansion

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, N, d] -> [B, N, d_ff] -> [B, N, d]
        x = self.linear1(x)      # GPU: cuBLAS GEMM
        x = self.activation(x)   # GPU: Elementwise kernel
        x = self.dropout(x)
        x = self.linear2(x)      # GPU: cuBLAS GEMM
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-normalization.

    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Pre-norm + residual for attention
        x = x + self.dropout(self.attn(self.norm1(x), mask))

        # Pre-norm + residual for FFN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


class Transformer(nn.Module):
    """
    Complete Transformer model.

    Args:
        vocab_size:   Size of vocabulary (V)
        d_model:      Hidden dimension (d)
        n_heads:      Number of attention heads (h)
        n_layers:     Number of transformer blocks (L)
        d_ff:         Feed-forward hidden dimension (d_ff, default 4*d)
        max_seq_len:  Maximum sequence length (N_max)
        dropout:      Dropout probability

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights with token embeddings (common practice)
        self.output.weight = self.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape

        # Create causal mask
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        mask = ~mask  # Invert: True = attend, False = mask

        # Embeddings
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        x = self.token_emb(x) * math.sqrt(self.d_model)
        x = x + self.pos_emb(positions)
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output
        x = self.norm(x)
        logits = self.output(x)

        return logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def print_model_summary(model: Transformer):
    """Print a summary of the model architecture."""
    print("=" * 70)
    print("TRANSFORMER MODEL SUMMARY")
    print("=" * 70)

    d = model.d_model
    h = model.layers[0].attn.n_heads
    d_k = model.layers[0].attn.d_k
    d_ff = model.layers[0].ffn.linear1.out_features
    L = len(model.layers)
    V = model.token_emb.num_embeddings
    N_max = model.pos_emb.num_embeddings

    print(f"\n{'NOTATION':-^70}")
    print(f"  B     = Batch size")
    print(f"  N     = Sequence length (max {N_max})")
    print(f"  d     = Hidden dimension = {d}")
    print(f"  h     = Number of heads = {h}")
    print(f"  d_k   = Head dimension = d/h = {d_k}")
    print(f"  d_ff  = FFN hidden = {d_ff}")
    print(f"  L     = Number of layers = {L}")
    print(f"  V     = Vocabulary size = {V}")

    total_params = model.count_parameters()
    print(f"\n{'PARAMETERS':-^70}")
    print(f"  Total: {total_params:,} ({total_params / 1e6:.2f}M)")

    # Parameter breakdown
    qkv = 3 * d * d
    out_proj = d * d
    ffn = 2 * d * d_ff
    layer_norm = 4 * d
    per_layer = qkv + out_proj + ffn + layer_norm

    print(f"\n{'PER LAYER BREAKDOWN':-^70}")
    print(f"  QKV projections (W_q, W_k, W_v): 3 × d × d = 3 × {d}² = {qkv:,}")
    print(f"  Output projection (W_o):         d × d = {d}² = {out_proj:,}")
    print(f"  FFN (Linear1 + Linear2):         2 × d × d_ff = 2 × {d} × {d_ff} = {ffn:,}")
    print(f"  LayerNorm (γ, β × 2):            4 × d = {layer_norm:,}")
    print(f"  {'':->50}")
    print(f"  Total per layer:                 {per_layer:,}")
    print(f"  Total for {L} layers:              {per_layer * L:,}")

    print(f"\n{'COMPLEXITY (per layer, batch=1)':-^70}")
    print(f"  QKV projection FLOPs:    6 × N × d² = 6N × {d}²")
    print(f"  Attention (QKᵀ + ×V):    4 × N² × d = 4N² × {d}  ← O(N²)!")
    print(f"  Output projection:       2 × N × d² = 2N × {d}²")
    print(f"  FFN:                     4 × N × d × d_ff = 4N × {d} × {d_ff}")
    print(f"  Softmax:                 5 × h × N² = 5 × {h} × N²")

    print(f"\n{'MEMORY (per layer, FP32)':-^70}")
    print(f"  Attention scores:  h × N × N × 4B = {h} × N² × 4 bytes  ← O(N²)!")
    print(f"  KV cache (infer):  2 × N × d × 4B = 2 × N × {d} × 4 bytes")
    print(f"  Activations:       N × d × 4B = N × {d} × 4 bytes")

    print("\n" + "=" * 70)


def compute_flops(B: int, N: int, d: int, h: int, d_ff: int, L: int) -> dict:
    """Compute FLOPs breakdown for forward pass."""
    # Per layer
    qkv = 6 * B * N * d * d
    qkt = 2 * B * N * N * d
    softmax = 5 * B * h * N * N
    av = 2 * B * N * N * d
    out_proj = 2 * B * N * d * d
    ffn = 4 * B * N * d * d_ff

    per_layer = qkv + qkt + softmax + av + out_proj + ffn
    total = L * per_layer

    return {
        'qkv_projection': qkv * L,
        'qk_transpose': qkt * L,
        'softmax': softmax * L,
        'attn_v': av * L,
        'out_projection': out_proj * L,
        'ffn': ffn * L,
        'per_layer': per_layer,
        'total': total
    }


def compute_memory(B: int, N: int, d: int, h: int, dtype_bytes: int = 4) -> dict:
    """Compute memory breakdown per layer."""
    return {
        'attention_scores': B * h * N * N * dtype_bytes,
        'kv_cache': 2 * B * N * d * dtype_bytes,
        'activations': B * N * d * dtype_bytes,
    }


def format_number(n: float) -> str:
    """Format large numbers with K/M/G/T suffixes."""
    if n >= 1e12:
        return f"{n/1e12:.2f}T"
    if n >= 1e9:
        return f"{n/1e9:.2f}G"
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"


def format_bytes(b: float) -> str:
    """Format bytes with KB/MB/GB suffixes."""
    if b >= 1e9:
        return f"{b/1e9:.2f} GB"
    if b >= 1e6:
        return f"{b/1e6:.2f} MB"
    if b >= 1e3:
        return f"{b/1e3:.2f} KB"
    return f"{b:.0f} B"


def demo_forward_pass():
    """Demonstrate a forward pass with shape annotations."""
    print("\n" + "=" * 70)
    print("DEMO: Forward Pass with Shape Tracking")
    print("=" * 70)

    # Model config
    B = 2       # Batch size
    N = 16      # Sequence length
    V = 1000    # Vocab size
    d = 256     # Hidden dimension
    h = 8       # Number of heads
    L = 4       # Number of layers
    d_ff = 4 * d

    print(f"\nConfiguration:")
    print(f"  B (batch)     = {B}")
    print(f"  N (seq len)   = {N}")
    print(f"  d (hidden)    = {d}")
    print(f"  h (heads)     = {h}")
    print(f"  d_k (per head)= {d // h}")
    print(f"  d_ff (FFN)    = {d_ff}")
    print(f"  L (layers)    = {L}")
    print(f"  V (vocab)     = {V}")

    # Create model
    model = Transformer(
        vocab_size=V,
        d_model=d,
        n_heads=h,
        n_layers=L,
        max_seq_len=512
    )

    # Print summary
    print_model_summary(model)

    # Create dummy input
    x = torch.randint(0, V, (B, N))
    print(f"\nInput shape:  [{B}, {N}] = [batch, sequence]")

    # Forward pass
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape: [{B}, {N}, {V}] = [batch, sequence, vocab]")

    # Compute FLOPs
    flops = compute_flops(B, N, d, h, d_ff, L)

    print(f"\n{'FLOPs BREAKDOWN (all layers)':-^70}")
    print(f"  QKV projection:  {format_number(flops['qkv_projection']):>10} FLOPs")
    print(f"  QKᵀ matmul:      {format_number(flops['qk_transpose']):>10} FLOPs")
    print(f"  Softmax:         {format_number(flops['softmax']):>10} FLOPs")
    print(f"  Attention × V:   {format_number(flops['attn_v']):>10} FLOPs")
    print(f"  Output proj:     {format_number(flops['out_projection']):>10} FLOPs")
    print(f"  FFN:             {format_number(flops['ffn']):>10} FLOPs")
    print(f"  {'':->50}")
    print(f"  Total:           {format_number(flops['total']):>10} FLOPs")

    # Compute memory
    mem = compute_memory(B, N, d, h)

    print(f"\n{'MEMORY PER LAYER (FP32)':-^70}")
    print(f"  Attention scores: {format_bytes(mem['attention_scores']):>10}  (O(N²) - bottleneck!)")
    print(f"  KV cache:         {format_bytes(mem['kv_cache']):>10}  (O(N))")
    print(f"  Activations:      {format_bytes(mem['activations']):>10}  (O(N))")

    print("\n" + "=" * 70)
    print("✓ Forward pass complete!")
    print("=" * 70)


def demo_scaling():
    """Show how complexity scales with different parameters."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: How complexity grows")
    print("=" * 70)

    configs = [
        ("Small (GPT-2)",    512,  8, 12, 2048),
        ("Medium",           1024, 16, 24, 2048),
        ("Large (GPT-3 6B)", 4096, 32, 32, 2048),
        ("XL (GPT-3 175B)",  12288, 96, 96, 2048),
    ]

    print(f"\n{'Model':<20} {'d':>6} {'h':>4} {'L':>4} {'N':>6} {'FLOPs':>12} {'Attn Mem':>12}")
    print("-" * 70)

    for name, d, h, L, N in configs:
        d_ff = 4 * d
        flops = compute_flops(1, N, d, h, d_ff, L)
        mem = compute_memory(1, N, d, h)
        attn_mem_total = mem['attention_scores'] * L

        print(f"{name:<20} {d:>6} {h:>4} {L:>4} {N:>6} {format_number(flops['total']):>12} {format_bytes(attn_mem_total):>12}")

    print("\n" + "=" * 70)

    # Show N² scaling
    print("\nSequence length scaling (d=4096, h=32, L=32):")
    print(f"\n{'N':>8} {'Attention FLOPs':>18} {'Attention Memory':>18}")
    print("-" * 50)

    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        flops = compute_flops(1, N, 4096, 32, 16384, 32)
        mem = compute_memory(1, N, 4096, 32)

        attn_flops = flops['qk_transpose'] + flops['attn_v']
        attn_mem = mem['attention_scores'] * 32

        print(f"{N:>8} {format_number(attn_flops):>18} {format_bytes(attn_mem):>18}")

    print("\n⚠️  Note: Attention memory scales O(N²) - this is why FlashAttention exists!")


if __name__ == "__main__":
    demo_forward_pass()
    demo_scaling()
