"""
Minimal Transformer Implementation (decoder-only, GPT-style)
Run: python transformer.py

This is a DECODER-ONLY (GPT-2-style) causal language model built from the
blocks of "Attention Is All You Need" (Vaswani et al., 2017). It is NOT the
paper's encoder-decoder model. Differences from the 2017 paper:
  - decoder only: no encoder and no cross-attention
  - pre-LayerNorm + a final LayerNorm   (paper: post-LN, x = LN(x + Sublayer(x)))
  - GELU in the FFN                     (paper: ReLU)
  - learned positional embeddings       (paper: sinusoidal; learned PE was an ablation)
  - no sqrt(d_model) scaling of the token embedding (GPT-2 convention; the paper
    multiplies by sqrt(d_model) and pairs it with sinusoidal PE)
  - no biases in the attention projections, GPT-2 init (normal, std 0.02)
Shared with the paper: scaled dot-product multi-head attention, the
position-wise FFN with d_ff = 4*d, residual connections, and tying the
embedding and output (LM-head) weights (paper Sec. 3.4).

This implementation prioritizes clarity over efficiency.
For production, use F.scaled_dot_product_attention / HuggingFace / TensorRT-LLM.

References:
- Paper: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)
- PyTorch: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Args:
        d_model: Hidden dimension (embedding size)
        n_heads: Number of attention heads
        dropout: Dropout probability

    Shapes:
        Input:  [B, N, d_model]
        mask:   bool tensor broadcastable to [B, h, N, N].
                True = this query MAY attend to this key.
                Same convention as the bool attn_mask of F.scaled_dot_product_attention,
                but the OPPOSITE of nn.MultiheadAttention / nn.Transformer, where
                True (or -inf) means "blocked". A row with no True entry gives NaN.
        Output: [B, N, d_model]

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
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
        self.last_attn = None  # attention weights of the last call (for inspection/demos)

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
        # GPU: cuBLAS batched GEMM (one small GEMM per (batch, head))
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask (for causal attention): blocked positions get -inf,
        # so exp(-inf) = 0 after the softmax.
        if mask is not None:
            assert mask.dtype == torch.bool, "mask must be bool, True = may attend"
            scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax and dropout
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attn = attn_weights.detach()
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

    FFN(x) = Linear2(GELU(Linear1(x)))      (the 2017 paper uses ReLU)

    Holds ~2/3 of each block's parameters (8d² of 12d², ignoring biases).

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
    Single Transformer block with pre-normalization (GPT-2 style).

    Structure:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    (The 2017 paper uses post-norm: x = LN(x + Sublayer(x)).)

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
    Complete decoder-only Transformer language model.

    Args:
        vocab_size:   Size of vocabulary (V)
        d_model:      Hidden dimension (d)
        n_heads:      Number of attention heads (h)
        n_layers:     Number of transformer blocks (L)
        d_ff:         Feed-forward hidden dimension (d_ff, default 4*d)
        max_seq_len:  Maximum sequence length (N_max)
        dropout:      Dropout probability

    Shapes:
        Input:  [B, N] token ids
        Output: [B, N, V] logits (row i = scores for the token at position i+1)

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

        # Output projection to vocabulary (the "LM head")
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
        assert N <= self.pos_emb.num_embeddings, (
            f"sequence length {N} > max_seq_len {self.pos_emb.num_embeddings} "
            "(learned positional embeddings cannot extrapolate)")

        # Causal mask: True = may attend. Query i sees keys 0..i (lower triangle).
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=x.device))

        # Embeddings: row lookup in the [V, d] table + a learned position vector.
        # GPT-2 style: no sqrt(d_model) scaling (see module docstring).
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        x = self.token_emb(x)
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
        """Count total trainable parameters (tied weights counted once)."""
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
    ffn = 2 * d * d_ff + d_ff + d   # weights + biases
    layer_norm = 4 * d
    per_layer = qkv + out_proj + ffn + layer_norm
    tok_emb = V * d                 # tied with the LM head, so counted once
    pos_emb = N_max * d
    final_ln = 2 * d

    print(f"\n{'PER LAYER BREAKDOWN':-^70}")
    print(f"  QKV projections (W_q, W_k, W_v): 3 × d × d = 3 × {d}² = {qkv:,}")
    print(f"  Output projection (W_o):         d × d = {d}² = {out_proj:,}")
    print(f"  FFN (Linear1 + Linear2):         2 × d × d_ff + d_ff + d = {ffn:,}")
    print(f"  LayerNorm (γ, β × 2):            4 × d = {layer_norm:,}")
    print(f"  {'':->50}")
    print(f"  Total per layer:                 {per_layer:,}")
    print(f"  Total for {L} layers:              {per_layer * L:,}")
    print(f"  + token embedding (= LM head):   V × d = {tok_emb:,}")
    print(f"  + position embedding:            N_max × d = {pos_emb:,}")
    print(f"  + final LayerNorm:               2 × d = {final_ln:,}")
    breakdown = per_layer * L + tok_emb + pos_emb + final_ln
    print(f"  = {breakdown:,}  (matches Total)")
    assert breakdown == total_params, "parameter breakdown does not add up"

    print(f"\n{'COMPLEXITY (per layer, batch=1)':-^70}")
    print(f"  QKV projection FLOPs:    6 × N × d² = 6N × {d}²")
    print(f"  Attention (QKᵀ + ×V):    4 × N² × d = 4N² × {d}  ← O(N²)!")
    print(f"  Output projection:       2 × N × d² = 2N × {d}²")
    print(f"  FFN:                     4 × N × d × d_ff = 4N × {d} × {d_ff}")
    print(f"  Softmax (estimate):      5 × h × N² = 5 × {h} × N²")
    print(f"  (+ once per model: LM head 2 × N × d × V = 2N × {d} × {V})")

    print(f"\n{'MEMORY (per layer, FP32)':-^70}")
    print(f"  Attention scores:  h × N × N × 4B = {h} × N² × 4 bytes  ← O(N²)!")
    print(f"  KV cache (infer):  2 × N × d × 4B = 2 × N × {d} × 4 bytes")
    print(f"  Activations:       N × d × 4B = N × {d} × 4 bytes")

    print("\n" + "=" * 70)


def compute_flops(B: int, N: int, d: int, h: int, d_ff: int, L: int, V: int = 0) -> dict:
    """
    FLOPs of one forward pass over N tokens (2 FLOPs per multiply-accumulate).

    Counts the matmuls of the L blocks and the LM head (2*B*N*d*V; pass V=0 to
    skip it), plus a 5*h*N² softmax estimate. Ignores LayerNorm, GELU, residual
    adds, the embedding gather, and the ~2x saving a causal kernel can get by
    skipping masked scores.

    All keys except 'per_layer' are summed over the L layers;
    'per_layer' is ONE layer (softmax included, LM head excluded).
    """
    # Per layer
    qkv = 6 * B * N * d * d
    qkt = 2 * B * N * N * d
    softmax = 5 * B * h * N * N
    av = 2 * B * N * N * d
    out_proj = 2 * B * N * d * d
    ffn = 4 * B * N * d * d_ff

    per_layer = qkv + qkt + softmax + av + out_proj + ffn
    lm_head = 2 * B * N * d * V
    total = L * per_layer + lm_head

    return {
        'qkv_projection': qkv * L,
        'qk_transpose': qkt * L,
        'softmax': softmax * L,
        'attn_v': av * L,
        'out_projection': out_proj * L,
        'ffn': ffn * L,
        'lm_head': lm_head,
        'per_layer': per_layer,
        'total': total
    }


def compute_memory(B: int, N: int, d: int, h: int, dtype_bytes: int = 4) -> dict:
    """
    Memory of ONE layer, in bytes.
    attention_scores: the naive [B, h, N, N] score matrix. At inference it is
                      freed before the next layer runs; FlashAttention never stores it.
    kv_cache:         K and V for N tokens with classic multi-head attention
                      (with grouped-query attention, multiply by n_kv_heads / h).
    """
    return {
        'attention_scores': B * h * N * N * dtype_bytes,
        'kv_cache': 2 * B * N * d * dtype_bytes,
        'activations': B * N * d * dtype_bytes,
    }


def format_number(n: float) -> str:
    """Format large numbers with K/M/G/T/P suffixes (powers of 1000)."""
    for div, suffix in ((1e15, "P"), (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(n) >= div:
            return f"{n/div:.2f}{suffix}"
    return f"{n:.0f}"


def format_bytes(b: float) -> str:
    """Format bytes with binary units (KiB/MiB/GiB/TiB, powers of 1024)."""
    for div, suffix in ((2**40, "TiB"), (2**30, "GiB"), (2**20, "MiB"), (2**10, "KiB")):
        if b >= div:
            return f"{b/div:.2f} {suffix}"
    return f"{b:.0f} B"


def demo_forward_pass():
    """Demonstrate a forward pass with shape annotations."""
    print("\n" + "=" * 70)
    print("DEMO: Forward Pass with Shape Tracking")
    print("=" * 70)

    torch.manual_seed(0)

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
    model.eval()  # disable dropout for inference (torch.no_grad only disables autograd)

    # Print summary
    print_model_summary(model)

    # Create dummy input
    x = torch.randint(0, V, (B, N))
    print(f"\nInput shape:  {list(x.shape)} = [batch, sequence]")

    # Forward pass
    with torch.no_grad():
        logits = model(x)
    print(f"Output shape: {list(logits.shape)} = [batch, sequence, vocab]")
    assert logits.shape == (B, N, V)

    # Compute FLOPs
    flops = compute_flops(B, N, d, h, d_ff, L, V)

    print(f"\n{'FLOPs BREAKDOWN (all layers)':-^70}")
    print(f"  QKV projection:  {format_number(flops['qkv_projection']):>10} FLOPs")
    print(f"  QKᵀ matmul:      {format_number(flops['qk_transpose']):>10} FLOPs")
    print(f"  Softmax (est.):  {format_number(flops['softmax']):>10} FLOPs")
    print(f"  Attention × V:   {format_number(flops['attn_v']):>10} FLOPs")
    print(f"  Output proj:     {format_number(flops['out_projection']):>10} FLOPs")
    print(f"  FFN:             {format_number(flops['ffn']):>10} FLOPs")
    print(f"  LM head:         {format_number(flops['lm_head']):>10} FLOPs")
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


def demo_causality():
    """Show the causal attention pattern and why it makes a KV cache valid."""
    print("\n" + "=" * 70)
    print("DEMO: Causal mask => past tokens never change => KV cache is valid")
    print("=" * 70)

    torch.manual_seed(0)
    model = Transformer(vocab_size=1000, d_model=64, n_heads=4, n_layers=2, max_seq_len=64)
    model.eval()
    x = torch.randint(0, 1000, (1, 8))

    with torch.no_grad():
        logits = model(x)
        w = model.layers[0].attn.last_attn[0, 0, :5, :5]  # layer 0, head 0, first 5 tokens
    print("\nAttention weights, layer 0 / head 0 (rows = query i, cols = key j):")
    for i, row in enumerate(w.tolist()):
        print(f"  q{i}: " + " ".join(f"{v:5.2f}" for v in row))
    print("  -> upper triangle is exactly 0 (no peeking ahead); each row sums to 1.")
    print("     An untrained model spreads weight almost evenly: ~1, 1/2, 1/3, 1/4, 1/5.")

    with torch.no_grad():
        prefix_logits = model(x[:, :5])
    same = torch.allclose(logits[:, :5], prefix_logits, atol=1e-5)
    print(f"\nLogits of tokens 0..4 unchanged when tokens 5..7 are appended? {same}")
    print("  -> K, V of old tokens never change, so they can be computed once and cached.")


def demo_scaling():
    """Show how complexity scales with different parameters."""
    print("\n" + "=" * 70)
    print("SCALING ANALYSIS: How complexity grows")
    print("=" * 70)

    V = 50257  # GPT-2 / GPT-3 vocabulary
    configs = [
        # name,                 d,     h,  L,  N (context)
        ("GPT-2 Small (124M)",  768,   12, 12, 1024),
        ("GPT-2 Medium (355M)", 1024,  16, 24, 1024),
        ("GPT-3 6.7B",          4096,  32, 32, 2048),
        ("GPT-3 175B",          12288, 96, 96, 2048),
    ]

    print("\nOne forward pass over the full context, batch 1, LM head included.")
    print("'Attn scores/layer' = one layer's naive [h, N, N] FP32 score matrix.")
    print(f"\n{'Model':<20} {'d':>6} {'h':>4} {'L':>4} {'N':>6} {'FLOPs':>10} {'Attn scores/layer':>18}")
    print("-" * 72)

    for name, d, h, L, N in configs:
        d_ff = 4 * d
        flops = compute_flops(1, N, d, h, d_ff, L, V)
        mem = compute_memory(1, N, d, h)

        print(f"{name:<20} {d:>6} {h:>4} {L:>4} {N:>6} {format_number(flops['total']):>10} "
              f"{format_bytes(mem['attention_scores']):>18}")

    print("\n" + "=" * 70)

    # Show N² scaling
    print("\nSequence length scaling (d=4096, h=32, L=32):")
    print(f"\n{'N':>8} {'Attention FLOPs (all L)':>24} {'Attn scores / layer (FP32)':>28}")
    print("-" * 62)

    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        flops = compute_flops(1, N, 4096, 32, 16384, 32)
        mem = compute_memory(1, N, 4096, 32)

        attn_flops = flops['qk_transpose'] + flops['attn_v']
        attn_mem = mem['attention_scores']  # one layer; freed before the next layer at inference

        print(f"{N:>8} {format_number(attn_flops):>24} {format_bytes(attn_mem):>28}")

    print("\n⚠️  Note: Attention memory scales O(N²) - this is why FlashAttention exists!")
    print("   (Training keeps every layer's activations for backward: multiply by L.)")


if __name__ == "__main__":
    demo_forward_pass()
    demo_causality()
    demo_scaling()
