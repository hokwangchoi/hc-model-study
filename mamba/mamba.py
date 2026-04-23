"""
Mamba — Selective State Space Model Reference Implementation
Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (2023)

Usage:
    python mamba.py

Architecture:
    x [B, L, D]
      ↓ In-projection (expand 2×)
      ↓ 1D conv (local mixing)
      ↓ SiLU activation
      ↓ Selective SSM (the core operation — input-dependent B, C, Δ)
      ↓ Gate × SiLU(residual stream)
      ↓ Out-projection (contract to D)
    y [B, L, D]

Model configs (parameter counts roughly match transformer equivalents):
    - Mamba-S: ~40M   (8 layers, d_model=512)
    - Mamba-B: ~130M  (24 layers, d_model=768)
    - Mamba-L: ~370M  (48 layers, d_model=1024)

KEY INSIGHTS this file tries to teach:

1. DUAL-MODE COMPUTATION:
   - Training: parallel scan over L positions (log-depth via prefix-sum)
   - Inference: recurrent, O(1) memory per step (just carry state)
   Same math, two implementations. This file shows both.

2. NO KV CACHE:
   The state is a fixed-size tensor h of shape [B, D, N] (N ≈ 16).
   Memory is O(D·N), NOT O(L·D) like a transformer's KV cache.
   10× longer context = same memory.

3. "SELECTIVE" means Δ, B, C are functions of the input.
   Plain SSMs (S4) use fixed A, B, C — no gating. Mamba makes them
   data-dependent, closing the gap to attention.

Note: this sequential selective_scan is SLOW (Python loop over L). The real
Mamba uses a custom CUDA kernel (selective_scan_cuda) that fuses the discretize +
scan + output into one SRAM-bound op. See mamba_ssm on GitHub.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Core operation: selective scan
# ============================================================================

def selective_scan_reference(
    u: torch.Tensor,          # [B, L, D]  — input sequence
    delta: torch.Tensor,      # [B, L, D]  — per-token, per-channel timestep
    A: torch.Tensor,          # [D, N]     — state matrix (log-parameterized)
    B: torch.Tensor,          # [B, L, N]  — input→state projection (data-dependent)
    C: torch.Tensor,          # [B, L, N]  — state→output projection (data-dependent)
    D_skip: torch.Tensor,     # [D]        — direct skip from input
) -> torch.Tensor:
    """
    Sequential selective scan. PEDAGOGICALLY CLEAR, SLOW IN PRACTICE.

    For each time step t:
        dA_t = exp(delta_t · A)              # [D, N] per batch — state decay
        dB_t = delta_t · B_t                 # [D, N]           — input injection
        h_t = dA_t * h_{t-1} + dB_t * u_t    # [B, D, N]        — recurrence
        y_t = sum_n C_t[n] * h_t[:,:,n]      # [B, D]           — output
    Final: y = y + D_skip · u (per-channel skip connection).

    Each channel d (of D) runs its own SSM with state of size N; they don't
    interact in the scan. The channels are mixed by the Linear layers
    surrounding this function.

    The real implementation fuses discretization + scan + output into one CUDA
    kernel; this Python version is only for understanding.
    """
    B_size, L, D_size = u.shape
    N = A.shape[-1]

    # Discretize (zero-order hold): dA = exp(delta * A), dB = delta * B.
    # delta: [B, L, D], A: [D, N] → dA: [B, L, D, N]
    deltaA = torch.exp(delta.unsqueeze(-1) * A)                      # [B, L, D, N]
    # Bmat shaped per-channel: same B vector for every channel d.
    deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # [B, L, D, N]

    # Initialize state and scan.
    h = torch.zeros(B_size, D_size, N, device=u.device, dtype=u.dtype)
    ys = []
    for t in range(L):
        h = deltaA[:, t] * h + deltaB_u[:, t]                        # [B, D, N]
        # Output is C_t dotted with state along N, summed per channel.
        y_t = torch.einsum("bdn,bn->bd", h, C[:, t])                 # [B, D]
        ys.append(y_t)
    y = torch.stack(ys, dim=1)                                        # [B, L, D]

    # Per-channel skip connection (scalar gate per channel).
    return y + u * D_skip


# ============================================================================
# Mamba Block
# ============================================================================

class MambaBlock(nn.Module):
    """
    A single Mamba block.

    Structure:
        1. in_proj: Linear [d_model → 2 · d_inner]  (split into two streams)
        2. conv1d: depthwise 1D conv over the sequence dim (local mixing)
        3. SiLU
        4. x_proj: Linear produces data-dependent Δ, B, C
        5. selective_scan (using A, D as learned parameters)
        6. gate: elementwise multiply by SiLU(residual stream)
        7. out_proj: Linear [d_inner → d_model]

    d_inner = expand × d_model (typically expand=2, so d_inner = 2 · d_model).
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,    # rank of the Δ projection; default ceil(d_model/16)
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state          # N in the math
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model  # D in the math (applied per inner channel)
        if dt_rank is None:
            dt_rank = math.ceil(d_model / 16)
        self.dt_rank = dt_rank

        # Projection into the two streams: SSM input, and gating stream
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Depthwise conv over the sequence for local context within the SSM stream
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,          # depthwise — each channel is independent
            padding=d_conv - 1,
            bias=True,
        )

        # Data-dependent projection that produces (Δ_rank, B [N], C [N])
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        # Low-rank projection for Δ itself: dt_rank → d_inner per-channel timesteps
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # State matrix A parameterized as log(-A) so it stays negative after exp.
        # Shape [d_inner, d_state] — one SSM per inner channel, state size N.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))           # learn log|A|; true A = -exp(A_log)
        # Direct skip connection (learned scalar per inner channel)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Final projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Initialize Δ bias so that initial timesteps are moderate (stability trick)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        # Inverse softplus: softplus(bias) == dt
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        returns: [B, L, d_model]
        """
        B_size, L, _ = x.shape

        # In-projection, split into two streams of d_inner each
        xz = self.in_proj(x)                             # [B, L, 2·d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                   # each [B, L, d_inner]

        # Depthwise 1D conv along sequence dim (need to transpose for Conv1d's layout)
        x_ssm = x_ssm.transpose(1, 2)                    # [B, d_inner, L]
        x_ssm = self.conv1d(x_ssm)[..., :L]              # causal-trim the padding
        x_ssm = x_ssm.transpose(1, 2)                    # [B, L, d_inner]
        x_ssm = F.silu(x_ssm)                            # activation before SSM

        # Data-dependent Δ, B, C via a single linear projection
        x_dbl = self.x_proj(x_ssm)                                    # [B, L, dt_rank + 2N]
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        # Δ is softplus-activated for positivity; shape [B, L, d_inner]
        delta = F.softplus(self.dt_proj(dt))

        # True A is negative (so dA = exp(delta · A) decays to 0 at large δ)
        A = -torch.exp(self.A_log.float())               # [d_inner, d_state]

        # Run the selective scan — this is the compute bottleneck in practice
        y = selective_scan_reference(x_ssm, delta, A, B, C, self.D)  # [B, L, d_inner]

        # Gate with the z stream (SiLU-activated) — this is the "gated" part of Mamba
        y = y * F.silu(z)                                # [B, L, d_inner]

        # Project back to d_model
        return self.out_proj(y)                          # [B, L, d_model]

    # ------------------------------------------------------------------------
    # Recurrent mode: process one token, carry a fixed-size state.
    # This is what gets used at INFERENCE — O(1) memory per token, no KV cache.
    # ------------------------------------------------------------------------
    def step(
        self,
        x_t: torch.Tensor,             # [B, d_model]  — ONE token
        h: torch.Tensor,               # [B, d_inner, d_state]  — carried state
        conv_cache: torch.Tensor,      # [B, d_inner, d_conv]    — last d_conv inputs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single-step recurrent forward. Ideal for autoregressive generation:
        constant memory per step, constant FLOPs per step (no attention over
        a growing cache).
        """
        xz = self.in_proj(x_t)                           # [B, 2·d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)                   # [B, d_inner] each

        # Roll the conv cache and apply the conv on the last d_conv inputs
        conv_cache = torch.roll(conv_cache, shifts=-1, dims=-1)
        conv_cache[..., -1] = x_ssm
        # Depthwise conv as a dot product across kernel positions
        x_ssm = torch.einsum(
            "bdk,dk->bd", conv_cache, self.conv1d.weight.squeeze(1)
        ) + self.conv1d.bias
        x_ssm = F.silu(x_ssm)

        # Data-dependent Δ, B, C for this single token
        x_dbl = self.x_proj(x_ssm)                       # [B, dt_rank + 2N]
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(dt))             # [B, d_inner]

        A = -torch.exp(self.A_log.float())               # [d_inner, d_state]
        dA = torch.exp(delta.unsqueeze(-1) * A)          # [B, d_inner, d_state]
        dB = delta.unsqueeze(-1) * B.unsqueeze(1)        # [B, d_inner, d_state]

        # Update state: h = dA * h + dB * x
        h = dA * h + dB * x_ssm.unsqueeze(-1)            # [B, d_inner, d_state]

        # Output: y = C · h + D · x
        y = torch.einsum("bdn,bn->bd", h, C) + self.D * x_ssm

        # Gate
        y = y * F.silu(z)
        y = self.out_proj(y)                              # [B, d_model]

        return y, h, conv_cache


# ============================================================================
# Full model — stack of Mamba blocks with residuals and RMSNorm
# ============================================================================

class RMSNorm(nn.Module):
    """Root-mean-square LayerNorm — what modern LLMs (Llama, Mamba) use instead of LayerNorm."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


class MambaLayer(nn.Module):
    """Residual wrapper: x + Mamba(RMSNorm(x))."""
    def __init__(self, d_model: int, **mamba_kwargs):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaBlock(d_model, **mamba_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))


class Mamba(nn.Module):
    """
    Full Mamba language model: token embedding → L Mamba layers → RMSNorm → LM head.
    No positional encoding — the recurrence is intrinsically positional.
    """
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_layer: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layer)
        ])
        self.norm_f = RMSNorm(d_model)
        # LM head — tie weights to embedding in real Mamba
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: [B, L]
        Returns logits [B, L, vocab_size].
        """
        x = self.embedding(token_ids)         # [B, L, d_model]
        for layer in self.layers:
            x = layer(x)                      # residual inside
        x = self.norm_f(x)
        return self.lm_head(x)                # [B, L, vocab_size]


# ============================================================================
# Model configurations
# ============================================================================

def mamba_small() -> Mamba:
    """Mamba-S: ~40M params. Tiny, for experimentation."""
    return Mamba(vocab_size=50257, d_model=512, n_layer=8)


def mamba_base() -> Mamba:
    """Mamba-B: ~130M params. Comparable to GPT-2 small / BERT-base."""
    return Mamba(vocab_size=50257, d_model=768, n_layer=24)


def mamba_large() -> Mamba:
    """Mamba-L: ~370M params. Comparable to GPT-2 medium."""
    return Mamba(vocab_size=50257, d_model=1024, n_layer=48)


# ============================================================================
# Analysis
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: Mamba):
    """Parameter breakdown by component."""
    print("=" * 60)
    print("Mamba Model Summary")
    print("=" * 60)

    emb = model.embedding.weight.numel()
    # Sum one layer's parameters and multiply, since all layers are identical
    layer_params = sum(p.numel() for p in model.layers[0].parameters())
    all_layers = layer_params * model.n_layer
    final_norm = sum(p.numel() for p in model.norm_f.parameters())

    # Break down one Mamba block
    block = model.layers[0].mixer
    in_proj = sum(p.numel() for p in block.in_proj.parameters())
    conv = sum(p.numel() for p in block.conv1d.parameters())
    x_proj = sum(p.numel() for p in block.x_proj.parameters())
    dt_proj = sum(p.numel() for p in block.dt_proj.parameters())
    out_proj = sum(p.numel() for p in block.out_proj.parameters())
    A_D = block.A_log.numel() + block.D.numel()
    norm = sum(p.numel() for p in model.layers[0].norm.parameters())

    total = count_parameters(model)
    print(f"Embedding (tied w/ LM head): {emb:>12,} params")
    print(f"Final RMSNorm:               {final_norm:>12,} params")
    print(f"All {model.n_layer} layers total:"
          f"            {all_layers:>12,} params")
    print(f"  Per-layer breakdown:")
    print(f"    RMSNorm:          {norm:>10,}")
    print(f"    in_proj (→2·D_i): {in_proj:>10,}")
    print(f"    conv1d:           {conv:>10,}")
    print(f"    x_proj (Δ,B,C):   {x_proj:>10,}")
    print(f"    dt_proj:          {dt_proj:>10,}")
    print(f"    out_proj:         {out_proj:>10,}")
    print(f"    A_log + D:        {A_D:>10,}")
    print("-" * 60)
    print(f"Total:                       {total:>12,} ({total / 1e6:.2f}M)")
    print("=" * 60)


def compute_flops(model: Mamba, seq_len: int) -> dict:
    """
    Approximate FLOPs for one forward pass over a sequence of length L.

    Per layer per token:
      - in_proj:   2 · d · 2·d_i
      - conv1d:    2 · d_i · d_conv        (depthwise, small)
      - x_proj:    2 · d_i · (dt_rank + 2N)
      - dt_proj:   2 · dt_rank · d_i
      - scan:      ~O(d_i · N) per step; linear in L across the sequence
      - out_proj:  2 · d_i · d
    """
    block = model.layers[0].mixer
    d = model.d_model
    d_i = block.d_inner
    N = block.d_state
    L_c = block.d_conv
    dt_rank = block.dt_rank

    per_token = (
        2 * d * (2 * d_i)                      # in_proj
        + 2 * d_i * L_c                        # conv1d (depthwise)
        + 2 * d_i * (dt_rank + 2 * N)          # x_proj
        + 2 * dt_rank * d_i                    # dt_proj
        + 2 * d_i * N                          # selective scan step (dominant: 4·d_i·N in real impl)
        + 2 * d_i * d                          # out_proj
    )
    per_layer = seq_len * per_token
    total = model.n_layer * per_layer

    # Add LM head: 2 · L · d · vocab
    lm_head = 2 * seq_len * d * model.vocab_size

    return {
        "per_token_per_layer_gflops": per_token / 1e9,
        "per_layer_gflops": per_layer / 1e9,
        "all_layers_gflops": total / 1e9,
        "lm_head_gflops": lm_head / 1e9,
        "total_gflops": (total + lm_head) / 1e9,
        "seq_len": seq_len,
    }


def demo_forward_pass():
    """One forward pass with shape annotations (parallel mode)."""
    print("\n" + "=" * 60)
    print("Forward Pass Demo (Mamba-S, parallel / training mode)")
    print("=" * 60)

    model = mamba_small()
    model.eval()

    B, L = 2, 64
    tokens = torch.randint(0, model.vocab_size, (B, L))
    print(f"\n1. Input tokens:       {list(tokens.shape)}")

    with torch.no_grad():
        x = model.embedding(tokens)
        print(f"2. Embedded:           {list(x.shape)}")

        x = model.layers[0](x)
        print(f"3. After layer 0:      {list(x.shape)}  (shape preserved across all layers)")

        logits = model(tokens)
        print(f"4. Output logits:      {list(logits.shape)}")

    print("\n" + "=" * 60)


def demo_recurrent_step():
    """Autoregressive inference: one token, O(1) memory per step."""
    print("\n" + "=" * 60)
    print("Recurrent Step Demo (inference mode — O(1) memory per step)")
    print("=" * 60)

    model = mamba_small()
    model.eval()

    block = model.layers[0].mixer
    B = 1

    # Initialize state: [B, d_inner, d_state] and conv cache [B, d_inner, d_conv]
    h = torch.zeros(B, block.d_inner, block.d_state)
    conv = torch.zeros(B, block.d_inner, block.d_conv)
    x_t = torch.randn(B, model.d_model)

    print(f"\nState (h):       {list(h.shape)}  ← fixed size, independent of L")
    print(f"Conv cache:      {list(conv.shape)}  ← small FIR tap history")
    print(f"Input token emb: {list(x_t.shape)}")

    with torch.no_grad():
        y, h_new, conv_new = block.step(x_t, h, conv)
    print(f"Output:          {list(y.shape)}")
    print(f"Updated state:   {list(h_new.shape)}  ← replace h for next step")
    print(f"\nMemory cost: O(d_inner · d_state) = O({block.d_inner} · {block.d_state}) per layer per batch,")
    print(f"independent of how many tokens we've already generated.")
    print(f"Compare with transformer KV cache: O(L · d · n_layers), which grows with L.")
    print("\n" + "=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Mamba (Selective SSM) Reference Implementation")
    print("=" * 60)

    model = mamba_base()
    print_model_summary(model)

    flops = compute_flops(model, seq_len=1024)
    print(f"\nFLOPs for seq_len=1024:")
    print(f"  Per-token per-layer:  {flops['per_token_per_layer_gflops']:.4f} GFLOPs")
    print(f"  Per-layer (all 1024): {flops['per_layer_gflops']:.2f} GFLOPs")
    print(f"  All {model.n_layer} layers:         "
          f"{flops['all_layers_gflops']:.2f} GFLOPs")
    print(f"  LM head:              {flops['lm_head_gflops']:.2f} GFLOPs")
    print(f"  Total forward:        {flops['total_gflops']:.2f} GFLOPs")

    try:
        demo_forward_pass()
        demo_recurrent_step()
    except Exception as e:
        print(f"\n(Skipping demo — torch not available here: {e})")

    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    configs = [
        ("Mamba-S", mamba_small),
        ("Mamba-B", mamba_base),
        ("Mamba-L", mamba_large),
    ]
    print(f"{'Model':<10} {'Params':>10} {'GFLOPs @ L=1024':>18}")
    print("-" * 42)
    for name, fn in configs:
        m = fn()
        p = count_parameters(m)
        g = compute_flops(m, 1024)["total_gflops"]
        print(f"{name:<10} {p/1e6:>8.1f}M {g:>16.2f}")
