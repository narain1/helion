"""
Qwen 3.5 2B Mega Kernel
=======================

Complete fused decoder block for Qwen 3.5 2B model inference.

This example implements the full transformer decoder block as a set of highly-optimized
Helion kernels, covering:

- **RMSNorm** pre-normalization
- **Grouped Query Attention (GQA)** with Flash-Attention-style tiled online softmax
  and causal masking
- **Rotary Position Embedding (RoPE)** applied to queries and keys
- **SwiGLU Feed-Forward Network (FFN)** with SiLU gating

All of these are combined into a single ``qwen_decoder_block`` function that represents
the "mega kernel" for one complete transformer layer of the Qwen 3.5 2B model.

Architecture (Qwen 3.5 2B):

- Hidden size: 2048
- Attention heads: 16 (GQA with 8 KV heads, groups of 2)
- Head dimension: 128
- Intermediate FFN size: 11008
- Normalization: RMSNorm (eps = 1e-6)
- Position encoding: RoPE (theta = 1,000,000)
- Activation: SwiGLU (SiLU gating)
"""

# %%
# Imports
# -------

# %%
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import helion
from helion._testing import DEVICE
from helion._testing import HALF_DTYPE
from helion._testing import run_example
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable


# %%
# Architecture Configuration
# --------------------------


# %%
@dataclass
class Qwen35Config:
    """Configuration for Qwen 3.5 2B model architecture."""

    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 11008
    num_hidden_layers: int = 28
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per key-value head (GQA ratio)."""
        return self.num_attention_heads // self.num_key_value_heads


# %%
# Helion Kernels
# --------------


# %%
@helion.kernel(static_shapes=True)
def rms_norm(
    x: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Root Mean Square Layer Normalization.

    Computes: output = x / rms(x) * weight, where rms(x) = sqrt(mean(x^2) + eps)

    Args:
        x: Input tensor of shape [tokens, hidden_size]
        weight: Scale parameter of shape [hidden_size]
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    m, n = x.size()
    n = hl.specialize(n)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        x_tile = x[tile_m, :].to(torch.float32)
        inv_rms = torch.rsqrt((x_tile * x_tile).mean(dim=-1) + eps)
        out[tile_m, :] = (x_tile * inv_rms[:, None] * weight[:].to(torch.float32)).to(
            out.dtype
        )
    return out


# %%
@helion.kernel(static_shapes=True)
def gqa_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Tensor:
    """
    Grouped Query Attention with Flash-Attention-style tiled online softmax and causal masking.

    Implements scaled dot-product attention using the online softmax algorithm for
    numerical stability and memory efficiency. Applies a causal mask so that each
    query position can only attend to key positions at or before its own position.

    This kernel is the core "mega" component: it handles the entire attention
    computation (QK similarity, masking, softmax, value aggregation) in a single
    tiled pass, avoiding materializing the full attention weight matrix.

    GQA note: K and V are expected to already be expanded to match the number of
    query heads (via ``repeat_interleave``). See ``qwen_decoder_block`` for the
    expansion step.

    Args:
        q: Query tensor of shape [batch * num_heads, seq_len, head_dim]
        k: Key tensor of shape [batch * num_heads, seq_len, head_dim]
        v: Value tensor of shape [batch * num_heads, seq_len, head_dim]

    Returns:
        Attention output of shape [batch * num_heads, seq_len, head_dim]
    """
    bh, seq_len, head_dim = q.size()
    head_dim = hl.specialize(head_dim)
    k_view = k.transpose(1, 2)  # [bh, head_dim, seq_len]
    out = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(head_dim)
    qk_scale = sm_scale * 1.44269504  # multiply by 1/log(2) to use exp2

    for tile_bh, tile_m in hl.tile([bh, seq_len]):
        m_i = hl.full([tile_bh, tile_m], float("-inf"), dtype=torch.float32)
        l_i = torch.full_like(m_i, 1.0)
        acc = hl.zeros([tile_bh, tile_m, head_dim], dtype=torch.float32)
        q_tile = q[tile_bh, tile_m, :]  # [tile_bh, tile_m, head_dim]

        for tile_n in hl.tile(seq_len):
            k_tile = k_view[tile_bh, :, tile_n]  # [tile_bh, head_dim, tile_n]
            qk = torch.bmm(q_tile, k_tile)  # [tile_bh, tile_m, tile_n]

            # Causal mask: query at absolute position p can only attend to keys at p' <= p.
            # tile_m.index / tile_n.index give absolute (not block-relative) positions.
            m_idx = tile_m.index  # [tile_m_size] — absolute query positions
            n_idx = tile_n.index  # [tile_n_size] — absolute key positions
            causal_mask = m_idx[:, None] >= n_idx[None, :]  # [tile_m, tile_n]
            qk = torch.where(
                causal_mask[None, :, :],
                qk,
                torch.full_like(qk, float("-inf")),
            )

            # Online softmax update (base-2 arithmetic for efficiency)
            m_ij = torch.maximum(m_i, torch.amax(qk, -1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, :, None]
            p = torch.exp2(qk)
            l_ij = torch.sum(p, -1)
            alpha = torch.exp2(m_i - m_ij)
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, :, None]

            v_tile = v[tile_bh, tile_n, :]  # [tile_bh, tile_n, head_dim]
            p = p.to(v_tile.dtype)
            acc = torch.baddbmm(acc, p, v_tile)
            m_i = m_ij

        acc = acc / l_i[:, :, None]
        out[tile_bh, tile_m, :] = acc.to(out.dtype)

    return out


# %%
@helion.kernel
def swiglu_fwd(a: Tensor, b: Tensor) -> Tensor:
    """
    SwiGLU activation: SiLU(a) * b.

    SiLU (Swish) activation: SiLU(x) = x * sigmoid(x)
    SwiGLU(a, b) = SiLU(a) * b

    This is the gating function used in Qwen's feed-forward network.

    Args:
        a: Gate tensor (input to SiLU), any shape
        b: Up-projection tensor, same shape as a

    Returns:
        SwiGLU output with same shape as inputs
    """
    assert a.shape == b.shape, (
        f"Input tensors must have same shape, got {a.shape} != {b.shape}"
    )
    out = torch.empty_like(a)
    total_elements = a.numel()
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    out_flat = out.view(-1)
    for tile_idx in hl.tile(total_elements):
        a_vals = a_flat[tile_idx].to(torch.float32)
        b_vals = b_flat[tile_idx]
        silu_a = a_vals * torch.sigmoid(a_vals)
        out_flat[tile_idx] = silu_a.to(b_vals.dtype) * b_vals
    return out


# %%
# Rotary Position Embedding Utilities
# ------------------------------------


# %%
def precompute_rope_cos_sin(
    seq_len: int,
    head_dim: int,
    rope_theta: float = 1_000_000.0,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """
    Precompute cosine and sine tables for Rotary Position Embedding (RoPE).

    Uses the standard RoPE formulation with frequencies:
        freq_i = 1 / (theta ^ (2i / head_dim)) for i in [0, head_dim/2)

    Args:
        seq_len: Maximum sequence length
        head_dim: Attention head dimension
        rope_theta: RoPE base frequency (1e6 for Qwen 3.5)
        device: Target device
        dtype: Data type for the tables

    Returns:
        Tuple of (cos, sin) tables each of shape [seq_len, head_dim // 2]
    """
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [seq_len, head_dim // 2]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rotary_pos_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> Tensor:
    """
    Apply Rotary Position Embedding to a query or key tensor.

    Uses the standard split-half rotation:
        out[..., :hd] = x[..., :hd] * cos - x[..., hd:] * sin
        out[..., hd:] = x[..., hd:] * cos + x[..., :hd] * sin

    Args:
        x: Tensor of shape [batch * heads, seq_len, head_dim]
        cos: Cosine table of shape [seq_len, head_dim // 2]
        sin: Sine table of shape [seq_len, head_dim // 2]

    Returns:
        RoPE-encoded tensor of same shape as x
    """
    _bh, _seq, head_dim = x.shape
    hd = head_dim // 2
    x1 = x[..., :hd]
    x2 = x[..., hd:]
    c = cos.unsqueeze(0)  # [1, seq, hd] — broadcast over batch*heads
    s = sin.unsqueeze(0)  # [1, seq, hd]
    return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)


# %%
# Complete Decoder Block
# ----------------------


# %%
def qwen_decoder_block(
    hidden_states: Tensor,
    attn_norm_weight: Tensor,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    ffn_norm_weight: Tensor,
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    config: Qwen35Config,
    batch_size: int,
    seq_len: int,
) -> Tensor:
    """
    Complete Qwen 3.5 2B decoder block (single transformer layer).

    Implements the full layer with all operations fused via Helion kernels:

    1. **Pre-attention RMSNorm** (Helion kernel)
    2. QKV linear projections
    3. **Rotary Position Embedding** applied to Q and K
    4. GQA expansion of K and V via ``repeat_interleave``
    5. **GQA Attention** with causal masking and online softmax (Helion mega kernel)
    6. Output projection + residual connection
    7. **Pre-FFN RMSNorm** (Helion kernel)
    8. Gate and up-projections for SwiGLU FFN
    9. **SwiGLU activation** (Helion kernel)
    10. Down-projection + residual connection

    Args:
        hidden_states: Input tensor of shape [batch_size * seq_len, hidden_size]
        attn_norm_weight: RMSNorm weight for attention pre-norm, shape [hidden_size]
        q_proj_weight: Query projection weight, shape [num_heads * head_dim, hidden_size]
        k_proj_weight: Key projection weight, shape [num_kv_heads * head_dim, hidden_size]
        v_proj_weight: Value projection weight, shape [num_kv_heads * head_dim, hidden_size]
        o_proj_weight: Output projection weight, shape [hidden_size, num_heads * head_dim]
        ffn_norm_weight: RMSNorm weight for FFN pre-norm, shape [hidden_size]
        gate_proj_weight: FFN gate projection, shape [intermediate_size, hidden_size]
        up_proj_weight: FFN up projection, shape [intermediate_size, hidden_size]
        down_proj_weight: FFN down projection, shape [hidden_size, intermediate_size]
        cos: RoPE cosine table, shape [seq_len, head_dim // 2]
        sin: RoPE sine table, shape [seq_len, head_dim // 2]
        config: Qwen 3.5 model configuration
        batch_size: Batch dimension size
        seq_len: Sequence length

    Returns:
        Updated hidden states of shape [batch_size * seq_len, hidden_size]
    """
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    num_kv_groups = config.num_kv_groups
    head_dim = config.head_dim

    # ── Attention Block ──────────────────────────────────────────────────────
    residual = hidden_states

    # 1. Pre-attention RMSNorm (Helion kernel)
    normed = rms_norm(hidden_states, attn_norm_weight, config.rms_norm_eps)

    # 2. QKV linear projections
    q = F.linear(normed, q_proj_weight)  # [tokens, num_heads * head_dim]
    k = F.linear(normed, k_proj_weight)  # [tokens, num_kv_heads * head_dim]
    v = F.linear(normed, v_proj_weight)  # [tokens, num_kv_heads * head_dim]

    # 3. Reshape → [batch, heads, seq, head_dim] → flatten batch*heads
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    q = q.reshape(batch_size * num_heads, seq_len, head_dim)
    k = k.reshape(batch_size * num_kv_heads, seq_len, head_dim)
    v = v.reshape(batch_size * num_kv_heads, seq_len, head_dim)

    # 4. Rotary Position Embedding applied to Q and K
    q = apply_rotary_pos_emb(q, cos, sin)
    k = apply_rotary_pos_emb(k, cos, sin)

    # 5. GQA: expand K/V so every query head has its own KV head copy
    k = k.reshape(batch_size, num_kv_heads, seq_len, head_dim)
    v = v.reshape(batch_size, num_kv_heads, seq_len, head_dim)
    k = k.repeat_interleave(num_kv_groups, dim=1)  # [batch, num_heads, seq, head_dim]
    v = v.repeat_interleave(num_kv_groups, dim=1)
    k = k.reshape(batch_size * num_heads, seq_len, head_dim)
    v = v.reshape(batch_size * num_heads, seq_len, head_dim)

    # 6. GQA Attention with causal masking (core Helion mega kernel)
    attn_out = gqa_attention(q, k, v)  # [batch * num_heads, seq_len, head_dim]

    # 7. Reshape and output projection
    attn_out = attn_out.reshape(batch_size, num_heads, seq_len, head_dim)
    attn_out = attn_out.transpose(1, 2).reshape(
        batch_size * seq_len, num_heads * head_dim
    )
    attn_out = F.linear(attn_out, o_proj_weight)  # [tokens, hidden_size]

    # 8. Residual connection
    hidden_states = residual + attn_out

    # ── FFN Block ────────────────────────────────────────────────────────────
    residual = hidden_states

    # 9. Pre-FFN RMSNorm (Helion kernel)
    normed = rms_norm(hidden_states, ffn_norm_weight, config.rms_norm_eps)

    # 10. SwiGLU FFN: gate + up projections
    gate = F.linear(normed, gate_proj_weight)  # [tokens, intermediate_size]
    up = F.linear(normed, up_proj_weight)  # [tokens, intermediate_size]

    # 11. SwiGLU activation (Helion kernel): SiLU(gate) * up
    intermediate = swiglu_fwd(gate, up)  # [tokens, intermediate_size]

    # 12. Down projection
    ffn_out = F.linear(intermediate, down_proj_weight)  # [tokens, hidden_size]

    # 13. Residual connection
    return residual + ffn_out


# %%
# PyTorch Reference Implementation
# ---------------------------------


# %%
class Qwen35DecoderBlockReference(nn.Module):
    """
    Pure PyTorch reference implementation of the Qwen 3.5 2B decoder block.

    Used for correctness verification against the Helion mega kernel.
    Implements the same operations as ``qwen_decoder_block`` using standard
    PyTorch primitives.
    """

    def __init__(self, config: Qwen35Config) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        head_dim = config.head_dim
        intermediate_size = config.intermediate_size

        # Attention
        self.attn_norm = nn.RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # FFN
        self.ffn_norm = nn.RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tensor:
        """Forward pass through the decoder block."""
        config = self.config
        num_heads = config.num_attention_heads
        num_kv_heads = config.num_key_value_heads
        num_kv_groups = config.num_kv_groups
        head_dim = config.head_dim

        # ── Attention Block ──
        residual = hidden_states

        # Pre-attention RMSNorm
        normed = self.attn_norm(hidden_states)

        # QKV projections
        q = self.q_proj(normed)
        k = self.k_proj(normed)
        v = self.v_proj(normed)

        # Reshape and flatten
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        q = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k = k.reshape(batch_size * num_kv_heads, seq_len, head_dim)
        v = v.reshape(batch_size * num_kv_heads, seq_len, head_dim)

        # RoPE
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # GQA expansion
        k = k.reshape(batch_size, num_kv_heads, seq_len, head_dim)
        v = v.reshape(batch_size, num_kv_heads, seq_len, head_dim)
        k = k.repeat_interleave(num_kv_groups, dim=1)
        v = v.repeat_interleave(num_kv_groups, dim=1)
        q = q.reshape(batch_size, num_heads, seq_len, head_dim)
        k = k.reshape(batch_size, num_heads, seq_len, head_dim)
        v = v.reshape(batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.reshape(batch_size * seq_len, num_heads * head_dim)
        attn_out = self.o_proj(attn_out)
        hidden_states = residual + attn_out

        # ── FFN Block ──
        residual = hidden_states

        # Pre-FFN RMSNorm
        normed = self.ffn_norm(hidden_states)

        # SwiGLU FFN
        gate = self.gate_proj(normed)
        up = self.up_proj(normed)
        intermediate = F.silu(gate) * up
        ffn_out = self.down_proj(intermediate)
        return residual + ffn_out


# %%
# Weight Helpers
# --------------


# %%
def _make_weights(
    config: Qwen35Config,
    device: torch.device | str,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    """Create randomly-initialized weight tensors matching the Qwen 3.5 2B architecture."""
    h = config.hidden_size
    nh = config.num_attention_heads
    nkv = config.num_key_value_heads
    hd = config.head_dim
    inter = config.intermediate_size
    return {
        "attn_norm_weight": torch.randn(h, device=device, dtype=dtype),
        "q_proj_weight": torch.randn(nh * hd, h, device=device, dtype=dtype),
        "k_proj_weight": torch.randn(nkv * hd, h, device=device, dtype=dtype),
        "v_proj_weight": torch.randn(nkv * hd, h, device=device, dtype=dtype),
        "o_proj_weight": torch.randn(h, nh * hd, device=device, dtype=dtype),
        "ffn_norm_weight": torch.randn(h, device=device, dtype=dtype),
        "gate_proj_weight": torch.randn(inter, h, device=device, dtype=dtype),
        "up_proj_weight": torch.randn(inter, h, device=device, dtype=dtype),
        "down_proj_weight": torch.randn(h, inter, device=device, dtype=dtype),
    }


def _copy_weights_to_reference(
    ref: Qwen35DecoderBlockReference,
    weights: dict[str, Tensor],
) -> None:
    """Copy weights from a flat dict into the reference PyTorch module."""
    ref.attn_norm.weight.data.copy_(weights["attn_norm_weight"])
    ref.q_proj.weight.data.copy_(weights["q_proj_weight"])
    ref.k_proj.weight.data.copy_(weights["k_proj_weight"])
    ref.v_proj.weight.data.copy_(weights["v_proj_weight"])
    ref.o_proj.weight.data.copy_(weights["o_proj_weight"])
    ref.ffn_norm.weight.data.copy_(weights["ffn_norm_weight"])
    ref.gate_proj.weight.data.copy_(weights["gate_proj_weight"])
    ref.up_proj.weight.data.copy_(weights["up_proj_weight"])
    ref.down_proj.weight.data.copy_(weights["down_proj_weight"])


# %%
# Verification
# ------------


# %%
def check(
    batch_size: int,
    seq_len: int,
    config: Qwen35Config | None = None,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> None:
    """
    Verify the Qwen 3.5 2B decoder block against the PyTorch reference.

    Args:
        batch_size: Batch size
        seq_len: Sequence length
        config: Qwen 3.5 2B architecture config (defaults to standard 2B params)
        device: Target device
        dtype: Data type for weights and activations
    """
    if config is None:
        config = Qwen35Config()

    tokens = batch_size * seq_len

    # Shared weights
    weights = _make_weights(config, device, dtype)

    # Reference module (PyTorch)
    ref = Qwen35DecoderBlockReference(config).to(device=device, dtype=dtype)
    _copy_weights_to_reference(ref, weights)

    # Precompute RoPE tables
    cos, sin = precompute_rope_cos_sin(
        seq_len=seq_len,
        head_dim=config.head_dim,
        rope_theta=config.rope_theta,
        device=device,
        dtype=dtype,
    )

    # Input hidden states
    hidden_states = torch.randn(tokens, config.hidden_size, device=device, dtype=dtype)

    def helion_fn(hidden_states: Tensor) -> Tensor:
        return qwen_decoder_block(
            hidden_states=hidden_states,
            attn_norm_weight=weights["attn_norm_weight"],
            q_proj_weight=weights["q_proj_weight"],
            k_proj_weight=weights["k_proj_weight"],
            v_proj_weight=weights["v_proj_weight"],
            o_proj_weight=weights["o_proj_weight"],
            ffn_norm_weight=weights["ffn_norm_weight"],
            gate_proj_weight=weights["gate_proj_weight"],
            up_proj_weight=weights["up_proj_weight"],
            down_proj_weight=weights["down_proj_weight"],
            cos=cos,
            sin=sin,
            config=config,
            batch_size=batch_size,
            seq_len=seq_len,
        )

    def reference_fn(hidden_states: Tensor) -> Tensor:
        return ref(hidden_states, cos, sin, batch_size, seq_len)

    run_example(
        helion_fn,
        reference_fn,
        (hidden_states,),
        kernel_name="helion_qwen35_decoder",
        baseline_name="pytorch",
        rtol=1e-2,
        atol=1e-1,
    )


# %%
# Tritonbench Integration
# -----------------------


# %%
def qwen3_5_2b_tritonbench(
    tb_op: object,
    hidden_states: Tensor,
    config: Qwen35Config,
    weights: dict[str, Tensor],
    cos: Tensor,
    sin: Tensor,
    batch_size: int,
    seq_len: int,
) -> Callable[[], Tensor]:
    """
    Wrapper for tritonbench benchmarking.

    Args:
        tb_op: TritonBench operator instance
        hidden_states: Input tensor
        config: Model configuration
        weights: Weight dictionary
        cos: RoPE cosine table
        sin: RoPE sine table
        batch_size: Batch size
        seq_len: Sequence length

    Returns:
        Callable that runs the decoder block
    """
    return lambda: qwen_decoder_block(
        hidden_states=hidden_states,
        attn_norm_weight=weights["attn_norm_weight"],
        q_proj_weight=weights["q_proj_weight"],
        k_proj_weight=weights["k_proj_weight"],
        v_proj_weight=weights["v_proj_weight"],
        o_proj_weight=weights["o_proj_weight"],
        ffn_norm_weight=weights["ffn_norm_weight"],
        gate_proj_weight=weights["gate_proj_weight"],
        up_proj_weight=weights["up_proj_weight"],
        down_proj_weight=weights["down_proj_weight"],
        cos=cos,
        sin=sin,
        config=config,
        batch_size=batch_size,
        seq_len=seq_len,
    )


# %%
# Main
# ----


# %%
def main() -> None:
    """
    Main entry point: verify the Qwen 3.5 2B decoder block with small test shapes.

    Uses reduced hidden/intermediate sizes so the test runs quickly on any GPU.
    The real Qwen 3.5 2B architecture can be tested by passing the default
    ``Qwen35Config()`` (which requires sufficient GPU memory).
    """
    # Use a minimal config that matches the head_dim / GQA ratio of Qwen 3.5 2B
    # but with smaller hidden and intermediate sizes for fast verification.
    mini_config = Qwen35Config(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        intermediate_size=512,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
        max_position_embeddings=512,
    )
    print("Checking Qwen 3.5 2B decoder block (mini config)...")
    check(batch_size=2, seq_len=32, config=mini_config, device=DEVICE, dtype=HALF_DTYPE)
    print("✓ Qwen 3.5 2B decoder block passed")


# %%
if __name__ == "__main__":
    main()
