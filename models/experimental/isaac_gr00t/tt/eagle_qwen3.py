#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-1.7B (16-layer) decoder for Eagle-3-VL backbone — TTNN implementation.

Adapted from models/experimental/tt_transformers_v2/ds_r1_qwen.py.
Key differences vs the ds_r1_qwen Qwen2:
  - Per-head q_norm / k_norm (RMSNorm, weight [head_dim=128]) applied after QKV
    projection and before RoPE (Qwen3-specific)
  - GQA: 16 Q heads, 8 KV heads — K/V repeated 2× before attention
  - No KV cache: single prefill pass over the full sequence
  - rope_theta=1_000_000
  - rms_norm_eps=1e-6

Linear projections and FFN run in TTNN; attention reshape + q/k_norm + RoPE +
scaled dot-product stay in torch (shapes are awkward in TTNN 4D layout).
"""

from __future__ import annotations

import math
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch
import ttnn

from .tt_perf import TtOpProfiler

# ── Architecture constants ────────────────────────────────────────────────────
HIDDEN_SIZE       = 2048
NUM_HEADS         = 16
NUM_KV_HEADS      = 8
HEAD_DIM          = HIDDEN_SIZE // NUM_HEADS   # 128
KV_GROUPS         = NUM_HEADS // NUM_KV_HEADS  # 2  (repeat factor)
INTERMEDIATE_SIZE = 6144
NUM_LAYERS        = 16
RMS_NORM_EPS      = 1e-6
ROPE_THETA        = 1_000_000.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tt(arr: Any, device: ttnn.Device) -> ttnn.Tensor:
    """Upload a numpy/torch array as bfloat16 TILE_LAYOUT tensor."""
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr.astype(np.float32)).to(torch.bfloat16)
    else:
        t = arr.to(torch.bfloat16)
    if t.dim() < 2:
        t = t.unsqueeze(0)
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device)


def _to_np(x: ttnn.Tensor) -> np.ndarray:
    return ttnn.to_torch(x).float().numpy().astype(np.float32)


def _rope_tables(seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build bfloat16 cos/sin RoPE tables of shape [seq_len, HEAD_DIM]."""
    half   = HEAD_DIM // 2
    inv_f  = 1.0 / (ROPE_THETA ** (torch.arange(0, half).float() / half))
    pos    = torch.arange(seq_len).float()
    freqs  = torch.outer(pos, inv_f)           # [S, half]
    emb    = torch.cat([freqs, freqs], dim=-1)  # [S, HEAD_DIM]
    return emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE.  x: [B, H, S, D].  cos/sin: [S, D]."""
    def rotate_half(t: torch.Tensor) -> torch.Tensor:
        h = t.shape[-1] // 2
        return torch.cat([-t[..., h:], t[..., :h]], dim=-1)
    c = cos.unsqueeze(0).unsqueeze(0)   # [1, 1, S, D]
    s = sin.unsqueeze(0).unsqueeze(0)
    return x * c + rotate_half(x) * s


def _rms_norm_torch(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Per-head RMSNorm (torch).  x: [..., D],  w: [D]."""
    rms = x.pow(2).mean(-1, keepdim=True).add(RMS_NORM_EPS).sqrt()
    return (x / rms) * w


def _to_bf16_torch(v: Any) -> torch.Tensor:
    """Coerce numpy or torch value to bfloat16 torch.Tensor."""
    if isinstance(v, np.ndarray):
        return torch.from_numpy(v.astype(np.float32)).to(torch.bfloat16)
    return v.to(torch.bfloat16)


# ── TtQwen3Decoder ────────────────────────────────────────────────────────────

class TtQwen3Decoder:
    """
    Qwen3-1.7B transformer decoder with weights preloaded to TT device.

    weight_dict keys (after stripping 'backbone.model.language_model.model.' prefix):
      embed_tokens.weight                                   [vocab, 2048]
      layers.{i}.input_layernorm.weight                    [2048]
      layers.{i}.self_attn.q_proj.weight                   [2048, 2048]
      layers.{i}.self_attn.k_proj.weight                   [1024, 2048]
      layers.{i}.self_attn.v_proj.weight                   [1024, 2048]
      layers.{i}.self_attn.o_proj.weight                   [2048, 2048]
      layers.{i}.self_attn.q_norm.weight                   [128]
      layers.{i}.self_attn.k_norm.weight                   [128]
      layers.{i}.mlp.gate_proj.weight                      [6144, 2048]
      layers.{i}.mlp.up_proj.weight                        [6144, 2048]
      layers.{i}.mlp.down_proj.weight                      [2048, 6144]
      layers.{i}.post_attention_layernorm.weight           [2048]
      norm.weight                                          [2048]
    """

    def __init__(
        self,
        weight_dict: dict[str, Any],
        device: ttnn.Device,
        num_active_layers: int = NUM_LAYERS,
    ) -> None:
        self.device = device
        self.num_active_layers = num_active_layers
        self._prof: TtOpProfiler | None = None  # set during forward()

        # Token embedding stays on CPU (lookup is fast; no need for device memory)
        self.embed_tokens_w = _to_bf16_torch(weight_dict["embed_tokens.weight"])  # [vocab, 2048]

        # Final RMSNorm weight on device
        t0_norm = perf_counter_ns()
        self.norm_w = _tt(weight_dict["norm.weight"].reshape(1, -1), device)      # [1, 2048]
        norm_ms = (perf_counter_ns() - t0_norm) / 1e6
        print(f"[upload] qwen3  norm       {norm_ms:8.1f} ms  (1 tensor)")

        # Per-layer weights
        self.layers: list[dict[str, Any]] = []
        total_ms = norm_ms
        for i in range(num_active_layers):
            t0_layer = perf_counter_ns()
            p   = f"layers.{i}"
            lw: dict[str, Any] = {
                # RMSNorm weights on device (broadcast over [B, T, D])
                "ln1_w": _tt(weight_dict[f"{p}.input_layernorm.weight"].reshape(1, -1), device),
                "ln2_w": _tt(weight_dict[f"{p}.post_attention_layernorm.weight"].reshape(1, -1), device),
                # Attention projections — transposed for x @ W matmul convention
                # q: [2048, 2048], k/v: [2048, 1024]
                "q_w":  _tt(weight_dict[f"{p}.self_attn.q_proj.weight"].T, device),
                "k_w":  _tt(weight_dict[f"{p}.self_attn.k_proj.weight"].T, device),
                "v_w":  _tt(weight_dict[f"{p}.self_attn.v_proj.weight"].T, device),
                "o_w":  _tt(weight_dict[f"{p}.self_attn.o_proj.weight"].T, device),
                # q_norm / k_norm stay on CPU (applied per-head after reshape, before RoPE)
                "q_norm_w": _to_bf16_torch(weight_dict[f"{p}.self_attn.q_norm.weight"]),  # [128]
                "k_norm_w": _to_bf16_torch(weight_dict[f"{p}.self_attn.k_norm.weight"]),  # [128]
                # SwiGLU FFN — gate/up: [2048, 6144], down: [6144, 2048]
                "gate_w": _tt(weight_dict[f"{p}.mlp.gate_proj.weight"].T, device),
                "up_w":   _tt(weight_dict[f"{p}.mlp.up_proj.weight"].T,   device),
                "down_w": _tt(weight_dict[f"{p}.mlp.down_proj.weight"].T,  device),
            }
            layer_ms = (perf_counter_ns() - t0_layer) / 1e6
            print(f"[upload] qwen3  layer.{i:02d}  {layer_ms:8.1f} ms  (9 device tensors)")
            total_ms += layer_ms
            self.layers.append(lw)

        print(f"[upload] qwen3  TOTAL      {total_ms:8.1f} ms  ({num_active_layers} layers + norm)")

    # ── Op primitives ─────────────────────────────────────────────────────────

    def _t(self, label: str, fn, *args, **kwargs):
        """Dispatch a TTNN op, recording timing+shapes via self._prof when active."""
        if self._prof is not None:
            return self._prof.timed(label, fn, *args, **kwargs)
        return fn(*args, **kwargs)

    def _rms_norm(self, x: ttnn.Tensor, w: ttnn.Tensor, pfx: str = "") -> ttnn.Tensor:
        """RMSNorm in TTNN: normalize + scale fused into one op."""
        return self._t(f"{pfx}.rms_norm", ttnn.rms_norm, x, weight=w, epsilon=RMS_NORM_EPS)

    def _attention(
        self,
        x_tt: ttnn.Tensor,
        lw: dict,
        n: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        pfx: str = "",
    ) -> ttnn.Tensor:
        """
        Qwen3 multi-head self-attention — QKV/FFN on device, preprocessing in torch.

        x_tt : [1, N, 2048]

        Kept in torch (no clean TTNN equivalent):
          - q_norm / k_norm: per-head RMSNorm (Qwen3-specific)
          - RoPE: rotate_half requires last-dim slicing
          - GQA expand: repeat_interleave needs 5D tensors

        On device (TTNN):
          - QKV projections, QK^T, scale, causal mask, softmax, PV, merge heads,
            output projection
        """
        # ── 1. QKV projections (TTNN) ─────────────────────────────────────────
        q_tt = self._t(f"{pfx}.q.linear", ttnn.linear, x_tt, lw["q_w"])   # [1, N, 2048]
        k_tt = self._t(f"{pfx}.k.linear", ttnn.linear, x_tt, lw["k_w"])   # [1, N, 1024]
        v_tt = self._t(f"{pfx}.v.linear", ttnn.linear, x_tt, lw["v_w"])   # [1, N, 1024]

        # ── 2. Preprocessing in torch (q_norm, k_norm, RoPE, GQA) ────────────
        t0_torch = perf_counter_ns()
        q = ttnn.to_torch(q_tt).float().squeeze(0)   # [N, 2048]
        k = ttnn.to_torch(k_tt).float().squeeze(0)   # [N, 1024]
        v = ttnn.to_torch(v_tt).float().squeeze(0)   # [N, 1024]

        q = q.reshape(1, n, NUM_HEADS,    HEAD_DIM).permute(0, 2, 1, 3)   # [1, 16, N, 128]
        k = k.reshape(1, n, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)   # [1, 8,  N, 128]
        v = v.reshape(1, n, NUM_KV_HEADS, HEAD_DIM).permute(0, 2, 1, 3)   # [1, 8,  N, 128]

        q = _rms_norm_torch(q, lw["q_norm_w"])
        k = _rms_norm_torch(k, lw["k_norm_w"])

        q = _apply_rope(q.to(torch.bfloat16), cos[:n], sin[:n]).float()
        k = _apply_rope(k.to(torch.bfloat16), cos[:n], sin[:n]).float()

        k = k.repeat_interleave(KV_GROUPS, dim=1)   # [1, 16, N, 128]
        v = v.repeat_interleave(KV_GROUPS, dim=1)   # [1, 16, N, 128]

        # ── 3. Upload preprocessed Q, K, V to device ─────────────────────────
        q_tt = _tt(q.numpy().astype(np.float32), self.device)   # [1, 16, N, 128]
        k_tt = _tt(k.numpy().astype(np.float32), self.device)
        v_tt = _tt(v.numpy().astype(np.float32), self.device)
        if self._prof is not None:
            self._prof.record(
                f"{pfx}.torch_prep",
                (perf_counter_ns() - t0_torch) / 1e6,
                (1, NUM_HEADS, n, HEAD_DIM), (1, NUM_HEADS, n, HEAD_DIM),
            )

        # ── 4. Fused causal flash-attention (is_causal=True applies upper-tri mask) ──
        scale   = 1.0 / math.sqrt(HEAD_DIM)
        ctx_tt  = self._t(f"{pfx}.sdpa", ttnn.transformer.scaled_dot_product_attention,
                          q_tt, k_tt, v_tt, is_causal=True, scale=scale)

        # ── 5. Merge heads: [1, H, N, head_dim] → [1, 1, N, 2048] → [1, N, 2048] ──
        ctx_tt = self._t(f"{pfx}.ctx.concat_heads", ttnn.experimental.nlp_concat_heads, ctx_tt)
        ctx_tt = self._t(f"{pfx}.ctx.reshape", ttnn.reshape, ctx_tt, (1, n, HIDDEN_SIZE))

        # ── 6. Output projection in TTNN ──────────────────────────────────────
        return self._t(f"{pfx}.o.linear", ttnn.linear, ctx_tt, lw["o_w"])

    def _ffn(self, x_tt: ttnn.Tensor, lw: dict, pfx: str = "") -> ttnn.Tensor:
        """SwiGLU: silu(gate_proj(x)) ⊙ up_proj(x) → down_proj."""
        gate = self._t(f"{pfx}.gate.linear", ttnn.linear,   x_tt, lw["gate_w"])
        gate = self._t(f"{pfx}.gate.silu",   ttnn.silu,     gate)
        up   = self._t(f"{pfx}.up.linear",   ttnn.linear,   x_tt, lw["up_w"])
        fuse = self._t(f"{pfx}.gate_up.mul", ttnn.multiply, gate, up)
        return self._t(f"{pfx}.down.linear", ttnn.linear,   fuse, lw["down_w"])

    # ── Public API ────────────────────────────────────────────────────────────

    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Token embedding lookup (CPU).

        Args:
            input_ids: int64 [B, T]

        Returns:
            float32 [B, T, 2048]
        """
        ids = torch.from_numpy(input_ids.astype(np.int64))
        emb = self.embed_tokens_w[ids]   # [B, T, 2048] bfloat16
        return emb.float().numpy().astype(np.float32)

    def forward(self, hidden_states_np: np.ndarray) -> np.ndarray:
        """
        Run 16 Qwen3 decoder layers + final RMSNorm.

        Args:
            hidden_states_np: float32 [B, T, 2048]
                Image embeddings must already be injected at image-token positions
                before calling this method.

        Returns:
            float32 [B, T, 2048]
        """
        B, T, _ = hidden_states_np.shape

        self._prof = TtOpProfiler(device=self.device)

        # RoPE tables for the full sequence length (built once per call)
        t0 = perf_counter_ns()
        cos, sin = _rope_tables(T)   # [T, HEAD_DIM], bfloat16

        # Upload to TTNN: [B, T, 2048]
        h_tt = _tt(hidden_states_np.reshape(B * T, HIDDEN_SIZE), self.device)
        h_tt = ttnn.reshape(h_tt, (B, T, HIDDEN_SIZE))
        self._prof.record(
            "qw.rope_tables+upload", (perf_counter_ns() - t0) / 1e6,
            tuple(hidden_states_np.shape), tuple(h_tt.shape),
        )

        # Transformer layers
        for i, lw in enumerate(self.layers):
            pfx = f"qw.{i:02d}"
            # Self-attention block
            residual = h_tt
            ln1_out  = self._rms_norm(h_tt, lw["ln1_w"], f"{pfx}.ln1")
            attn_out = self._attention(ln1_out, lw, T, cos, sin, f"{pfx}.attn")
            h_tt     = self._t(f"{pfx}.attn.res", ttnn.add, residual, attn_out)

            # FFN block
            residual = h_tt
            ln2_out  = self._rms_norm(h_tt, lw["ln2_w"], f"{pfx}.ln2")
            ffn_out  = self._ffn(ln2_out, lw, f"{pfx}.ffn")
            h_tt     = self._t(f"{pfx}.ffn.res", ttnn.add, residual, ffn_out)

        # Final RMSNorm
        h_tt = self._rms_norm(h_tt, self.norm_w, "qw.final_norm")

        self._prof.dump("qwen3")
        self._prof = None
        return _to_np(h_tt)   # [B, T, 2048]
