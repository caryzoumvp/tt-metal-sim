#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
SigLIP2 ViT encoder for Eagle-3-VL backbone — TTNN implementation.

Adapted from models/experimental/openvla/tt/tt_optimized_openvla_vision.py.
Key differences vs the openvla SigLIP:
  - 336px input  → patch_count=24  → 576 patches  (openvla: 224px → 16 → 256)
  - Position embeddings bicubic-interpolated from base (16×16=256) to (24×24=576)
  - Separate Q/K/V weight matrices (no merged QKV in GR00T safetensors)
  - Attention matmul stays in torch (shape flexibility); linear projections in TTNN
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
PATCH_SIZE        = 14
HIDDEN_SIZE       = 1152
NUM_HEADS         = 16
HEAD_DIM          = HIDDEN_SIZE // NUM_HEADS   # 72
INTERMEDIATE_SIZE = 4304
NUM_LAYERS        = 27
BASE_GRID         = 16   # 224px / 14px = 16; base position embedding grid


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


def _interpolate_pos_embed(pos_np: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Bicubic-interpolate position embeddings from (BASE_GRID²=256, D) to (target_h*target_w, D).
    Called once per resolution change — result cached by caller if needed.
    """
    n, d = pos_np.shape
    assert n == BASE_GRID * BASE_GRID, f"Expected {BASE_GRID**2} base positions, got {n}"
    pe = (pos_np if isinstance(pos_np, torch.Tensor) else torch.from_numpy(pos_np)).float()  # [256, D]
    pe = pe.reshape(1, BASE_GRID, BASE_GRID, d).permute(0, 3, 1, 2) # [1, D, 16, 16]
    pe = torch.nn.functional.interpolate(
        pe, size=(target_h, target_w), mode="bicubic", align_corners=False
    )                                                                # [1, D, th, tw]
    return pe.permute(0, 2, 3, 1).reshape(target_h * target_w, d).numpy().astype(np.float32)


# ── TtSigLIP2Encoder ──────────────────────────────────────────────────────────

class TtSigLIP2Encoder:
    """
    SigLIP2 ViT encoder with weights preloaded to TT device.

    weight_dict keys (after stripping 'backbone.model.vision_model.vision_model.' prefix):
      embeddings.patch_embedding.{weight,bias}
      embeddings.position_embedding.weight
      encoder.layers.{i}.layer_norm{1,2}.{weight,bias}
      encoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
      encoder.layers.{i}.self_attn.out_proj.{weight,bias}
      encoder.layers.{i}.mlp.{fc1,fc2}.{weight,bias}
      post_layernorm.{weight,bias}
    """

    def __init__(
        self,
        weight_dict: dict[str, np.ndarray],
        device: ttnn.Device,
        num_active_layers: int = NUM_LAYERS,
    ) -> None:
        self.device = device
        self.num_active_layers = num_active_layers
        self._prof: TtOpProfiler | None = None  # set during forward()

        # Patch embedding linear: weight (1152, 588) → T → (588, 1152) for x @ W
        pe_w = weight_dict["embeddings.patch_embedding.weight"]  # [1152, 588]
        pe_b = weight_dict["embeddings.patch_embedding.bias"]    # [1152]

        # Base position embedding (interpolated at runtime)
        self.pos_embed_np = weight_dict["embeddings.position_embedding.weight"]  # [256, 1152]

        t0_global = perf_counter_ns()
        self.patch_proj_w = _tt(pe_w.T, device)                 # [588, 1152]
        self.patch_proj_b = _tt(pe_b.reshape(1, -1), device)    # [1, 1152]
        self.post_ln_w = _tt(weight_dict["post_layernorm.weight"].reshape(1, -1), device)
        self.post_ln_b = _tt(weight_dict["post_layernorm.bias"].reshape(1, -1), device)
        global_ms = (perf_counter_ns() - t0_global) / 1e6
        print(f"[upload] siglip2  global    {global_ms:8.1f} ms  (4 tensors)")

        # Per-encoder-layer weights
        self.layers: list[dict[str, ttnn.Tensor]] = []
        total_ms = global_ms
        for i in range(num_active_layers):
            t0_layer = perf_counter_ns()
            p = f"encoder.layers.{i}"
            lw: dict[str, ttnn.Tensor] = {
                "ln1_w": _tt(weight_dict[f"{p}.layer_norm1.weight"].reshape(1, -1), device),
                "ln1_b": _tt(weight_dict[f"{p}.layer_norm1.bias"].reshape(1, -1), device),
                "ln2_w": _tt(weight_dict[f"{p}.layer_norm2.weight"].reshape(1, -1), device),
                "ln2_b": _tt(weight_dict[f"{p}.layer_norm2.bias"].reshape(1, -1), device),
                # Attention: transposed for x @ W matmul convention
                "q_w": _tt(weight_dict[f"{p}.self_attn.q_proj.weight"].T, device),
                "q_b": _tt(weight_dict[f"{p}.self_attn.q_proj.bias"].reshape(1, -1), device),
                "k_w": _tt(weight_dict[f"{p}.self_attn.k_proj.weight"].T, device),
                "k_b": _tt(weight_dict[f"{p}.self_attn.k_proj.bias"].reshape(1, -1), device),
                "v_w": _tt(weight_dict[f"{p}.self_attn.v_proj.weight"].T, device),
                "v_b": _tt(weight_dict[f"{p}.self_attn.v_proj.bias"].reshape(1, -1), device),
                "o_w": _tt(weight_dict[f"{p}.self_attn.out_proj.weight"].T, device),
                "o_b": _tt(weight_dict[f"{p}.self_attn.out_proj.bias"].reshape(1, -1), device),
                # FFN (GELU)
                "fc1_w": _tt(weight_dict[f"{p}.mlp.fc1.weight"].T, device),
                "fc1_b": _tt(weight_dict[f"{p}.mlp.fc1.bias"].reshape(1, -1), device),
                "fc2_w": _tt(weight_dict[f"{p}.mlp.fc2.weight"].T, device),
                "fc2_b": _tt(weight_dict[f"{p}.mlp.fc2.bias"].reshape(1, -1), device),
            }
            layer_ms = (perf_counter_ns() - t0_layer) / 1e6
            print(f"[upload] siglip2  layer.{i:02d}  {layer_ms:8.1f} ms  (16 tensors)")
            total_ms += layer_ms
            self.layers.append(lw)

        print(f"[upload] siglip2  TOTAL     {total_ms:8.1f} ms  ({num_active_layers} layers + global)")

    # ── Op primitives ─────────────────────────────────────────────────────────

    def _t(self, label: str, fn, *args, **kwargs):
        """Dispatch a TTNN op, recording timing+shapes via self._prof when active."""
        if self._prof is not None:
            return self._prof.timed(label, fn, *args, **kwargs)
        return fn(*args, **kwargs)

    def _linear(self, x: ttnn.Tensor, w: ttnn.Tensor, b: ttnn.Tensor, pfx: str = "") -> ttnn.Tensor:
        return self._t(f"{pfx}.linear", ttnn.linear, x, w, bias=b)

    def _layer_norm(self, x: ttnn.Tensor, w: ttnn.Tensor, b: ttnn.Tensor, pfx: str = "") -> ttnn.Tensor:
        return self._t(f"{pfx}.layer_norm", ttnn.layer_norm, x, weight=w, bias=b)

    def _attention(self, x_tt: ttnn.Tensor, lw: dict, n: int, pfx: str = "") -> ttnn.Tensor:
        """
        Multi-head self-attention for SigLIP2 — fully on-device in TTNN.

        x_tt : [1, N, 1152]
        Shapes through attention:
          QKV proj → [1, N, 1152]
          reshape  → [1, H, N, head_dim]   H=16, head_dim=72
          QK^T     → [1, H, N, N]
          softmax  → [1, H, N, N]
          PV       → [1, H, N, head_dim]
          merge    → [1, N, 1152]
        """
        # QKV projections in TTNN: [1, N, 1152]
        q_tt = self._linear(x_tt, lw["q_w"], lw["q_b"], f"{pfx}.q")
        k_tt = self._linear(x_tt, lw["k_w"], lw["k_b"], f"{pfx}.k")
        v_tt = self._linear(x_tt, lw["v_w"], lw["v_b"], f"{pfx}.v")

        # [1, N, 1152] → [1, H, N, head_dim]
        q_tt = self._t(f"{pfx}.q.reshape", ttnn.reshape, q_tt, (1, n, NUM_HEADS, HEAD_DIM))
        q_tt = self._t(f"{pfx}.q.perm",   ttnn.permute, q_tt, (0, 2, 1, 3))
        k_tt = self._t(f"{pfx}.k.reshape", ttnn.reshape, k_tt, (1, n, NUM_HEADS, HEAD_DIM))
        k_tt = self._t(f"{pfx}.k.perm",   ttnn.permute, k_tt, (0, 2, 1, 3))
        v_tt = self._t(f"{pfx}.v.reshape", ttnn.reshape, v_tt, (1, n, NUM_HEADS, HEAD_DIM))
        v_tt = self._t(f"{pfx}.v.perm",   ttnn.permute, v_tt, (0, 2, 1, 3))

        # Scaled QK^T: [1, H, N, N]
        # Note: HEAD_DIM=72 is not tile-aligned (not a multiple of 32), so
        # ttnn.transformer.scaled_dot_product_attention cannot be used here.
        scale  = 1.0 / math.sqrt(HEAD_DIM)
        k_t    = self._t(f"{pfx}.k.t",       ttnn.permute,  k_tt, (0, 1, 3, 2))
        scores = self._t(f"{pfx}.qk.matmul", ttnn.matmul,   q_tt, k_t)
        scores = self._t(f"{pfx}.qk.scale",  ttnn.multiply, scores, scale)
        probs  = self._t(f"{pfx}.softmax",   ttnn.softmax,  scores, dim=-1)

        # Context: [1, H, N, head_dim] → [1, N, 1152]
        ctx_tt = self._t(f"{pfx}.pv.matmul",   ttnn.matmul, probs, v_tt)
        ctx_tt = self._t(f"{pfx}.ctx.perm",    ttnn.permute, ctx_tt, (0, 2, 1, 3))
        ctx_tt = self._t(f"{pfx}.ctx.reshape", ttnn.reshape, ctx_tt, (1, n, HIDDEN_SIZE))

        # Output projection
        return self._linear(ctx_tt, lw["o_w"], lw["o_b"], f"{pfx}.o")

    def _ffn(self, x_tt: ttnn.Tensor, lw: dict, pfx: str = "") -> ttnn.Tensor:
        """Feed-forward: GELU(fc1(x)) → fc2.  gelu fused into ttnn.linear."""
        h = self._t(f"{pfx}.fc1.linear_gelu", ttnn.linear, x_tt, lw["fc1_w"],
                    bias=lw["fc1_b"], activation="gelu")
        return self._t(f"{pfx}.fc2.linear", ttnn.linear, h, lw["fc2_w"], bias=lw["fc2_b"])

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, pixel_values_np: np.ndarray) -> np.ndarray:
        """
        Args:
            pixel_values_np: float32 [B, 3, 336, 336] preprocessed image.

        Returns:
            float32 [B, 576, 1152] patch features (post-layernorm applied).
        """
        B, C, H, W = pixel_values_np.shape
        ph = H // PATCH_SIZE   # 24
        pw = W // PATCH_SIZE   # 24
        n_patches = ph * pw    # 576

        self._prof = TtOpProfiler()

        # ── Patch embedding ───────────────────────────────────────────────────
        # CPU unfold + upload (not a TTNN op, time as one block)
        t0 = perf_counter_ns()
        img = torch.from_numpy(pixel_values_np).float()   # [B, 3, H, W]
        # Unfold spatial dims → [B, 3, ph, pw, PS, PS] → [B, ph*pw, 3*PS*PS]
        patches = img.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, n_patches, C * PATCH_SIZE * PATCH_SIZE)
        patches_np = patches.numpy().astype(np.float32)   # [B, 576, 588]
        patches_tt = _tt(patches_np.reshape(B * n_patches, -1), self.device)  # [B*576, 588]
        self._prof.record(
            "sg.patch.unfold_upload", (perf_counter_ns() - t0) / 1e6,
            (B, n_patches, C * PATCH_SIZE * PATCH_SIZE), tuple(patches_tt.shape),
        )

        embeds_tt = self._linear(patches_tt, self.patch_proj_w, self.patch_proj_b, "sg.patch")
        embeds_tt = self._t("sg.patch.reshape", ttnn.reshape, embeds_tt, (B, n_patches, HIDDEN_SIZE))

        # ── Position embedding (bicubic-interpolated) ─────────────────────────
        t0 = perf_counter_ns()
        pos_np = _interpolate_pos_embed(self.pos_embed_np, ph, pw)  # [576, 1152]
        pos_tt = _tt(pos_np.reshape(1, n_patches, HIDDEN_SIZE), self.device)
        self._prof.record(
            "sg.pos.interp_upload", (perf_counter_ns() - t0) / 1e6,
            (BASE_GRID * BASE_GRID, HIDDEN_SIZE), (1, n_patches, HIDDEN_SIZE),
        )
        h_tt = self._t("sg.pos.add", ttnn.add, embeds_tt, pos_tt)

        # ── Encoder layers ────────────────────────────────────────────────────
        for i, lw in enumerate(self.layers):
            pfx = f"sg.{i:02d}"
            # Self-attention with pre-norm
            residual = h_tt
            ln1_out  = self._layer_norm(h_tt, lw["ln1_w"], lw["ln1_b"], f"{pfx}.ln1")
            attn_out = self._attention(ln1_out, lw, n_patches, f"{pfx}.attn")
            h_tt     = self._t(f"{pfx}.attn.res", ttnn.add, residual, attn_out)

            # FFN with pre-norm
            residual = h_tt
            ln2_out  = self._layer_norm(h_tt, lw["ln2_w"], lw["ln2_b"], f"{pfx}.ln2")
            ffn_out  = self._ffn(ln2_out, lw, f"{pfx}.ffn")
            h_tt     = self._t(f"{pfx}.ffn.res", ttnn.add, residual, ffn_out)

        # ── Post layer norm ───────────────────────────────────────────────────
        h_tt = self._layer_norm(h_tt, self.post_ln_w, self.post_ln_b, "sg.post_ln")

        self._prof.dump("siglip2")
        self._prof = None
        return _to_np(h_tt)   # [B, 576, 1152]
