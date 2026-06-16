#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Eagle-3-VL backbone — TTNN orchestrator.

Assembles TtSigLIP2Encoder + TtMlp1Connector + TtQwen3Decoder into a full
vision-language backbone that mirrors Eagle3_VL.forward():

  images [B, 3, 336, 336]  ──┐
  texts  list[str]          ──┤  Eagle processor (CPU, unchanged)
                               ↓
  pixel_values [B, 3, 336, 336],  input_ids [B, T],  attention_mask [B, T]
                               │
  TtSigLIP2Encoder (TTNN)     ↓
  [B, 576, 1152] ─── pixel_shuffle 2×2 ──→ [B, 144, 4608]
                               │
  TtMlp1Connector (TTNN)      ↓
  [B, 144, 2048] ────────────┐
                              │
  TtQwen3Decoder.embed       ↓
  [B, T, 2048]  (image token positions replaced by connector output)
                               │
  TtQwen3Decoder (TTNN)       ↓
  [B, T, 2048]  ← final hidden states

Returns:
  vl_embeds               [B, T, 2048]  float32
  backbone_attention_mask [B, T]        bool
  image_mask              [B, T]        bool  (True = image token)
"""

from __future__ import annotations

import math
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch
import ttnn

from .eagle_siglip2 import TtSigLIP2Encoder
from .eagle_qwen3 import TtQwen3Decoder
from .tt_perf import TtOpProfiler

IMAGE_TOKEN_INDEX  = 151_669   # Eagle-3-VL "<image>" token id
PIXEL_SHUFFLE_MERGE = 2        # 2×2 spatial merge: 576 → 144 visual tokens


# ── Weight-dict helpers ───────────────────────────────────────────────────────

def _filter(d: dict, prefix: str) -> dict:
    """Return sub-dict with prefix stripped from matching keys."""
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


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


# ── Pixel shuffle (spatial 2×2 merge) ────────────────────────────────────────

def _pixel_shuffle(x: np.ndarray, merge: int = PIXEL_SHUFFLE_MERGE) -> np.ndarray:
    """
    Merge m×m spatial tiles in the patch grid.

    Args:
        x: float32 [B, ph*pw, D]  (e.g. [B, 576, 1152])

    Returns:
        float32 [B, (ph/m)*(pw/m), D*m*m]  (e.g. [B, 144, 4608])
    """
    B, N, D = x.shape
    ph = pw = int(math.isqrt(N))
    assert ph * pw == N, f"Expected square patch grid, got N={N}"
    m  = merge
    # [B, ph/m, m, pw/m, m, D]
    x = x.reshape(B, ph // m, m, pw // m, m, D)
    # [B, ph/m, pw/m, m, m, D] → [B, (ph/m)*(pw/m), m*m*D]
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, (ph // m) * (pw // m), m * m * D)
    return x.astype(np.float32)


# ── mlp1 connector ────────────────────────────────────────────────────────────

class TtMlp1Connector:
    """
    Eagle connector: LayerNorm(4608) → Linear(4608→2048) → GELU → Linear(2048→2048).

    weight_dict keys (after stripping 'backbone.model.'):
      mlp1.0.weight  [4608]        — LayerNorm weight
      mlp1.0.bias    [4608]        — LayerNorm bias
      mlp1.1.weight  [2048, 4608]  — first Linear weight
      mlp1.1.bias    [2048]        — first Linear bias
      mlp1.3.weight  [2048, 2048]  — second Linear weight  (index 2 = GELU, no params)
      mlp1.3.bias    [2048]        — second Linear bias
    """

    def __init__(self, weight_dict: dict[str, Any], device: ttnn.Device) -> None:
        self.device = device
        self._prof: TtOpProfiler | None = None  # set during forward()

        def _timed(arr) -> tuple:
            t0 = perf_counter_ns()
            r = _tt(arr, device)
            return r, (perf_counter_ns() - t0) / 1e6

        self.ln_w,  t_ln_w  = _timed(weight_dict["mlp1.0.weight"].reshape(1, -1))
        self.ln_b,  t_ln_b  = _timed(weight_dict["mlp1.0.bias"].reshape(1, -1))
        self.fc1_w, t_fc1_w = _timed(weight_dict["mlp1.1.weight"].T)   # [4608, 2048]
        self.fc1_b, t_fc1_b = _timed(weight_dict["mlp1.1.bias"].reshape(1, -1))
        self.fc2_w, t_fc2_w = _timed(weight_dict["mlp1.3.weight"].T)   # [2048, 2048]
        self.fc2_b, t_fc2_b = _timed(weight_dict["mlp1.3.bias"].reshape(1, -1))
        for name, ms, tensor in [
            ("ln_w",  t_ln_w,  self.ln_w),  ("ln_b",  t_ln_b,  self.ln_b),
            ("fc1_w", t_fc1_w, self.fc1_w), ("fc1_b", t_fc1_b, self.fc1_b),
            ("fc2_w", t_fc2_w, self.fc2_w), ("fc2_b", t_fc2_b, self.fc2_b),
        ]:
            print(f"[upload] mlp1  {name:5s}  {ms:7.1f} ms  {tuple(tensor.shape)}")
        total_ms = t_ln_w + t_ln_b + t_fc1_w + t_fc1_b + t_fc2_w + t_fc2_b
        print(f"[upload] mlp1  TOTAL  {total_ms:7.1f} ms  (6 tensors)")

    def _t(self, label: str, fn, *args, **kwargs):
        """Dispatch a TTNN op, recording timing+shapes via self._prof when active."""
        if self._prof is not None:
            return self._prof.timed(label, fn, *args, **kwargs)
        return fn(*args, **kwargs)

    def forward(self, x_np: np.ndarray) -> np.ndarray:
        """
        Args:
            x_np: float32 [B, 576, 1152]  — SigLIP2 output

        Returns:
            float32 [B, 144, 2048]
        """
        B = x_np.shape[0]
        self._prof = TtOpProfiler(device=self.device)

        # pixel_shuffle + upload are CPU/numpy + one ttnn.from_torch — time as one block
        t0 = perf_counter_ns()
        merged = _pixel_shuffle(x_np)   # [B, 144, 4608]
        n_vis  = merged.shape[1]        # 144
        h_tt   = _tt(merged.reshape(B * n_vis, merged.shape[2]), self.device)
        h_tt   = ttnn.reshape(h_tt, (B, n_vis, merged.shape[2]))
        self._prof.record("mlp1.shuffle_upload", (perf_counter_ns() - t0) / 1e6,
                          tuple(x_np.shape), tuple(h_tt.shape))

        # LayerNorm fused (norm + scale + bias in one op)
        h_tt = self._t("mlp1.ln.layer_norm", ttnn.layer_norm, h_tt,
                        weight=self.ln_w, bias=self.ln_b)

        # fc1 fused with gelu
        h_tt = self._t("mlp1.fc1.linear_gelu", ttnn.linear, h_tt, self.fc1_w,
                        bias=self.fc1_b, activation="gelu")

        # fc2
        h_tt = self._t("mlp1.fc2.linear", ttnn.linear, h_tt, self.fc2_w, bias=self.fc2_b)

        self._prof.dump("mlp1")
        self._prof = None
        return _to_np(h_tt)   # [B, 144, 2048]


# ── TtEagleBackbone ───────────────────────────────────────────────────────────

class TtEagleBackbone:
    """
    Full Eagle-3-VL backbone running on a TT device.

    Usage:
        backbone = TtEagleBackbone.load_weights(store, device, processor)
        vl_embeds, attn_mask, image_mask = backbone.forward(images_np, texts)
    """

    def __init__(
        self,
        siglip: TtSigLIP2Encoder,
        connector: TtMlp1Connector,
        qwen3: TtQwen3Decoder,
        processor: Any,
    ) -> None:
        self.siglip    = siglip
        self.connector = connector
        self.qwen3     = qwen3
        self.processor = processor

    @classmethod
    def load_weights(
        cls,
        store: dict,
        device: ttnn.Device,
        processor: Any,
        siglip_layers: int = 27,
        qwen3_layers:  int = 16,
    ) -> "TtEagleBackbone":
        """
        Build a TtEagleBackbone from the raw GR00T safetensors weight dict.

        Args:
            store:     full GR00T weight dict (keys like 'backbone.model.*')
            device:    open TT device
            processor: Eagle processor (for tokenization / image preprocessing)
        """
        bb = _filter(store, "backbone.model.")

        print("  Loading SigLIP2 encoder ...")
        siglip = TtSigLIP2Encoder(
            weight_dict      = _filter(bb, "vision_model.vision_model."),
            device           = device,
            num_active_layers= siglip_layers,
        )

        print("  Loading mlp1 connector ...")
        connector = TtMlp1Connector(weight_dict=bb, device=device)

        print("  Loading Qwen3 decoder ...")
        qwen3 = TtQwen3Decoder(
            weight_dict      = _filter(bb, "language_model.model."),
            device           = device,
            num_active_layers= qwen3_layers,
        )

        return cls(siglip=siglip, connector=connector, qwen3=qwen3, processor=processor)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images_np: np.ndarray,    # float32 [B, 3, 336, 336]  in [0, 1]
        texts:     list[str],      # instruction per batch item
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full Eagle-3-VL forward pass.

        Returns:
            vl_embeds               float32 [B, T, 2048]
            backbone_attention_mask bool    [B, T]   True = valid token
            image_mask              bool    [B, T]   True = image token position
        """
        from PIL import Image as PILImage

        B   = images_np.shape[0]
        dev = self.processor

        # ── Step 1: Eagle processor (CPU) ────────────────────────────────────
        # Convert [B, C, H, W] float [0,1] → PIL images
        imgs_u8 = (images_np.transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
        pil_imgs = [PILImage.fromarray(imgs_u8[i]) for i in range(B)]

        conversations = [
            [{"role": "user", "content": [
                {"type": "text",  "text": texts[i]},
                {"type": "image", "image": pil_imgs[i]},
            ]}]
            for i in range(B)
        ]
        text_prompts = [
            self.processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
            for conv in conversations
        ]
        image_inputs, _ = self.processor.process_vision_info(conversations)
        vl_inputs = self.processor(
            text=text_prompts, images=image_inputs, return_tensors="pt", padding=True
        )

        input_ids    = vl_inputs["input_ids"].numpy()                       # [B, T] int64
        attn_mask    = vl_inputs["attention_mask"].bool().numpy()           # [B, T] bool

        # pixel_values: [B, 3, 336, 336] or list; coerce to float32 numpy
        pv = vl_inputs["pixel_values"]
        if isinstance(pv, list):
            pv = pv[0]
        pv_np = pv.float().cpu().numpy()                                    # [B, 3, 336, 336]

        # ── Step 2: SigLIP2 vision encoder (TTNN) ────────────────────────────
        siglip_out = self.siglip.forward(pv_np)                             # [B, 576, 1152]

        # ── Step 3: pixel_shuffle + mlp1 connector (TTNN) ────────────────────
        visual_embeds = self.connector.forward(siglip_out)                  # [B, 144, 2048]
        n_vis = visual_embeds.shape[1]   # 144

        # ── Step 4: token embeddings (CPU) ───────────────────────────────────
        hidden = self.qwen3.embed(input_ids)                                # [B, T, 2048]

        # ── Step 5: inject visual embeddings at image token positions ─────────
        image_mask = (input_ids == IMAGE_TOKEN_INDEX)                       # [B, T] bool
        for b in range(B):
            img_pos = np.where(image_mask[b])[0]
            if len(img_pos) == 0:
                continue
            if len(img_pos) != n_vis:
                # Truncate or pad to match connector output — handles edge cases
                img_pos = img_pos[:n_vis]
            hidden[b, img_pos] = visual_embeds[b, :len(img_pos)]

        # ── Step 6: Qwen3 LLM (TTNN) ─────────────────────────────────────────
        vl_embeds = self.qwen3.forward(hidden)                              # [B, T, 2048]

        return vl_embeds, attn_mask, image_mask
