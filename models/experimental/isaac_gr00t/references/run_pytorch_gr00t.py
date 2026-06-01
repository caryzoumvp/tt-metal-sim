#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GR00T N1.6 — PyTorch reference implementation.

Mirrors run_pytorch_openvla.py: loads weights from safetensors directly,
implements every component inline with explicit ops (layernorm, softmax, matmul),
no Isaac-GR00T package import required.  The per-op decomposition makes each
step a drop-in replacement target for a TTNN kernel.

Architecture (config.json / Polaris profiling on Wormhole n150):

  Image (336×336) → Eagle-3-VL vision encoder (SigLIP2) → pixel_shuffle → [B, 64, 1152*4]
                 → mlp1 projector Linear(4608→2048)                      → [B, 64, 2048]
  Text (N tok)   → Qwen3-1.7B embed + 16 LLM layers                     → [B, N,  2048]
  cat → [B, 64+N, 2048]  ← backbone output (backbone_embedding_dim=2048)

  vlln   LayerNorm(2048)
  state  [B, 1,  128] → state_enc  CategorySpecificMLP(128→1024→1536) → [B, 1, 1536]
  action [B, T,  128] → action_enc MultiEmbodimentActionEncoder  (128→1536) → [B, T, 1536]
  sa_embs = cat([state, action], dim=1)  →  [B, T+1, 1536]

  AlternateVLDiT (32 layers, 32H×48d = 1536):
    odd  layers: self-attention    Q=K=V=[B, T+1, 1536]
    even layers: cross-attention   Q=[B, T+1, 1536], KV=[B, 64+N, 2048]
    × 4 denoising steps (flow matching, Euler integration)

  proj_out_2  Linear(1536→1024)
  action_dec  CategorySpecificMLP(1024→1024→128)
  drop state token  →  [B, T, 128]

Weight prefix map (GR00T-N1.6-3B safetensors):
  backbone.model.*           Eagle-3-VL backbone
  action_head.model.*        AlternateVLDiT
  action_head.state_encoder.* CategorySpecificMLP (state)
  action_head.action_encoder.* MultiEmbodimentActionEncoder
  action_head.action_decoder.* CategorySpecificMLP (action)
  action_head.vlln.*         LayerNorm
  action_head.position_embedding.* nn.Embedding

Usage:
    # Full end-to-end run (image + text → action):
    python run_pytorch_gr00t.py --model-path GR00T-N1.6-3B

    # Benchmark:
    python run_pytorch_gr00t.py --model-path GR00T-N1.6-3B --benchmark --iterations 20

    # Save intermediate tensors for TTNN parity testing:
    python run_pytorch_gr00t.py --model-path GR00T-N1.6-3B --output /tmp/gr00t_ref.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Inline model components — no Isaac-GR00T package import
# All ops are spelled out so TTNN replacements are drop-in.
# ---------------------------------------------------------------------------

# ── Sinusoidal positional encoding (used by MultiEmbodimentActionEncoder) ──

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # timesteps: [B, T]
        B, T = timesteps.shape
        device = timesteps.device
        half_dim = self.embedding_dim // 2
        exponent = -torch.arange(half_dim, dtype=torch.float32, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        freqs = timesteps.float().unsqueeze(-1) * exponent.exp()  # [B, T, half_dim]
        return torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1)  # [B, T, embedding_dim]


# ── Category-specific linear (per-embodiment weight selection) ──

class CategorySpecificLinear(nn.Module):
    """BMM-based linear: selects per-embodiment weight matrices at runtime."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]   cat_ids: [B]
        # → matmul per batch item with its embodiment weight, then add bias
        W = self.W[cat_ids]           # [B, input_dim, hidden_dim]
        b = self.b[cat_ids]           # [B, hidden_dim]
        return torch.bmm(x, W) + b.unsqueeze(1)   # [B, T, hidden_dim]


class CategorySpecificMLP(nn.Module):
    """Two-layer ReLU MLP with per-embodiment weight matrices."""

    def __init__(self, num_categories: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, cat_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, T, input_dim]
        return self.layer2(F.relu(self.layer1(x, cat_ids)), cat_ids)


# ── Action encoder (action + sinusoidal timestep → 1536-dim features) ──

class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim: int, hidden_size: int, num_embodiments: int):
        super().__init__()
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(
        self,
        actions: torch.Tensor,     # [B, T, action_dim]
        timesteps: torch.Tensor,   # [B]
        cat_ids: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        B, T, _ = actions.shape
        tau = timesteps.unsqueeze(1).expand(-1, T)   # [B, T]
        a_emb  = self.W1(actions, cat_ids)            # [B, T, hidden_size]
        tau_emb = self.pos_encoding(tau).to(a_emb.dtype)  # [B, T, hidden_size]
        x = torch.cat([a_emb, tau_emb], dim=-1)      # [B, T, 2*hidden_size]
        x = self.W2(x, cat_ids)
        x = x * torch.sigmoid(x)                     # swish
        return self.W3(x, cat_ids)                   # [B, T, hidden_size]


# ── DiT timestep encoder ──

from diffusers.models.embeddings import TimestepEmbedding, Timesteps


class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        dtype = next(self.parameters()).dtype
        proj = self.time_proj(timesteps).to(dtype)
        return self.timestep_embedder(proj)


# ── AdaLayerNorm (timestep-conditioned layernorm used in DiT blocks) ──

class AdaLayerNorm(nn.Module):
    def __init__(self, embedding_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 2 * embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, norm_eps, elementwise_affine=False)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]   temb: [B, D]
        cond = self.linear(self.silu(temb))            # [B, 2D]
        scale, shift = cond.chunk(2, dim=1)            # each [B, D]
        x = self.norm(x) * (1 + scale[:, None]) + shift[:, None]
        return x


# ── Inline attention — key names match checkpoint: attn1.to_q/k/v, to_out.0 ──

class _Attn(nn.Module):
    def __init__(self, query_dim: int, kv_dim: int, inner_dim: int,
                 num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = inner_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.to_q   = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k   = nn.Linear(kv_dim,    inner_dim, bias=True)
        self.to_v   = nn.Linear(kv_dim,    inner_dim, bias=True)
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=True),
            nn.Dropout(dropout),
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kv = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        B, T, _ = hidden_states.shape
        S = kv.shape[1]
        H, d = self.num_heads, self.head_dim

        q = self.to_q(hidden_states).reshape(B, T, H, d).permute(0, 2, 1, 3)
        k = self.to_k(kv).reshape(B, S, H, d).permute(0, 2, 1, 3)
        v = self.to_v(kv).reshape(B, S, H, d).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale   # [B, H, T, S]
        if attention_mask is not None:
            # attention_mask: [B, S] bool, True = keep
            bias = torch.where(
                attention_mask[:, None, None, :],
                torch.zeros(1, device=scores.device, dtype=scores.dtype),
                torch.full((1,), -1e9, device=scores.device, dtype=scores.dtype),
            )
            scores = scores + bias
        probs = F.softmax(scores, dim=-1)
        ctx = torch.matmul(probs, v).permute(0, 2, 1, 3).reshape(B, T, H * d)
        return self.to_out[0](ctx)   # output linear (dropout skipped in eval)


# ── Inline FFN — key names match checkpoint: ff.net.0.proj, ff.net.2 ──

class _GELUProj(nn.Module):
    """Wrapper so ff.net.0.proj.weight maps correctly."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(x))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, final_dropout: bool = False):
        super().__init__()
        ff_dim = dim * 4
        self.net = nn.ModuleList([
            _GELUProj(dim, ff_dim),    # net.0  (has .proj sub-weight)
            nn.Dropout(dropout),        # net.1
            nn.Linear(ff_dim, dim, bias=True),  # net.2
        ])
        self._final_dropout = nn.Dropout(dropout) if final_dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        if self._final_dropout is not None:
            x = self._final_dropout(x)
        return x


# ── Single DiT transformer block ──

class DiTBlock(nn.Module):
    """
    BasicTransformerBlock with a single attention (self or cross) + FFN.
    cross_attention_dim=None  →  self-attention (K/V come from hidden_states)
    cross_attention_dim=D     →  cross-attention (K/V from encoder_hidden_states)

    Attribute names match the GR00T-N1.6-3B safetensors key layout:
      norm1.linear.*  norm3 (no affine)  attn1.to_{q,k,v,out.0}.*  ff.net.{0.proj,2}.*
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        cross_attention_dim: Optional[int] = None,
        dropout: float = 0.0,
        final_dropout: bool = False,
    ):
        super().__init__()
        inner_dim = num_heads * head_dim
        kv_dim    = cross_attention_dim if cross_attention_dim is not None else dim

        self.norm1 = AdaLayerNorm(dim)
        self.attn1 = _Attn(dim, kv_dim, inner_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)
        self.ff    = _FeedForward(dim, dropout=dropout, final_dropout=final_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed = self.norm1(hidden_states, temb)
        attn_out = self.attn1(
            normed,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
        )
        hidden_states = attn_out + hidden_states

        normed = self.norm3(hidden_states)
        hidden_states = self.ff(normed) + hidden_states
        return hidden_states


# ── AlternateVLDiT: 32 layers alternating self/cross attention ──

class AlternateVLDiT(nn.Module):
    """
    GR00T N1.6 diffusion transformer.

    Layer layout (attend_text_every_n_blocks=2):
      idx % 2 == 1  →  self-attention
      idx % 2 == 0  and  idx % (2*2) == 0  →  cross-attn on text tokens
      idx % 2 == 0  and  idx % (2*2) != 0  →  cross-attn on image tokens
    """

    def __init__(
        self,
        num_layers: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        final_dropout: bool = False,
        attend_text_every_n_blocks: int = 2,
    ):
        super().__init__()
        self.attend_text_every_n_blocks = attend_text_every_n_blocks
        inner_dim = num_attention_heads * attention_head_dim

        self.timestep_encoder = TimestepEncoder(inner_dim)

        self.transformer_blocks = nn.ModuleList()
        for idx in range(num_layers):
            is_self_attn = (idx % 2 == 1)
            self.transformer_blocks.append(DiTBlock(
                dim=inner_dim,
                num_heads=num_attention_heads,
                head_dim=attention_head_dim,
                cross_attention_dim=None if is_self_attn else cross_attention_dim,
                dropout=dropout,
                final_dropout=final_dropout,
            ))

        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(inner_dim, 2 * inner_dim)
        self.proj_out_2 = nn.Linear(inner_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, T, inner_dim]
        encoder_hidden_states: torch.Tensor,   # [B, S, cross_dim]
        timestep: torch.Tensor,                # [B]  long
        image_mask: torch.Tensor,              # [B, S]  bool, True=image token
        backbone_attention_mask: torch.Tensor, # [B, S]  bool
    ) -> torch.Tensor:
        temb = self.timestep_encoder(timestep)  # [B, inner_dim]

        image_attn_mask   = image_mask & backbone_attention_mask   # [B, S]
        text_attn_mask    = (~image_mask) & backbone_attention_mask

        for idx, block in enumerate(self.transformer_blocks):
            if idx % 2 == 1:
                # Self-attention
                hidden_states = block(hidden_states, temb)
            else:
                # Cross-attention: alternate image / text context
                if idx % (2 * self.attend_text_every_n_blocks) == 0:
                    mask = text_attn_mask
                else:
                    mask = image_attn_mask
                hidden_states = block(
                    hidden_states, temb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=mask,
                )

        # Output modulation (same as DiT)
        shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=1)  # each [B, inner_dim]
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        return self.proj_out_2(hidden_states)   # [B, T, output_dim]


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards from model_dir into one dict."""
    try:
        from safetensors import safe_open
    except ImportError:
        raise RuntimeError("pip install safetensors")

    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    elif single_path.exists():
        shard_files = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")

    weights: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    print(f"Loaded {len(weights)} tensors from {model_dir}")
    return weights


def _filter(weights: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    n = len(prefix)
    return {k[n:]: v for k, v in weights.items() if k.startswith(prefix)}


# ---------------------------------------------------------------------------
# Eagle-3-VL backbone loader (no gr00t package — AutoModel from local nvidia dir)
# ---------------------------------------------------------------------------

def load_eagle_backbone(model_dir: Path, weights: dict[str, torch.Tensor]):
    """
    Load the Eagle-3-VL backbone (Eagle-Block2A-2B-v2) and its processor.

    Uses trust_remote_code to handle the relative-import package structure.
    AutoConfig.from_pretrained imports configuration_eagle3_vl which in turn
    imports modeling_siglip2 — so that module is in sys.modules before the model
    is ever instantiated.  We then patch Siglip2PreTrainedModel._init_weights
    in-place (wrapping with try-except) to skip the two isinstance() branches that
    reference undefined class names (Siglip2Model, Siglip2ForImageClassification).
    AutoModel.from_config then instantiates the model using the patched class.

    The Qwen3-1.7B LLM is truncated to 16 layers (GR00T config: select_layer=16).

    Returns (backbone model, Eagle processor).
    """
    from transformers import AutoConfig, AutoModel, AutoProcessor

    eagle_path = str(
        Path("/workspaces/tensix/Isaac-GR00T/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2")
    )
    print(f"  Eagle model path: {eagle_path}")

    # Step 1: load config — this imports modeling_siglip2 via configuration_eagle3_vl.
    config = AutoConfig.from_pretrained(eagle_path, trust_remote_code=True)

    # Step 2: patch Siglip2PreTrainedModel._init_weights in the Eagle-local module.
    # configuration_eagle3_vl imports modeling_siglip2 via the transformers_modules
    # cache package (e.g. transformers_modules.Eagle_hyphen_Block2A_hyphen_2B_hyphen_v2.
    # modeling_siglip2).  That file references Siglip2Model and Siglip2ForImageClassification
    # which are never defined there; wrapping _init_weights with try-except skips them.
    # We target the Eagle-specific module (not the built-in transformers.models.siglip2).
    for mod_name, mod in list(sys.modules.items()):
        if ("Eagle" in mod_name and "siglip2" in mod_name
                and hasattr(mod, "Siglip2PreTrainedModel")):
            _orig = mod.Siglip2PreTrainedModel._init_weights
            def _safe(self, module, _o=_orig):
                try:
                    _o(self, module)
                except NameError:
                    pass
            mod.Siglip2PreTrainedModel._init_weights = _safe
            print(f"  [patch] Siglip2PreTrainedModel._init_weights ({mod_name})")
            break

    # Step 3: instantiate the model using the patched class.
    config._attn_implementation = "eager"
    backbone = AutoModel.from_config(config, trust_remote_code=True)

    # Load GR00T backbone weights (prefix "backbone.model.")
    backbone_weights = _filter(weights, "backbone.model.")
    missing, unexpected = backbone.load_state_dict(backbone_weights, strict=False)
    if unexpected:
        print(f"  [backbone] unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  [backbone] missing keys:    {len(missing)}")

    # Truncate Qwen3-1.7B LLM to select_layer=16 layers (GR00T config: select_layer=16)
    llm_layers = backbone.language_model.model.layers
    while len(llm_layers) > 16:
        llm_layers.pop(-1)
    print(f"  Qwen3 LLM truncated to {len(llm_layers)} layers")

    processor = AutoProcessor.from_pretrained(eagle_path, trust_remote_code=True)
    print("  Eagle processor loaded")

    return backbone, processor


# ---------------------------------------------------------------------------
# Eagle backbone forward pass
# ---------------------------------------------------------------------------

def run_backbone(
    backbone: nn.Module,
    processor: Any,
    images_np: np.ndarray,   # [B, C, H, W]  float32 in [0, 1]
    texts: list[str],        # one instruction per batch item
    device: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Eagle-3-VL forward: image + text → backbone_features [B, T, 2048].

    Returns:
        features            [B, T, 2048]  — last LLM hidden state
        backbone_attn_mask  [B, T]  bool  — True = valid token
        image_mask          [B, T]  bool  — True = image token
    """
    from PIL import Image as PILImage

    B = images_np.shape[0]
    # [B, C, H, W] float [0,1] → [B, H, W, C] uint8 for PIL
    imgs_u8 = (images_np.transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    pil_images = [PILImage.fromarray(imgs_u8[i]) for i in range(B)]

    # Build conversations in Eagle chat-template format
    conversations = [
        [{"role": "user", "content": [
            {"type": "text",  "text": texts[i]},
            {"type": "image", "image": pil_images[i]},
        ]}]
        for i in range(B)
    ]
    text_prompts = [
        processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
        for conv in conversations
    ]

    # Eagle-specific vision-info processing
    image_inputs, _ = processor.process_vision_info(conversations)

    # Tokenize + prepare pixel_values
    vl_inputs = processor(
        text=text_prompts, images=image_inputs, return_tensors="pt", padding=True
    )

    # Move all inputs to device; pixel_values may be a list of tensors
    def _to_dev(v: Any) -> Any:
        if isinstance(v, torch.Tensor):
            return v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
        if isinstance(v, list):
            return [_to_dev(x) for x in v]
        return v

    vl_inputs = {k: _to_dev(v) for k, v in vl_inputs.items()}

    # Eagle3_VL.forward() only accepts input_ids, attention_mask, pixel_values
    # (plus optional flags).  The processor may return extra keys (image_sizes, etc.)
    # that the model doesn't accept; filter to the three required keys.
    fwd_inputs = {k: vl_inputs[k] for k in ("input_ids", "attention_mask", "pixel_values")
                  if k in vl_inputs}

    backbone.eval()
    with torch.no_grad():
        out = backbone(**fwd_inputs, output_hidden_states=True)

    features    = out["hidden_states"][-1]                                           # [B, T, 2048]
    image_mask  = (vl_inputs["input_ids"] == backbone.config.image_token_index)     # [B, T]
    attn_mask   = vl_inputs["attention_mask"].bool()                                 # [B, T]

    return features, attn_mask, image_mask


# ---------------------------------------------------------------------------
# GR00T Action Head (assembled from inline components)
# ---------------------------------------------------------------------------

class GR00TActionHead(nn.Module):
    """
    Standalone action head — loads weights from safetensors, no Isaac-GR00T import.

    Parameters match GR00T-N1.6-3B config.json:
      max_state_dim=128, max_action_dim=128, action_horizon=50
      hidden_size=1024, input_embedding_dim=1536, backbone_embedding_dim=2048
      num_inference_timesteps=4, max_num_embodiments=32
    """

    def __init__(
        self,
        max_state_dim: int = 128,
        max_action_dim: int = 128,
        hidden_size: int = 1024,
        input_embedding_dim: int = 1536,
        backbone_embedding_dim: int = 2048,
        num_embodiments: int = 32,
        num_inference_timesteps: int = 4,
        action_horizon: int = 16,
    ):
        super().__init__()
        self.action_dim    = max_action_dim
        self.action_horizon = action_horizon
        self.num_inference_timesteps = num_inference_timesteps

        self.vlln = nn.LayerNorm(backbone_embedding_dim)
        self.state_encoder = CategorySpecificMLP(
            num_embodiments, max_state_dim, hidden_size, input_embedding_dim
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            max_action_dim, input_embedding_dim, num_embodiments
        )
        self.model = AlternateVLDiT(
            num_layers=32,
            num_attention_heads=32,
            attention_head_dim=48,
            cross_attention_dim=backbone_embedding_dim,
            output_dim=hidden_size,          # 1024
            dropout=0.2,
            final_dropout=True,
            attend_text_every_n_blocks=2,
        )
        self.action_decoder = CategorySpecificMLP(
            num_embodiments, hidden_size, hidden_size, max_action_dim
        )
        self.position_embedding = nn.Embedding(1024, input_embedding_dim)

    def load_weights(self, weights: dict[str, torch.Tensor]) -> None:
        head_w = _filter(weights, "action_head.")

        def _load(module: nn.Module, prefix: str, label: str) -> None:
            sub = _filter(head_w, prefix)
            m, u = module.load_state_dict(sub, strict=False)
            if u: print(f"  [{label}] unexpected: {len(u)}")
            if m: print(f"  [{label}] missing:    {len(m)}")

        _load(self.vlln,            "vlln.",            "vlln")
        _load(self.state_encoder,   "state_encoder.",   "state_encoder")
        _load(self.action_encoder,  "action_encoder.",  "action_encoder")
        _load(self.model,           "model.",           "dit")
        _load(self.action_decoder,  "action_decoder.",  "action_decoder")
        _load(self.position_embedding, "position_embedding.", "pos_embed")

    @torch.no_grad()
    def get_action(
        self,
        vl_embeds: torch.Tensor,              # [B, S, 2048]  backbone output
        backbone_attention_mask: torch.Tensor, # [B, S]  bool
        image_mask: torch.Tensor,             # [B, S]  bool  True=image token
        state: torch.Tensor,                  # [B, 1, 128]
        embodiment_id: torch.Tensor,          # [B]  long
    ) -> torch.Tensor:
        """
        Flow-matching denoising loop → predicted action chunk [B, T, 128].
        Euler integration, num_inference_timesteps=4.
        """
        B = vl_embeds.shape[0]
        device = vl_embeds.device
        dtype  = vl_embeds.dtype

        # 1. VLM LayerNorm
        vl_embeds = self.vlln(vl_embeds)          # [B, S, 2048]

        # 2. Encode state
        state_features = self.state_encoder(state, embodiment_id)  # [B, 1, 1536]

        # 3. Start from noise
        actions = torch.randn(B, self.action_horizon, self.action_dim,
                              dtype=dtype, device=device)
        dt = 1.0 / self.num_inference_timesteps

        # 4. Denoising loop
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_disc = int(t_cont * 1000)
            timesteps_t = torch.full((B,), t_disc, dtype=torch.long, device=device)

            # Encode noisy actions
            action_features = self.action_encoder(actions, timesteps_t, embodiment_id)  # [B, T, 1536]

            # Add positional embedding
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

            # Concat [state | action] tokens
            sa_embs = torch.cat([state_features, action_features], dim=1)  # [B, T+1, 1536]

            # DiT forward
            model_out = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=timesteps_t,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )  # [B, T+1, 1024]

            # Decode velocity
            pred = self.action_decoder(model_out, embodiment_id)  # [B, T+1, 128]
            pred_velocity = pred[:, -self.action_horizon:]         # [B, T, 128]

            # Euler step
            actions = actions + dt * pred_velocity

        return actions   # [B, T, 128]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GR00T N1.6 PyTorch reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-path",
        default="models/experimental/isaac_gr00t/GR00T-N1.6-3B",
        help="Path to GR00T-N1.6-3B checkpoint directory",
    )
    p.add_argument("--device",     default="cpu")
    p.add_argument("--batch",      type=int, default=1)
    p.add_argument("--action-horizon", type=int, default=16,
                   help="Action chunk length (GR1=16)")
    p.add_argument("--image",      default="",
                   help="Path to an RGB image file (PNG/JPG, any size; resized to 336×336 internally). "
                        "If omitted, a black 336×336 image is used.")
    p.add_argument("--instruction", default="",
                   help="Language instruction passed to the Eagle backbone")
    p.add_argument("--benchmark",  action="store_true")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--output",     default="",
                   help="Optional path to save reference tensors (.pt)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    model_dir  = Path(args.model_path).resolve()
    device     = args.device
    iterations = args.iterations if args.iterations is not None else (20 if args.benchmark else 3)

    print("=" * 64)
    print("GR00T N1.6 — PyTorch reference")
    print("=" * 64)
    print(f"Model:   {model_dir}")
    print(f"Device:  {device}")
    print(f"Iters:   {iterations}  ({'benchmark' if args.benchmark else 'run'})")
    print()

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load weights ─────────────────────────────────────────────────────────
    print("Loading safetensors …")
    t0 = time.perf_counter()
    weights = _load_safetensors(model_dir)
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    print()

    # ── Build action head ─────────────────────────────────────────────────────
    print("Building GR00TActionHead …")
    action_head = GR00TActionHead(action_horizon=args.action_horizon)
    action_head.load_weights(weights)
    action_head.to(device=device, dtype=torch.bfloat16)
    action_head.eval()
    n_params = sum(p.numel() for p in action_head.parameters())
    print(f"  action-head params: {n_params/1e6:.1f} M")
    print()

    # ── Build Eagle backbone ──────────────────────────────────────────────────
    print("Loading Eagle-3-VL backbone …")
    t0 = time.perf_counter()
    backbone, processor = load_eagle_backbone(model_dir, weights)
    backbone.to(device=device, dtype=torch.bfloat16)
    backbone.eval()
    n_bb = sum(p.numel() for p in backbone.parameters())
    print(f"  backbone params: {n_bb/1e6:.1f} M")
    print(f"  done in {time.perf_counter()-t0:.1f}s")
    print()

    # ── Prepare raw inputs ───────────────────────────────────────────────────
    # GR00T takes: image (B, C, H, W), text instruction, robot state.
    # Eagle-3-VL processes image+text → vl_embeds; action head uses vl_embeds+state.
    dtype = torch.bfloat16
    B = args.batch

    if args.image:
        from PIL import Image as _PILImage
        _img = _PILImage.open(args.image).convert("RGB").resize((336, 336))
        _arr = np.asarray(_img, dtype=np.float32) / 255.0   # [336, 336, 3]
        image = np.tile(_arr.transpose(2, 0, 1)[None], (B, 1, 1, 1))  # [B, 3, 336, 336]
    else:
        image = np.zeros((B, 3, 336, 336), dtype=np.float32)
    state          = torch.zeros(B, 1, 128, dtype=dtype, device=device)
    embodiment_id  = torch.zeros(B, dtype=torch.long, device=device)
    instruction    = args.instruction

    print("Raw model inputs:")
    _img_src = args.image if args.image else "zeros (black)"
    print(f"  image          {image.shape}  ({_img_src})")
    print(f"  instruction    {repr(instruction)}")
    print(f"  state          {tuple(state.shape)}  (robot joint/pose)")
    print(f"  embodiment_id  {tuple(embodiment_id.shape)}")
    print()

    # ── Eagle backbone: image + text → vl_embeds ─────────────────────────────
    print("Eagle-3-VL backbone forward …")
    t0 = time.perf_counter()
    vl_embeds, backbone_attention_mask, image_mask = run_backbone(
        backbone, processor, image, [instruction] * B, device, dtype
    )
    print(f"  done in {time.perf_counter()-t0:.1f}s")

    n_img_tok  = int(image_mask[0].sum().item())
    n_text_tok = int((~image_mask[0] & backbone_attention_mask[0]).sum().item())
    print(f"  vl_embeds              {tuple(vl_embeds.shape)}")
    print(f"  backbone_attention_mask {tuple(backbone_attention_mask.shape)}")
    print(f"  image_mask             {tuple(image_mask.shape)}  ({n_img_tok} image / {n_text_tok} text tokens)")
    print()

    print("Action-head inputs (backbone output + state):")
    print(f"  vl_embeds              {tuple(vl_embeds.shape)}")
    print(f"  backbone_attention_mask {tuple(backbone_attention_mask.shape)}")
    print(f"  image_mask             {tuple(image_mask.shape)}")
    print(f"  state                  {tuple(state.shape)}")
    print(f"  embodiment_id          {tuple(embodiment_id.shape)}")
    print()

    # ── Inference loop ────────────────────────────────────────────────────────
    print(f"Running {iterations} action-head inference iteration(s) …")
    latencies: list[float] = []
    last_action: torch.Tensor | None = None

    for i in range(iterations):
        t0 = time.perf_counter()
        action = action_head.get_action(
            vl_embeds=vl_embeds,
            backbone_attention_mask=backbone_attention_mask,
            image_mask=image_mask,
            state=state,
            embodiment_id=embodiment_id,
        )
        latencies.append(time.perf_counter() - t0)
        last_action = action
        if not args.benchmark or i == 0:
            print(f"  iter {i:3d}: {latencies[-1]*1000:.1f} ms   output {tuple(action.shape)}")

    print()

    # ── Timing ───────────────────────────────────────────────────────────────
    if len(latencies) > 1:
        warm = latencies[1:]
        mean_ms = 1000 * float(np.mean(warm))
        std_ms  = 1000 * float(np.std(warm))
        fps     = 1.0 / float(np.mean(warm))
        print(f"Timing ({len(warm)} warm iterations):")
        print(f"  mean: {mean_ms:.1f} ms   std: {std_ms:.1f} ms   fps: {fps:.2f}")
    else:
        ms = latencies[0] * 1000
        print(f"Latency: {ms:.1f} ms ({1000/ms:.2f} fps)")

    # ── Print outputs ─────────────────────────────────────────────────────────
    assert last_action is not None
    print()
    print("=== GR00T action-head output ===")
    print(f"  action shape: {tuple(last_action.shape)}")
    print(f"  action[0, 0, :8] = {last_action[0, 0, :8].float().tolist()}")
    print(f"  action mean={last_action.float().mean():.6f}  std={last_action.float().std():.6f}")

    # ── Save reference tensors (for TTNN parity) ──────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "vl_embeds":               vl_embeds.float().cpu(),
                "backbone_attention_mask": backbone_attention_mask.cpu(),
                "image_mask":              image_mask.cpu(),
                "state":                   state.float().cpu(),
                "embodiment_id":           embodiment_id.cpu(),
                "action":                  last_action.float().cpu(),
            },
            out_path,
        )
        print(f"\nSaved reference tensors to: {out_path}")


if __name__ == "__main__":
    main()
