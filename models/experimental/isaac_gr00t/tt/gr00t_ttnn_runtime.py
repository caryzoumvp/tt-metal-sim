#!/usr/bin/env python3
"""TTNN runtime for GR00T N1.6 action head.

TtGr00tActionHeadRuntime implements the AlternateVLDiT denoising loop
using ttnn ops (matmul, layernorm, softmax, silu, gelu, …).

Weights are loaded directly from safetensors via a flat dict[str, Tensor]
— no Isaac-GR00T package import required.

Usage from demo/demo.py:
    runtime = TtGr00tActionHeadRuntime.load_weights(
        store=safetensors_dict, args=rt_args, device=tt_device
    )
    actions = runtime.run_denoise_loop(rt_args, shared_inputs=shared_inputs)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import torch
import ttnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_shared_inputs(shared_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in shared_inputs.items()}


def _load_model_cfg(model_dir: str | Path) -> dict[str, Any]:
    config_path = Path(model_dir).resolve() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in model directory: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


# Accumulates upload stats across all _timed_upload calls; flushed by _dump_upload_total().
_upload_accum = {"count": 0, "mb": 0.0, "tilize_ms": 0.0, "device_ms": 0.0}


def _timed_upload(t_bf16: torch.Tensor, device: ttnn.Device, *, label: str) -> ttnn.Tensor:
    """Upload a bfloat16 torch tensor to TT device, printing per-phase timing."""
    nbytes = t_bf16.numel() * 2
    t0 = perf_counter_ns()
    t_host = ttnn.from_torch(t_bf16, layout=ttnn.TILE_LAYOUT)
    t1 = perf_counter_ns()
    t_dev = ttnn.to_device(t_host, device)
    t2 = perf_counter_ns()
    tilize_ms = (t1 - t0) / 1e6
    upload_ms = (t2 - t1) / 1e6
    total_ms  = (t2 - t0) / 1e6
    mb = nbytes / 1e6
    bw = mb / (upload_ms / 1e3) if upload_ms > 0 else float("inf")
    print(
        f"[upload] {label:60s} shape={tuple(t_bf16.shape)}  {mb:6.1f} MB"
        f"  tilize={tilize_ms:7.1f} ms  device={upload_ms:7.1f} ms"
        f"  total={total_ms:7.1f} ms  bw={bw:6.0f} MB/s"
    )
    _upload_accum["count"]     += 1
    _upload_accum["mb"]        += mb
    _upload_accum["tilize_ms"] += tilize_ms
    _upload_accum["device_ms"] += upload_ms
    return t_dev


def _dump_upload_total() -> None:
    """Print cumulative upload stats and reset the accumulator."""
    a = _upload_accum
    if a["count"] == 0:
        return
    total_ms = a["tilize_ms"] + a["device_ms"]
    bw = a["mb"] / (a["device_ms"] / 1e3) if a["device_ms"] > 0 else float("inf")
    print(
        f"[upload] TOTAL: {a['count']} tensors  {a['mb']:.1f} MB"
        f"  tilize={a['tilize_ms']:.1f} ms  device={a['device_ms']:.1f} ms"
        f"  total={total_ms:.1f} ms  bw={bw:.0f} MB/s"
    )
    a["count"] = a["mb"] = a["tilize_ms"] = a["device_ms"] = 0


def get_tensor_np(store: Any, key: str) -> np.ndarray:
    """Extract a tensor from the safetensors store as a float32 numpy array."""
    return store.get(key).to(torch.float32).cpu().numpy()


# ---------------------------------------------------------------------------
# Sinusoidal embeddings (CPU numpy — uploaded to device as needed)
# ---------------------------------------------------------------------------

def timestep_embedding_np(
    timesteps: np.ndarray,
    embedding_dim: int = 256,
    *,
    flip_sin_to_cos: bool = True,
    downscale_freq_shift: float = 1.0,
    max_period: float = 10000.0,
) -> np.ndarray:
    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * np.arange(half_dim, dtype=np.float32)
    exponent = exponent / float(half_dim - downscale_freq_shift)
    freqs = np.exp(exponent)[None, :]
    args = timesteps.astype(np.float32)[:, None] * freqs
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=-1)
    if flip_sin_to_cos:
        emb = np.concatenate([emb[:, half_dim:], emb[:, :half_dim]], axis=-1)
    if embedding_dim % 2 == 1:
        emb = np.pad(emb, ((0, 0), (0, 1)))
    return emb.astype(np.float32)


def action_pos_encoding_np(timesteps_bt: np.ndarray, embedding_dim: int) -> np.ndarray:
    """Sinusoidal positional encoding for action timestep tokens."""
    half_dim = embedding_dim // 2
    exponent = -np.arange(half_dim, dtype=np.float32) * (math.log(10000.0) / float(half_dim))
    freqs = timesteps_bt[:, :, None].astype(np.float32) * np.exp(exponent)[None, None, :]
    return np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# TtGr00tActionHeadRuntime
# ---------------------------------------------------------------------------

class TtGr00tActionHeadRuntime:
    """
    GR00T N1.6 action head — AlternateVLDiT denoising loop via ttnn ops.

    Every linear, layernorm, softmax, silu, gelu, and elementwise op is
    dispatched to ttnn, making each step a drop-in replacement target for
    a TTNN kernel on Wormhole hardware.

    Weight layout (from GR00T-N1.6-3B safetensors):
      action_head.state_encoder.layer{1,2}.{W,b}[embodiment_id]
      action_head.action_encoder.W{1,2,3}.{W,b}[embodiment_id]
      action_head.action_decoder.layer{1,2}.{W,b}[embodiment_id]
      action_head.model.timestep_encoder.timestep_embedder.linear_{1,2}.{weight,bias}
      action_head.model.transformer_blocks.{i}.{norm1,attn1,ff}.*
      action_head.model.{proj_out_1,proj_out_2}.{weight,bias}
      action_head.position_embedding.weight        (optional)
      action_head.vlln.{weight,bias}               (optional)
    """

    def __init__(
        self,
        *,
        device: ttnn.Device | None,
        weights: dict[str, Any],
        cfg: dict[str, Any],
    ):
        self.device = device
        self.w = weights
        self.cfg = cfg
        self._tt_weight_cache: dict[str, Any] = {}
        self._tt_bias_cache: dict[str, Any] = {}
        self._op_timings: list[tuple[str, float, list[tuple], tuple | None]] = []

    @classmethod
    def load_weights(
        cls, *, store: Any, args: argparse.Namespace, device: ttnn.Device | None
    ) -> "TtGr00tActionHeadRuntime":
        """
        Build the runtime from a safetensors store and config.

        args fields used:
          model_dir        — path to GR00T-N1.6-3B dir (for config.json)
          embodiment_id    — index into CategorySpecific* weight tables
          ttnn_num_layers  — cap DiT layers (0 = all from config)
        """
        cfg = _load_model_cfg(args.model_dir)
        emb = int(args.embodiment_id)

        num_layers = int(cfg["diffusion_model_cfg"]["num_layers"])
        if args.ttnn_num_layers > 0:
            num_layers = min(num_layers, int(args.ttnn_num_layers))

        weights: dict[str, Any] = {
            "num_layers":                  num_layers,
            "num_heads":                   int(cfg["diffusion_model_cfg"]["num_attention_heads"]),
            "inner_dim":                   int(cfg["diffusion_model_cfg"]["num_attention_heads"])
                                           * int(cfg["diffusion_model_cfg"]["attention_head_dim"]),
            "norm_eps":                    float(cfg["diffusion_model_cfg"].get("norm_eps", 1e-5)),
            "interleave_self_attention":   bool(cfg["diffusion_model_cfg"].get("interleave_self_attention", False)),
            "use_alternate_vl_dit":        bool(cfg.get("use_alternate_vl_dit", False)),
            "attend_text_every_n_blocks":  int(cfg.get("attend_text_every_n_blocks", 2)),
            "add_pos_embed":               bool(cfg.get("add_pos_embed", False)),
            "use_vlln":                    bool(cfg.get("use_vlln", False)),
            "num_timestep_buckets":        int(cfg.get("num_timestep_buckets", 1000)),
            "action_dim":                  int(cfg.get("max_action_dim", 128)),
            "state_dim":                   int(cfg.get("max_state_dim", 128)),
            "backbone_dim":                int(cfg.get("backbone_embedding_dim", 2048)),
            "input_embedding_dim":         int(cfg.get("input_embedding_dim", 1536)),
        }

        # Per-embodiment state encoder (CategorySpecificMLP)
        weights["state_l1_w"] = get_tensor_np(store, "action_head.state_encoder.layer1.W")[emb]
        weights["state_l1_b"] = get_tensor_np(store, "action_head.state_encoder.layer1.b")[emb]
        weights["state_l2_w"] = get_tensor_np(store, "action_head.state_encoder.layer2.W")[emb]
        weights["state_l2_b"] = get_tensor_np(store, "action_head.state_encoder.layer2.b")[emb]

        # Per-embodiment action encoder (MultiEmbodimentActionEncoder W1/W2/W3)
        weights["act_w1_w"] = get_tensor_np(store, "action_head.action_encoder.W1.W")[emb]
        weights["act_w1_b"] = get_tensor_np(store, "action_head.action_encoder.W1.b")[emb]
        weights["act_w2_w"] = get_tensor_np(store, "action_head.action_encoder.W2.W")[emb]
        weights["act_w2_b"] = get_tensor_np(store, "action_head.action_encoder.W2.b")[emb]
        weights["act_w3_w"] = get_tensor_np(store, "action_head.action_encoder.W3.W")[emb]
        weights["act_w3_b"] = get_tensor_np(store, "action_head.action_encoder.W3.b")[emb]

        # Per-embodiment action decoder (CategorySpecificMLP)
        weights["dec_l1_w"] = get_tensor_np(store, "action_head.action_decoder.layer1.W")[emb]
        weights["dec_l1_b"] = get_tensor_np(store, "action_head.action_decoder.layer1.b")[emb]
        weights["dec_l2_w"] = get_tensor_np(store, "action_head.action_decoder.layer2.W")[emb]
        weights["dec_l2_b"] = get_tensor_np(store, "action_head.action_decoder.layer2.b")[emb]

        # DiT timestep encoder (diffusers TimestepEmbedding linear_1 / linear_2)
        weights["time_l1_w"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight")
        weights["time_l1_b"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_1.bias")
        weights["time_l2_w"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_2.weight")
        weights["time_l2_b"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_2.bias")

        # DiT output projection
        weights["proj_out_1_w"] = get_tensor_np(store, "action_head.model.proj_out_1.weight")
        weights["proj_out_1_b"] = get_tensor_np(store, "action_head.model.proj_out_1.bias")
        weights["proj_out_2_w"] = get_tensor_np(store, "action_head.model.proj_out_2.weight")
        weights["proj_out_2_b"] = get_tensor_np(store, "action_head.model.proj_out_2.bias")

        # Optional positional embedding and VLM layernorm
        weights["pos_embedding"] = (
            get_tensor_np(store, "action_head.position_embedding.weight")
            if weights["add_pos_embed"] else None
        )
        if weights["use_vlln"]:
            weights["vlln_w"] = get_tensor_np(store, "action_head.vlln.weight")
            weights["vlln_b"] = get_tensor_np(store, "action_head.vlln.bias")
        else:
            weights["vlln_w"] = weights["vlln_b"] = None

        # Per-block weights for AlternateVLDiT transformer blocks
        blocks: list[dict[str, np.ndarray]] = []
        for i in range(num_layers):
            p = f"action_head.model.transformer_blocks.{i}"
            blocks.append({
                # AdaLayerNorm (norm1): silu(temb) → linear → scale/shift
                "norm1_linear_w": get_tensor_np(store, f"{p}.norm1.linear.weight"),
                "norm1_linear_b": get_tensor_np(store, f"{p}.norm1.linear.bias"),
                # Attention projections (attn1): Q/K/V + output
                "attn_q_w": get_tensor_np(store, f"{p}.attn1.to_q.weight"),
                "attn_q_b": get_tensor_np(store, f"{p}.attn1.to_q.bias"),
                "attn_k_w": get_tensor_np(store, f"{p}.attn1.to_k.weight"),
                "attn_k_b": get_tensor_np(store, f"{p}.attn1.to_k.bias"),
                "attn_v_w": get_tensor_np(store, f"{p}.attn1.to_v.weight"),
                "attn_v_b": get_tensor_np(store, f"{p}.attn1.to_v.bias"),
                "attn_o_w": get_tensor_np(store, f"{p}.attn1.to_out.0.weight"),
                "attn_o_b": get_tensor_np(store, f"{p}.attn1.to_out.0.bias"),
                # Feed-forward (ff): net.0.proj (GELU) + net.2 (output linear)
                "ff_in_w":  get_tensor_np(store, f"{p}.ff.net.0.proj.weight"),
                "ff_in_b":  get_tensor_np(store, f"{p}.ff.net.0.proj.bias"),
                "ff_out_w": get_tensor_np(store, f"{p}.ff.net.2.weight"),
                "ff_out_b": get_tensor_np(store, f"{p}.ff.net.2.bias"),
            })
        weights["blocks"] = blocks

        return cls(device=device, weights=weights, cfg=cfg)

    # ── Op timing helpers ─────────────────────────────────────────────────────

    def _timed_ttnn(self, label: str, fn, *args, **kwargs):
        """Call fn(*args, **kwargs), record wall-clock time + all tensor arg shapes."""
        in_shapes = [tuple(a.shape) for a in args if hasattr(a, "shape")]

        t0 = perf_counter_ns()
        result = fn(*args, **kwargs)
        elapsed_ms = (perf_counter_ns() - t0) / 1e6

        out_shape: tuple | None = tuple(result.shape) if hasattr(result, "shape") else None

        self._op_timings.append((label, elapsed_ms, in_shapes, out_shape))
        return result

    def _dump_op_timings(self, prefix: str = "") -> None:
        """Print accumulated op timings with tensor shapes, then clear the list.

        prefix is typically "t=<n>" where n is the flow-matching timestep bucket
        (0 = fully denoised / clean action, num_timestep_buckets = noisy).
        """
        if not self._op_timings:
            return
        hdr = f"[op-perf]{f' {prefix}' if prefix else ''}"
        total = sum(ms for _, ms, _, _ in self._op_timings)

        # Annotate the t=<n> header so readers know what the number means
        note = ""
        if prefix.startswith("t="):
            note = "  [flow-matching timestep bucket: 0=clean, 1000=noise]"

        _dump_upload_total()
        print(f"{hdr} --- {len(self._op_timings)} ops  total={total:.1f} ms{note} ---")
        for label, ms, in_shapes, out_shape in self._op_timings:
            if in_shapes and out_shape is not None:
                shape_str = f"  {', '.join(str(s) for s in in_shapes)} -> {out_shape}"
            elif in_shapes:
                shape_str = f"  {', '.join(str(s) for s in in_shapes)}"
            else:
                shape_str = ""
            print(f"{hdr}  {label:55s}  {ms:8.3f} ms{shape_str}")
        self._op_timings.clear()

    # ── Weight cache helpers ──────────────────────────────────────────────────

    def _weight_to_tt(self, name: str, weight_io: np.ndarray) -> ttnn.Tensor:
        if name not in self._tt_weight_cache:
            self._tt_weight_cache[name] = _timed_upload(
                torch.from_numpy(weight_io).to(torch.bfloat16),
                self.device,
                label=f"action.{name}",
            )
        return self._tt_weight_cache[name]

    def _bias_to_tt(self, name: str, bias: np.ndarray) -> ttnn.Tensor:
        if name not in self._tt_bias_cache:
            self._tt_bias_cache[name] = _timed_upload(
                torch.from_numpy(bias.reshape(1, -1)).to(torch.bfloat16),
                self.device,
                label=f"action.{name}",
            )
        return self._tt_bias_cache[name]

    # ── Core TTNN op primitives ───────────────────────────────────────────────

    def _linear_lastdim(
        self, x: Any, weight_io: np.ndarray, bias: np.ndarray | None, *, op_name: str
    ) -> ttnn.Tensor:
        """Linear over the last dimension: x @ W + b, with shape broadcast."""
        shape = tuple(x.shape)
        in_dim = shape[-1]
        w = weight_io.astype(np.float32)
        if w.shape[0] != in_dim:
            if w.shape[1] == in_dim:
                w = w.T
            else:
                raise ValueError(f"{op_name}: weight/input mismatch x_in={in_dim} weight={w.shape}")
        x_tt = self._ensure_tt(x, op_name=op_name)
        if len(shape) > 2:
            m = 1
            for d in shape[:-1]:
                m *= int(d)
            x_tt = self._timed_ttnn(f"{op_name}.reshape_in", ttnn.reshape, x_tt, (m, in_dim))
        y_tt = self._timed_ttnn(f"{op_name}.matmul", ttnn.matmul, x_tt, self._weight_to_tt(op_name, w))
        if bias is not None:
            y_tt = self._timed_ttnn(f"{op_name}.bias_add", ttnn.add, y_tt,
                                    self._bias_to_tt(f"{op_name}.b", bias.astype(np.float32)))
        if len(shape) > 2:
            out_dim = int(tuple(y_tt.shape)[-1])
            y_tt = self._timed_ttnn(f"{op_name}.reshape_out", ttnn.reshape, y_tt, shape[:-1] + (out_dim,))
        return y_tt

    def _to_tt(self, x: np.ndarray, *, op_name: str) -> ttnn.Tensor:
        if self.device is None:
            raise RuntimeError(f"TTNN device unavailable at {op_name}")
        tx = torch.from_numpy(x)
        if tx.dtype in (torch.float16, torch.float32, torch.float64):
            tx = tx.to(torch.bfloat16)
        return ttnn.from_torch(tx, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _to_np(self, x: ttnn.Tensor) -> np.ndarray:
        return ttnn.to_torch(x).to(torch.float32).cpu().numpy().astype(np.float32)

    def _ensure_tt(self, x: Any, *, op_name: str) -> ttnn.Tensor:
        if isinstance(x, ttnn.Tensor):
            return x
        return self._to_tt(np.asarray(x, dtype=np.float32), op_name=op_name)

    def _tt_silu(self, x: Any, *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.silu, self._ensure_tt(x, op_name=op_name))

    def _tt_relu(self, x: Any, *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.relu, self._ensure_tt(x, op_name=op_name))

    def _tt_gelu(self, x: Any, *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.gelu, self._ensure_tt(x, op_name=op_name))

    def _tt_add(self, x: Any, y: Any, *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.add,
                                self._ensure_tt(x, op_name=f"{op_name}.lhs"),
                                self._ensure_tt(y, op_name=f"{op_name}.rhs"))

    def _tt_mul(self, x: Any, y: Any, *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.multiply,
                                self._ensure_tt(x, op_name=f"{op_name}.lhs"),
                                self._ensure_tt(y, op_name=f"{op_name}.rhs"))

    def _tt_mul_scalar(self, x: Any, scalar: float, *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.multiply,
                                self._ensure_tt(x, op_name=op_name), np.float32(scalar))

    def _tt_repeat(self, x: Any, repeats: list[int], *, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.repeat, self._ensure_tt(x, op_name=op_name), repeats)

    def _tt_concat(self, xs: list, *, dim: int, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.concat,
                                [self._ensure_tt(x, op_name=f"{op_name}.in{i}") for i, x in enumerate(xs)],
                                dim=dim)

    def _tt_split_half(self, x: Any, *, dim: int, op_name: str) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        x_shape = tuple(x.shape)
        if x_shape[dim] % 2 != 0:
            raise ValueError(f"{op_name}: split dim {dim} size {x_shape[dim]} not divisible by 2")
        tx = self._ensure_tt(x, op_name=op_name)
        parts = self._timed_ttnn(op_name, ttnn.split, tx, x_shape[dim] // 2, dim=dim)
        if len(parts) != 2:
            raise RuntimeError(f"{op_name}: expected 2 chunks, got {len(parts)}")
        return parts[0], parts[1]

    def _tt_layer_norm(self, x: Any, *, eps: float, op_name: str) -> ttnn.Tensor:
        return self._timed_ttnn(op_name, ttnn.layer_norm,
                                self._ensure_tt(x, op_name=op_name), epsilon=float(eps))

    def _tt_bcast_h(self, x: Any, y: Any, *, math_op: Any, op_name: str) -> ttnn.Tensor:
        x_shape = tuple(x.shape)
        if isinstance(y, np.ndarray) and y.ndim == len(x_shape) - 1:
            y = np.expand_dims(y.astype(np.float32), axis=-2)
        y_shape = tuple(y.shape)
        if y_shape[-2] != 1:
            raise ValueError(f"{op_name}: rhs H must be 1 for BcastOpDim.H, got {y_shape[-2]}")
        if y_shape[-1] != x_shape[-1]:
            raise ValueError(f"{op_name}: rhs W {y_shape[-1]} != lhs W {x_shape[-1]}")
        return self._timed_ttnn(op_name, ttnn.bcast,
                                self._ensure_tt(x, op_name=f"{op_name}.lhs"),
                                self._ensure_tt(y, op_name=f"{op_name}.rhs"),
                                math_op, ttnn.BcastOpDim.H)

    def _tt_bcast_h_add(self, x: Any, y: Any, *, op_name: str) -> ttnn.Tensor:
        return self._tt_bcast_h(x, y, math_op=ttnn.BcastOpMath.ADD, op_name=op_name)

    def _tt_bcast_h_mul(self, x: Any, y: Any, *, op_name: str) -> ttnn.Tensor:
        return self._tt_bcast_h(x, y, math_op=ttnn.BcastOpMath.MUL, op_name=op_name)

    # ── Multi-head attention ──────────────────────────────────────────────────

    def _mh_attention(
        self,
        q: ttnn.Tensor,
        k: ttnn.Tensor,
        v: ttnn.Tensor,
        *,
        attention_mask: np.ndarray | None,
        op_name: str = "attn",
    ) -> ttnn.Tensor:
        """Scaled dot-product multi-head attention via ttnn ops.

        Q [B, T, D]  K/V [B, S, D]  → output [B, T, D]
        attention_mask [B, S] bool, True = keep token.
        """
        batch, q_len, dim = tuple(q.shape)
        k_len = int(tuple(k.shape)[1])
        heads = int(self.w["num_heads"])
        head_dim = dim // heads

        # Reshape to [B, H, T/S, d] for batched matmul
        q_tt = self._timed_ttnn(f"{op_name}.q_permute",
                                ttnn.permute,
                                self._timed_ttnn(f"{op_name}.q_reshape", ttnn.reshape, q, (batch, q_len, heads, head_dim)),
                                (0, 2, 1, 3))
        k_tt = self._timed_ttnn(f"{op_name}.k_permute",
                                ttnn.permute,
                                self._timed_ttnn(f"{op_name}.k_reshape", ttnn.reshape, k, (batch, k_len, heads, head_dim)),
                                (0, 2, 3, 1))
        v_tt = self._timed_ttnn(f"{op_name}.v_permute",
                                ttnn.permute,
                                self._timed_ttnn(f"{op_name}.v_reshape", ttnn.reshape, v, (batch, k_len, heads, head_dim)),
                                (0, 2, 1, 3))

        # Attention scores and softmax
        scores_tt = self._timed_ttnn(f"{op_name}.qk_scale", ttnn.multiply,
                                     self._timed_ttnn(f"{op_name}.qk_matmul", ttnn.matmul, q_tt, k_tt),
                                     np.float32(1.0 / np.sqrt(float(head_dim))))
        if attention_mask is not None:
            keep = attention_mask.astype(bool)[:, None, None, :]
            bias_np = np.where(keep, np.float32(0.0), np.float32(-1e9))
            bias_np = bias_np.reshape(batch, 1, 1, k_len).astype(np.float32)
            if heads != 1 or q_len != 1:
                bias_tt = self._tt_repeat(bias_np, [1, heads, q_len, 1], op_name=f"{op_name}.mask_repeat")
            else:
                bias_tt = self._ensure_tt(bias_np, op_name=f"{op_name}.mask_bias")
            scores_tt = self._timed_ttnn(f"{op_name}.mask_add", ttnn.add, scores_tt, bias_tt)
        probs_tt = self._timed_ttnn(f"{op_name}.softmax", ttnn.softmax, scores_tt, dim=-1)

        # Weighted sum → [B, T, D]
        ctx_tt = self._timed_ttnn(f"{op_name}.pv_permute",
                                  ttnn.permute,
                                  self._timed_ttnn(f"{op_name}.pv_matmul", ttnn.matmul, probs_tt, v_tt),
                                  (0, 2, 1, 3))
        return self._timed_ttnn(f"{op_name}.out_reshape", ttnn.reshape, ctx_tt, (batch, q_len, dim))

    # ── Input builder (used when shared_inputs is None) ──────────────────────

    def _build_inputs(self, args: argparse.Namespace) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed + 1)
        batch    = int(args.batch)
        action_h = int(args.action_horizon)

        # Eagle-3-VL measured: 144 image tokens (SigLIP2 @ 336px → 576 patches,
        # pixel_shuffle(0.5) → 144) + 23 chat-template tokens = 167 total
        n_img_patches = 144
        vl_seq_len    = int(args.vl_seq_len)
        state    = rng.standard_normal((batch, 1, int(self.w["state_dim"])), dtype=np.float32)
        vl_embeds = rng.standard_normal((batch, vl_seq_len, int(self.w["backbone_dim"])), dtype=np.float32)
        image_mask = np.zeros((batch, vl_seq_len), dtype=bool)
        image_mask[:, :n_img_patches] = True   # first 144 tokens = image patches
        backbone_attention_mask = np.ones((batch, vl_seq_len), dtype=bool)
        actions = torch.randn((batch, action_h, int(self.w["action_dim"])),
                              dtype=torch.float32).cpu().numpy().astype(np.float32)
        return {
            "state":                   state,
            "vl_embeds":               vl_embeds,
            "image_mask":              image_mask,
            "backbone_attention_mask": backbone_attention_mask,
            "actions":                 actions,
        }

    # ── Single denoising step ─────────────────────────────────────────────────

    def run_step(
        self,
        *,
        actions: np.ndarray,
        state_features: np.ndarray,
        vl_embeds: np.ndarray,
        image_mask: np.ndarray,
        backbone_attention_mask: np.ndarray,
        t_discretized: int,
        action_horizon: int,
        return_debug: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
        """
        One Euler denoising step via ttnn ops.

        Returns (pred_velocity [B, T, 128], model_output [B, T+1, 1024], debug).
        """
        batch = int(tuple(actions.shape)[0])

        # ── Timestep embedding ──────────────────────────────────────────────
        # Sinusoidal projection (CPU) → two MLP linears (ttnn) → temb [B, 1536]
        t_bucket = np.full((batch,), int(t_discretized), dtype=np.int64)
        t_proj = timestep_embedding_np(t_bucket, embedding_dim=256)
        temb = self._linear_lastdim(t_proj, self.w["time_l1_w"].T, self.w["time_l1_b"], op_name="time_l1")
        temb = self._tt_silu(temb, op_name="time_silu")
        temb = self._linear_lastdim(temb, self.w["time_l2_w"].T, self.w["time_l2_b"], op_name="time_l2")

        # ── Action encoder ──────────────────────────────────────────────────
        # MultiEmbodimentActionEncoder: W1(action) + sinusoidal(t) → concat → W2 (swish) → W3
        action_features = self._linear_lastdim(
            actions, self.w["act_w1_w"], self.w["act_w1_b"], op_name="act_w1"
        )
        timestep_bt = np.repeat(t_bucket[:, None].astype(np.float32), action_horizon, axis=1)
        tau_emb = action_pos_encoding_np(timestep_bt, int(self.w["input_embedding_dim"]))
        action_features = self._tt_concat([action_features, tau_emb], dim=-1, op_name="action.concat_tau")
        action_features = self._linear_lastdim(
            action_features, self.w["act_w2_w"], self.w["act_w2_b"], op_name="act_w2"
        )
        action_features = self._tt_silu(action_features, op_name="action.silu")
        action_features = self._linear_lastdim(
            action_features, self.w["act_w3_w"], self.w["act_w3_b"], op_name="act_w3"
        )

        # ── Positional embedding ────────────────────────────────────────────
        if self.w["add_pos_embed"] and self.w["pos_embedding"] is not None:
            pos = self.w["pos_embedding"][:action_horizon][None, :, :].astype(np.float32)
            if batch != 1:
                pos = self._tt_repeat(pos, [batch, 1, 1], op_name="action.pos_repeat_batch")
            action_features = self._tt_add(action_features, pos, op_name="action.add_pos")

        # ── Concat [state | action] tokens ─────────────────────────────────
        hidden_states = self._tt_concat([state_features, action_features], dim=1, op_name="sa.concat")

        block_hidden_states: list[np.ndarray] = [self._to_np(hidden_states)] if return_debug else []

        # ── AlternateVLDiT blocks ───────────────────────────────────────────
        # odd  blocks (idx%2==1): self-attention  (K/V = hidden_states)
        # even blocks (idx%2==0): cross-attention (K/V = vl_embeds, alt image/text)
        for i in range(int(self.w["num_layers"])):
            block = self.w["blocks"][i]

            # AdaLayerNorm: layernorm → silu(temb) → linear → scale/shift
            norm_hidden = self._tt_layer_norm(hidden_states, eps=float(self.w["norm_eps"]), op_name=f"block{i}.ln1")
            ada = self._linear_lastdim(
                self._tt_silu(temb, op_name=f"block{i}.ada_silu"),
                block["norm1_linear_w"].T, block["norm1_linear_b"],
                op_name=f"block{i}.norm1_linear",
            )
            scale, shift = self._tt_split_half(ada, dim=len(ada.shape) - 1, op_name=f"block{i}.ada_split")
            scale_plus_one = self._tt_add(scale, np.ones(tuple(scale.shape), dtype=np.float32), op_name=f"block{i}.scale_plus_one")
            norm_hidden = self._tt_bcast_h_mul(norm_hidden, scale_plus_one, op_name=f"block{i}.ada_mul")
            norm_hidden = self._tt_bcast_h_add(norm_hidden, shift,          op_name=f"block{i}.ada_add")

            # Attention: self or cross
            cross_mask: np.ndarray | None = None
            if bool(self.w["interleave_self_attention"]) and (i % 2 == 1):
                enc = norm_hidden  # self-attention
            else:
                enc = vl_embeds   # cross-attention
                if bool(self.w["use_alternate_vl_dit"]):
                    if i % (2 * int(self.w["attend_text_every_n_blocks"])) == 0:
                        cross_mask = (~image_mask) & backbone_attention_mask   # text tokens
                    else:
                        cross_mask = image_mask & backbone_attention_mask      # image tokens

            q = self._linear_lastdim(norm_hidden, block["attn_q_w"].T, block["attn_q_b"], op_name=f"block{i}.q")
            k = self._linear_lastdim(enc,         block["attn_k_w"].T, block["attn_k_b"], op_name=f"block{i}.k")
            v = self._linear_lastdim(enc,         block["attn_v_w"].T, block["attn_v_b"], op_name=f"block{i}.v")
            ctx = self._mh_attention(q, k, v, attention_mask=cross_mask, op_name=f"block{i}.attn")
            attn_out = self._linear_lastdim(ctx, block["attn_o_w"].T, block["attn_o_b"], op_name=f"block{i}.o")
            hidden_states = self._tt_add(hidden_states, attn_out, op_name=f"block{i}.resid_attn")

            # Feed-forward: layernorm → GELU(net.0.proj) → net.2
            ff_in = self._linear_lastdim(
                self._tt_layer_norm(hidden_states, eps=float(self.w["norm_eps"]), op_name=f"block{i}.ln2"),
                block["ff_in_w"].T, block["ff_in_b"], op_name=f"block{i}.ff_in",
            )
            ff_out_in_dim = int(block["ff_out_w"].shape[1])
            ff_in_last    = int(tuple(ff_in.shape)[-1])
            if ff_in_last == ff_out_in_dim:
                ff_act = self._tt_gelu(ff_in, op_name=f"block{i}.ff_gelu")
            elif ff_in_last == 2 * ff_out_in_dim:
                ff_x, ff_gate = self._tt_split_half(ff_in, dim=len(ff_in.shape) - 1, op_name=f"block{i}.ff_split")
                ff_act = self._tt_mul(ff_x, self._tt_gelu(ff_gate, op_name=f"block{i}.ff_gate_gelu"), op_name=f"block{i}.ff_gate_mul")
            else:
                raise ValueError(f"block{i}.ff: unsupported dimensions ff_in={ff_in_last}, ff_out_in_dim={ff_out_in_dim}")
            ff_out = self._linear_lastdim(ff_act, block["ff_out_w"].T, block["ff_out_b"], op_name=f"block{i}.ff_out")
            hidden_states = self._tt_add(hidden_states, ff_out, op_name=f"block{i}.resid_ff")

            if return_debug:
                block_hidden_states.append(self._to_np(hidden_states))

        # ── Output modulation (same as standard DiT) ────────────────────────
        # norm_out (no affine) → silu(temb) → proj_out_1 → scale/shift → proj_out_2
        out_norm = self._tt_layer_norm(hidden_states, eps=1e-6, op_name="out_norm")
        shift_scale = self._linear_lastdim(
            self._tt_silu(temb, op_name="proj_out_1_silu"),
            self.w["proj_out_1_w"].T, self.w["proj_out_1_b"], op_name="proj_out_1",
        )
        shift, scale = self._tt_split_half(shift_scale, dim=len(shift_scale.shape) - 1, op_name="proj_out_1_split")
        scale_plus_one = self._tt_add(scale, np.ones(tuple(scale.shape), dtype=np.float32), op_name="proj_out_scale_plus_one")
        out_norm = self._tt_bcast_h_mul(out_norm, scale_plus_one, op_name="proj_out_affine_mul")
        out_norm = self._tt_bcast_h_add(out_norm, shift,          op_name="proj_out_affine_add")
        model_output = self._linear_lastdim(
            out_norm, self.w["proj_out_2_w"].T, self.w["proj_out_2_b"], op_name="proj_out_2"
        )

        # ── Action decoder (CategorySpecificMLP) ────────────────────────────
        d1   = self._tt_relu(self._linear_lastdim(model_output, self.w["dec_l1_w"], self.w["dec_l1_b"], op_name="dec_l1"), op_name="dec_relu")
        pred = self._linear_lastdim(d1, self.w["dec_l2_w"], self.w["dec_l2_b"], op_name="dec_l2")

        pred_np         = self._to_np(pred)
        model_output_np = self._to_np(model_output)
        pred_velocity   = pred_np[:, -action_horizon:, :]

        step_debug: dict[str, np.ndarray] | None = None
        if return_debug:
            step_debug = {
                "timesteps":      t_bucket.astype(np.int64),
                "temb":           self._to_np(temb),
                "action_features": self._to_np(action_features),
                "model_output":   model_output_np,
                "pred_velocity":  pred_velocity.astype(np.float32),
            }
            for i, hs in enumerate(block_hidden_states):
                step_debug[f"block_hidden_{i:02d}"] = hs

        self._dump_op_timings(prefix=f"t={t_discretized}")
        return pred_velocity.astype(np.float32), model_output_np.astype(np.float32), step_debug

    # ── Full denoising loop ───────────────────────────────────────────────────

    def run_denoise_loop(
        self,
        args: argparse.Namespace,
        *,
        shared_inputs: dict[str, np.ndarray] | None = None,
        return_debug: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """
        Flow-matching denoising loop (Euler integration, num_inference_timesteps steps).

        shared_inputs keys: state, vl_embeds, image_mask, backbone_attention_mask, actions.
        Pass shared_inputs=None to use _build_inputs() for synthetic test data.

        Returns action array [B, T, 128], or (action, debug_dict) if return_debug=True.
        """
        inputs = _clone_shared_inputs(shared_inputs) if shared_inputs is not None else self._build_inputs(args)

        state                   = inputs["state"]
        vl_embeds               = inputs["vl_embeds"]
        image_mask              = inputs["image_mask"]
        backbone_attention_mask = inputs["backbone_attention_mask"]
        actions                 = inputs["actions"]
        action_horizon          = int(args.action_horizon)

        debug: dict[str, np.ndarray] = {}
        if return_debug:
            debug["input/state"]                   = state.astype(np.float32)
            debug["input/vl_embeds"]               = vl_embeds.astype(np.float32)
            debug["input/image_mask"]              = image_mask
            debug["input/backbone_attention_mask"] = backbone_attention_mask
            debug["input/actions_init"]            = actions.astype(np.float32)

        # ── Optional VLM LayerNorm ──────────────────────────────────────────
        if self.w["use_vlln"] and self.w["vlln_w"] is not None:
            vl_embeds = self._tt_layer_norm(vl_embeds, eps=1e-5, op_name="vlln.norm")
            vlln_w = self.w["vlln_w"].reshape(1, -1).astype(np.float32)
            vlln_b = self.w["vlln_b"].reshape(1, -1).astype(np.float32)
            vl_batch = int(tuple(vl_embeds.shape)[0])
            if vl_batch != 1:
                vlln_w = self._tt_repeat(vlln_w, [vl_batch, 1], op_name="vlln.w_repeat_batch")
                vlln_b = self._tt_repeat(vlln_b, [vl_batch, 1], op_name="vlln.b_repeat_batch")
            vl_embeds = self._tt_bcast_h_mul(vl_embeds, vlln_w, op_name="vlln.mul_weight")
            vl_embeds = self._tt_bcast_h_add(vl_embeds, vlln_b, op_name="vlln.add_bias")

        # ── State encoder (CategorySpecificMLP) ────────────────────────────
        state_features = self._tt_relu(
            self._linear_lastdim(state, self.w["state_l1_w"], self.w["state_l1_b"], op_name="state_l1"),
            op_name="state_relu",
        )
        state_features = self._linear_lastdim(
            state_features, self.w["state_l2_w"], self.w["state_l2_b"], op_name="state_l2"
        )

        if return_debug:
            debug["encoded/state_features"] = self._to_np(state_features)

        # Pre-upload initial actions once; keep on device across denoising steps
        actions_tt = self._ensure_tt(actions, op_name="pre.actions")
        dt = np.float32(1.0 / float(args.num_inference_timesteps))

        # ── Euler denoising steps ───────────────────────────────────────────
        for t in range(int(args.num_inference_timesteps)):
            t_cont        = float(t) / float(args.num_inference_timesteps)
            t_discretized = int(t_cont * int(self.w["num_timestep_buckets"]))

            pred_velocity, _, step_debug = self.run_step(
                actions=actions_tt,
                state_features=state_features,
                vl_embeds=vl_embeds,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                t_discretized=t_discretized,
                action_horizon=action_horizon,
                return_debug=return_debug,
            )
            delta_tt  = self._tt_mul_scalar(pred_velocity, float(dt), op_name=f"step{t}.delta_scale")
            actions_tt = self._tt_add(actions_tt, delta_tt, op_name=f"step{t}.actions_update")
            print(f"[ttnn] step={t}  t_disc={t_discretized}  actions={tuple(actions_tt.shape)}")

            if return_debug and step_debug is not None:
                for k, v in step_debug.items():
                    debug[f"step_{t:02d}/{k}"] = v
                debug[f"step_{t:02d}/actions"] = self._to_np(actions_tt).astype(np.float32)

        out = self._to_np(actions_tt).astype(np.float32)
        if return_debug:
            debug["output/actions"] = out
            return out, debug
        return out
