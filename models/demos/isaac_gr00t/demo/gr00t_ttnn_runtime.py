#!/usr/bin/env python3
"""TTNN runtime implementation for GR00T action head."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import json
import math
from pathlib import Path
import sys
from time import perf_counter_ns
import types
from typing import Any

import numpy as np
import torch
import ttnn

def _clone_shared_inputs(shared_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in shared_inputs.items()}


def _load_model_cfg(model_dir: str | Path) -> dict[str, Any]:
    config_path = Path(model_dir).resolve() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in model directory: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _extract_prefixed_tensors(
    flat_tensors: dict[str, np.ndarray], prefix: str
) -> dict[str, np.ndarray]:
    normalized_prefix = prefix.rstrip("/")
    out: dict[str, np.ndarray] = {}
    for key, value in flat_tensors.items():
        if key == normalized_prefix:
            out[""] = value
        elif key.startswith(f"{normalized_prefix}/"):
            out[key[len(normalized_prefix) + 1 :]] = value
    return out


def _to_torch_system1_tensor(name: str, value: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(value)
    if name in {"input_ids", "attention_mask", "image_sizes"}:
        return tensor.to(torch.long)
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        return tensor.to(torch.float32)
    return tensor


def _build_system1_inputs_from_shared(shared_inputs: dict[str, np.ndarray]) -> dict[str, Any]:
    prepared_system1 = _extract_prefixed_tensors(shared_inputs, "system1_input")
    if not prepared_system1:
        raise RuntimeError(
            "Shared inputs missing system1_input/* tensors. "
            "Regenerate shared inputs with cpu_ref after latest gr00t.py changes."
        )

    backbone_inputs: dict[str, Any] = {}
    pixel_values: list[tuple[int, torch.Tensor]] = []
    for key, value in prepared_system1.items():
        if key.startswith("pixel_values/"):
            idx_str = key.split("/", 1)[1]
            if idx_str.isdigit():
                pixel_values.append((int(idx_str), _to_torch_system1_tensor("pixel_values", value)))
            continue
        if key == "pixel_values":
            continue
        backbone_inputs[key] = _to_torch_system1_tensor(key, value)
    if pixel_values:
        pixel_values.sort(key=lambda x: x[0])
        backbone_inputs["pixel_values"] = [t for _, t in pixel_values]
    return backbone_inputs


class TtGr00tSystem1BackboneRuntime:
    """System1 backbone runtime that routes Linear ops through TTNN."""

    def __init__(self, *, args: argparse.Namespace, device: ttnn.Device):
        self.args = args
        self.device = device
        self.policy = self._load_policy()

    def _load_policy(self):
        isaac_root = Path(self.args.isaac_gr00t_root).resolve()
        if not isaac_root.exists():
            raise FileNotFoundError(f"Isaac-GR00T root not found: {isaac_root}")
        if str(isaac_root) not in sys.path:
            sys.path.insert(0, str(isaac_root))
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        embodiment = EmbodimentTag[self.args.embodiment_tag]
        return Gr00tPolicy(
            embodiment_tag=embodiment,
            model_path=self.args.model_dir,
            device="cpu",
            strict=False,
        )

    def _ttnn_linear_with_cached(
        self,
        x2d: np.ndarray,
        weight_tt: ttnn.Tensor,
        bias_tt: ttnn.Tensor | None,
    ) -> np.ndarray:
        m, _ = x2d.shape
        padded_m = ((m + 31) // 32) * 32
        if padded_m != m:
            x_work = np.zeros((padded_m, x2d.shape[1]), dtype=np.float32)
            x_work[:m, :] = x2d
        else:
            x_work = x2d

        tx = ttnn.from_torch(
            torch.from_numpy(x_work).to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        y = ttnn.matmul(tx, weight_tt)
        if bias_tt is not None:
            y = ttnn.add(y, bias_tt)
        out = ttnn.to_torch(y).to(torch.float32).cpu().numpy()
        return out[:m, :].astype(np.float32)

    def _patch_backbone_linears(self) -> dict[str, Any]:
        originals: dict[str, Any] = {}
        backbone = self.policy.model.backbone
        for module_name, module in backbone.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            originals[module_name] = module.forward
            weight_io = module.weight.detach().cpu().to(torch.float32).T.numpy().astype(np.float32)
            bias_np = (
                module.bias.detach().cpu().to(torch.float32).numpy().astype(np.float32)
                if module.bias is not None
                else None
            )
            weight_tt = ttnn.from_torch(
                torch.from_numpy(weight_io).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
            bias_tt = (
                ttnn.from_torch(
                    torch.from_numpy(bias_np.reshape(1, -1)).to(torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                if bias_np is not None
                else None
            )

            def _patched_forward(self_mod, input_tensor, _wtt=weight_tt, _btt=bias_tt):
                x_np = input_tensor.detach().cpu().to(torch.float32).numpy().astype(np.float32)
                x2d = x_np.reshape(-1, x_np.shape[-1]).astype(np.float32)
                y2d = self._ttnn_linear_with_cached(x2d, _wtt, _btt)
                y_np = y2d.reshape(*x_np.shape[:-1], y2d.shape[-1]).astype(np.float32)
                out = torch.from_numpy(y_np).to(input_tensor.device)
                if input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                    out = out.to(input_tensor.dtype)
                return out

            module.forward = types.MethodType(_patched_forward, module)
        return originals

    def _restore_backbone_linears(self, originals: dict[str, Any]) -> None:
        backbone = self.policy.model.backbone
        for module_name, original_forward in originals.items():
            module = backbone
            for token in module_name.split("."):
                module = module[int(token)] if token.isdigit() else getattr(module, token)
            module.forward = original_forward

    def run(self, backbone_inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        originals = self._patch_backbone_linears()
        try:
            with torch.no_grad():
                backbone_output = self.policy.model.backbone(backbone_inputs)
                backbone_output = self.policy.model.action_head.process_backbone_output(backbone_output)
        finally:
            self._restore_backbone_linears(originals)
        return {
            "backbone_features": backbone_output.backbone_features.detach().cpu().to(torch.float32).numpy(),
            "backbone_attention_mask": backbone_output.backbone_attention_mask.detach().cpu().numpy().astype(bool),
            "image_mask": backbone_output.image_mask.detach().cpu().numpy().astype(bool),
        }


def _shape_repr(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "(n/a)"
    return str(tuple(shape))


def _collect_op_tensor_inputs(value: Any, prefix: str, out: list[str]) -> None:
    if isinstance(value, ttnn.Tensor):
        out.append(f"{prefix}=TTNNTensor(shape={_shape_repr(value)}, dtype={getattr(value, 'dtype', 'n/a')})")
        return
    if isinstance(value, torch.Tensor):
        out.append(f"{prefix}=TorchTensor(shape={_shape_repr(value)}, dtype={value.dtype})")
        return
    if isinstance(value, np.ndarray):
        out.append(f"{prefix}=NdArray(shape={_shape_repr(value)}, dtype={value.dtype})")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_op_tensor_inputs(item, f"{prefix}.{key}", out)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            _collect_op_tensor_inputs(item, f"{prefix}[{idx}]", out)


def _make_ttnn_pre_operation_hook(op_start_times: dict[int, int] | None = None):
    def hook(operation, args, kwargs):
        op_key = id(operation)
        if op_start_times is not None:
            op_start_times[op_key] = perf_counter_ns()
        op_name = getattr(operation, "python_fully_qualified_name", None)
        if op_name is None:
            op_name = str(operation)
        op_kwargs = kwargs if isinstance(kwargs, dict) else {}
        tensor_inputs: list[str] = []
        for idx, value in enumerate(args):
            _collect_op_tensor_inputs(value, f"arg{idx}", tensor_inputs)
        for key in sorted(op_kwargs.keys()):
            _collect_op_tensor_inputs(op_kwargs[key], f"kw:{key}", tensor_inputs)
        tensor_msg = ", ".join(tensor_inputs) if tensor_inputs else "no_tensor_inputs"
        print(f"[ttnn-op] {op_name}: {tensor_msg}")
        return None

    return hook


def _make_ttnn_post_operation_hook(op_start_times: dict[int, int]):
    def hook(operation, args, kwargs, output):
        op_name = getattr(operation, "python_fully_qualified_name", None)
        if op_name is None:
            op_name = str(operation)
        start_ns = op_start_times.pop(id(operation), None)
        if start_ns is None:
            print(f"[ttnn-op-time] {op_name}: unavailable")
        else:
            elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
            print(f"[ttnn-op-time] {op_name}: {elapsed_ms:.3f} ms")
        return None

    return hook


def get_tensor_np(store: Any, key: str) -> np.ndarray:
    return store.get(key).to(torch.float32).cpu().numpy()

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
    half_dim = embedding_dim // 2
    exponent = -np.arange(half_dim, dtype=np.float32) * (math.log(10000.0) / float(half_dim))
    freqs = timesteps_bt[:, :, None].astype(np.float32) * np.exp(exponent)[None, None, :]
    return np.concatenate([np.sin(freqs), np.cos(freqs)], axis=-1).astype(np.float32)


class TtGr00tActionHeadRuntime:
    """Incremental TTNN action-head runtime."""

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

    @classmethod
    def load_weights(
        cls, *, store: Any, args: argparse.Namespace, device: ttnn.Device | None
    ) -> "TtGr00tActionHeadRuntime":
        cfg = _load_model_cfg(args.model_dir)
        emb = int(args.embodiment_id)

        num_layers = int(cfg["diffusion_model_cfg"]["num_layers"])
        if args.ttnn_num_layers > 0:
            num_layers = min(num_layers, int(args.ttnn_num_layers))

        weights: dict[str, Any] = {
            "num_layers": num_layers,
            "num_heads": int(cfg["diffusion_model_cfg"]["num_attention_heads"]),
            "inner_dim": int(cfg["diffusion_model_cfg"]["num_attention_heads"])
            * int(cfg["diffusion_model_cfg"]["attention_head_dim"]),
            "norm_eps": float(cfg["diffusion_model_cfg"].get("norm_eps", 1e-5)),
            "interleave_self_attention": bool(cfg["diffusion_model_cfg"].get("interleave_self_attention", False)),
            "use_alternate_vl_dit": bool(cfg.get("use_alternate_vl_dit", False)),
            "attend_text_every_n_blocks": int(cfg.get("attend_text_every_n_blocks", 2)),
            "add_pos_embed": bool(cfg.get("add_pos_embed", False)),
            "use_vlln": bool(cfg.get("use_vlln", False)),
            "num_timestep_buckets": int(cfg.get("num_timestep_buckets", 1000)),
            "action_dim": int(cfg.get("max_action_dim", 128)),
            "state_dim": int(cfg.get("max_state_dim", 128)),
            "backbone_dim": int(cfg.get("backbone_embedding_dim", 2048)),
            "input_embedding_dim": int(cfg.get("input_embedding_dim", 1536)),
        }

        weights["state_l1_w"] = get_tensor_np(store, "action_head.state_encoder.layer1.W")[emb]
        weights["state_l1_b"] = get_tensor_np(store, "action_head.state_encoder.layer1.b")[emb]
        weights["state_l2_w"] = get_tensor_np(store, "action_head.state_encoder.layer2.W")[emb]
        weights["state_l2_b"] = get_tensor_np(store, "action_head.state_encoder.layer2.b")[emb]

        weights["act_w1_w"] = get_tensor_np(store, "action_head.action_encoder.W1.W")[emb]
        weights["act_w1_b"] = get_tensor_np(store, "action_head.action_encoder.W1.b")[emb]
        weights["act_w2_w"] = get_tensor_np(store, "action_head.action_encoder.W2.W")[emb]
        weights["act_w2_b"] = get_tensor_np(store, "action_head.action_encoder.W2.b")[emb]
        weights["act_w3_w"] = get_tensor_np(store, "action_head.action_encoder.W3.W")[emb]
        weights["act_w3_b"] = get_tensor_np(store, "action_head.action_encoder.W3.b")[emb]

        weights["dec_l1_w"] = get_tensor_np(store, "action_head.action_decoder.layer1.W")[emb]
        weights["dec_l1_b"] = get_tensor_np(store, "action_head.action_decoder.layer1.b")[emb]
        weights["dec_l2_w"] = get_tensor_np(store, "action_head.action_decoder.layer2.W")[emb]
        weights["dec_l2_b"] = get_tensor_np(store, "action_head.action_decoder.layer2.b")[emb]

        weights["time_l1_w"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight")
        weights["time_l1_b"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_1.bias")
        weights["time_l2_w"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_2.weight")
        weights["time_l2_b"] = get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_2.bias")

        weights["proj_out_1_w"] = get_tensor_np(store, "action_head.model.proj_out_1.weight")
        weights["proj_out_1_b"] = get_tensor_np(store, "action_head.model.proj_out_1.bias")
        weights["proj_out_2_w"] = get_tensor_np(store, "action_head.model.proj_out_2.weight")
        weights["proj_out_2_b"] = get_tensor_np(store, "action_head.model.proj_out_2.bias")

        if weights["add_pos_embed"]:
            weights["pos_embedding"] = get_tensor_np(store, "action_head.position_embedding.weight")
        else:
            weights["pos_embedding"] = None

        if weights["use_vlln"]:
            weights["vlln_w"] = get_tensor_np(store, "action_head.vlln.weight")
            weights["vlln_b"] = get_tensor_np(store, "action_head.vlln.bias")
        else:
            weights["vlln_w"] = None
            weights["vlln_b"] = None

        blocks: list[dict[str, np.ndarray]] = []
        for i in range(num_layers):
            p = f"action_head.model.transformer_blocks.{i}"
            blocks.append(
                {
                    "norm1_linear_w": get_tensor_np(store, f"{p}.norm1.linear.weight"),
                    "norm1_linear_b": get_tensor_np(store, f"{p}.norm1.linear.bias"),
                    "attn_q_w": get_tensor_np(store, f"{p}.attn1.to_q.weight"),
                    "attn_q_b": get_tensor_np(store, f"{p}.attn1.to_q.bias"),
                    "attn_k_w": get_tensor_np(store, f"{p}.attn1.to_k.weight"),
                    "attn_k_b": get_tensor_np(store, f"{p}.attn1.to_k.bias"),
                    "attn_v_w": get_tensor_np(store, f"{p}.attn1.to_v.weight"),
                    "attn_v_b": get_tensor_np(store, f"{p}.attn1.to_v.bias"),
                    "attn_o_w": get_tensor_np(store, f"{p}.attn1.to_out.0.weight"),
                    "attn_o_b": get_tensor_np(store, f"{p}.attn1.to_out.0.bias"),
                    "ff_in_w": get_tensor_np(store, f"{p}.ff.net.0.proj.weight"),
                    "ff_in_b": get_tensor_np(store, f"{p}.ff.net.0.proj.bias"),
                    "ff_out_w": get_tensor_np(store, f"{p}.ff.net.2.weight"),
                    "ff_out_b": get_tensor_np(store, f"{p}.ff.net.2.bias"),
                }
            )
        weights["blocks"] = blocks

        return cls(device=device, weights=weights, cfg=cfg)

    def _weight_to_tt(self, name: str, weight_io: np.ndarray):
        if name not in self._tt_weight_cache:
            self._tt_weight_cache[name] = ttnn.from_torch(
                torch.from_numpy(weight_io).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        return self._tt_weight_cache[name]

    def _bias_to_tt(self, name: str, bias: np.ndarray):
        if name not in self._tt_bias_cache:
            self._tt_bias_cache[name] = ttnn.from_torch(
                torch.from_numpy(bias.reshape(1, -1)).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )
        return self._tt_bias_cache[name]

    def _ttnn_linear_2d(
        self, x2d: np.ndarray, weight_io: np.ndarray, bias: np.ndarray | None, *, op_name: str
    ) -> np.ndarray:
        in_dim = int(x2d.shape[1])
        w = weight_io
        if w.shape[0] != in_dim:
            if w.shape[1] == in_dim:
                w = w.T
            else:
                raise ValueError(
                    f"{op_name}: weight/input mismatch, x_in={in_dim}, weight_shape={w.shape}"
                )

        if self.device is None:
            raise RuntimeError(f"TTNN device unavailable at op {op_name}")

        m, _ = x2d.shape
        padded_m = ((m + 31) // 32) * 32
        if padded_m != m:
            x_work = np.zeros((padded_m, x2d.shape[1]), dtype=np.float32)
            x_work[:m, :] = x2d
        else:
            x_work = x2d

        try:
            tx = ttnn.from_torch(
                torch.from_numpy(x_work).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=self.device
            )
            tw = self._weight_to_tt(op_name, w)
            y = ttnn.matmul(tx, tw)
            if bias is not None:
                y = ttnn.add(y, self._bias_to_tt(f"{op_name}.b", bias))
            out = ttnn.to_torch(y).to(torch.float32).cpu().numpy()
            return out[:m, :]
        except Exception as exc:
            raise RuntimeError(f"TTNN op failed at {op_name}") from exc

    def _linear_lastdim(
        self, x: np.ndarray, weight_io: np.ndarray, bias: np.ndarray | None, *, op_name: str
    ) -> np.ndarray:
        x2d = x.reshape(-1, x.shape[-1]).astype(np.float32)
        y2d = self._ttnn_linear_2d(x2d, weight_io.astype(np.float32), bias, op_name=op_name)
        return y2d.reshape(*x.shape[:-1], y2d.shape[-1]).astype(np.float32)

    def _to_tt(self, x: np.ndarray, *, op_name: str) -> ttnn.Tensor:
        if self.device is None:
            raise RuntimeError(f"TTNN device unavailable at {op_name}")
        tx = torch.from_numpy(x)
        if tx.dtype in (torch.float16, torch.float32, torch.float64):
            tx = tx.to(torch.bfloat16)
        return ttnn.from_torch(tx, layout=ttnn.TILE_LAYOUT, device=self.device)

    def _to_np(self, x: ttnn.Tensor) -> np.ndarray:
        return ttnn.to_torch(x).to(torch.float32).cpu().numpy().astype(np.float32)

    def _tt_unary(self, x: np.ndarray, op_fn: Any, *, op_name: str) -> np.ndarray:
        tx = self._to_tt(x.astype(np.float32), op_name=op_name)
        ty = op_fn(tx)
        return self._to_np(ty)

    def _tt_binary(self, x: np.ndarray, y: np.ndarray, op_fn: Any, *, op_name: str) -> np.ndarray:
        tx = self._to_tt(x.astype(np.float32), op_name=f"{op_name}.lhs")
        ty = self._to_tt(y.astype(np.float32), op_name=f"{op_name}.rhs")
        tz = op_fn(tx, ty)
        return self._to_np(tz)

    def _tt_mul_scalar(self, x: np.ndarray, scalar: float, *, op_name: str) -> np.ndarray:
        tx = self._to_tt(x.astype(np.float32), op_name=op_name)
        ty = ttnn.multiply(tx, np.float32(scalar))
        return self._to_np(ty)

    def _tt_silu(self, x: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_unary(x, ttnn.silu, op_name=op_name)

    def _tt_relu(self, x: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_unary(x, ttnn.relu, op_name=op_name)

    def _tt_gelu(self, x: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_unary(x, ttnn.gelu, op_name=op_name)

    def _tt_add(self, x: np.ndarray, y: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_binary(x, y, ttnn.add, op_name=op_name)

    def _tt_mul(self, x: np.ndarray, y: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_binary(x, y, ttnn.multiply, op_name=op_name)

    def _tt_repeat(self, x: np.ndarray, repeats: list[int], *, op_name: str) -> np.ndarray:
        tx = self._to_tt(x.astype(np.float32), op_name=op_name)
        ty = ttnn.repeat(tx, repeats)
        return self._to_np(ty)

    def _tt_bcast_h(self, x: np.ndarray, y: np.ndarray, *, math_op: Any, op_name: str) -> np.ndarray:
        y_work = y.astype(np.float32)
        if y_work.ndim == x.ndim - 1:
            y_work = np.expand_dims(y_work, axis=-2)
        if y_work.ndim != x.ndim:
            raise ValueError(f"{op_name}: rhs rank mismatch lhs={x.ndim}, rhs={y_work.ndim}")
        if y_work.shape[-2] != 1:
            raise ValueError(f"{op_name}: rhs H dimension must be 1 for BcastOpDim.H, got {y_work.shape[-2]}")
        if y_work.shape[-1] != x.shape[-1]:
            raise ValueError(f"{op_name}: rhs W dimension must match lhs W ({x.shape[-1]}), got {y_work.shape[-1]}")
        if y_work.shape[:-2] != x.shape[:-2]:
            raise ValueError(
                f"{op_name}: leading dims must match for bcast, lhs={x.shape[:-2]}, rhs={y_work.shape[:-2]}"
            )
        tx = self._to_tt(x.astype(np.float32), op_name=f"{op_name}.lhs")
        ty = self._to_tt(y_work.astype(np.float32), op_name=f"{op_name}.rhs")
        tz = ttnn.bcast(tx, ty, math_op, ttnn.BcastOpDim.H)
        return self._to_np(tz)

    def _tt_bcast_h_add(self, x: np.ndarray, y: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_bcast_h(x, y, math_op=ttnn.BcastOpMath.ADD, op_name=op_name)

    def _tt_bcast_h_mul(self, x: np.ndarray, y: np.ndarray, *, op_name: str) -> np.ndarray:
        return self._tt_bcast_h(x, y, math_op=ttnn.BcastOpMath.MUL, op_name=op_name)

    def _tt_concat(self, xs: list[np.ndarray], *, dim: int, op_name: str) -> np.ndarray:
        tt_tensors = [self._to_tt(x.astype(np.float32), op_name=f"{op_name}.in{idx}") for idx, x in enumerate(xs)]
        y = ttnn.concat(tt_tensors, dim=dim)
        return self._to_np(y)

    def _tt_split_half(self, x: np.ndarray, *, dim: int, op_name: str) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[dim] % 2 != 0:
            raise ValueError(f"{op_name}: split dimension {dim} size {x.shape[dim]} is not divisible by 2")
        tx = self._to_tt(x.astype(np.float32), op_name=op_name)
        parts = ttnn.split(tx, x.shape[dim] // 2, dim=dim)
        if len(parts) != 2:
            raise RuntimeError(f"{op_name}: expected 2 chunks from split, got {len(parts)}")
        return self._to_np(parts[0]), self._to_np(parts[1])

    def _tt_layer_norm(self, x: np.ndarray, *, eps: float, op_name: str) -> np.ndarray:
        tx = self._to_tt(x.astype(np.float32), op_name=op_name)
        ty = ttnn.layer_norm(tx, epsilon=float(eps))
        return self._to_np(ty)

    def _mh_attention(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        *,
        attention_mask: np.ndarray | None,
    ) -> np.ndarray:
        batch, q_len, dim = q.shape
        k_len = k.shape[1]
        heads = int(self.w["num_heads"])
        head_dim = dim // heads

        if self.device is None:
            raise RuntimeError("TTNN device unavailable at attention")

        q_tt = self._to_tt(q.astype(np.float32), op_name="attn.q_in")
        k_tt = self._to_tt(k.astype(np.float32), op_name="attn.k_in")
        v_tt = self._to_tt(v.astype(np.float32), op_name="attn.v_in")

        q_tt = ttnn.reshape(q_tt, (batch, q_len, heads, head_dim))
        q_tt = ttnn.permute(q_tt, (0, 2, 1, 3))
        k_tt = ttnn.reshape(k_tt, (batch, k_len, heads, head_dim))
        k_tt = ttnn.permute(k_tt, (0, 2, 3, 1))
        v_tt = ttnn.reshape(v_tt, (batch, k_len, heads, head_dim))
        v_tt = ttnn.permute(v_tt, (0, 2, 1, 3))

        scores_tt = ttnn.matmul(q_tt, k_tt)
        scores_tt = ttnn.multiply(scores_tt, np.float32(1.0 / np.sqrt(float(head_dim))))
        if attention_mask is not None:
            keep = attention_mask.astype(bool)[:, None, None, :]
            bias = np.where(keep, np.float32(0.0), np.float32(-1e9))
            bias = bias.reshape(batch, 1, 1, k_len).astype(np.float32)
            if heads != 1 or q_len != 1:
                bias = self._tt_repeat(bias, [1, heads, q_len, 1], op_name="attn.mask_repeat")
            tbias = self._to_tt(bias, op_name="attn.mask_bias")
            scores_tt = ttnn.add(scores_tt, tbias)
        probs_tt = ttnn.softmax(scores_tt, dim=-1)
        ctx_tt = ttnn.matmul(probs_tt, v_tt)
        ctx_tt = ttnn.permute(ctx_tt, (0, 2, 1, 3))
        ctx_tt = ttnn.reshape(ctx_tt, (batch, q_len, dim))
        return self._to_np(ctx_tt)

    def _build_inputs(self, args: argparse.Namespace) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed + 1)
        batch = int(args.batch)
        action_h = int(args.action_horizon)

        state = rng.standard_normal((batch, 1, int(self.w["state_dim"])), dtype=np.float32)
        vl_embeds = rng.standard_normal(
            (batch, int(args.vl_seq_len), int(self.w["backbone_dim"])), dtype=np.float32
        )
        image_mask = np.zeros((batch, int(args.vl_seq_len)), dtype=bool)
        image_mask[:, ::2] = True
        backbone_attention_mask = np.ones((batch, int(args.vl_seq_len)), dtype=bool)
        actions = (
            torch.randn((batch, action_h, int(self.w["action_dim"])), dtype=torch.float32).cpu().numpy()
        )
        return {
            "state": state,
            "vl_embeds": vl_embeds,
            "image_mask": image_mask,
            "backbone_attention_mask": backbone_attention_mask,
            "actions": actions.astype(np.float32),
        }

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
        batch = actions.shape[0]

        t_bucket = np.full((batch,), int(t_discretized), dtype=np.int64)
        t_proj = timestep_embedding_np(t_bucket, embedding_dim=256)
        temb = self._linear_lastdim(t_proj, self.w["time_l1_w"].T, self.w["time_l1_b"], op_name="time_l1")
        temb = self._tt_silu(temb, op_name="time_silu")
        temb = self._linear_lastdim(temb, self.w["time_l2_w"].T, self.w["time_l2_b"], op_name="time_l2")

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

        if self.w["add_pos_embed"] and self.w["pos_embedding"] is not None:
            pos = self.w["pos_embedding"][:action_horizon][None, :, :].astype(np.float32)
            if action_features.shape[0] != 1:
                pos = self._tt_repeat(pos, [action_features.shape[0], 1, 1], op_name="action.pos_repeat_batch")
            action_features = self._tt_add(action_features, pos, op_name="action.add_pos")

        hidden_states = self._tt_concat([state_features, action_features], dim=1, op_name="sa.concat")
        block_hidden_states: list[np.ndarray] = [hidden_states.astype(np.float32)] if return_debug else []
        for i in range(int(self.w["num_layers"])):
            block = self.w["blocks"][i]

            norm_hidden = self._tt_layer_norm(hidden_states, eps=float(self.w["norm_eps"]), op_name=f"block{i}.ln1")
            ada = self._linear_lastdim(
                self._tt_silu(temb, op_name=f"block{i}.ada_silu"),
                block["norm1_linear_w"].T,
                block["norm1_linear_b"],
                op_name=f"block{i}.norm1_linear",
            )
            scale, shift = self._tt_split_half(ada, dim=ada.ndim - 1, op_name=f"block{i}.ada_split")
            scale_plus_one = self._tt_add(scale, np.ones_like(scale, dtype=np.float32), op_name=f"block{i}.scale_plus_one")
            norm_hidden = self._tt_bcast_h_mul(norm_hidden, scale_plus_one, op_name=f"block{i}.ada_mul")
            norm_hidden = self._tt_bcast_h_add(norm_hidden, shift, op_name=f"block{i}.ada_add")

            cross_mask: np.ndarray | None = None
            if bool(self.w["interleave_self_attention"]) and (i % 2 == 1):
                enc = norm_hidden
            else:
                enc = vl_embeds
                if bool(self.w["use_alternate_vl_dit"]):
                    if i % (2 * int(self.w["attend_text_every_n_blocks"])) == 0:
                        cross_mask = (~image_mask) & backbone_attention_mask
                    else:
                        cross_mask = image_mask & backbone_attention_mask

            q = self._linear_lastdim(norm_hidden, block["attn_q_w"].T, block["attn_q_b"], op_name=f"block{i}.q")
            k = self._linear_lastdim(enc, block["attn_k_w"].T, block["attn_k_b"], op_name=f"block{i}.k")
            v = self._linear_lastdim(enc, block["attn_v_w"].T, block["attn_v_b"], op_name=f"block{i}.v")
            ctx = self._mh_attention(q, k, v, attention_mask=cross_mask)
            attn_out = self._linear_lastdim(
                ctx, block["attn_o_w"].T, block["attn_o_b"], op_name=f"block{i}.o"
            )
            hidden_states = self._tt_add(hidden_states, attn_out, op_name=f"block{i}.resid_attn")

            ff_in = self._linear_lastdim(
                self._tt_layer_norm(hidden_states, eps=float(self.w["norm_eps"]), op_name=f"block{i}.ln2"),
                block["ff_in_w"].T,
                block["ff_in_b"],
                op_name=f"block{i}.ff_in",
            )
            ff_out_in_dim = int(block["ff_out_w"].shape[1])
            if ff_in.shape[-1] == ff_out_in_dim:
                ff_act = self._tt_gelu(ff_in, op_name=f"block{i}.ff_gelu")
            elif ff_in.shape[-1] == 2 * ff_out_in_dim:
                ff_x, ff_gate = self._tt_split_half(ff_in, dim=ff_in.ndim - 1, op_name=f"block{i}.ff_split")
                ff_act = self._tt_mul(
                    ff_x,
                    self._tt_gelu(ff_gate, op_name=f"block{i}.ff_gate_gelu"),
                    op_name=f"block{i}.ff_gate_mul",
                )
            else:
                raise ValueError(
                    f"block{i}.ff: unsupported dimensions ff_in={ff_in.shape[-1]}, "
                    f"ff_out_in_dim={ff_out_in_dim}"
                )
            ff_out = self._linear_lastdim(
                ff_act, block["ff_out_w"].T, block["ff_out_b"], op_name=f"block{i}.ff_out"
            )
            hidden_states = self._tt_add(hidden_states, ff_out, op_name=f"block{i}.resid_ff")
            if return_debug:
                block_hidden_states.append(hidden_states.astype(np.float32))

        out_norm = self._tt_layer_norm(hidden_states, eps=1e-6, op_name="out_norm")
        shift_scale = self._linear_lastdim(
            self._tt_silu(temb, op_name="proj_out_1_silu"),
            self.w["proj_out_1_w"].T,
            self.w["proj_out_1_b"],
            op_name="proj_out_1",
        )
        shift, scale = self._tt_split_half(shift_scale, dim=shift_scale.ndim - 1, op_name="proj_out_1_split")
        scale_plus_one = self._tt_add(scale, np.ones_like(scale, dtype=np.float32), op_name="proj_out_scale_plus_one")
        out_norm = self._tt_bcast_h_mul(out_norm, scale_plus_one, op_name="proj_out_affine_mul")
        out_norm = self._tt_bcast_h_add(out_norm, shift, op_name="proj_out_affine_add")
        model_output = self._linear_lastdim(
            out_norm, self.w["proj_out_2_w"].T, self.w["proj_out_2_b"], op_name="proj_out_2"
        )

        d1 = self._linear_lastdim(model_output, self.w["dec_l1_w"], self.w["dec_l1_b"], op_name="dec_l1")
        d1 = self._tt_relu(d1, op_name="dec_relu")
        pred = self._linear_lastdim(d1, self.w["dec_l2_w"], self.w["dec_l2_b"], op_name="dec_l2")
        pred_velocity = pred[:, -action_horizon:, :]
        step_debug: dict[str, np.ndarray] | None = None
        if return_debug:
            step_debug = {
                "timesteps": t_bucket.astype(np.int64),
                "temb": temb.astype(np.float32),
                "action_features": action_features.astype(np.float32),
                "sa_embs": self._tt_concat([state_features, action_features], dim=1, op_name="debug.sa_embs"),
                "model_output": model_output.astype(np.float32),
                "pred_velocity": pred_velocity.astype(np.float32),
            }
            for i, hs in enumerate(block_hidden_states):
                step_debug[f"block_hidden_{i:02d}"] = hs
        return pred_velocity.astype(np.float32), model_output.astype(np.float32), step_debug

    def run_denoise_loop(
        self,
        args: argparse.Namespace,
        *,
        shared_inputs: dict[str, np.ndarray] | None = None,
        return_debug: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        if shared_inputs is None:
            inputs = self._build_inputs(args)
        else:
            inputs = _clone_shared_inputs(shared_inputs)
        state = inputs["state"]
        vl_embeds = inputs["vl_embeds"]
        image_mask = inputs["image_mask"]
        backbone_attention_mask = inputs["backbone_attention_mask"]
        actions = inputs["actions"]
        action_horizon = int(args.action_horizon)
        debug: dict[str, np.ndarray] = {}
        if return_debug:
            debug["input/state"] = state.astype(np.float32)
            debug["input/vl_embeds"] = vl_embeds.astype(np.float32)
            debug["input/image_mask"] = image_mask
            debug["input/backbone_attention_mask"] = backbone_attention_mask
            debug["input/actions_init"] = actions.astype(np.float32)
            if "embodiment_id" in inputs:
                debug["input/embodiment_id"] = inputs["embodiment_id"].astype(np.int64)

        if self.w["use_vlln"] and self.w["vlln_w"] is not None and self.w["vlln_b"] is not None:
            vl_embeds = self._tt_layer_norm(vl_embeds, eps=1e-5, op_name="vlln.norm")
            vlln_w = self.w["vlln_w"].reshape(1, -1).astype(np.float32)
            vlln_b = self.w["vlln_b"].reshape(1, -1).astype(np.float32)
            if vl_embeds.shape[0] != 1:
                vlln_w = self._tt_repeat(vlln_w, [vl_embeds.shape[0], 1], op_name="vlln.w_repeat_batch")
                vlln_b = self._tt_repeat(vlln_b, [vl_embeds.shape[0], 1], op_name="vlln.b_repeat_batch")
            vl_embeds = self._tt_bcast_h_mul(vl_embeds, vlln_w, op_name="vlln.mul_weight")
            vl_embeds = self._tt_bcast_h_add(vl_embeds, vlln_b, op_name="vlln.add_bias")

        state_features = self._linear_lastdim(state, self.w["state_l1_w"], self.w["state_l1_b"], op_name="state_l1")
        state_features = self._tt_relu(state_features, op_name="state_relu")
        state_features = self._linear_lastdim(
            state_features, self.w["state_l2_w"], self.w["state_l2_b"], op_name="state_l2"
        )
        if return_debug:
            debug["encoded/vl_embeds"] = vl_embeds.astype(np.float32)
            debug["encoded/state_features"] = state_features.astype(np.float32)

        dt = np.float32(1.0 / float(args.num_inference_timesteps))
        for t in range(int(args.num_inference_timesteps)):
            t_cont = float(t) / float(args.num_inference_timesteps)
            t_discretized = int(t_cont * int(self.w["num_timestep_buckets"]))
            pred_velocity, _, step_debug = self.run_step(
                actions=actions,
                state_features=state_features,
                vl_embeds=vl_embeds,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                t_discretized=t_discretized,
                action_horizon=action_horizon,
                return_debug=return_debug,
            )
            delta = self._tt_mul_scalar(pred_velocity, float(dt), op_name=f"step{t}.delta_scale")
            actions = self._tt_add(actions, delta, op_name=f"step{t}.actions_update")
            print(
                f"[ttnn_full] step={t} t_discretized={t_discretized} actions_shape={actions.shape}"
            )
            if return_debug and step_debug is not None:
                step_prefix = f"step_{t:02d}"
                for k, v in step_debug.items():
                    debug[f"{step_prefix}/{k}"] = v
                debug[f"{step_prefix}/actions"] = actions.astype(np.float32)

        out = actions.astype(np.float32)
        if return_debug:
            debug["output/actions"] = out
            return out, debug
        return out


def run_ttnn_full_runtime(
    store: Any,
    args: argparse.Namespace,
    *,
    shared_inputs: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if shared_inputs is None:
        raise ValueError("run_ttnn_full_runtime requires shared_inputs")
    inputs = _clone_shared_inputs(shared_inputs)
    tt_device = None
    with ExitStack() as hook_stack:
        if args.log_ttnn_ops:
            if getattr(ttnn.CONFIG, "enable_fast_runtime_mode", False):
                print(
                    "[ttnn-op-hook] Hooks are disabled in fast runtime mode. "
                    "Set TTNN_CONFIG_OVERRIDES='{\"enable_fast_runtime_mode\": false}' to enable op timing logs."
                )
            op_start_times: dict[int, int] = {}
            hook_stack.enter_context(
                ttnn.register_pre_operation_hook(_make_ttnn_pre_operation_hook(op_start_times=op_start_times))
            )
            hook_stack.enter_context(
                ttnn.register_post_operation_hook(_make_ttnn_post_operation_hook(op_start_times))
            )
        else:
            hook_stack.enter_context(nullcontext())
        try:
            tt_device = ttnn.open_device(device_id=args.tt_device_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to open TT device {args.tt_device_id}") from exc
        try:
            # System1 TTNN backbone compute from prepared system1_input/* tensors.
            system1_inputs = _build_system1_inputs_from_shared(inputs)
            system1_runtime = TtGr00tSystem1BackboneRuntime(args=args, device=tt_device)
            system1_output = system1_runtime.run(system1_inputs)
            inputs["vl_embeds"] = system1_output["backbone_features"].astype(np.float32)
            inputs["backbone_attention_mask"] = system1_output["backbone_attention_mask"].astype(bool)
            inputs["image_mask"] = system1_output["image_mask"].astype(bool)

            runtime = TtGr00tActionHeadRuntime.load_weights(store=store, args=args, device=tt_device)
            ttnn_actions, ttnn_debug = runtime.run_denoise_loop(
                args,
                shared_inputs=inputs,
                return_debug=True,
            )
            ttnn_debug["system1_output/backbone_features"] = inputs["vl_embeds"].astype(np.float32)
            ttnn_debug["system1_output/backbone_attention_mask"] = inputs["backbone_attention_mask"].astype(bool)
            ttnn_debug["system1_output/image_mask"] = inputs["image_mask"].astype(bool)
        finally:
            if tt_device is not None:
                ttnn.close_device(tt_device)
    ttnn_debug["meta/used_ttnn_device"] = np.asarray([1], dtype=np.int64)
    return ttnn_actions, ttnn_debug
