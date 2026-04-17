#!/usr/bin/env python3
"""TTNN-native Isaac-GR00T migration demo.

Graph modes:
- `cpu_ref`: CPU reference compute for the migrated graph path.
- `ttnn_full`: TTNN runtime for the migrated graph path, compared to CPU reference.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
from pathlib import Path
import sys
import types
from typing import Any

import numpy as np
import torch
import ttnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TTNN-native Isaac-GR00T migration demo")
    parser.add_argument(
        "--model-dir",
        default="/workspaces/tensix/tt-metal/models/demos/isaac_gr00t/GR00T-N1.6-3B",
        help="Local GR00T model directory (HF-style with safetensors/bin)",
    )
    parser.add_argument("--tt-device-id", type=int, default=0, help="TTNN device id")
    parser.add_argument(
        "--strict-ttnn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require TTNN device execution (no CPU fallback path)",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--num-inference-timesteps", type=int, default=4)
    parser.add_argument("--vl-seq-len", type=int, default=32)
    parser.add_argument(
        "--graph-mode",
        choices=("cpu_ref", "ttnn_full"),
        default="ttnn_full",
        help="`cpu_ref`: CPU reference compute; `ttnn_full`: TTNN runtime compare against CPU reference",
    )
    parser.add_argument(
        "--isaac-gr00t-root",
        default="/workspaces/tensix/Isaac-GR00T",
        help="Path to Isaac-GR00T source tree used for full-graph execution",
    )
    parser.add_argument("--embodiment-id", type=int, default=0, help="Category index for category-specific weights")
    parser.add_argument("--list-keys", action="store_true", help="List candidate linear keys and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-source",
        choices=("synthetic", "gr1"),
        default="gr1",
        help="Input source for migrated graph runs: `synthetic` random tensors, or `gr1` real GR1 dataset sample",
    )
    parser.add_argument(
        "--dataset-path",
        default="/workspaces/tensix/Isaac-GR00T/demo_data/gr1.PickNPlace",
        help="LeRobot dataset path used when --input-source gr1",
    )
    parser.add_argument(
        "--video-backend",
        default="opencv",
        choices=("ffmpeg", "opencv", "decord", "torchcodec", "torchvision_av"),
        help="Video decoding backend used when --input-source gr1",
    )
    parser.add_argument("--traj-id", type=int, default=0, help="Trajectory index for --input-source gr1")
    parser.add_argument("--step", type=int, default=0, help="Step index within trajectory for --input-source gr1")
    parser.add_argument(
        "--task-text",
        default="",
        help="Optional language override for --input-source gr1",
    )
    parser.add_argument(
        "--embodiment-tag",
        default="GR1",
        help="EmbodimentTag enum name used when --input-source gr1 (e.g., GR1)",
    )
    parser.add_argument(
        "--ttnn-num-layers",
        type=int,
        default=0,
        help="Optional limit for migrated TTNN action-head transformer blocks (0 = use all config layers)",
    )
    parser.add_argument(
        "--dump-dir",
        default="",
        help="Optional directory to dump input/intermediate/final tensors (.npy) for comparison",
    )
    parser.add_argument(
        "--dump-assembly-dir",
        default="",
        help="Optional directory to dump CPU assembly tensors (processor/collate/prepare/backbone) as .npy",
    )
    parser.add_argument(
        "--load-assembly-dir",
        default="",
        help="Directory containing dumped assembly tensors (from --dump-assembly-dir) for replay/compare",
    )
    parser.add_argument(
        "--backbone-only-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run isolated CPU System1 backbone replay from --load-assembly-dir and compare outputs",
    )
    parser.add_argument(
        "--backbone-proj-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run isolated System1 q_proj op compare (PyTorch vs TTNN matmul) from --load-assembly-dir",
    )
    parser.add_argument(
        "--backbone-proj-layer",
        type=int,
        default=12,
        help="Language-model layer index for --backbone-proj-compare",
    )
    parser.add_argument(
        "--backbone-phaseb-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Phase-B linear-stack replay compare for System1 (selected layers/linears)",
    )
    parser.add_argument(
        "--backbone-layer-start",
        type=int,
        default=12,
        help="Start layer index (inclusive) for --backbone-phaseb-compare",
    )
    parser.add_argument(
        "--backbone-layer-end",
        type=int,
        default=15,
        help="End layer index (inclusive) for --backbone-phaseb-compare",
    )
    parser.add_argument(
        "--backbone-linear-set",
        choices=("attn", "mlp", "all"),
        default="all",
        help="Linear groups to migrate in --backbone-phaseb-compare",
    )
    parser.add_argument(
        "--system2-encoder-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Phase-C encoder parity report (state_features/action_features/sa_embs)",
    )
    parser.add_argument(
        "--system2-block-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Phase-C diffusion block hidden-state parity report for --phasec-step",
    )
    parser.add_argument(
        "--system2-trajectory-compare",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Phase-C denoise trajectory parity report over step_XX/actions + output/actions",
    )
    parser.add_argument(
        "--phasec-step",
        type=int,
        default=0,
        help="Diffusion step index used by --system2-block-compare",
    )
    parser.add_argument(
        "--save-shared-inputs-dir",
        default="",
        help="Optional directory to save deterministic shared inputs (.npy + meta.json)",
    )
    parser.add_argument(
        "--load-shared-inputs-dir",
        default="",
        help="Optional directory to load deterministic shared inputs (.npy + meta.json)",
    )
    parser.add_argument(
        "--check-input-determinism",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build GR1 shared inputs twice and fail if tensors differ",
    )
    parser.add_argument(
        "--dump-max-entries",
        type=int,
        default=40,
        help="Max number of tensor-diff entries printed in compare summary",
    )
    parser.add_argument(
        "--log-ttnn-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log each TTNN op call with input tensor shapes",
    )
    parser.add_argument("--save-json", default="", help="Optional path to save action JSON")
    return parser.parse_args()


def action_to_json(action: np.ndarray) -> dict[str, Any]:
    return {"action": action.tolist()}


def _load_model_cfg(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.model_dir).resolve() / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in model directory: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def build_shared_action_head_inputs(
    args: argparse.Namespace, *, state_dim: int, backbone_dim: int, action_dim: int
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed + 1)

    batch = int(args.batch)
    action_horizon = int(args.action_horizon)
    vl_seq_len = int(args.vl_seq_len)

    state = rng.standard_normal((batch, 1, state_dim), dtype=np.float32)
    vl_embeds = rng.standard_normal((batch, vl_seq_len, backbone_dim), dtype=np.float32)
    image_mask = np.zeros((batch, vl_seq_len), dtype=bool)
    image_mask[:, ::2] = True
    backbone_attention_mask = np.ones((batch, vl_seq_len), dtype=bool)
    actions = torch.randn((batch, action_horizon, action_dim), dtype=torch.float32).cpu().numpy()
    embodiment_id = np.full((batch,), int(args.embodiment_id), dtype=np.int64)
    return {
        "state": state.astype(np.float32),
        "vl_embeds": vl_embeds.astype(np.float32),
        "image_mask": image_mask,
        "backbone_attention_mask": backbone_attention_mask,
        "actions": actions.astype(np.float32),
        "embodiment_id": embodiment_id,
    }


def _clone_shared_inputs(shared_inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.array(v, copy=True) for k, v in shared_inputs.items()}


def _shape_and_dtype(value: Any) -> tuple[str, str]:
    shape = tuple(value.shape) if hasattr(value, "shape") else "(n/a)"
    dtype = str(value.dtype) if hasattr(value, "dtype") else type(value).__name__
    return str(shape), dtype


def _shape_repr(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return "(n/a)"
    try:
        return str(tuple(int(dim) for dim in shape))
    except Exception:
        try:
            return str(tuple(shape))
        except Exception:
            return str(shape)


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


def _make_ttnn_pre_operation_hook():
    state = {"op_index": 0}

    def _hook(operation: Any, op_args: tuple[Any, ...], op_kwargs: dict[str, Any]) -> None:
        state["op_index"] += 1
        op_name = getattr(operation, "python_fully_qualified_name", str(operation))
        tensor_inputs: list[str] = []
        for idx, value in enumerate(op_args):
            _collect_op_tensor_inputs(value, f"arg{idx}", tensor_inputs)
        for key in sorted(op_kwargs.keys()):
            _collect_op_tensor_inputs(op_kwargs[key], f"kw:{key}", tensor_inputs)
        if tensor_inputs:
            print(f"[ttnn-op] #{state['op_index']} {op_name} | " + ", ".join(tensor_inputs))
        else:
            print(f"[ttnn-op] #{state['op_index']} {op_name} | no tensor inputs")

    return _hook


def _flatten_tensor_tree(prefix: str, value: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            tensor = tensor.to(torch.float32)
        out[prefix] = tensor.numpy()
        return
    if isinstance(value, np.ndarray):
        out[prefix] = np.array(value, copy=True)
        return
    if value is None:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_tensor_tree(next_prefix, item, out)
        return
    if hasattr(value, "items") and callable(getattr(value, "items")):
        for key, item in value.items():
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_tensor_tree(next_prefix, item, out)
        return
    if isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}/{idx}" if prefix else str(idx)
            _flatten_tensor_tree(next_prefix, item, out)
        return
    if isinstance(value, (bool, int, float, np.number, np.bool_)):
        out[prefix] = np.asarray(value)
        return


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_named_tensors(out_dir: Path, tensors: dict[str, np.ndarray]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in sorted(tensors.items()):
        parts = [p for p in key.split("/") if p]
        tensor_base = out_dir.joinpath(*parts)
        tensor_path = Path(f"{tensor_base.as_posix()}.npy")
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(tensor_path, value)


def _load_named_tensors(in_dir: Path) -> dict[str, np.ndarray]:
    if not in_dir.exists():
        raise FileNotFoundError(f"Input tensor directory not found: {in_dir}")
    tensors: dict[str, np.ndarray] = {}
    for npy_path in sorted(in_dir.rglob("*.npy")):
        rel = npy_path.relative_to(in_dir).as_posix()
        key = rel[:-4] if rel.endswith(".npy") else rel
        tensors[key] = np.load(npy_path, allow_pickle=False)
    return tensors


def _require_shared_input_keys(shared_inputs: dict[str, np.ndarray]) -> None:
    required = {
        "state",
        "vl_embeds",
        "image_mask",
        "backbone_attention_mask",
        "actions",
        "embodiment_id",
    }
    missing = sorted(required - set(shared_inputs.keys()))
    if missing:
        raise KeyError(f"Missing required shared-input tensors: {missing}")


def _save_shared_inputs(
    out_dir: Path,
    shared_inputs: dict[str, np.ndarray],
    metadata: dict[str, Any] | None = None,
) -> None:
    _save_named_tensors(out_dir, shared_inputs)
    if metadata is not None:
        _write_json(out_dir / "meta.json", metadata)


def _load_shared_inputs(in_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any] | None]:
    shared_inputs = _load_named_tensors(in_dir)
    _require_shared_input_keys(shared_inputs)
    meta_path = in_dir / "meta.json"
    metadata = None
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return shared_inputs, metadata


def _max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
    return 0.0 if np.array_equal(a, b) else float("inf")


def _assert_shared_inputs_deterministic(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> tuple[str, float]:
    keys = sorted(set(first.keys()) | set(second.keys()))
    worst_key = ""
    worst_diff = 0.0
    for key in keys:
        if key not in first or key not in second:
            raise RuntimeError(f"Determinism check failed: key mismatch at {key}")
        diff = _max_abs_diff(first[key], second[key])
        if diff > worst_diff:
            worst_diff = diff
            worst_key = key
    if worst_diff != 0.0:
        raise RuntimeError(
            f"Determinism check failed: max_abs_diff={worst_diff:.6f} at key={worst_key}"
        )
    return worst_key, worst_diff


def _resolve_assembly_tensor_dir(load_assembly_dir: str) -> Path:
    base = Path(load_assembly_dir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Assembly directory not found: {base}")
    assembly_dir = base / "assembly"
    if assembly_dir.exists():
        return assembly_dir
    return base


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


def _load_backbone_inputs_outputs_from_assembly(
    load_assembly_dir: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any] | None]:
    assembly_dir = _resolve_assembly_tensor_dir(load_assembly_dir)
    flat_tensors = _load_named_tensors(assembly_dir)
    prepared_system1 = _extract_prefixed_tensors(flat_tensors, "prepared/system1")
    system1_output = _extract_prefixed_tensors(flat_tensors, "system1_output")
    if not prepared_system1:
        raise KeyError(
            f"No prepared/system1 tensors found under {assembly_dir}. "
            "Generate assembly dump first with --dump-assembly-dir."
        )
    if not system1_output:
        raise KeyError(
            f"No system1_output tensors found under {assembly_dir}. "
            "Generate assembly dump first with --dump-assembly-dir."
        )

    meta_path = assembly_dir / "meta.json"
    metadata = None
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return prepared_system1, system1_output, metadata


def _to_torch_tensor(name: str, value: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(value)
    if name in {"input_ids", "attention_mask", "image_sizes", "embodiment_id"}:
        return tensor.to(torch.long)
    if tensor.dtype in (torch.float16, torch.float32, torch.float64):
        return tensor.to(torch.float32)
    return tensor


def _build_backbone_inputs_from_prepared(
    prepared_system1: dict[str, np.ndarray],
) -> dict[str, Any]:
    backbone_inputs: dict[str, Any] = {}
    pixel_values: list[tuple[int, torch.Tensor]] = []
    for key, value in prepared_system1.items():
        if key.startswith("pixel_values/"):
            idx_str = key.split("/", 1)[1]
            if not idx_str.isdigit():
                continue
            pixel_values.append((int(idx_str), _to_torch_tensor("pixel_values", value)))
            continue
        if key == "pixel_values":
            continue
        backbone_inputs[key] = _to_torch_tensor(key, value)

    if pixel_values:
        pixel_values.sort(key=lambda x: x[0])
        backbone_inputs["pixel_values"] = [t for _, t in pixel_values]
    return backbone_inputs


def _load_policy_for_backbone(
    args: argparse.Namespace,
    metadata: dict[str, Any] | None,
):
    embodiment_tag = args.embodiment_tag
    if metadata is not None and "embodiment_tag" in metadata:
        embodiment_tag = str(metadata["embodiment_tag"])

    isaac_root = Path(args.isaac_gr00t_root).resolve()
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    embodiment = EmbodimentTag[embodiment_tag]
    policy = Gr00tPolicy(
        embodiment_tag=embodiment,
        model_path=args.model_dir,
        device="cpu",
        strict=False,
    )
    return policy, embodiment


def run_cpu_backbone_replay_compare(args: argparse.Namespace) -> None:
    if not args.load_assembly_dir:
        raise ValueError("--backbone-only-compare requires --load-assembly-dir")

    prepared_system1, system1_output_expected, metadata = _load_backbone_inputs_outputs_from_assembly(
        args.load_assembly_dir
    )
    backbone_inputs = _build_backbone_inputs_from_prepared(prepared_system1)

    policy, _ = _load_policy_for_backbone(args, metadata)

    with torch.no_grad():
        backbone_output = policy.model.backbone(backbone_inputs)
        backbone_output = policy.model.action_head.process_backbone_output(backbone_output)

    system1_output_actual: dict[str, np.ndarray] = {
        "backbone_features": backbone_output.backbone_features.detach().cpu().to(torch.float32).numpy(),
        "backbone_attention_mask": backbone_output.backbone_attention_mask.detach().cpu().numpy(),
        "image_mask": backbone_output.image_mask.detach().cpu().numpy(),
    }

    worst_diff, worst_key, diff_lines = compare_tensor_maps(
        system1_output_actual, system1_output_expected, max_entries=int(args.dump_max_entries)
    )

    print("=== System1 Backbone Replay Compare (CPU) ===")
    print(f"Assembly source: {_resolve_assembly_tensor_dir(args.load_assembly_dir)}")
    if metadata is not None:
        print(
            "Case: "
            f"dataset={metadata.get('dataset_path', 'n/a')}, "
            f"traj={metadata.get('trajectory_id', 'n/a')}, step={metadata.get('step', 'n/a')}"
        )
    print(f"Worst tensor diff: {worst_diff:.6f} at key: {worst_key or 'n/a'}")
    print("Tensor diff summary:")
    for line in diff_lines:
        print(f"  - {line}")


def _ttnn_linear_compare(
    x2d: np.ndarray,
    weight_io: np.ndarray,
    *,
    bias: np.ndarray | None,
    args: argparse.Namespace,
    tt_device: Any | None = None,
    use_ttnn: bool = True,
) -> tuple[np.ndarray, bool]:
    m, _ = x2d.shape
    padded_m = ((m + 31) // 32) * 32
    if padded_m != m:
        x_work = np.zeros((padded_m, x2d.shape[1]), dtype=np.float32)
        x_work[:m, :] = x2d
    else:
        x_work = x2d.astype(np.float32)

    used_ttnn = bool(use_ttnn and tt_device is not None)
    if used_ttnn:
        try:
            tx = ttnn.from_torch(
                torch.from_numpy(x_work).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
            )
            tw = ttnn.from_torch(
                torch.from_numpy(weight_io.astype(np.float32)).to(torch.bfloat16),
                layout=ttnn.TILE_LAYOUT,
                device=tt_device,
            )
            y = ttnn.matmul(tx, tw)
            if bias is not None:
                tb = ttnn.from_torch(
                    torch.from_numpy(bias.reshape(1, -1).astype(np.float32)).to(torch.bfloat16),
                    layout=ttnn.TILE_LAYOUT,
                    device=tt_device,
                )
                y = ttnn.add(y, tb)
            out = ttnn.to_torch(y).to(torch.float32).cpu().numpy()[:m, :]
            return out.astype(np.float32), used_ttnn
        except Exception as exc:
            if args.strict_ttnn:
                raise RuntimeError("TTNN linear compare failed and strict_ttnn=True") from exc
            used_ttnn = False
    x_t = torch.from_numpy(x2d.astype(np.float32)).to(torch.bfloat16)
    w_t = torch.from_numpy(weight_io.astype(np.float32)).to(torch.bfloat16)
    y_t = torch.matmul(x_t, w_t).to(torch.float32)
    y = y_t.cpu().numpy()
    if bias is not None:
        y = y + bias.astype(np.float32)
    return y.astype(np.float32), used_ttnn


def run_backbone_q_proj_compare(args: argparse.Namespace) -> None:
    if not args.load_assembly_dir:
        raise ValueError("--backbone-proj-compare requires --load-assembly-dir")
    prepared_system1, _, metadata = _load_backbone_inputs_outputs_from_assembly(args.load_assembly_dir)
    backbone_inputs = _build_backbone_inputs_from_prepared(prepared_system1)
    policy, _ = _load_policy_for_backbone(args, metadata)

    try:
        layers = policy.model.backbone.model.language_model.model.layers
    except Exception as exc:
        raise RuntimeError("Unable to locate language-model layers in backbone") from exc

    layer_idx = int(args.backbone_proj_layer)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(f"--backbone-proj-layer {layer_idx} out of range [0, {len(layers)-1}]")
    q_proj = layers[layer_idx].self_attn.q_proj

    captured: dict[str, torch.Tensor] = {}

    def _pre_hook(_module, module_inputs):
        x = module_inputs[0]
        captured["x_raw"] = x.detach().cpu()

    def _fwd_hook(_module, _module_inputs, module_output):
        captured["y_ref_raw"] = module_output.detach().cpu()

    pre_handle = q_proj.register_forward_pre_hook(_pre_hook)
    fwd_handle = q_proj.register_forward_hook(_fwd_hook)
    try:
        with torch.no_grad():
            _ = policy.model.backbone(backbone_inputs)
    finally:
        pre_handle.remove()
        fwd_handle.remove()

    if "x_raw" not in captured or "y_ref_raw" not in captured:
        raise RuntimeError("Failed to capture q_proj input/output from backbone replay")

    x_raw = captured["x_raw"]
    y_ref_raw = captured["y_ref_raw"]
    x_np = x_raw.to(torch.float32).numpy().astype(np.float32)
    y_ref_np = y_ref_raw.to(torch.float32).numpy().astype(np.float32)

    weight = q_proj.weight.detach().cpu().to(torch.float32)
    bias_t = q_proj.bias.detach().cpu().to(torch.float32) if q_proj.bias is not None else None
    weight_io = weight.T.numpy().astype(np.float32)
    bias_np = bias_t.numpy().astype(np.float32) if bias_t is not None else None

    x2d = x_np.reshape(-1, x_np.shape[-1]).astype(np.float32)
    x2d_bf16_t = torch.from_numpy(x2d).to(torch.bfloat16)
    weight_io_bf16_t = torch.from_numpy(weight_io).to(torch.bfloat16)
    y2d_manual_t = torch.matmul(x2d_bf16_t, weight_io_bf16_t).to(torch.float32)
    y2d_manual = y2d_manual_t.cpu().numpy()
    if bias_np is not None:
        y2d_manual = y2d_manual + bias_np
    y_manual_np = y2d_manual.reshape(y_ref_np.shape).astype(np.float32)
    manual_max_abs = float(np.max(np.abs(y_manual_np - y_ref_np)))

    tt_device = None
    use_ttnn = True
    try:
        try:
            tt_device = ttnn.open_device(device_id=args.tt_device_id)
        except Exception as exc:
            if args.strict_ttnn:
                raise RuntimeError("Unable to open TT device for backbone proj compare") from exc
            use_ttnn = False
        y2d_ttnn, used_ttnn = _ttnn_linear_compare(
            x2d,
            weight_io,
            bias=bias_np,
            args=args,
            tt_device=tt_device,
            use_ttnn=use_ttnn,
        )
    finally:
        if tt_device is not None:
            ttnn.close_device(tt_device)
    y_ttnn_np = y2d_ttnn.reshape(y_ref_np.shape).astype(np.float32)
    ttnn_max_abs = float(np.max(np.abs(y_ttnn_np - y_ref_np)))

    print("=== System1 q_proj Op Compare ===")
    print(f"Assembly source: {_resolve_assembly_tensor_dir(args.load_assembly_dir)}")
    print(f"Layer index: {layer_idx}")
    print(f"Input shape: {tuple(x_np.shape)}")
    print(f"Weight shape: {tuple(weight.shape)}")
    print(f"Output shape: {tuple(y_ref_np.shape)}")
    print(f"PyTorch BF16-manual max_abs_diff: {manual_max_abs:.6f}")
    print(
        f"{'TTNN' if used_ttnn else 'CPU-fallback'}-vs-PyTorch max_abs_diff: {ttnn_max_abs:.6f}"
    )
    if not used_ttnn:
        print("TTNN execution unavailable; used CPU fallback because --no-strict-ttnn was set.")

    if args.dump_dir:
        dump_root = Path(args.dump_dir).resolve()
        dump_tensor_map(
            dump_root,
            "backbone_q_proj_compare",
            {
                "input/x": x_np,
                "weight/q_proj_weight": weight.numpy().astype(np.float32),
                "output/y_ref": y_ref_np,
                "output/y_manual": y_manual_np,
                "output/y_ttnn_or_fallback": y_ttnn_np,
            },
        )
        print(f"Dumped q_proj compare tensors to: {dump_root / 'backbone_q_proj_compare'}")


def _get_module_by_path(root: torch.nn.Module, module_path: str) -> torch.nn.Module:
    module: Any = root
    for segment in module_path.split("."):
        if segment.isdigit():
            module = module[int(segment)]
        else:
            module = getattr(module, segment)
    if not isinstance(module, torch.nn.Module):
        raise TypeError(f"Resolved object is not a torch.nn.Module for path: {module_path}")
    return module


def _build_backbone_linear_paths(
    *, layer_start: int, layer_end: int, linear_set: str
) -> list[str]:
    if layer_end < layer_start:
        raise ValueError(
            f"Invalid layer range: start={layer_start}, end={layer_end} (end must be >= start)"
        )
    paths: list[str] = []
    for layer_idx in range(layer_start, layer_end + 1):
        if linear_set in {"attn", "all"}:
            paths.extend(
                [
                    f"layers.{layer_idx}.self_attn.q_proj",
                    f"layers.{layer_idx}.self_attn.k_proj",
                    f"layers.{layer_idx}.self_attn.v_proj",
                    f"layers.{layer_idx}.self_attn.o_proj",
                ]
            )
        if linear_set in {"mlp", "all"}:
            paths.extend(
                [
                    f"layers.{layer_idx}.mlp.gate_proj",
                    f"layers.{layer_idx}.mlp.up_proj",
                    f"layers.{layer_idx}.mlp.down_proj",
                ]
            )
    return paths


def _run_backbone_with_module_capture(
    policy: Any,
    backbone_inputs: dict[str, Any],
    module_paths: list[str],
    *,
    layer_paths: list[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    lm_root = policy.model.backbone.model.language_model.model
    captured: dict[str, np.ndarray] = {}
    captured_layers: dict[str, np.ndarray] = {}
    handles: list[Any] = []

    def _mk_hook(path: str):
        def _hook(_module, _module_inputs, module_output):
            captured[path] = module_output.detach().cpu().to(torch.float32).numpy()

        return _hook

    def _mk_layer_hook(path: str):
        def _hook(_module, _module_inputs, module_output):
            out = module_output
            if isinstance(out, (tuple, list)) and len(out) > 0:
                out = out[0]
            if not isinstance(out, torch.Tensor):
                return
            captured_layers[path] = out.detach().cpu().to(torch.float32).numpy()

        return _hook

    for path in module_paths:
        module = _get_module_by_path(lm_root, path)
        handles.append(module.register_forward_hook(_mk_hook(path)))
    for path in layer_paths or []:
        layer_module = _get_module_by_path(lm_root, path)
        handles.append(layer_module.register_forward_hook(_mk_layer_hook(path)))

    try:
        with torch.no_grad():
            backbone_output = policy.model.backbone(backbone_inputs)
            backbone_output = policy.model.action_head.process_backbone_output(backbone_output)
    finally:
        for handle in handles:
            handle.remove()

    system1_outputs: dict[str, np.ndarray] = {
        "backbone_features": backbone_output.backbone_features.detach().cpu().to(torch.float32).numpy(),
        "backbone_attention_mask": backbone_output.backbone_attention_mask.detach().cpu().numpy(),
        "image_mask": backbone_output.image_mask.detach().cpu().numpy(),
    }
    return system1_outputs, captured, captured_layers


def _patch_linear_modules_for_phaseb(
    lm_root: Any,
    module_paths: list[str],
    *,
    args: argparse.Namespace,
    tt_device: Any | None,
    use_ttnn: bool,
    stats: dict[str, int],
) -> dict[str, Any]:
    originals: dict[str, Any] = {}

    for path in module_paths:
        module = _get_module_by_path(lm_root, path)
        if not isinstance(module, torch.nn.Linear):
            raise TypeError(f"Target module is not Linear: {path} ({type(module)})")

        originals[path] = module.forward
        weight_io = module.weight.detach().cpu().to(torch.float32).T.numpy().astype(np.float32)
        bias_np = (
            module.bias.detach().cpu().to(torch.float32).numpy().astype(np.float32)
            if module.bias is not None
            else None
        )

        def _patched_forward(self, input_tensor, _w=weight_io, _b=bias_np):
            x_np = input_tensor.detach().cpu().to(torch.float32).numpy().astype(np.float32)
            x2d = x_np.reshape(-1, x_np.shape[-1]).astype(np.float32)
            y2d, used_ttnn = _ttnn_linear_compare(
                x2d,
                _w,
                bias=_b,
                args=args,
                tt_device=tt_device,
                use_ttnn=use_ttnn,
            )
            stats["total"] += 1
            if used_ttnn:
                stats["ttnn"] += 1
            else:
                stats["fallback"] += 1
            y_np = y2d.reshape(*x_np.shape[:-1], y2d.shape[-1]).astype(np.float32)
            out = torch.from_numpy(y_np).to(input_tensor.device)
            if input_tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
                out = out.to(input_tensor.dtype)
            return out

        module.forward = types.MethodType(_patched_forward, module)

    return originals


def _restore_patched_linears(lm_root: Any, originals: dict[str, Any]) -> None:
    for path, original_forward in originals.items():
        module = _get_module_by_path(lm_root, path)
        module.forward = original_forward


def run_backbone_phaseb_compare(args: argparse.Namespace) -> None:
    if not args.load_assembly_dir:
        raise ValueError("--backbone-phaseb-compare requires --load-assembly-dir")

    prepared_system1, system1_output_expected, metadata = _load_backbone_inputs_outputs_from_assembly(
        args.load_assembly_dir
    )
    backbone_inputs = _build_backbone_inputs_from_prepared(prepared_system1)
    policy, _ = _load_policy_for_backbone(args, metadata)
    lm_root = policy.model.backbone.model.language_model.model
    num_layers = len(lm_root.layers)

    layer_start = int(args.backbone_layer_start)
    layer_end = int(args.backbone_layer_end)
    if layer_start < 0 or layer_start >= num_layers or layer_end < 0 or layer_end >= num_layers:
        raise IndexError(
            f"Layer range [{layer_start}, {layer_end}] out of bounds for model with {num_layers} layers"
        )

    module_paths = _build_backbone_linear_paths(
        layer_start=layer_start,
        layer_end=layer_end,
        linear_set=str(args.backbone_linear_set),
    )
    layer_paths = [f"layers.{layer_idx}" for layer_idx in range(layer_start, layer_end + 1)]
    if not module_paths:
        raise RuntimeError("No linear module paths selected for Phase-B compare")

    system1_ref, module_ref, layer_ref = _run_backbone_with_module_capture(
        policy,
        backbone_inputs,
        module_paths,
        layer_paths=layer_paths,
    )
    expected_worst, expected_key, _ = compare_tensor_maps(
        system1_ref, system1_output_expected, max_entries=int(args.dump_max_entries)
    )

    tt_device = None
    use_ttnn = True
    try:
        try:
            tt_device = ttnn.open_device(device_id=args.tt_device_id)
        except Exception as exc:
            if args.strict_ttnn:
                raise RuntimeError("Unable to open TT device for Phase-B compare") from exc
            use_ttnn = False

        patch_stats = {"total": 0, "ttnn": 0, "fallback": 0}
        originals = _patch_linear_modules_for_phaseb(
            lm_root,
            module_paths,
            args=args,
            tt_device=tt_device,
            use_ttnn=use_ttnn,
            stats=patch_stats,
        )
        try:
            system1_phaseb, module_phaseb, layer_phaseb = _run_backbone_with_module_capture(
                policy,
                backbone_inputs,
                module_paths,
                layer_paths=layer_paths,
            )
        finally:
            _restore_patched_linears(lm_root, originals)
    finally:
        if tt_device is not None:
            ttnn.close_device(tt_device)

    module_worst, module_worst_key, module_lines = compare_tensor_maps(
        module_phaseb, module_ref, max_entries=int(args.dump_max_entries)
    )
    layer_worst, layer_worst_key, layer_lines = compare_tensor_maps(
        layer_phaseb, layer_ref, max_entries=int(args.dump_max_entries)
    )
    system1_worst, system1_worst_key, system1_lines = compare_tensor_maps(
        system1_phaseb, system1_ref, max_entries=int(args.dump_max_entries)
    )

    print("=== Phase B Backbone Linear Stack Compare ===")
    print(f"Assembly source: {_resolve_assembly_tensor_dir(args.load_assembly_dir)}")
    print(f"Layer range: [{layer_start}, {layer_end}] / num_layers={num_layers}")
    print(f"Linear set: {args.backbone_linear_set}")
    print(f"Modules selected: {len(module_paths)}")
    print(
        "Linear execution stats: "
        f"total_calls={patch_stats['total']}, ttnn_calls={patch_stats['ttnn']}, "
        f"fallback_calls={patch_stats['fallback']}"
    )
    print(
        "CPU replay vs assembly expected System1 outputs: "
        f"worst={expected_worst:.6f} at {expected_key or 'n/a'}"
    )
    print(
        "Patched-vs-reference module outputs: "
        f"worst={module_worst:.6f} at {module_worst_key or 'n/a'}"
    )
    print(
        "Patched-vs-reference layer outputs: "
        f"worst={layer_worst:.6f} at {layer_worst_key or 'n/a'}"
    )
    print(
        "Patched-vs-reference System1 outputs: "
        f"worst={system1_worst:.6f} at {system1_worst_key or 'n/a'}"
    )
    print("Module diff summary:")
    for line in module_lines:
        print(f"  - {line}")
    print("System1 diff summary:")
    for line in system1_lines:
        print(f"  - {line}")
    print("Layer-output diff summary:")
    for line in layer_lines:
        print(f"  - {line}")

    if args.dump_dir:
        dump_root = Path(args.dump_dir).resolve()
        ref_dump: dict[str, np.ndarray] = {}
        test_dump: dict[str, np.ndarray] = {}
        for key, value in module_ref.items():
            ref_dump[f"module/{key}"] = value
        for key, value in module_phaseb.items():
            test_dump[f"module/{key}"] = value
        for key, value in layer_ref.items():
            ref_dump[f"layer/{key}"] = value
        for key, value in layer_phaseb.items():
            test_dump[f"layer/{key}"] = value
        for key, value in system1_ref.items():
            ref_dump[f"system1/{key}"] = value
        for key, value in system1_phaseb.items():
            test_dump[f"system1/{key}"] = value
        dump_tensor_map(dump_root, "phaseb_reference", ref_dump)
        dump_tensor_map(dump_root, "phaseb_patched", test_dump)
        print(f"Dumped Phase-B compare tensors to: {dump_root}")


def _decode_action_for_demo(
    args: argparse.Namespace, normalized_action: np.ndarray, shared_inputs: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray] | None, Any | None]:
    if args.input_source != "gr1":
        return None, None

    state_prefix = "raw_state."
    batched_states = {
        key[len(state_prefix) :]: value.astype(np.float32)
        for key, value in shared_inputs.items()
        if key.startswith(state_prefix)
    }
    if not batched_states:
        return None, None

    isaac_root = Path(args.isaac_gr00t_root).resolve()
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))
    from gr00t.data.embodiment_tags import EmbodimentTag
    from transformers import AutoProcessor

    embodiment = EmbodimentTag[args.embodiment_tag]
    processor = AutoProcessor.from_pretrained(Path(args.model_dir).resolve())
    action_for_decode = normalized_action.astype(np.float32)
    expected_horizon = len(processor.modality_configs[embodiment.value]["action"].delta_indices)
    current_horizon = int(action_for_decode.shape[1])
    if current_horizon < expected_horizon:
        pad_len = expected_horizon - current_horizon
        if current_horizon > 0:
            pad = np.repeat(action_for_decode[:, -1:, :], pad_len, axis=1)
        else:
            pad = np.zeros(
                (action_for_decode.shape[0], pad_len, action_for_decode.shape[2]), dtype=np.float32
            )
        action_for_decode = np.concatenate([action_for_decode, pad], axis=1)
    elif current_horizon > expected_horizon:
        action_for_decode = action_for_decode[:, :expected_horizon, :]

    try:
        decoded_action = processor.decode_action(action_for_decode, embodiment, batched_states)
        casted = {k: v.astype(np.float32) for k, v in decoded_action.items()}
        return casted, embodiment
    except Exception:
        # Fallback: split concatenated normalized action by joint-group dimensions.
        out: dict[str, np.ndarray] = {}
        start_idx = 0
        joint_groups = processor.modality_configs[embodiment.value]["action"].modality_keys
        for key in joint_groups:
            joint_dim = int(
                processor.state_action_processor.norm_params[embodiment.value]["action"][key][
                    "dim"
                ].item()
            )
            out[key] = action_for_decode[:, :, start_idx : start_idx + joint_dim].astype(np.float32)
            start_idx += joint_dim
        return out, embodiment


def _print_demo_result(
    *,
    args: argparse.Namespace,
    decoded_action: dict[str, np.ndarray],
    embodiment: Any,
) -> None:
    print("=== GR00T Demo Result ===")
    print(f"Model: {Path(args.model_dir).resolve()}")
    print(f"Dataset: {Path(args.dataset_path).resolve()}")
    print(f"Embodiment: {embodiment.name} ({embodiment.value})")
    print(f"Trajectory: {args.traj_id}, step: {args.step}")
    print("Action keys and shapes:")
    for key, value in decoded_action.items():
        print(f"  - {key}: {value.shape}")

    print("\nFirst predicted action timestep (batch=0, t=0):")
    for key, value in decoded_action.items():
        print(f"  - {key}: {value[0, 0].tolist()}")


def _build_observation(step_data: Any, language_key: str) -> dict[str, Any]:
    observation: dict[str, Any] = {"video": {}, "state": {}, "language": {}}
    for key, frames in step_data.images.items():
        observation["video"][key] = np.stack(frames, axis=0)[None, :]
    for key, state in step_data.states.items():
        observation["state"][key] = np.asarray(state, dtype=np.float32)[None, :]
    observation["language"][language_key] = [[step_data.text]]
    return observation


def build_shared_action_head_inputs_from_gr1(
    args: argparse.Namespace,
    *,
    dump_system_inputs: bool = False,
    capture_assembly_tensors: dict[str, np.ndarray] | None = None,
    capture_assembly_meta: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    isaac_root = Path(args.isaac_gr00t_root).resolve()
    if not isaac_root.exists():
        raise FileNotFoundError(f"Isaac-GR00T root not found: {isaac_root}")
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))

    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.data.types import MessageType
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    try:
        embodiment = EmbodimentTag[args.embodiment_tag]
    except KeyError as exc:
        valid = ", ".join(e.name for e in EmbodimentTag)
        raise ValueError(
            f"Unknown embodiment tag '{args.embodiment_tag}'. Valid values: {valid}"
        ) from exc

    policy = Gr00tPolicy(
        embodiment_tag=embodiment,
        model_path=args.model_dir,
        device="cpu",
        strict=False,
    )
    modality_configs = policy.get_modality_config()
    loader = LeRobotEpisodeLoader(
        str(dataset_path),
        modality_configs=modality_configs,
        video_backend=args.video_backend,
    )
    traj = loader[args.traj_id]
    if args.step < 0 or args.step >= len(traj):
        raise IndexError(f"Step {args.step} out of range for trajectory {args.traj_id} (len={len(traj)})")

    step_data = extract_step_data(
        traj,
        args.step,
        modality_configs=modality_configs,
        embodiment_tag=embodiment,
        allow_padding=True,
    )
    if args.task_text:
        step_data.text = args.task_text
    language_key = modality_configs["language"].modality_keys[0]
    observation = _build_observation(step_data, language_key)

    if capture_assembly_meta is not None:
        capture_assembly_meta.update(
            {
                "dataset_path": dataset_path.as_posix(),
                "trajectory_id": int(args.traj_id),
                "step": int(args.step),
                "embodiment_tag": str(args.embodiment_tag),
                "video_backend": str(args.video_backend),
                "task_text": str(step_data.text),
                "video_keys": sorted(list(observation["video"].keys())),
                "state_keys": sorted(list(observation["state"].keys())),
                "language_key": str(language_key),
            }
        )
    if capture_assembly_tensors is not None:
        _flatten_tensor_tree("observation/video", observation["video"], capture_assembly_tensors)
        _flatten_tensor_tree("observation/state", observation["state"], capture_assembly_tensors)

    unbatched = policy._unbatch_observation(observation)
    processed_inputs = []
    for obs in unbatched:
        vla_step_data = policy._to_vla_step_data(obs)
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        processed_inputs.append(policy.processor(messages))
    collated_inputs = policy.collate_fn(processed_inputs)
    inputs = collated_inputs["inputs"]

    if capture_assembly_tensors is not None:
        if processed_inputs:
            _flatten_tensor_tree("processor_output", processed_inputs[0], capture_assembly_tensors)
        _flatten_tensor_tree("collated_inputs", inputs, capture_assembly_tensors)

    if dump_system_inputs:
        print("=== Assembly Trace ===")
        print("observation -> processor(messages) -> collate_fn -> model.prepare_input")
        print("processor(messages) output keys:")
        for key in sorted(processed_inputs[0].keys()):
            shape, dtype = _shape_and_dtype(processed_inputs[0][key])
            print(f"  - {key}: shape={shape}, dtype={dtype}")

        print("Collated tensors (input to Gr00tN1d6.prepare_input):")
        for key, value in inputs.items():
            shape, dtype = _shape_and_dtype(value)
            print(f"  - {key}: shape={shape}, dtype={dtype}")

    with torch.no_grad():
        model_inputs = copy.deepcopy(inputs)
        backbone_inputs, action_inputs = policy.model.prepare_input(model_inputs)
        backbone_output = policy.model.backbone(backbone_inputs)
        backbone_output = policy.model.action_head.process_backbone_output(backbone_output)

        if capture_assembly_tensors is not None:
            _flatten_tensor_tree("prepared/system1", backbone_inputs, capture_assembly_tensors)
            _flatten_tensor_tree("prepared/system2", action_inputs, capture_assembly_tensors)
            _flatten_tensor_tree(
                "system1_output/backbone_features",
                backbone_output.backbone_features,
                capture_assembly_tensors,
            )
            _flatten_tensor_tree(
                "system1_output/backbone_attention_mask",
                backbone_output.backbone_attention_mask,
                capture_assembly_tensors,
            )
            _flatten_tensor_tree(
                "system1_output/image_mask",
                backbone_output.image_mask,
                capture_assembly_tensors,
            )

        if dump_system_inputs:
            print("System1 (Vision-Language Backbone) raw prepared inputs:")
            for key, value in sorted(backbone_inputs.items()):
                shape, dtype = _shape_and_dtype(value)
                print(f"  - {key}: shape={shape}, dtype={dtype}")

            print("System2 (Diffusion Action Head) raw prepared inputs:")
            for key, value in sorted(action_inputs.items()):
                shape, dtype = _shape_and_dtype(value)
                print(f"  - {key}: shape={shape}, dtype={dtype}")

            print("System1 actual model-consumed inputs (EagleBackbone.forward):")
            for key in ("input_ids", "attention_mask", "pixel_values"):
                if key not in backbone_inputs:
                    continue
                value = backbone_inputs[key]
                if isinstance(value, list):
                    print(f"  - {key}: list(len={len(value)})")
                    if len(value) > 0 and hasattr(value[0], "shape"):
                        print(f"    first[0]: shape={tuple(value[0].shape)}, dtype={value[0].dtype}")
                else:
                    shape, dtype = _shape_and_dtype(value)
                    print(f"  - {key}: shape={shape}, dtype={dtype}")

            vl_embeds_t = backbone_output.backbone_features
            vl_attn_mask_t = backbone_output.backbone_attention_mask
            image_mask_t = backbone_output.image_mask
            valid_tokens = int(vl_attn_mask_t.sum().item())
            image_tokens = int((image_mask_t & vl_attn_mask_t).sum().item())
            text_tokens = valid_tokens - image_tokens

            print("System1 output -> System2 cross-attention context:")
            print(
                "  - backbone_features: "
                f"shape={tuple(vl_embeds_t.shape)}, dtype={vl_embeds_t.dtype}"
            )
            print(
                "  - backbone_attention_mask: "
                f"shape={tuple(vl_attn_mask_t.shape)}, dtype={vl_attn_mask_t.dtype}"
            )
            print(f"  - token counts: valid={valid_tokens}, image={image_tokens}, text_or_other={text_tokens}")

            action_head = policy.model.action_head
            features = action_head._encode_features(backbone_output, action_inputs)
            state_features = features.state_features
            batch_size = vl_embeds_t.shape[0]
            device = vl_embeds_t.device

            action_seed = torch.zeros(
                size=(batch_size, int(args.action_horizon), action_head.action_dim),
                dtype=vl_embeds_t.dtype,
                device=device,
            )
            timestep0 = torch.zeros(size=(batch_size,), dtype=torch.long, device=device)
            action_tokens = action_head.action_encoder(action_seed, timestep0, action_inputs.embodiment_id)
            if action_head.config.add_pos_embed:
                pos_ids = torch.arange(action_tokens.shape[1], dtype=torch.long, device=device)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_tokens = action_tokens + pos_embs
            sa_embs = torch.cat((state_features, action_tokens), dim=1)

            print("System2 diffusion model assembled inputs (DiT/AlternateVLDiT):")
            print(
                "  - encoder_hidden_states (from system1): "
                f"shape={tuple(vl_embeds_t.shape)}, dtype={vl_embeds_t.dtype}"
            )
            print(
                "  - hidden_states.state_tokens: "
                f"shape={tuple(state_features.shape)}, dtype={state_features.dtype}"
            )
            print(
                "  - hidden_states.action_tokens: "
                f"shape={tuple(action_tokens.shape)}, dtype={action_tokens.dtype}"
            )
            print(f"  - hidden_states.concat(sa_embs): shape={tuple(sa_embs.shape)}, dtype={sa_embs.dtype}")
            print(f"  - timestep: shape={tuple(timestep0.shape)}, dtype={timestep0.dtype}")
            if action_head.config.use_alternate_vl_dit:
                print(f"  - image_mask: shape={tuple(image_mask_t.shape)}, dtype={image_mask_t.dtype}")
                print(
                    "  - backbone_attention_mask: "
                    f"shape={tuple(vl_attn_mask_t.shape)}, dtype={vl_attn_mask_t.dtype}"
                )

    state = action_inputs["state"].detach().cpu().to(torch.float32).numpy()
    embodiment_id = action_inputs["embodiment_id"].detach().cpu().to(torch.long).numpy().astype(np.int64)
    vl_embeds = backbone_output.backbone_features.detach().cpu().to(torch.float32).numpy()
    image_mask = backbone_output.image_mask.detach().cpu().numpy().astype(bool)
    backbone_attention_mask = backbone_output.backbone_attention_mask.detach().cpu().numpy().astype(bool)

    action_dim = int(policy.model.config.max_action_dim)
    torch.manual_seed(args.seed + 1)
    actions = torch.randn(
        (state.shape[0], int(args.action_horizon), action_dim), dtype=torch.float32
    ).cpu().numpy()

    print("=== Shared Input Source (GR1 sample) ===")
    print(f"Dataset: {dataset_path.as_posix()}")
    print(f"Trajectory: {args.traj_id}, step: {args.step}")
    print(f"Text: {step_data.text}")
    print(f"state: {state.shape}, vl_embeds: {vl_embeds.shape}, image_mask: {image_mask.shape}")

    shared_inputs = {
        "state": state.astype(np.float32),
        "vl_embeds": vl_embeds.astype(np.float32),
        "image_mask": image_mask,
        "backbone_attention_mask": backbone_attention_mask,
        "actions": actions.astype(np.float32),
        "embodiment_id": embodiment_id,
        **{f"raw_state.{k}": v.astype(np.float32) for k, v in observation["state"].items()},
    }
    if capture_assembly_tensors is not None:
        _flatten_tensor_tree("shared_inputs", shared_inputs, capture_assembly_tensors)
    return shared_inputs


def dump_tensor_map(base_dir: Path, run_name: str, tensors: dict[str, np.ndarray]) -> None:
    run_dir = base_dir / run_name
    for key, value in sorted(tensors.items()):
        parts = [p for p in key.split("/") if p]
        out_base = run_dir.joinpath(*parts)
        out_path = Path(f"{out_base.as_posix()}.npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, value)


def compare_tensor_maps(
    lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray], max_entries: int
) -> tuple[float, str, list[str]]:
    shared = sorted(set(lhs.keys()) & set(rhs.keys()))
    worst_key = ""
    worst_diff = 0.0
    diffs: list[tuple[float, str]] = []
    non_numeric_msgs: list[str] = []

    for key in shared:
        a = lhs[key]
        b = rhs[key]
        if a.shape != b.shape:
            non_numeric_msgs.append(f"{key}: shape mismatch {a.shape} vs {b.shape}")
            continue
        if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
            diff = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
            diffs.append((diff, key))
            if diff > worst_diff:
                worst_diff = diff
                worst_key = key
        else:
            same = bool(np.array_equal(a, b))
            non_numeric_msgs.append(f"{key}: exact_equal={same}")

    diffs.sort(key=lambda x: x[0], reverse=True)
    lines = [f"{k}: max_abs_diff={d:.6f}" for d, k in diffs[: max(0, int(max_entries))]]
    lines.extend(non_numeric_msgs[: max(0, int(max_entries) - len(lines))])

    lhs_only = sorted(set(lhs.keys()) - set(rhs.keys()))
    rhs_only = sorted(set(rhs.keys()) - set(lhs.keys()))
    if lhs_only:
        lines.append(f"lhs_only_keys={len(lhs_only)} (example: {lhs_only[0]})")
    if rhs_only:
        lines.append(f"rhs_only_keys={len(rhs_only)} (example: {rhs_only[0]})")
    return worst_diff, worst_key, lines


def _single_tensor_diff(
    lhs: dict[str, np.ndarray], rhs: dict[str, np.ndarray], key: str
) -> tuple[float, str]:
    if key not in lhs or key not in rhs:
        return float("inf"), "missing"
    a = lhs[key]
    b = rhs[key]
    if a.shape != b.shape:
        return float("inf"), f"shape mismatch {a.shape} vs {b.shape}"
    if np.issubdtype(a.dtype, np.number) and np.issubdtype(b.dtype, np.number):
        diff = float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
        return diff, "max_abs_diff"
    same = bool(np.array_equal(a, b))
    return (0.0 if same else float("inf")), ("exact_equal" if same else "not_equal")


def summarize_phasec_parity(
    ttnn_debug: dict[str, np.ndarray],
    ref_debug: dict[str, np.ndarray],
) -> list[str]:
    lines: list[str] = []

    groups: dict[str, list[str]] = {
        "C0 interface": [
            "input/vl_embeds",
            "input/backbone_attention_mask",
            "input/image_mask",
            "input/state",
            "input/embodiment_id",
        ],
        "C1 encoder": [
            "encoded/state_features",
            "step_00/action_features",
            "step_00/sa_embs",
        ],
        "C2/C3 diffusion_step0": [
            "step_00/model_output",
            "step_00/pred_velocity",
            "step_00/actions",
        ],
        "C4 final": [
            "output/actions",
        ],
    }

    for group_name, keys in groups.items():
        worst = -1.0
        worst_key = ""
        detail = ""
        for key in keys:
            diff, mode = _single_tensor_diff(ttnn_debug, ref_debug, key)
            if diff > worst:
                worst = diff
                worst_key = key
                detail = mode
        if worst < 0:
            lines.append(f"{group_name}: no_keys")
        elif np.isinf(worst):
            lines.append(f"{group_name}: worst=inf at {worst_key} ({detail})")
        else:
            lines.append(f"{group_name}: worst={worst:.6f} at {worst_key} ({detail})")

    step_action_keys = sorted(
        key for key in set(ttnn_debug.keys()) & set(ref_debug.keys()) if key.endswith("/actions")
    )
    step_worst = 0.0
    step_worst_key = ""
    for key in step_action_keys:
        diff, _ = _single_tensor_diff(ttnn_debug, ref_debug, key)
        if np.isfinite(diff) and diff > step_worst:
            step_worst = diff
            step_worst_key = key
    if step_action_keys:
        lines.append(
            f"C3 trajectory: worst={step_worst:.6f} at {step_worst_key or step_action_keys[0]}"
        )
    return lines


def _subset_tensor_map(
    tensors: dict[str, np.ndarray],
    keys: list[str],
) -> dict[str, np.ndarray]:
    return {k: tensors[k] for k in keys if k in tensors}


def _run_phasec_reports(
    args: argparse.Namespace,
    *,
    ttnn_debug: dict[str, np.ndarray],
    ref_debug: dict[str, np.ndarray],
) -> None:
    if args.system2_encoder_compare:
        encoder_keys = [
            "encoded/state_features",
            "step_00/action_features",
            "step_00/sa_embs",
        ]
        lhs = _subset_tensor_map(ttnn_debug, encoder_keys)
        rhs = _subset_tensor_map(ref_debug, encoder_keys)
        worst, worst_key, lines = compare_tensor_maps(lhs, rhs, max_entries=int(args.dump_max_entries))
        print("\n=== Phase C Encoder Compare ===")
        print(f"Worst tensor diff: {worst:.6f} at key: {worst_key or 'n/a'}")
        for line in lines:
            print(f"  - {line}")

    if args.system2_block_compare:
        step = int(args.phasec_step)
        prefix = f"step_{step:02d}/block_hidden_"
        keys = sorted(k for k in ref_debug.keys() if k.startswith(prefix))
        lhs = _subset_tensor_map(ttnn_debug, keys)
        rhs = _subset_tensor_map(ref_debug, keys)
        worst, worst_key, lines = compare_tensor_maps(lhs, rhs, max_entries=int(args.dump_max_entries))
        print("\n=== Phase C Diffusion Block Compare ===")
        print(f"Step: {step}")
        print(f"Block tensors: {len(keys)}")
        print(f"Worst tensor diff: {worst:.6f} at key: {worst_key or 'n/a'}")
        for line in lines:
            print(f"  - {line}")

    if args.system2_trajectory_compare:
        keys = sorted(k for k in ref_debug.keys() if k.startswith("step_") and k.endswith("/actions"))
        if "output/actions" in ref_debug:
            keys.append("output/actions")
        lhs = _subset_tensor_map(ttnn_debug, keys)
        rhs = _subset_tensor_map(ref_debug, keys)
        worst, worst_key, lines = compare_tensor_maps(lhs, rhs, max_entries=int(args.dump_max_entries))
        print("\n=== Phase C Trajectory Compare ===")
        print(f"Trajectory tensors: {len(keys)}")
        print(f"Worst tensor diff: {worst:.6f} at key: {worst_key or 'n/a'}")
        for line in lines:
            print(f"  - {line}")


def _resolve_shared_inputs(
    args: argparse.Namespace,
    *,
    dump_system_inputs: bool,
) -> dict[str, np.ndarray]:
    if args.load_shared_inputs_dir:
        in_dir = Path(args.load_shared_inputs_dir).resolve()
        shared_inputs, metadata = _load_shared_inputs(in_dir)
        print(f"Loaded shared inputs from: {in_dir}")
        if metadata:
            source = metadata.get("input_source", "unknown")
            case = f"traj={metadata.get('trajectory_id', 'n/a')}, step={metadata.get('step', 'n/a')}"
            print(f"Shared input metadata: source={source}, {case}")
        return _clone_shared_inputs(shared_inputs)

    assembly_tensors: dict[str, np.ndarray] | None = None
    assembly_meta: dict[str, Any] | None = None
    if args.dump_assembly_dir:
        assembly_tensors = {}
        assembly_meta = {}

    if args.input_source == "gr1":
        shared_inputs = build_shared_action_head_inputs_from_gr1(
            args,
            dump_system_inputs=dump_system_inputs,
            capture_assembly_tensors=assembly_tensors,
            capture_assembly_meta=assembly_meta,
        )
        if args.check_input_determinism:
            second = build_shared_action_head_inputs_from_gr1(
                args,
                dump_system_inputs=False,
                capture_assembly_tensors=None,
                capture_assembly_meta=None,
            )
            worst_key, worst_diff = _assert_shared_inputs_deterministic(shared_inputs, second)
            print(
                "Shared-input determinism check passed "
                f"(worst_key={worst_key or 'n/a'}, max_abs_diff={worst_diff:.6f})"
            )
    else:
        cfg = _load_model_cfg(args)
        shared_inputs = build_shared_action_head_inputs(
            args,
            state_dim=int(cfg["max_state_dim"]),
            backbone_dim=int(cfg["backbone_embedding_dim"]),
            action_dim=int(cfg["max_action_dim"]),
        )
        if args.check_input_determinism:
            second = build_shared_action_head_inputs(
                args,
                state_dim=int(cfg["max_state_dim"]),
                backbone_dim=int(cfg["backbone_embedding_dim"]),
                action_dim=int(cfg["max_action_dim"]),
            )
            worst_key, worst_diff = _assert_shared_inputs_deterministic(shared_inputs, second)
            print(
                "Shared-input determinism check passed "
                f"(worst_key={worst_key or 'n/a'}, max_abs_diff={worst_diff:.6f})"
            )

    if args.save_shared_inputs_dir:
        out_dir = Path(args.save_shared_inputs_dir).resolve()
        metadata: dict[str, Any] = {
            "input_source": str(args.input_source),
            "seed": int(args.seed),
            "batch": int(args.batch),
            "action_horizon": int(args.action_horizon),
            "num_inference_timesteps": int(args.num_inference_timesteps),
            "vl_seq_len": int(args.vl_seq_len),
            "model_dir": str(Path(args.model_dir).resolve()),
            "isaac_gr00t_root": str(Path(args.isaac_gr00t_root).resolve()),
        }
        if args.input_source == "gr1":
            metadata.update(
                {
                    "dataset_path": str(Path(args.dataset_path).resolve()),
                    "trajectory_id": int(args.traj_id),
                    "step": int(args.step),
                    "task_text_override": str(args.task_text),
                    "embodiment_tag": str(args.embodiment_tag),
                    "video_backend": str(args.video_backend),
                }
            )
        if assembly_meta is not None:
            metadata.update(assembly_meta)
        _save_shared_inputs(out_dir, shared_inputs, metadata)
        print(f"Saved shared inputs to: {out_dir}")

    if args.dump_assembly_dir and assembly_tensors is not None:
        dump_root = Path(args.dump_assembly_dir).resolve()
        dump_tensor_map(dump_root, "assembly", assembly_tensors)
        if assembly_meta is not None:
            _write_json(dump_root / "assembly" / "meta.json", assembly_meta)
        print(f"Dumped assembly tensors to: {dump_root / 'assembly'}")

    return _clone_shared_inputs(shared_inputs)


class TensorStore:
    def keys(self) -> list[str]:
        raise NotImplementedError

    def get(self, key: str) -> torch.Tensor:
        raise NotImplementedError


class DictStore(TensorStore):
    def __init__(self, state_dict: dict[str, torch.Tensor]):
        self.state_dict = state_dict

    def keys(self) -> list[str]:
        return list(self.state_dict.keys())

    def get(self, key: str) -> torch.Tensor:
        return self.state_dict[key]


class SafeTensorsStore(TensorStore):
    def __init__(self, file_map: dict[str, Path]):
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError("Please install safetensors: pip install safetensors") from exc
        self._safe_open = safe_open
        self.file_map = file_map

    def keys(self) -> list[str]:
        return list(self.file_map.keys())

    def get(self, key: str) -> torch.Tensor:
        file_path = self.file_map[key]
        with self._safe_open(str(file_path), framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def load_store(model_dir: Path) -> TensorStore:
    safetensors_index = model_dir / "model.safetensors.index.json"
    safetensors_single = model_dir / "model.safetensors"
    pytorch_bin = model_dir / "pytorch_model.bin"

    if safetensors_index.exists():
        data = json.loads(safetensors_index.read_text(encoding="utf-8"))
        weight_map = data["weight_map"]
        file_map = {k: model_dir / v for k, v in weight_map.items()}
        return SafeTensorsStore(file_map)

    if safetensors_single.exists():
        from safetensors import safe_open

        file_map: dict[str, Path] = {}
        with safe_open(str(safetensors_single), framework="pt", device="cpu") as f:
            for key in f.keys():
                file_map[key] = safetensors_single
        return SafeTensorsStore(file_map)

    if pytorch_bin.exists():
        obj = torch.load(pytorch_bin, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
            obj = obj["state_dict"]
        if not isinstance(obj, dict):
            raise RuntimeError(f"Unsupported pytorch_model.bin format at {pytorch_bin}")
        return DictStore(obj)

    raise FileNotFoundError(f"No supported checkpoint found in {model_dir}")


def load_model_dir_and_store(args: argparse.Namespace) -> tuple[Path, TensorStore]:
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    return model_dir, load_store(model_dir)


def maybe_list_keys(store: TensorStore, args: argparse.Namespace) -> None:
    if args.list_keys:
        for key in sorted(k for k in store.keys() if k.startswith("action_head.")):
            print(key)
        raise SystemExit(0)


def get_tensor_np(store: TensorStore, key: str) -> np.ndarray:
    return store.get(key).to(torch.float32).cpu().numpy()

def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def _load_action_head_state_dict(store: TensorStore) -> dict[str, torch.Tensor]:
    prefix = "action_head."
    return {k[len(prefix):]: store.get(k).to(torch.float32) for k in store.keys() if k.startswith(prefix)}


def run_full_action_head_reference(
    store: TensorStore,
    args: argparse.Namespace,
    *,
    shared_inputs: dict[str, np.ndarray] | None = None,
    return_debug: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    isaac_root = Path(args.isaac_gr00t_root).resolve()
    if not isaac_root.exists():
        raise FileNotFoundError(f"Isaac-GR00T root not found: {isaac_root}")
    if str(isaac_root) not in sys.path:
        sys.path.insert(0, str(isaac_root))

    from transformers.feature_extraction_utils import BatchFeature

    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6ActionHead

    cfg_data = _load_model_cfg(args)
    config = Gr00tN1d6Config(**cfg_data)
    config.action_horizon = args.action_horizon
    config.num_inference_timesteps = args.num_inference_timesteps

    action_head = Gr00tN1d6ActionHead(config).to(torch.float32).eval()
    state_dict = _load_action_head_state_dict(store)
    missing, unexpected = action_head.load_state_dict(state_dict, strict=False)
    missing_required = [k for k in missing if k != "mask_token"]
    if missing_required:
        raise RuntimeError(f"Missing required action-head keys: {missing_required[:10]}")
    if unexpected:
        raise RuntimeError(f"Unexpected action-head keys: {unexpected[:10]}")

    if shared_inputs is None:
        shared_inputs = build_shared_action_head_inputs(
            args,
            state_dim=int(config.max_state_dim),
            backbone_dim=int(config.backbone_embedding_dim),
            action_dim=int(config.max_action_dim),
        )
    else:
        shared_inputs = _clone_shared_inputs(shared_inputs)

    state = torch.from_numpy(shared_inputs["state"]).to(torch.float32)
    backbone_features = torch.from_numpy(shared_inputs["vl_embeds"]).to(torch.float32)
    image_mask = torch.from_numpy(shared_inputs["image_mask"]).to(torch.bool)
    backbone_attention_mask = torch.from_numpy(shared_inputs["backbone_attention_mask"]).to(torch.bool)
    embodiment_id = torch.from_numpy(shared_inputs["embodiment_id"]).to(torch.long)
    actions = torch.from_numpy(shared_inputs["actions"]).to(torch.float32)
    batch = int(state.shape[0])
    action_horizon = int(args.action_horizon)
    num_steps = int(args.num_inference_timesteps)

    backbone_output = BatchFeature(
        data={
            "backbone_features": backbone_features,
            "backbone_attention_mask": backbone_attention_mask,
            "image_mask": image_mask,
        }
    )
    action_input = BatchFeature(
        data={
            "state": state,
            "embodiment_id": embodiment_id,
        }
    )

    debug: dict[str, np.ndarray] = {}
    debug["input/state"] = shared_inputs["state"].astype(np.float32)
    debug["input/vl_embeds"] = shared_inputs["vl_embeds"].astype(np.float32)
    debug["input/image_mask"] = shared_inputs["image_mask"]
    debug["input/backbone_attention_mask"] = shared_inputs["backbone_attention_mask"]
    debug["input/actions_init"] = shared_inputs["actions"].astype(np.float32)
    debug["input/embodiment_id"] = shared_inputs["embodiment_id"]

    with torch.no_grad():
        backbone_output = action_head.process_backbone_output(backbone_output)
        vl_embeds = backbone_output.backbone_features
        state_features = action_head.state_encoder(action_input.state, embodiment_id)
        debug["encoded/vl_embeds"] = vl_embeds.detach().cpu().to(torch.float32).numpy()
        debug["encoded/state_features"] = state_features.detach().cpu().to(torch.float32).numpy()

        dt = 1.0 / float(num_steps)
        for t in range(num_steps):
            t_cont = float(t) / float(num_steps)
            t_discretized = int(t_cont * int(action_head.num_timestep_buckets))
            timesteps_tensor = torch.full(size=(batch,), fill_value=t_discretized, dtype=torch.long)

            action_features = action_head.action_encoder(actions, timesteps_tensor, embodiment_id)
            if config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long)
                pos_embs = action_head.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs
            sa_embs = torch.cat((state_features, action_features), dim=1)

            if config.use_alternate_vl_dit:
                model_output, all_hidden_states = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    return_all_hidden_states=True,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output, all_hidden_states = action_head.model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embeds,
                    timestep=timesteps_tensor,
                    return_all_hidden_states=True,
                )
            pred = action_head.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -action_horizon:, :]
            actions = actions + dt * pred_velocity

            step_prefix = f"step_{t:02d}"
            debug[f"{step_prefix}/timesteps"] = timesteps_tensor.detach().cpu().numpy()
            debug[f"{step_prefix}/action_features"] = (
                action_features.detach().cpu().to(torch.float32).numpy()
            )
            debug[f"{step_prefix}/sa_embs"] = sa_embs.detach().cpu().to(torch.float32).numpy()
            debug[f"{step_prefix}/model_output"] = (
                model_output.detach().cpu().to(torch.float32).numpy()
            )
            debug[f"{step_prefix}/pred_velocity"] = (
                pred_velocity.detach().cpu().to(torch.float32).numpy()
            )
            debug[f"{step_prefix}/actions"] = actions.detach().cpu().to(torch.float32).numpy()
            for i, hs in enumerate(all_hidden_states):
                debug[f"{step_prefix}/block_hidden_{i:02d}"] = (
                    hs.detach().cpu().to(torch.float32).numpy()
                )

    action_np = actions.detach().cpu().to(torch.float32).numpy()
    debug["output/actions"] = action_np
    if return_debug:
        return action_np, debug
    return action_np


def relu_np(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def gelu_approx_np(x: np.ndarray) -> np.ndarray:
    c = np.float32(np.sqrt(2.0 / np.pi))
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * np.power(x, 3))))


def layer_norm_np(
    x: np.ndarray, eps: float, weight: np.ndarray | None = None, bias: np.ndarray | None = None
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean(np.square(x - mean), axis=-1, keepdims=True)
    y = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        y = y * weight
    if bias is not None:
        y = y + bias
    return y


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
    """Incremental TTNN action-head runtime.

    Runtime contract:
    - `load_weights`: map checkpoint tensors into runtime structure
    - `run_step`: execute one diffusion denoise step
    - `run_denoise_loop`: execute full denoising loop
    """

    def __init__(
        self,
        *,
        device: ttnn.Device | None,
        strict_ttnn: bool,
        weights: dict[str, Any],
        cfg: dict[str, Any],
    ):
        self.device = device
        self.strict_ttnn = strict_ttnn
        self.w = weights
        self.cfg = cfg
        self._tt_weight_cache: dict[str, Any] = {}
        self._tt_bias_cache: dict[str, Any] = {}

    @classmethod
    def load_weights(
        cls, *, store: TensorStore, args: argparse.Namespace, device: ttnn.Device | None
    ) -> "TtGr00tActionHeadRuntime":
        config_path = Path(args.model_dir).resolve() / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in model directory: {config_path}")
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
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

        return cls(device=device, strict_ttnn=bool(args.strict_ttnn), weights=weights, cfg=cfg)

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
            if self.strict_ttnn:
                raise RuntimeError(
                    f"TTNN device unavailable and strict_ttnn=True at op {op_name}"
                )
            y = x2d @ w
            if bias is not None:
                y = y + bias
            return y.astype(np.float32)

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
            if self.strict_ttnn:
                raise RuntimeError(f"TTNN op failed at {op_name}") from exc
            y = x2d @ w
            if bias is not None:
                y = y + bias
            return y.astype(np.float32)

    def _linear_lastdim(
        self, x: np.ndarray, weight_io: np.ndarray, bias: np.ndarray | None, *, op_name: str
    ) -> np.ndarray:
        x2d = x.reshape(-1, x.shape[-1]).astype(np.float32)
        y2d = self._ttnn_linear_2d(x2d, weight_io.astype(np.float32), bias, op_name=op_name)
        return y2d.reshape(*x.shape[:-1], y2d.shape[-1]).astype(np.float32)

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
        qh = q.reshape(batch, q_len, heads, head_dim).transpose(0, 2, 1, 3)
        kh = k.reshape(batch, k_len, heads, head_dim).transpose(0, 2, 1, 3)
        vh = v.reshape(batch, k_len, heads, head_dim).transpose(0, 2, 1, 3)

        scores = np.matmul(qh, np.swapaxes(kh, -1, -2)) / np.sqrt(float(head_dim))
        if attention_mask is not None:
            keep = attention_mask.astype(bool)[:, None, None, :]
            scores = np.where(keep, scores, np.float32(-1e9))
        probs = softmax_np(scores, axis=-1).astype(np.float32)
        ctx = np.matmul(probs, vh)
        return ctx.transpose(0, 2, 1, 3).reshape(batch, q_len, dim).astype(np.float32)

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
        temb = silu_np(temb)
        temb = self._linear_lastdim(temb, self.w["time_l2_w"].T, self.w["time_l2_b"], op_name="time_l2")

        action_features = self._linear_lastdim(
            actions, self.w["act_w1_w"], self.w["act_w1_b"], op_name="act_w1"
        )
        timestep_bt = np.repeat(t_bucket[:, None].astype(np.float32), action_horizon, axis=1)
        tau_emb = action_pos_encoding_np(timestep_bt, int(self.w["input_embedding_dim"]))
        action_features = np.concatenate([action_features, tau_emb], axis=-1).astype(np.float32)
        action_features = self._linear_lastdim(
            action_features, self.w["act_w2_w"], self.w["act_w2_b"], op_name="act_w2"
        )
        action_features = silu_np(action_features)
        action_features = self._linear_lastdim(
            action_features, self.w["act_w3_w"], self.w["act_w3_b"], op_name="act_w3"
        )

        if self.w["add_pos_embed"] and self.w["pos_embedding"] is not None:
            action_features = action_features + self.w["pos_embedding"][:action_horizon][None, :, :]

        hidden_states = np.concatenate([state_features, action_features], axis=1).astype(np.float32)
        block_hidden_states: list[np.ndarray] = [hidden_states.astype(np.float32)] if return_debug else []
        for i in range(int(self.w["num_layers"])):
            block = self.w["blocks"][i]

            norm_hidden = layer_norm_np(hidden_states, eps=float(self.w["norm_eps"]))
            ada = self._linear_lastdim(
                silu_np(temb), block["norm1_linear_w"].T, block["norm1_linear_b"], op_name=f"block{i}.norm1_linear"
            )
            scale, shift = np.split(ada, 2, axis=-1)
            norm_hidden = norm_hidden * (1.0 + scale[:, None, :]) + shift[:, None, :]

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
            hidden_states = hidden_states + attn_out

            ff_in = self._linear_lastdim(
                layer_norm_np(hidden_states, eps=float(self.w["norm_eps"])),
                block["ff_in_w"].T,
                block["ff_in_b"],
                op_name=f"block{i}.ff_in",
            )
            ff_out_in_dim = int(block["ff_out_w"].shape[1])
            if ff_in.shape[-1] == ff_out_in_dim:
                # Non-gated FFN variant.
                ff_act = gelu_approx_np(ff_in)
            elif ff_in.shape[-1] == 2 * ff_out_in_dim:
                # GEGLU/SwiGLU-style gated FFN variant.
                ff_x, ff_gate = np.split(ff_in, 2, axis=-1)
                ff_act = ff_x * gelu_approx_np(ff_gate)
            else:
                raise ValueError(
                    f"block{i}.ff: unsupported dimensions ff_in={ff_in.shape[-1]}, "
                    f"ff_out_in_dim={ff_out_in_dim}"
                )
            ff_out = self._linear_lastdim(
                ff_act, block["ff_out_w"].T, block["ff_out_b"], op_name=f"block{i}.ff_out"
            )
            hidden_states = hidden_states + ff_out
            if return_debug:
                block_hidden_states.append(hidden_states.astype(np.float32))

        out_norm = layer_norm_np(hidden_states, eps=1e-6)
        shift_scale = self._linear_lastdim(
            silu_np(temb), self.w["proj_out_1_w"].T, self.w["proj_out_1_b"], op_name="proj_out_1"
        )
        shift, scale = np.split(shift_scale, 2, axis=-1)
        out_norm = out_norm * (1.0 + scale[:, None, :]) + shift[:, None, :]
        model_output = self._linear_lastdim(
            out_norm, self.w["proj_out_2_w"].T, self.w["proj_out_2_b"], op_name="proj_out_2"
        )

        d1 = self._linear_lastdim(model_output, self.w["dec_l1_w"], self.w["dec_l1_b"], op_name="dec_l1")
        d1 = relu_np(d1)
        pred = self._linear_lastdim(d1, self.w["dec_l2_w"], self.w["dec_l2_b"], op_name="dec_l2")
        pred_velocity = pred[:, -action_horizon:, :]
        step_debug: dict[str, np.ndarray] | None = None
        if return_debug:
            step_debug = {
                "timesteps": t_bucket.astype(np.int64),
                "temb": temb.astype(np.float32),
                "action_features": action_features.astype(np.float32),
                "sa_embs": np.concatenate([state_features, action_features], axis=1).astype(np.float32),
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
            vl_embeds = layer_norm_np(vl_embeds, eps=1e-5, weight=self.w["vlln_w"], bias=self.w["vlln_b"])

        state_features = self._linear_lastdim(state, self.w["state_l1_w"], self.w["state_l1_b"], op_name="state_l1")
        state_features = relu_np(state_features)
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
            actions = actions + dt * pred_velocity
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
    store: TensorStore,
    args: argparse.Namespace,
    *,
    shared_inputs: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if shared_inputs is None:
        shared_inputs = _resolve_shared_inputs(args, dump_system_inputs=False)
    else:
        shared_inputs = _clone_shared_inputs(shared_inputs)
    ref_actions, ref_debug = run_full_action_head_reference(
        store,
        args,
        shared_inputs=shared_inputs,
        return_debug=True,
    )
    tt_device = None
    used_ttnn_device = True
    op_log_context = (
        ttnn.register_pre_operation_hook(_make_ttnn_pre_operation_hook())
        if args.log_ttnn_ops
        else nullcontext()
    )
    with op_log_context:
        try:
            tt_device = ttnn.open_device(device_id=args.tt_device_id)
        except Exception as exc:
            if args.strict_ttnn:
                raise RuntimeError(
                    f"Failed to open TT device {args.tt_device_id} with strict_ttnn=True"
                ) from exc
            used_ttnn_device = False
            print(
                f"[ttnn_full] TT device unavailable on id={args.tt_device_id}; "
                "falling back to BF16 CPU path because --no-strict-ttnn is set."
            )
        try:
            runtime = TtGr00tActionHeadRuntime.load_weights(store=store, args=args, device=tt_device)
            ttnn_actions, ttnn_debug = runtime.run_denoise_loop(
                args,
                shared_inputs=shared_inputs,
                return_debug=True,
            )
        finally:
            if tt_device is not None:
                ttnn.close_device(tt_device)
    ttnn_debug["meta/used_ttnn_device"] = np.asarray([1 if used_ttnn_device else 0], dtype=np.int64)
    return ttnn_actions, ref_actions, ttnn_debug, ref_debug


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    print_preview = True
    save_payload: dict[str, Any] | None = None

    if args.backbone_only_compare:
        run_cpu_backbone_replay_compare(args)
        return

    if args.backbone_proj_compare:
        run_backbone_q_proj_compare(args)
        return

    if args.backbone_phaseb_compare:
        run_backbone_phaseb_compare(args)
        return

    if args.system2_encoder_compare or args.system2_block_compare or args.system2_trajectory_compare:
        _, store = load_model_dir_and_store(args)
        maybe_list_keys(store, args)
        shared_inputs = _resolve_shared_inputs(args, dump_system_inputs=False)
        _, _, ttnn_debug, ref_debug = run_ttnn_full_runtime(
            store,
            args,
            shared_inputs=shared_inputs,
        )
        used_ttnn_device = bool(
            int(ttnn_debug.get("meta/used_ttnn_device", np.asarray([0], dtype=np.int64))[0])
        )
        print("=== Phase C Standalone Compare ===")
        print(f"Model dir: {model_dir}")
        print(f"Input source: {args.input_source}")
        print(f"Strict TTNN mode: {args.strict_ttnn}")
        print(f"TTNN device used: {used_ttnn_device}")
        _run_phasec_reports(args, ttnn_debug=ttnn_debug, ref_debug=ref_debug)
        return

    if args.graph_mode == "cpu_ref":
        _, store = load_model_dir_and_store(args)
        maybe_list_keys(store, args)
        shared_inputs = _resolve_shared_inputs(
            args,
            dump_system_inputs=(args.input_source == "gr1"),
        )
        action_pred = run_full_action_head_reference(store, args, shared_inputs=shared_inputs)
        if args.input_source == "gr1":
            decoded_action, embodiment = _decode_action_for_demo(args, action_pred, shared_inputs)
        else:
            decoded_action, embodiment = None, None
        if decoded_action is not None and embodiment is not None:
            _print_demo_result(args=args, decoded_action=decoded_action, embodiment=embodiment)
            save_payload = {k: v.tolist() for k, v in decoded_action.items()}
            print_preview = False
        else:
            print("=== Isaac-GR00T CPU Reference Compute ===")
            print(f"Model dir: {model_dir}")
            print(f"Isaac-GR00T root: {Path(args.isaac_gr00t_root).resolve()}")
            print(f"Input source: {args.input_source}")
            print(f"Batch: {args.batch}, vl_seq_len: {args.vl_seq_len}")
            print(f"Embodiment id: {args.embodiment_id}")
            print(f"Action horizon: {args.action_horizon}")
            print(f"Diffusion timesteps: {args.num_inference_timesteps}")
            print(f"Output action shape: {action_pred.shape}")
        preview_tensor = action_pred
    else:
        _, store = load_model_dir_and_store(args)
        maybe_list_keys(store, args)
        shared_inputs = _resolve_shared_inputs(args, dump_system_inputs=False)
        action_pred, action_ref, ttnn_debug, ref_debug = run_ttnn_full_runtime(
            store,
            args,
            shared_inputs=shared_inputs,
        )
        max_abs_diff = float(np.max(np.abs(action_pred - action_ref)))
        worst_diff, worst_key, diff_lines = compare_tensor_maps(
            ttnn_debug, ref_debug, max_entries=int(args.dump_max_entries)
        )
        used_ttnn_device = bool(
            int(ttnn_debug.get("meta/used_ttnn_device", np.asarray([0], dtype=np.int64))[0])
        )
        phasec_lines = summarize_phasec_parity(ttnn_debug, ref_debug)
        print("=== Isaac-GR00T TTNN Full Runtime ===")
        print(f"Model dir: {model_dir}")
        print(f"Isaac-GR00T root: {Path(args.isaac_gr00t_root).resolve()}")
        print(f"Input source: {args.input_source}")
        print(f"TTNN device id: {args.tt_device_id}")
        print(f"Strict TTNN mode: {args.strict_ttnn}")
        print(f"TTNN device used: {used_ttnn_device}")
        print(f"Batch: {args.batch}, vl_seq_len: {args.vl_seq_len}")
        print(f"Embodiment id: {args.embodiment_id}")
        print(f"Action horizon: {args.action_horizon}")
        print(f"Diffusion timesteps: {args.num_inference_timesteps}")
        print(f"Transformer blocks executed: {args.ttnn_num_layers or 'all'}")
        print(f"Output action shape: {action_pred.shape}")
        print(f"max_abs_diff(TTNN vs CPU reference): {max_abs_diff:.6f}")
        print(f"Worst tensor diff: {worst_diff:.6f} at key: {worst_key or 'n/a'}")
        print("\nTensor diff summary:")
        for line in diff_lines:
            print(f"  - {line}")
        print("\nPhase C gate summary:")
        for line in phasec_lines:
            print(f"  - {line}")
        if args.dump_dir:
            dump_root = Path(args.dump_dir).resolve()
            dump_tensor_map(dump_root, "ttnn", ttnn_debug)
            dump_tensor_map(dump_root, "reference", ref_debug)
            print(f"\nDumped tensors to: {dump_root}")
        preview_tensor = action_pred

    if print_preview:
        print("\nFirst output row preview:")
        preview = preview_tensor[0, : min(8, preview_tensor.shape[1])].tolist()
        print(preview)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            if save_payload is not None:
                json.dump(save_payload, f, indent=2)
            else:
                json.dump(action_to_json(preview_tensor), f, indent=2)
        print(f"\nSaved action JSON to: {output_path}")


if __name__ == "__main__":
    main()
