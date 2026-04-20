#!/usr/bin/env python3
"""TTNN-native Isaac-GR00T demo entrypoint.

This cleaned script keeps only the runtime paths used by `run_gr00t.sh`:
- `cpu_ref`
- `ttnn_full`
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

try:
    from .gr00t_host_reference import run_full_action_head_reference
    from .gr00t_ttnn_runtime import run_ttnn_full_runtime
except ImportError:
    from gr00t_host_reference import run_full_action_head_reference
    from gr00t_ttnn_runtime import run_ttnn_full_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TTNN-native Isaac-GR00T demo")
    parser.add_argument(
        "--model-dir",
        default="/workspaces/tensix/tt-metal/models/demos/isaac_gr00t/GR00T-N1.6-3B",
        help="Local GR00T model directory",
    )
    parser.add_argument("--tt-device-id", type=int, default=0, help="TTNN device id")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--num-inference-timesteps", type=int, default=4)
    parser.add_argument("--vl-seq-len", type=int, default=32)
    parser.add_argument(
        "--graph-mode",
        choices=("cpu_ref", "ttnn_full"),
        default="ttnn_full",
        help="`cpu_ref`: CPU reference compute; `ttnn_full`: TTNN runtime compared to CPU reference",
    )
    parser.add_argument(
        "--isaac-gr00t-root",
        default="/workspaces/tensix/Isaac-GR00T",
        help="Path to Isaac-GR00T source tree",
    )
    parser.add_argument("--embodiment-id", type=int, default=0, help="Category index for category-specific weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-source",
        choices=("synthetic", "gr1"),
        default="gr1",
        help="Input source for runtime execution",
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
    parser.add_argument("--task-text", default="", help="Optional language override for --input-source gr1")
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
    parser.add_argument("--dump-dir", default="", help="Optional directory to dump reference/ttnn tensors")
    parser.add_argument(
        "--dump-assembly-dir",
        default="",
        help="Optional directory to dump CPU assembly tensors",
    )
    parser.add_argument(
        "--save-shared-inputs-dir",
        default="",
        help="Optional directory to save shared inputs (.npy + meta.json)",
    )
    parser.add_argument(
        "--load-shared-inputs-dir",
        default="",
        help="Optional directory to load shared inputs (.npy + meta.json)",
    )
    parser.add_argument(
        "--log-ttnn-ops",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Log each TTNN op call with input tensor shapes",
    )
    parser.add_argument("--save-json", default="", help="Optional path to save output action JSON")
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
    # Save prepared System1 inputs so ttnn_full can execute System1 backbone runtime.
    system1_shared_flat: dict[str, np.ndarray] = {}
    _flatten_tensor_tree("system1_input", backbone_inputs, system1_shared_flat)
    shared_inputs.update(system1_shared_flat)

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
    lhs: dict[str, np.ndarray],
    rhs: dict[str, np.ndarray],
    *,
    max_entries: int = 40,
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
    return lines


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
    else:
        cfg = _load_model_cfg(args)
        shared_inputs = build_shared_action_head_inputs(
            args,
            state_dim=int(cfg["max_state_dim"]),
            backbone_dim=int(cfg["backbone_embedding_dim"]),
            action_dim=int(cfg["max_action_dim"]),
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


def main() -> None:
    args = parse_args()
    # TTNN runtime is strict-only.
    args.strict_ttnn = True
    model_dir = Path(args.model_dir).resolve()
    print_preview = True
    save_payload: dict[str, Any] | None = None

    if args.graph_mode == "cpu_ref":
        _, store = load_model_dir_and_store(args)
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
        shared_inputs = _resolve_shared_inputs(args, dump_system_inputs=True)
        action_pred, ttnn_debug = run_ttnn_full_runtime(
            store,
            args,
            shared_inputs=shared_inputs,
        )
        used_ttnn_device = bool(
            int(ttnn_debug.get("meta/used_ttnn_device", np.asarray([0], dtype=np.int64))[0])
        )
        if args.input_source == "gr1":
            decoded_action, embodiment = _decode_action_for_demo(args, action_pred, shared_inputs)
        else:
            decoded_action, embodiment = None, None
        if decoded_action is not None and embodiment is not None:
            _print_demo_result(args=args, decoded_action=decoded_action, embodiment=embodiment)
            save_payload = {k: v.tolist() for k, v in decoded_action.items()}
            print_preview = False

        print("=== Isaac-GR00T TTNN Full Runtime ===")
        print(f"Model dir: {model_dir}")
        print(f"Isaac-GR00T root: {Path(args.isaac_gr00t_root).resolve()}")
        print(f"Input source: {args.input_source}")
        print(f"TTNN device id: {args.tt_device_id}")
        print(f"TTNN device used: {used_ttnn_device}")
        print(f"Batch: {args.batch}, vl_seq_len: {args.vl_seq_len}")
        print(f"Embodiment id: {args.embodiment_id}")
        print(f"Action horizon: {args.action_horizon}")
        print(f"Diffusion timesteps: {args.num_inference_timesteps}")
        print(f"Transformer blocks executed: {args.ttnn_num_layers or 'all'}")
        print(f"Output action shape: {action_pred.shape}")
        if args.dump_dir:
            dump_root = Path(args.dump_dir).resolve()
            dump_tensor_map(dump_root, "ttnn", ttnn_debug)
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
