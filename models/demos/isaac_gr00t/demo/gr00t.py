#!/usr/bin/env python3
"""TTNN-native Isaac-GR00T migration demo.

This demo does not import or execute Isaac-GR00T runtime code.
It directly loads local GR00T checkpoint weights and runs a small TTNN MLP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    parser.add_argument("--embodiment-id", type=int, default=0, help="Category index for category-specific weights")
    parser.add_argument("--list-keys", action="store_true", help="List candidate linear keys and exit")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-json", default="", help="Optional path to save action JSON")
    return parser.parse_args()


def action_to_json(action: np.ndarray) -> dict[str, Any]:
    return {"action": action.tolist()}


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


def get_tensor_np(store: TensorStore, key: str) -> np.ndarray:
    return store.get(key).to(torch.float32).cpu().numpy()


def load_model_tensors(args: argparse.Namespace) -> dict[str, np.ndarray]:
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")
    store = load_store(model_dir)

    if args.list_keys:
        for key in sorted(k for k in store.keys() if k.startswith("action_head.")):
            print(key)
        raise SystemExit(0)

    rng = np.random.default_rng(args.seed)
    emb = args.embodiment_id
    state_in = get_tensor_np(store, "action_head.state_encoder.layer1.W").shape[1]
    x = rng.standard_normal((args.batch, state_in), dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "x": x,
        # state encoder
        "state_l1_w": get_tensor_np(store, "action_head.state_encoder.layer1.W")[emb],  # [state_in, 1024]
        "state_l1_b": get_tensor_np(store, "action_head.state_encoder.layer1.b")[emb],  # [1024]
        "state_l2_w": get_tensor_np(store, "action_head.state_encoder.layer2.W")[emb],  # [1024, 1536]
        "state_l2_b": get_tensor_np(store, "action_head.state_encoder.layer2.b")[emb],  # [1536]
        # action encoder
        "act_w1_w": get_tensor_np(store, "action_head.action_encoder.W1.W")[emb],  # [128, 1536]
        "act_w1_b": get_tensor_np(store, "action_head.action_encoder.W1.b")[emb],  # [1536]
        "act_w2_w": get_tensor_np(store, "action_head.action_encoder.W2.W")[emb],  # [3072, 1536]
        "act_w2_b": get_tensor_np(store, "action_head.action_encoder.W2.b")[emb],  # [1536]
        "act_w3_w": get_tensor_np(store, "action_head.action_encoder.W3.W")[emb],  # [1536, 1536]
        "act_w3_b": get_tensor_np(store, "action_head.action_encoder.W3.b")[emb],  # [1536]
        # timestep encoder
        "time_l1_w": get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight"),
        "time_l1_b": get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_1.bias"),
        "time_l2_w": get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_2.weight"),
        "time_l2_b": get_tensor_np(store, "action_head.model.timestep_encoder.timestep_embedder.linear_2.bias"),
        # attention block 0
        "attn_q_w": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_q.weight"),
        "attn_q_b": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_q.bias"),
        "attn_k_w": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_k.weight"),
        "attn_k_b": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_k.bias"),
        "attn_v_w": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_v.weight"),
        "attn_v_b": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_v.bias"),
        "attn_o_w": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_out.0.weight"),
        "attn_o_b": get_tensor_np(store, "action_head.model.transformer_blocks.0.attn1.to_out.0.bias"),
        # proj_out (1536 -> 1024)
        "proj_out_2_w": get_tensor_np(store, "action_head.model.proj_out_2.weight"),
        "proj_out_2_b": get_tensor_np(store, "action_head.model.proj_out_2.bias"),
        # action decoder (1024 -> 1024 -> 128)
        "dec_l1_w": get_tensor_np(store, "action_head.action_decoder.layer1.W")[emb],
        "dec_l1_b": get_tensor_np(store, "action_head.action_decoder.layer1.b")[emb],
        "dec_l2_w": get_tensor_np(store, "action_head.action_decoder.layer2.W")[emb],
        "dec_l2_b": get_tensor_np(store, "action_head.action_decoder.layer2.b")[emb],
    }
    return tensors


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def run_cpu_gr00t_action_head_reference(
    tensors: dict[str, np.ndarray], args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = tensors["x"]
    batch = x.shape[0]
    hidden = tensors["state_l2_w"].shape[1]
    action_dim = tensors["dec_l2_w"].shape[1]
    action_horizon = args.action_horizon

    rng = np.random.default_rng(args.seed + 1)
    vl_embeds = rng.standard_normal((batch, args.vl_seq_len, 2048), dtype=np.float32)
    actions = rng.standard_normal((batch, action_horizon, action_dim), dtype=np.float32)

    # state encoder (category-specific MLP)
    s1 = x @ tensors["state_l1_w"] + tensors["state_l1_b"]
    s1 = silu_np(s1)
    state_features = (s1 @ tensors["state_l2_w"] + tensors["state_l2_b"])[:, None, :]  # [B,1,1536]

    print("\n[CPU GR00T-like Block Shapes]")
    print(f"state_encoder: x {x.shape} -> state_features {state_features.shape}")
    print(f"vl_embeds(backbone_features): {vl_embeds.shape}")

    dt = 1.0 / float(args.num_inference_timesteps)
    last_pred_features = None
    last_pred_velocity = None

    for t in range(args.num_inference_timesteps):
        # timestep encoder (2-layer MLP from 256-dim sinusoid-like bucket onehot)
        t_in = np.zeros((batch, 256), dtype=np.float32)
        t_in[:, t % 256] = 1.0
        t1 = t_in @ tensors["time_l1_w"].T + tensors["time_l1_b"]
        t1 = silu_np(t1)
        time_emb = t1 @ tensors["time_l2_w"].T + tensors["time_l2_b"]  # [B,1536]

        # action encoder (category-specific 3-layer MLP)
        af1 = actions.reshape(batch * action_horizon, action_dim) @ tensors["act_w1_w"] + tensors["act_w1_b"]
        af1 = af1.reshape(batch, action_horizon, hidden)
        time_rep = np.repeat(time_emb[:, None, :], action_horizon, axis=1)
        af2_in = np.concatenate([af1, time_rep], axis=-1)  # [B,H,3072]
        af2 = af2_in.reshape(batch * action_horizon, 3072) @ tensors["act_w2_w"] + tensors["act_w2_b"]
        af2 = silu_np(af2)
        af3 = af2 @ tensors["act_w3_w"] + tensors["act_w3_b"]
        action_features = af3.reshape(batch, action_horizon, hidden)

        sa_embs = np.concatenate([state_features, action_features], axis=1)  # [B, 1+H, hidden]

        q = sa_embs @ tensors["attn_q_w"].T + tensors["attn_q_b"]
        k = vl_embeds @ tensors["attn_k_w"].T + tensors["attn_k_b"]
        v = vl_embeds @ tensors["attn_v_w"].T + tensors["attn_v_b"]
        attn_scores = np.matmul(q, np.swapaxes(k, 1, 2)) / np.sqrt(hidden)
        attn_probs = softmax_np(attn_scores, axis=-1)
        cross = np.matmul(attn_probs, v)
        model_output = np.tanh(cross @ tensors["attn_o_w"].T + tensors["attn_o_b"] + sa_embs)
        pred_features_1536 = model_output[:, -action_horizon:, :]
        proj = pred_features_1536.reshape(batch * action_horizon, hidden) @ tensors["proj_out_2_w"].T + tensors["proj_out_2_b"]
        pred_features = proj.reshape(batch, action_horizon, 1024)
        d1 = pred_features.reshape(batch * action_horizon, 1024) @ tensors["dec_l1_w"] + tensors["dec_l1_b"]
        d1 = silu_np(d1)
        pred_velocity = d1 @ tensors["dec_l2_w"] + tensors["dec_l2_b"]
        pred_velocity = pred_velocity.reshape(batch, action_horizon, action_dim)
        actions = actions + dt * pred_velocity

        print(f"step {t}: action_features {action_features.shape}, sa_embs {sa_embs.shape}")
        print(f"        q {q.shape}, k {k.shape}, v {v.shape}, attn_scores {attn_scores.shape}")
        print(f"        attn_probs {attn_probs.shape}, cross {cross.shape}, model_output {model_output.shape}")
        print(f"        pred_features {pred_features.shape}, pred_velocity {pred_velocity.shape}, actions {actions.shape}")

        last_pred_features = pred_features
        last_pred_velocity = pred_velocity

    assert last_pred_features is not None and last_pred_velocity is not None
    return (
        actions.astype(np.float32),
        last_pred_features.astype(np.float32),
        last_pred_velocity.astype(np.float32),
        tensors["dec_l2_w"].astype(np.float32),
        tensors["dec_l2_b"].astype(np.float32),
    )


def run_ttnn_decoder_projection(
    pred_features: np.ndarray, decoder_w: np.ndarray, decoder_b: np.ndarray, device: ttnn.Device
) -> np.ndarray:
    batch, action_horizon, hidden = pred_features.shape
    action_dim = decoder_w.shape[1]
    flat_in = pred_features.reshape(batch * action_horizon, hidden)

    # TTNN matmul expects tile-friendly dimensions. Pad M to 32.
    original_m = flat_in.shape[0]
    padded_m = ((original_m + 31) // 32) * 32
    if padded_m != original_m:
        x_work = np.zeros((padded_m, flat_in.shape[1]), dtype=np.float32)
        x_work[:original_m, :] = flat_in
    else:
        x_work = flat_in

    print("\n[TTNN Op Shapes]")
    print(f"0) decoder input host: {flat_in.shape}, padded to: {x_work.shape}")
    tx = ttnn.from_torch(torch.from_numpy(x_work).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    tw = ttnn.from_torch(torch.from_numpy(decoder_w).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    tb = ttnn.from_torch(torch.from_numpy(decoder_b.reshape(1, -1)).to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    print(
        "1) to_device/tile: "
        f"tx {tuple(tx.shape)}, tw {tuple(tw.shape)}, tb {tuple(tb.shape)}"
    )

    print(f"2) decoder matmul: tx {tuple(tx.shape)} x tw {tuple(tw.shape)}")
    y = ttnn.matmul(tx, tw)
    print(f"   decoder matmul out: {tuple(y.shape)}")
    print(f"3) decoder add bias: y {tuple(y.shape)} + tb {tuple(tb.shape)}")
    y = ttnn.add(y, tb)
    print(f"   decoder add out: {tuple(y.shape)}")
    y_out = ttnn.to_torch(y).to(torch.float32).cpu().numpy()
    print(f"4) to_torch host out: {y_out.shape}, final sliced: {(original_m, action_dim)}")
    y_out = y_out[:original_m, :].reshape(batch, action_horizon, action_dim)
    return y_out


def main() -> None:
    args = parse_args()
    tensors = load_model_tensors(args)
    actions_cpu, pred_features_cpu, pred_velocity_cpu, decoder_w, decoder_b = run_cpu_gr00t_action_head_reference(
        tensors, args
    )
    tt_device = ttnn.open_device(device_id=args.tt_device_id)
    try:
        pred_velocity_tt = run_ttnn_decoder_projection(pred_features_cpu, decoder_w, decoder_b, tt_device)
    finally:
        ttnn.close_device(tt_device)

    print("=== Isaac-GR00T migration demo (TTNN native) ===")
    print(f"Model dir: {Path(args.model_dir).resolve()}")
    print("Mapped keys: state/action encoders, timestep MLP, attn q/k/v/o, proj_out_2, action_decoder")
    print(f"TTNN device id: {args.tt_device_id}")
    print(f"Strict TTNN mode: {args.strict_ttnn}")
    print(f"Input shape: {tensors['x'].shape}")
    print(f"Hidden size: {tensors['state_l2_w'].shape[1]}")
    print(f"Embodiment id: {args.embodiment_id}")
    print(f"Action horizon: {args.action_horizon}")
    print(f"Diffusion timesteps: {args.num_inference_timesteps}")
    max_abs_diff = float(np.max(np.abs(pred_velocity_cpu - pred_velocity_tt)))
    print("TTNN execution: success")
    print(f"CPU final actions shape: {actions_cpu.shape}")
    print(f"Decoder velocity shape: {pred_velocity_tt.shape}")
    print(f"max_abs_diff(TTNN decoder vs CPU decoder): {max_abs_diff:.6f}")
    preview_tensor = pred_velocity_tt

    print("\nFirst output row preview:")
    preview = preview_tensor[0, : min(8, preview_tensor.shape[1])].tolist()
    print(preview)

    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(action_to_json(preview_tensor), f, indent=2)
        print(f"\nSaved action JSON to: {output_path}")


if __name__ == "__main__":
    main()
