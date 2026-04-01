#!/usr/bin/env python3
"""Export TTNN demo reference tensors from a local GR00T model checkpoint.

Outputs:
  - x.npy
  - w1.npy
  - b1.npy
  - w2.npy
  - b2.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export reference tensors from local GR00T model")
    p.add_argument("--model-dir", required=True, help="Local model directory (HF-style)")
    p.add_argument("--out-dir", required=True, help="Output directory for x/w/b npy files")
    p.add_argument("--w1-key", default="", help="Optional explicit first weight key")
    p.add_argument("--w2-key", default="", help="Optional explicit second weight key")
    p.add_argument("--batch", type=int, default=1, help="Input batch size")
    p.add_argument("--seed", type=int, default=0, help="Seed for random x")
    p.add_argument("--list-only", action="store_true", help="Only list candidate 2D weight keys")
    return p.parse_args()


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
        try:
            from safetensors import safe_open
        except Exception as exc:
            raise RuntimeError("Please install safetensors: pip install safetensors") from exc
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


def linear_candidates(store: TensorStore) -> list[str]:
    keys = []
    for key in store.keys():
        if not key.endswith(".weight"):
            continue
        t = store.get(key)
        if t.ndim == 2 and t.dtype.is_floating_point:
            keys.append(key)
    return keys


def to_io_weight(weight: torch.Tensor) -> torch.Tensor:
    # PyTorch linear weight is [out, in]. Demo expects [in, out].
    return weight.transpose(0, 1).contiguous().to(torch.float32)


def find_pair(store: TensorStore, w1_key: str, w2_key: str) -> tuple[str, str]:
    if w1_key and w2_key:
        return w1_key, w2_key

    cands = linear_candidates(store)
    for k1 in cands:
        w1 = to_io_weight(store.get(k1))
        for k2 in cands:
            w2 = to_io_weight(store.get(k2))
            if w1.shape[1] == w2.shape[0]:
                return k1, k2
    raise RuntimeError("Could not auto-find a compatible pair. Pass --w1-key and --w2-key.")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    store = load_store(model_dir)
    cands = linear_candidates(store)
    if args.list_only:
        print("Candidate linear weight keys:")
        for key in cands:
            print(key)
        return

    w1_key, w2_key = find_pair(store, args.w1_key, args.w2_key)
    w1 = to_io_weight(store.get(w1_key))
    w2 = to_io_weight(store.get(w2_key))

    b1_key = w1_key.replace(".weight", ".bias")
    b2_key = w2_key.replace(".weight", ".bias")
    if b1_key in store.keys():
        b1 = store.get(b1_key).to(torch.float32)
    else:
        b1 = torch.zeros((w1.shape[1],), dtype=torch.float32)
    if b2_key in store.keys():
        b2 = store.get(b2_key).to(torch.float32)
    else:
        b2 = torch.zeros((w2.shape[1],), dtype=torch.float32)

    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal((args.batch, int(w1.shape[0])), dtype=np.float32)

    np.save(out_dir / "x.npy", x)
    np.save(out_dir / "w1.npy", w1.cpu().numpy())
    np.save(out_dir / "b1.npy", b1.cpu().numpy())
    np.save(out_dir / "w2.npy", w2.cpu().numpy())
    np.save(out_dir / "b2.npy", b2.cpu().numpy())

    print(f"Exported reference tensors to: {out_dir}")
    print(f"w1_key={w1_key}, shape={tuple(w1.shape)}")
    print(f"w2_key={w2_key}, shape={tuple(w2.shape)}")


if __name__ == "__main__":
    main()
