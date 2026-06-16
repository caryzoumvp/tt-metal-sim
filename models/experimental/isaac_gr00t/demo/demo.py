#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
GR00T N1.6 — TTNN action-head demo.

Runs the full end-to-end pipeline:
  1. Eagle-3-VL backbone (TTNN on TT device) — image + text → vl_embeds
  2. AlternateVLDiT denoising loop (TTNN ops on TT device) — vl_embeds + state → action

Pipeline:
  image  [B, 3, 336, 336] ──┐
  text   instruction       ──┤ TtEagleBackbone (SigLIP2 + mlp1 + Qwen3, TTNN)
                             ↓
  vl_embeds [B, S, 2048] ───┐
  state     [B, 1, 128]  ───┤ AlternateVLDiT × 4 denoising steps  (TTNN ops)
  embodiment_id [B]      ───┘
                             ↓
  action [B, 16, 128]

Usage (run from tt-metal root, or via demo/run_gr00t.sh):
    python models/experimental/isaac_gr00t/demo/demo.py
    python models/experimental/isaac_gr00t/demo/demo.py --benchmark --iterations 20
    python models/experimental/isaac_gr00t/demo/demo.py --output /tmp/gr00t_tt.npy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ttnn

# ── Paths ─────────────────────────────────────────────────────────────────────
_DEMO_DIR  = Path(__file__).resolve().parent   # demo/
_MODEL_DIR = _DEMO_DIR.parent                  # isaac_gr00t/
_REFS_DIR  = _MODEL_DIR / "references"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "GR00T-N1.6-3B"

for _p in (str(_MODEL_DIR), str(_REFS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tt.gr00t_ttnn_runtime import TtGr00tActionHeadRuntime  # noqa: E402
from tt.eagle_backbone import TtEagleBackbone               # noqa: E402

_EAGLE_MODEL_PATH = "/workspaces/tensix/Isaac-GR00T/gr00t/model/modules/nvidia/Eagle-Block2A-2B-v2"


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def _load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    """Load all safetensors shards into one flat dict.

    Python dicts expose .get() so the dict is passed directly as `store`
    to TtGr00tActionHeadRuntime — no adapter needed.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        raise RuntimeError("pip install safetensors")

    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"

    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif single_path.exists():
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")

    weights: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
    print(f"Loaded {len(weights)} tensors from {model_dir}")
    return weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GR00T N1.6 TTNN demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model-path", default=str(_DEFAULT_MODEL_PATH),
                   help="Path to GR00T-N1.6-3B checkpoint directory")
    p.add_argument("--tt-device-id",  type=int, default=0)
    p.add_argument("--embodiment-id", type=int, default=0,
                   help="Embodiment index (GR1=0)")
    p.add_argument("--batch",          type=int, default=1)
    p.add_argument("--action-horizon", type=int, default=16,
                   help="Action chunk length (GR1=16)")
    p.add_argument("--num-inference-timesteps", type=int, default=4)
    p.add_argument("--num-layers",     type=int, default=0,
                   help="Limit DiT layers (0 = all 32)")
    p.add_argument("--siglip-layers",  type=int, default=27,
                   help="Limit SigLIP2 vision backbone layers")
    p.add_argument("--qwen3-layers",   type=int, default=16,
                   help="Limit Qwen3 language backbone layers")
    p.add_argument("--image",          default="",
                   help="Path to an RGB image file (PNG/JPG, any size; resized to 336×336). "
                        "If omitted, a black 336×336 image is used.")
    p.add_argument("--instruction",    default="",
                   help="Language instruction passed to the Eagle-3-VL backbone.")
    p.add_argument("--seed",           type=int, default=0)
    p.add_argument("--benchmark",      action="store_true")
    p.add_argument("--iterations",     type=int, default=None)
    p.add_argument("--output",         default="",
                   help="Save final action array to this .npy path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    model_dir  = Path(args.model_path).resolve()
    iterations = args.iterations if args.iterations is not None else (20 if args.benchmark else 3)

    print("=" * 64)
    print("GR00T N1.6 — TTNN demo")
    print("=" * 64)
    print(f"Model:       {model_dir}")
    print(f"TT device:   {args.tt_device_id}")
    print(f"Embodiment:  {args.embodiment_id}")
    print(f"Action H:    {args.action_horizon}")
    print(f"DiT steps:   {args.num_inference_timesteps}")
    print(f"SigLIP2 L:   {args.siglip_layers}")
    print(f"Qwen3 L:     {args.qwen3_layers}")
    print(f"DiT L:       {args.num_layers if args.num_layers else 'all'}")
    print(f"Iterations:  {iterations}  ({'benchmark' if args.benchmark else 'run'})")
    print()

    if not model_dir.exists():
        print(f"ERROR: model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load weights ──────────────────────────────────────────────────────────
    print("Loading safetensors ...")
    t0 = time.perf_counter()
    store = _load_safetensors(model_dir)
    print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    # ── Raw model inputs ─────────────────────────────────────────────────────
    B           = args.batch
    instruction = args.instruction

    if args.image:
        from PIL import Image as _PILImage
        _img = _PILImage.open(args.image).convert("RGB").resize((336, 336))
        _arr = np.asarray(_img, dtype=np.float32) / 255.0
        image = np.tile(_arr.transpose(2, 0, 1)[None], (B, 1, 1, 1))
    else:
        image = np.zeros((B, 3, 336, 336), dtype=np.float32)
    state         = np.zeros((B, 1, 128), dtype=np.float32)
    embodiment_id = np.zeros(B, dtype=np.int64)

    _img_src = args.image if args.image else "zeros (black)"
    print("Raw model inputs:")
    print(f"  image          {image.shape}  ({_img_src})")
    print(f"  instruction    {repr(instruction)}")
    print(f"  state          {state.shape}  (robot joint/pose)")
    print(f"  embodiment_id  {embodiment_id.shape}")
    print()

    # ── Eagle processor (CPU, for tokenisation + image preprocessing) ────────
    print("Loading Eagle processor ...")
    t0 = time.perf_counter()
    from transformers import AutoProcessor as _AutoProcessor
    processor = _AutoProcessor.from_pretrained(_EAGLE_MODEL_PATH, trust_remote_code=True)
    print(f"  done in {time.perf_counter() - t0:.1f}s\n")

    # Initial noisy action sequence (flow matching starts from Gaussian noise)
    rng = np.random.default_rng(args.seed)
    actions_init = rng.standard_normal((B, args.action_horizon, 128)).astype(np.float32)

    # shared_inputs skeleton — vl_embeds / image_mask / backbone_attention_mask
    # are filled in after TtEagleBackbone runs (inside the device block below).
    shared_inputs: dict[str, np.ndarray] = {
        "state":   state,
        "actions": actions_init,
    }

    # ── Runtime args namespace (fields expected by TtGr00tActionHeadRuntime) ─
    # vl_seq_len is set dynamically after the backbone forward pass.
    rt_args = SimpleNamespace(
        model_dir=str(model_dir),
        embodiment_id=args.embodiment_id,
        ttnn_num_layers=args.num_layers,
        action_horizon=args.action_horizon,
        num_inference_timesteps=args.num_inference_timesteps,
        vl_seq_len=0,   # updated after backbone forward
        batch=B,
        seed=args.seed,
    )

    # ── Open TT device ────────────────────────────────────────────────────────
    print(f"Opening TT device {args.tt_device_id} ...")
    device = None
    try:
        device = ttnn.open_device(device_id=args.tt_device_id)
        print("  device opened\n")
    except Exception as exc:
        print(f"  WARNING: device open failed ({exc}); dry-run only\n")

    try:
        # ── Build TtEagleBackbone and upload weights to device ────────────────
        print("Building TtEagleBackbone and uploading weights ...")
        t0 = time.perf_counter()
        tt_backbone = TtEagleBackbone.load_weights(
            store=store,
            device=device,
            processor=processor,
            siglip_layers=args.siglip_layers,
            qwen3_layers=args.qwen3_layers,
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s\n")

        # ── Eagle-3-VL backbone: image + text → vl_embeds (TTNN) ─────────────
        print("Eagle-3-VL backbone forward (TTNN) ...")
        t0 = time.perf_counter()
        vl_embeds, backbone_attention_mask, image_mask = tt_backbone.forward(
            image, [instruction] * B
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s")

        vl_seq_len = vl_embeds.shape[1]
        n_img_tok  = int(image_mask[0].sum())
        n_text_tok = int((~image_mask[0] & backbone_attention_mask[0]).sum())
        print(f"  vl_embeds               {vl_embeds.shape}")
        print(f"  backbone_attention_mask {backbone_attention_mask.shape}")
        print(f"  image_mask              {image_mask.shape}  ({n_img_tok} image / {n_text_tok} text tokens)")
        print()

        # Propagate actual vl_seq_len to rt_args
        rt_args.vl_seq_len = vl_seq_len

        # shared_inputs for denoising loop (update now that we have backbone output)
        shared_inputs["vl_embeds"]               = vl_embeds
        shared_inputs["image_mask"]              = image_mask
        shared_inputs["backbone_attention_mask"] = backbone_attention_mask

        # ── Build TTNN runtime and upload weights to device ───────────────────
        print("Building TtGr00tActionHeadRuntime and uploading weights ...")
        t0 = time.perf_counter()
        runtime = TtGr00tActionHeadRuntime.load_weights(
            store=store, args=rt_args, device=device
        )
        print(f"  done in {time.perf_counter() - t0:.1f}s\n")

        # ── TTNN denoising loop ───────────────────────────────────────────────
        # Each iteration re-clones shared_inputs so noise init is identical
        # across iterations (consistent with the reference benchmark).
        print(f"Running {iterations} TTNN denoising loop iteration(s) ...")
        latencies: list[float] = []
        last_action: np.ndarray | None = None

        for i in range(iterations):
            t0 = time.perf_counter()
            result = runtime.run_denoise_loop(rt_args, shared_inputs=shared_inputs)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)

            action = result[0] if isinstance(result, tuple) else result
            last_action = action

            if not args.benchmark or i == 0:
                print(f"  iter {i:3d}: {elapsed * 1000:.1f} ms   output {tuple(action.shape)}")

        print()

        # ── Timing ───────────────────────────────────────────────────────────
        if len(latencies) > 1:
            warm    = latencies[1:]
            mean_ms = 1000 * float(np.mean(warm))
            std_ms  = 1000 * float(np.std(warm))
            fps     = 1.0 / float(np.mean(warm))
            print(f"Timing ({len(warm)} warm iterations):")
            print(f"  mean: {mean_ms:.1f} ms   std: {std_ms:.1f} ms   fps: {fps:.2f}")
        else:
            ms = latencies[0] * 1000
            print(f"Latency: {ms:.1f} ms ({1000 / ms:.2f} fps)")

        # ── Output ────────────────────────────────────────────────────────────
        assert last_action is not None
        print()
        print("=== GR00T TTNN action-head output ===")
        print(f"  action shape:    {tuple(last_action.shape)}")
        print(f"  action[0,0,:8] = {last_action[0, 0, :8].tolist()}")
        print(f"  mean={float(last_action.mean()):.6f}   std={float(last_action.std()):.6f}")

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, last_action)
            print(f"\nSaved action to: {out_path}")

    finally:
        if device is not None:
            ttnn.close_device(device)


if __name__ == "__main__":
    main()
