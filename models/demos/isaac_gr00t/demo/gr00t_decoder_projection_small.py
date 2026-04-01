#!/usr/bin/env python3
"""Minimal TTNN decoder-projection repro for Isaac-GR00T.

This isolates the TTNN matmul + bias add used by `run_ttnn_decoder_projection`
without loading the full GR00T checkpoint or running the CPU reference block.
"""

from __future__ import annotations

import argparse

import numpy as np
import ttnn

from models.demos.isaac_gr00t.demo.gr00t import run_ttnn_decoder_projection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal GR00T decoder projection repro")
    parser.add_argument("--tt-device-id", type=int, default=0, help="TTNN device id")
    parser.add_argument("--batch", type=int, default=1, help="Batch dimension")
    parser.add_argument("--action-horizon", type=int, default=1, help="Action horizon")
    parser.add_argument("--hidden", type=int, default=1024, help="Decoder input hidden size")
    parser.add_argument("--action-dim", type=int, default=128, help="Decoder output width")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    pred_features = rng.standard_normal(
        (args.batch, args.action_horizon, args.hidden), dtype=np.float32
    )
    decoder_w = rng.standard_normal((args.hidden, args.action_dim), dtype=np.float32)
    decoder_b = rng.standard_normal((args.action_dim,), dtype=np.float32)

    print("[Minimal GR00T Decoder Projection Repro]")
    print(f"pred_features: {pred_features.shape}")
    print(f"decoder_w:     {decoder_w.shape}")
    print(f"decoder_b:     {decoder_b.shape}")

    device = ttnn.open_device(device_id=args.tt_device_id)
    try:
        out = run_ttnn_decoder_projection(pred_features, decoder_w, decoder_b, device)
    finally:
        ttnn.close_device(device)

    print(f"output:        {out.shape}")


if __name__ == "__main__":
    main()
