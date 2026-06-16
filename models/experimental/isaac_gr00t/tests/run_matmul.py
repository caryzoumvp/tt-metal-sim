#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Run a single ttnn.matmul with shapes passed as CLI args and check PCC.

Usage:
    # 2D weight: activation (M, K) x weight (K, N)
    python run_matmul.py --a 1 576 1152 --b 1152 1152

    # 4D batched: (B, H, M, K) x (B, H, K, N)
    python run_matmul.py --a 1 16 576 72 --b 1 16 72 576

    # custom PCC threshold
    python run_matmul.py --a 1 171 2048 --b 2048 6144 --pcc 0.97
"""

import argparse
import sys
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--a", nargs="+", type=int, required=True, metavar="D",
                   help="Shape of input tensor A, e.g. --a 1 576 1152")
    p.add_argument("--b", nargs="+", type=int, required=True, metavar="D",
                   help="Shape of weight/input tensor B, e.g. --b 1152 1152")
    p.add_argument("--pcc", type=float, default=0.98, metavar="F",
                   help="PCC pass threshold (default: 0.98)")
    p.add_argument("--device", type=int, default=0, metavar="ID",
                   help="TT device id (default: 0)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    a_shape = tuple(args.a)
    b_shape = tuple(args.b)

    print(f"A: {a_shape}  x  B: {b_shape}  ->  pcc>={args.pcc}")

    a = torch.randn(a_shape, dtype=torch.bfloat16)
    b = torch.randn(b_shape, dtype=torch.bfloat16)
    ref = torch.matmul(a, b)
    print(f"Reference output shape: {tuple(ref.shape)}")

    device = ttnn.open_device(device_id=args.device)
    try:
        tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
        tt_out = ttnn.matmul(tt_a, tt_b)
        out = ttnn.to_torch(tt_out)
    finally:
        ttnn.close_device(device)

    passed, pcc = assert_with_pcc(ref, out, args.pcc)
    status = "PASS" if passed else "FAIL"
    print(f"{status}  PCC={pcc:.6f}  (threshold={args.pcc})")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
