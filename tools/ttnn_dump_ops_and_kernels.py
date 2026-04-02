#!/usr/bin/env python3
import argparse
import pathlib
import sqlite3
from collections import defaultdict


def query_ops(db_path: pathlib.Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT operation_id, name FROM operations ORDER BY operation_id")
    ops = cur.fetchall()

    tensor_shape = {}
    cur.execute("SELECT tensor_id, shape FROM tensors")
    for tid, shape in cur.fetchall():
        tensor_shape[tid] = shape

    in_shapes = defaultdict(list)
    cur.execute("SELECT operation_id, input_index, tensor_id FROM input_tensors ORDER BY operation_id, input_index")
    for op_id, _, tid in cur.fetchall():
        in_shapes[op_id].append(tensor_shape.get(tid, f"<tensor:{tid}>"))

    out_shapes = defaultdict(list)
    cur.execute("SELECT operation_id, output_index, tensor_id FROM output_tensors ORDER BY operation_id, output_index")
    for op_id, _, tid in cur.fetchall():
        out_shapes[op_id].append(tensor_shape.get(tid, f"<tensor:{tid}>"))

    conn.close()

    result = []
    for op_id, name in ops:
        result.append((op_id, name, in_shapes.get(op_id, []), out_shapes.get(op_id, [])))
    return result


def query_kernel_names(cache_root: pathlib.Path):
    kernels = set()
    for csv in cache_root.glob("*/kernels/kernel_args.csv"):
        try:
            for line in csv.read_text(errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                # line starts with .../kernels/<kernel_name>/<hash>/<risc>/, ...
                marker = "/kernels/"
                idx = line.find(marker)
                if idx == -1:
                    continue
                tail = line[idx + len(marker) :]
                kname = tail.split("/", 1)[0]
                if kname:
                    kernels.add(kname)
        except Exception:
            continue
    return sorted(kernels)


def main():
    ap = argparse.ArgumentParser(description="Dump TTNN op I/O shapes and compiled kernel names")
    ap.add_argument("--report-path", required=True, help="TTNN report path that contains db.sqlite")
    ap.add_argument(
        "--cache-root",
        default=str(pathlib.Path.home() / ".cache" / "tt-metal-cache"),
        help="TT_METAL cache root (default: ~/.cache/tt-metal-cache)",
    )
    ap.add_argument("--filter", default="", help="Substring filter for op name and kernel name")
    args = ap.parse_args()

    report = pathlib.Path(args.report_path)
    db = report / "db.sqlite"
    if not db.exists():
        raise SystemExit(f"db.sqlite not found at {db}")

    filt = args.filter.lower().strip()

    print("=== TTNN Operations (with input/output shapes) ===")
    for op_id, name, ins, outs in query_ops(db):
        if filt and filt not in name.lower():
            continue
        print(f"[{op_id:6d}] {name}")
        print(f"  in : {ins}")
        print(f"  out: {outs}")

    print("\n=== Compiled/Dispatched Kernel Names (from kernel_args.csv) ===")
    kernels = query_kernel_names(pathlib.Path(args.cache_root))
    for k in kernels:
        if filt and filt not in k.lower():
            continue
        print(k)


if __name__ == "__main__":
    main()
