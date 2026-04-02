#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/tensix/tt-metal

source ~/.conda_env.rc
conda activate tt-isaac-gr00t

source tt_env.sh

# Simulator/slow-dispatch friendly defaults.
export TT_METAL_CACHE=${TT_METAL_CACHE:-/tmp/tt-metal-cache-qwen25-b1}
export TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT=${TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT:-0}
mkdir -p "$TT_METAL_CACHE"

MESH_DEVICE=${MESH_DEVICE:-N150}
HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}
GRAPH_REPORT_DIR=${GRAPH_REPORT_DIR:-/tmp/qwen25_graph_capture}
GRAPH_CAPTURE_TARGET=${GRAPH_CAPTURE_TARGET:-prefill}
GRAPH_CAPTURE_CALL_INDEX=${GRAPH_CAPTURE_CALL_INDEX:-2}
mkdir -p "$GRAPH_REPORT_DIR"

MESH_DEVICE="$MESH_DEVICE" \
HF_MODEL="$HF_MODEL" \
GRAPH_REPORT_DIR="$GRAPH_REPORT_DIR" \
GRAPH_CAPTURE_TARGET="$GRAPH_CAPTURE_TARGET" \
GRAPH_CAPTURE_CALL_INDEX="$GRAPH_CAPTURE_CALL_INDEX" \
python - <<'PYCODE'
import json
import os
import re
from pathlib import Path

import pytest
import ttnn
from ttnn.graph_tracer_utils import GraphTracerUtils

report_dir = Path(os.environ["GRAPH_REPORT_DIR"])
target = os.environ.get("GRAPH_CAPTURE_TARGET", "prefill").strip().lower()
target_call_index = int(os.environ.get("GRAPH_CAPTURE_CALL_INDEX", "2"))

raw_path = report_dir / f"qwen25_{target}_capture_raw.json"
args_path = report_dir / f"qwen25_{target}_capture_args.json"
levelized_path = report_dir / f"qwen25_{target}_capture_levelized.json"
summary_path = report_dir / f"qwen25_{target}_ops.txt"
neighbors_json_path = report_dir / f"qwen25_{target}_op_neighbors.json"
neighbors_txt_path = report_dir / f"qwen25_{target}_op_neighbors.txt"

shape_pattern = re.compile(r"Shape\(\[([^\]]+)\]\)")


class CaptureComplete(RuntimeError):
    pass


def sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(key): sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return value


def is_tensor_vertex(vertex):
    return vertex.get("name", "").startswith("tensor[")


def resolve_upstream_ops(vertex, by_counter):
    upstream = []
    for edge_counter in vertex.get("in_edges", []):
        source = by_counter.get(edge_counter)
        if source is None:
            continue
        if is_tensor_vertex(source):
            if source.get("in_edges"):
                for producer_counter in source.get("in_edges", []):
                    producer = by_counter.get(producer_counter)
                    if producer is not None and not is_tensor_vertex(producer):
                        upstream.append(producer)
            else:
                upstream.append(source)
        else:
            upstream.append(source)
    return upstream


def resolve_downstream_ops(vertex, by_counter):
    downstream = []
    for edge_counter in vertex.get("out_edges", []):
        sink = by_counter.get(edge_counter)
        if sink is None:
            continue
        if is_tensor_vertex(sink):
            for consumer_counter in sink.get("out_edges", []):
                consumer = by_counter.get(consumer_counter)
                if consumer is not None and not is_tensor_vertex(consumer):
                    downstream.append(consumer)
        else:
            downstream.append(sink)
    return downstream


def write_capture(captured_graph):
    raw_path.write_text(json.dumps(captured_graph, indent=2), encoding="utf-8")

    serialized = sanitize_for_json(GraphTracerUtils.serialize_graph(captured_graph))
    args_path.write_text(json.dumps(serialized, indent=2), encoding="utf-8")

    levelized_graph = ttnn.graph.extract_levelized_graph(captured_graph)
    levelized_path.write_text(json.dumps(levelized_graph, indent=2), encoding="utf-8")
    by_counter = {vertex["counter"]: vertex for vertex in levelized_graph}

    lines = [
        f"target={target}",
        f"call_index={target_call_index}",
        f"num_nodes={len(captured_graph)}",
        "",
        "-- function_start ops --",
    ]

    for node in captured_graph:
        if node.get("node_type") != "function_start":
            continue
        op_name = node.get("params", {}).get("name", "<unknown>")
        arg_shapes = []
        for arg in node.get("arguments", []):
            for match in shape_pattern.findall(str(arg)):
                dims = [dim.strip() for dim in match.split(",") if dim.strip()]
                arg_shapes.append(dims)
        lines.append(f"{op_name}")
        if arg_shapes:
            lines.append(f"  arg_shapes={arg_shapes}")

    lines.extend(["", "-- levelized graph --"])
    for vertex in levelized_graph:
        name = vertex.get("name", "<unknown>")
        output_shape = vertex.get("output_shape", [])
        stacking_level = vertex.get("stacking_level")
        lines.append(f"{name}")
        lines.append(f"  level={stacking_level} output_shape={output_shape}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    neighbors = []
    neighbors_lines = []
    for vertex in levelized_graph:
        if is_tensor_vertex(vertex):
            continue

        upstream = resolve_upstream_ops(vertex, by_counter)
        downstream = resolve_downstream_ops(vertex, by_counter)

        entry = {
            "counter": vertex.get("counter"),
            "name": vertex.get("name", "<unknown>"),
            "input_shapes": [
                producer.get("output_shape", [])
                for producer in upstream
            ],
            "output_shape": vertex.get("output_shape", []),
            "upstream": [
                {
                    "counter": producer.get("counter"),
                    "name": producer.get("name", "<unknown>"),
                    "output_shape": producer.get("output_shape", []),
                }
                for producer in upstream
            ],
            "downstream": [
                {
                    "counter": consumer.get("counter"),
                    "name": consumer.get("name", "<unknown>"),
                    "output_shape": consumer.get("output_shape", []),
                }
                for consumer in downstream
            ],
        }
        neighbors.append(entry)

        neighbors_lines.append(f"{entry['counter']}: {entry['name']}")
        neighbors_lines.append(f"  inputs={entry['input_shapes']}")
        neighbors_lines.append(f"  output={entry['output_shape']}")
        neighbors_lines.append(
            "  upstream="
            + str([(item["counter"], item["name"]) for item in entry["upstream"]])
        )
        neighbors_lines.append(
            "  downstream="
            + str([(item["counter"], item["name"]) for item in entry["downstream"]])
        )

    neighbors_json_path.write_text(json.dumps(neighbors, indent=2), encoding="utf-8")
    neighbors_txt_path.write_text("\n".join(neighbors_lines) + "\n", encoding="utf-8")


def capture_once(method, call_name):
    call_counter = {"count": 0}

    def wrapped(*args, **kwargs):
        call_counter["count"] += 1
        should_capture = call_name == target and call_counter["count"] == target_call_index
        if not should_capture:
            return method(*args, **kwargs)

        ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NO_DISPATCH)
        try:
            method(*args, **kwargs)
        finally:
            captured_graph = ttnn.graph.end_graph_capture()
            write_capture(captured_graph)

        raise CaptureComplete(f"Captured {call_name} call {call_counter['count']}")

    return wrapped


def install_capture_hooks():
    from models.demos.qwen25_vl.tt.generator import Generator

    Generator.prefill_forward_text = capture_once(Generator.prefill_forward_text, "prefill")
    Generator.decode_forward = capture_once(Generator.decode_forward, "decode")


install_capture_hooks()

exit_code = 1
try:
    exit_code = pytest.main(["models/demos/qwen25_vl/demo/demo.py", "-k", "batch-1"])
except CaptureComplete as exc:
    print(str(exc))
    exit_code = 0
finally:
    print(f"Graph summary: {summary_path}")
    print(f"Raw capture: {raw_path}")
    print(f"Serialized args: {args_path}")
    print(f"Levelized graph: {levelized_path}")
    print(f"Op neighbors: {neighbors_txt_path}")

raise SystemExit(exit_code)
PYCODE
