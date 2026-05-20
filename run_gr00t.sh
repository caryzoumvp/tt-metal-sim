#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_gr00t.sh [--cpu-ref] [--ttnn-full] [--dump-tensor] [--help]

Modes:
  --cpu-ref      Run GR00T CPU reference flow and generate shared/assembly dumps.
  --ttnn-full    Run GR00T TTNN full flow using shared inputs.
  --dump-tensor  Print dumped tensor shapes from assembly/reference/ttnn directories.

Behavior:
  - If no mode flags are provided, all three modes run in order.
  - Running --ttnn-full alone requires existing shared inputs at /tmp/gr00t_shared.
EOF
}

RUN_CPU_REF=0
RUN_TTNN_FULL=0
RUN_DUMP_TENSOR=0

if [[ $# -eq 0 ]]; then
  RUN_CPU_REF=1
  RUN_TTNN_FULL=1
  RUN_DUMP_TENSOR=1
else
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --cpu-ref)
        RUN_CPU_REF=1
        shift
        ;;
      --ttnn-full)
        RUN_TTNN_FULL=1
        shift
        ;;
      --dump-tensor)
        RUN_DUMP_TENSOR=1
        shift
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 1
        ;;
    esac
  done
fi

source ~/.conda_env.rc
conda activate tt-isaac-gr00t
export TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}'

export LD_LIBRARY_PATH=/workspaces/tensix/tt-metal/build_Debug/lib:/workspaces/tensix/tt-metal/build_Debug/ttnn:/workspaces/tensix/tt-metal/build_Debug/tt_metal:${LD_LIBRARY_PATH:-}

ROOT="/workspaces/tensix/tt-metal"
PY="${ROOT}/models/demos/isaac_gr00t/demo/gr00t.py"
DATASET="/workspaces/tensix/Isaac-GR00T/demo_data/gr1.PickNPlace"
SHARED_DIR="/tmp/gr00t_shared"
ASSEMBLY_DIR="/tmp/gr00t_assembly"
COMPARE_DIR="/tmp/gr00t_compare"
OPS_LOG="/tmp/gr00t_ttnn_ops.log"

cd "${ROOT}"

if [[ "${RUN_CPU_REF}" -eq 1 ]]; then
  python3 "${PY}" \
    --graph-mode cpu_ref \
    --input-source gr1 \
    --dataset-path "${DATASET}" \
    --embodiment-tag GR1 \
    --traj-id 0 \
    --step 0 \
    --task-text "pick the pear from the counter and place it in the plate" \
    --save-shared-inputs-dir "${SHARED_DIR}" \
    --dump-assembly-dir "${ASSEMBLY_DIR}"
fi

if [[ "${RUN_TTNN_FULL}" -eq 1 ]]; then
  if [[ ! -d "${SHARED_DIR}" ]]; then
    echo "Missing shared inputs: ${SHARED_DIR}" >&2
    echo "Run with --cpu-ref first (or run without args to execute all modes)." >&2
    exit 1
  fi
  python3 "${PY}" \
    --graph-mode ttnn_full \
    --input-source gr1 \
    --load-shared-inputs-dir "${SHARED_DIR}" \
    --log-ttnn-ops \
    2>&1 | tee "${OPS_LOG}"
fi

if [[ "${RUN_DUMP_TENSOR}" -eq 1 ]]; then
  python3 - <<'PY'
from pathlib import Path
import numpy as np

for root in [
    Path("/tmp/gr00t_assembly/assembly"),
    Path("/tmp/gr00t_compare/ttnn"),
    Path("/tmp/gr00t_compare/reference"),
]:
    if not root.exists():
        continue
    print(f"\n=== {root} ===")
    for p in sorted(root.rglob("*.npy")):
        a = np.load(p, allow_pickle=False)
        print(f"{p.relative_to(root)}: shape={a.shape}, dtype={a.dtype}")
PY
fi
