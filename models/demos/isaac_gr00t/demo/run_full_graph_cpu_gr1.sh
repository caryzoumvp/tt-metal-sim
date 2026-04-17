#!/usr/bin/env bash
set -euo pipefail

# Full GR00T graph on host CPU using GR1 demo data.
# Extra CLI args are forwarded to examples/minimal_gr00t_demo.py.

source ~/.conda_env.rc
conda activate tt-isaac-gr00t

ISAAC_ROOT="/workspaces/tensix/Isaac-GR00T"
MODEL_PATH="/workspaces/tensix/tt-metal/models/demos/isaac_gr00t/GR00T-N1.6-3B"
DATASET_PATH="${ISAAC_ROOT}/demo_data/gr1.PickNPlace"

if [[ ! -d "${ISAAC_ROOT}" ]]; then
  echo "ERROR: Isaac-GR00T root not found: ${ISAAC_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -d "${DATASET_PATH}" ]]; then
  echo "ERROR: Dataset path not found: ${DATASET_PATH}" >&2
  exit 1
fi

FIRST_PARQUET="$(find "${DATASET_PATH}/data" -type f -name '*.parquet' | head -n 1 || true)"
if [[ -z "${FIRST_PARQUET}" ]]; then
  echo "ERROR: No parquet files found under ${DATASET_PATH}/data" >&2
  exit 1
fi
if head -n 1 "${FIRST_PARQUET}" | grep -q "git-lfs.github.com/spec/v1"; then
  echo "ERROR: Demo parquet files are Git LFS pointer files, not actual data." >&2
  echo "Fetch demo data first:" >&2
  echo "  cd ${ISAAC_ROOT}" >&2
  echo "  git lfs install" >&2
  echo "  git lfs pull --include='demo_data/gr1.PickNPlace/**'" >&2
  exit 1
fi

cd "${ISAAC_ROOT}"
export PYTHONPATH="${ISAAC_ROOT}:${PYTHONPATH:-}"
python examples/minimal_gr00t_demo.py \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --embodiment-tag GR1 \
  --device cpu \
  --video-backend opencv \
  "$@"
