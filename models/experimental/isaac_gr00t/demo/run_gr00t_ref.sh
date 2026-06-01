#!/usr/bin/env bash
# Run GR00T N1.6 PyTorch reference (references/run_pytorch_gr00t.py).
#
# Usage:
#   bash run_gr00t_ref.sh                            # full end-to-end run, 3 iterations
#   bash run_gr00t_ref.sh --benchmark --iterations 20
#   bash run_gr00t_ref.sh --output /tmp/gr00t_ref.pt
#   bash run_gr00t_ref.sh --image /path/to/image.png --instruction "open the drawer"
#   bash run_gr00t_ref.sh [any run_pytorch_gr00t.py args ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_BASE="/workspaces/tensix/tt-metal"
MODEL_PATH="${SCRIPT_DIR}/../GR00T-N1.6-3B"
PY="${SCRIPT_DIR}/../references/run_pytorch_gr00t.py"
DEFAULT_IMAGE="${SCRIPT_DIR}/../assets/test_image.png"
DEFAULT_INSTRUCTION="pick up the cup"

source ~/.conda_env.rc
conda activate tt-isaac-gr00t
source "${TT_METAL_BASE}/tt_env.sh"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${SCRIPT_DIR}"
python3 "${PY}" \
  --model-path  "${MODEL_PATH}" \
  --device      cpu \
  --image       "${DEFAULT_IMAGE}" \
  --instruction "${DEFAULT_INSTRUCTION}" \
  "$@"
