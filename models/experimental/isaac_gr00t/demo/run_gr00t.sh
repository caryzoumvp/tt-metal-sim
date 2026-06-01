#!/usr/bin/env bash
# Run GR00T N1.6 TTNN demo (demo.py).
#
# For the pure PyTorch reference use:
#   bash run_gr00t_ref.sh
#
# Usage:
#   bash run_gr00t.sh                            # full end-to-end run, 3 iterations
#   bash run_gr00t.sh --benchmark --iterations 20
#   bash run_gr00t.sh --output /tmp/gr00t_tt.npy
#   bash run_gr00t.sh --image /path/to/image.png --instruction "open the drawer"
#   bash run_gr00t.sh [any demo.py args ...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_METAL_BASE="/workspaces/tensix/tt-metal"
MODEL_PATH="${SCRIPT_DIR}/../GR00T-N1.6-3B"
PY="${SCRIPT_DIR}/demo.py"
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
  --image       "${DEFAULT_IMAGE}" \
  --instruction "${DEFAULT_INSTRUCTION}" \
  "$@"
