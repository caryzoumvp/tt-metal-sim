#!/usr/bin/env bash
set -euo pipefail

cd /workspaces/tensix/tt-metal

source ~/.conda_env.rc
conda activate tt-isaac-gr00t

source tt_env.sh


MESH_DEVICE=${MESH_DEVICE:-N150}
HF_ENDPOINT=https://hf-mirror.com
HF_MODEL=${HF_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}

MESH_DEVICE="$MESH_DEVICE" \
HF_MODEL="$HF_MODEL" \
python - <<'PYCODE'
import json
import os
import re
from pathlib import Path

import pytest
import ttnn

exit_code = 1
try:
    exit_code = pytest.main(["models/demos/qwen25_vl/demo/demo.py", "-k", "batch-1"])
except CaptureComplete as exc:
    print(str(exc))
    exit_code = 0
raise SystemExit(exit_code)
PYCODE
