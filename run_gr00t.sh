#!/usr/bin/env bash
set -euo pipefail

source ~/.conda_env.rc
conda activate tt-isaac-gr00t

export LD_LIBRARY_PATH=/workspaces/tensix/tt-metal/build_Debug/lib:/workspaces/tensix/tt-metal/build_Debug/ttnn:/workspaces/tensix/tt-metal/build_Debug/tt_metal:${LD_LIBRARY_PATH:-}

cd /workspaces/tensix/tt-metal
source tt_env.sh
python models/demos/isaac_gr00t/demo/gr00t.py
