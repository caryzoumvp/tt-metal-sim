#!/usr/bin/env bash
set -euo pipefail

source ~/.conda_env.rc
conda activate tt-isaac-gr00t

export LD_LIBRARY_PATH=/workspaces/tensix/tt-metal/build_Debug/lib:/workspaces/tensix/tt-metal/build_Debug/ttnn:/workspaces/tensix/tt-metal/build_Debug/tt_metal:${LD_LIBRARY_PATH:-}
source ~/.conda_env.rc && conda activate tt-isaac-gr00t
python /workspaces/tensix/tt-metal/models/demos/isaac_gr00t/demo/gr00t.py \
	--backbone-phaseb-compare \
	--load-assembly-dir /tmp/gr00t_assembly_case0 \
	--backbone-layer-start 12 \
	--backbone-layer-end 15 \
	--backbone-linear-set all \
	--no-strict-ttnn \
	--dump-max-entries 20 \
	--dump-dir /tmp/gr00t_phaseb_compare

