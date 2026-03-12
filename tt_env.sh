export TT_METAL_BASE=/workspaces/tensix/tt-metal/
export TT_UMD_SIMULATOR=$TT_METAL_BASE/tt_metal/third_party/umd/sim_dev/libttsim.so
export TT_METAL_SIMULATOR=$TT_METAL_BASE/tt_metal/third_party/umd/sim_dev/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_UMD_TTSIM_SLOW_PATH=1
export TT_LOGGER_LEVEL=info
export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
export TT_METAL_KERNEL_MAP=0

export TT_METAL_RISCV_DEBUG_INFO=1
export TT_METAL_WATCHER=1
export TT_METAL_DPRINT_CORES='(0,0),(0,1)'
export TT_METAL_RISCV_DEBUG_INFO=1
#export TT_METAL_WATCHER_CORES='all'
export TT_METAL_DPRINT_CORES='(0,0),(0,1)'
#export TT_METAL_DPRINT_ONE_FILE_PER_RISC=0
export TT_METAL_DPRINT_FILE=$TT_METAL_BASE/generated/dprint/dprint.log
export PYTHONPATH=$TT_METAL_BASE/../tt-forge
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/workspaces/tensix/tt-metal/build_Debug/lib:/workspaces/tensix/tt-metal/build_Debug/ttnn:/workspaces/tensix/tt-metal/build_Debug/tt_metal:${LD_LIBRARY_PATH:-}
#unset TT_UMD_TTSIM_SLOW_PATH
#unset TT_METAL_SLOW_DISPATCH_MODE
unset TT_METAL_WATCHER //unset watcher, wather will enble noc saniter check, this will slow down performance significatlly
