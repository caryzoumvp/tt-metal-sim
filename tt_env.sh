export TT_UMD_SIMULATOR=/workspaces/tensix/tt-metal/tt_metal/third_party/umd/sim_dev/libttsim.so
export TT_METAL_SIMULATOR=/workspaces/tensix/tt-metal/tt_metal/third_party/umd/sim_dev/libttsim.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_LOGGER_LEVEL=info
export TT_METAL_LOG_KERNELS_COMPILE_COMMANDS=1
export TT_METAL_KERNEL_MAP=1

export TT_METAL_RISCV_DEBUG_INFO=1
export TT_METAL_WATCHER=1
export TT_METAL_DPRINT_CORES='(0,0),(0,1)'
export TT_METAL_RISCV_DEBUG_INFO=1
export TT_METAL_WATCHER_CORES='(0,0),(0,1)' 
#export TT_METAL_DPRINT_ONE_FILE_PER_RISC=0
export TT_METAL_DPRINT_FILE=$HOME/tensix/tt-metal/generated/dprint/dprint.log
