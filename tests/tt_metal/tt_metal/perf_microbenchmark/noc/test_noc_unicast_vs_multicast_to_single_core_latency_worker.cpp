// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <map>
#include <string>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>

#include <tt_stl/assert.hpp>

using namespace tt;

namespace {

CoreCoord select_destination_worker(tt_metal::IDevice* device) {
    const auto grid = device->compute_with_storage_grid_size();
    if (grid.x > 1) {
        return {1, 0};
    }
    if (grid.y > 1) {
        return {0, 1};
    }
    return {0, 0};
}

void measure_latency(const std::string& kernel_name) {
    constexpr int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    const CoreCoord producer_logical_core{0, 0};
    const CoreCoord destination_logical_core = select_destination_worker(device);
    const CoreCoord destination_physical_core = device->worker_core_from_logical_core(destination_logical_core);

    std::map<std::string, std::string> defines = {
        {"WORKER_NOC_X", std::to_string(destination_physical_core.x)},
        {"WORKER_NOC_Y", std::to_string(destination_physical_core.y)},
    };

    tt_metal::Program program = tt_metal::CreateProgram();
    tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/perf_microbenchmark/noc/kernels/" + kernel_name + ".cpp",
        producer_logical_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .defines = defines});

    tt::tt_metal::detail::SetDeviceProfilerDir(kernel_name + "_worker_microbenchmark");
    tt::tt_metal::detail::FreshProfilerDeviceLog();
    tt::tt_metal::detail::CompileProgram(device, program);
    tt_metal::detail::LaunchProgram(device, program);
    tt_metal::CloseDevice(device);
}

}  // namespace

int main() {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") == nullptr) {
        TT_THROW("Test not supported w/ fast dispatch, exiting");
    }

    measure_latency("multicast_to_single_core");
    measure_latency("unicast_to_single_core");
}
