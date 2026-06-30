// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cerrno>
#include <fmt/base.h>
#include <cstdint>
#include <cstdlib>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_common.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/tt_metal/perf_microbenchmark/common/util.hpp"
#include <tt-metalium/distributed.hpp>

using namespace tt;
using namespace tt::tt_metal;
using std::chrono::duration_cast;
using std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////////////
// This test measures the performance of adjacent NOC data transfer. Every
// Tensix cores read from or write to the L1 of the neighbor Tensix core. The
// direction of the transfer is fixed within a test. A user can change the
// direction of the transfer by giving an input argument.
//
// Usage example:
//   ./test_noc_adjacent
//     --cores-r <number of cores in a row>
//     --cores-c <number of cores in a column>
//     --num-tiles <total number of tiles each core transfers>
//     --tiles-per-transfer <number of tiles for each transfer>
//     --noc-index <NOC index to use>
//     --noc-direction <direction of data transfer:
//                      0 for +x, 1 for -y, 2 for -x, and 3 for +y>
//     --access-type <0 for read access, 1 for write access>
//     --use-device-profiler (set to use device profiler for measurement)
//     --num-tests <count of tests>
//     --bypass-check (set to bypass checking performance criteria fulfillment)
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        log_error(tt::LogTest, "Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;
    std::vector<double> measured_device_bandwidth;
    std::vector<double> measured_host_wall_bandwidth;

    ////////////////////////////////////////////////////////////////////////////
    //                      Initial Runtime Args Parse
    ////////////////////////////////////////////////////////////////////////////
    std::vector<std::string> input_args(argv, argv + argc);
    uint32_t num_cores_r = 0;
    uint32_t num_cores_c = 0;
    uint32_t num_tiles = 204800;
    uint32_t noc_index = 0;
    uint32_t noc_direction = 0;
    uint32_t access_type = 0;
    uint32_t tiles_per_transfer;
    uint32_t num_tests = 10;
    bool use_device_profiler = false;
    bool bypass_check = false;
    try {
        std::tie(num_cores_r, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--cores-r", 0);
        std::tie(num_cores_c, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--cores-c", 0);

        std::tie(num_tiles, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tiles", 204800);

        std::tie(tiles_per_transfer, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--tiles-per-transfer", 1);

        std::tie(noc_index, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--noc-index", 0);

        std::tie(noc_direction, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--noc-direction", 0);

        std::tie(access_type, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--access-type", 0);

        std::tie(use_device_profiler, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--use-device-profiler");

        std::tie(num_tests, input_args) =
            test_args::get_command_option_uint32_and_remaining_args(input_args, "--num-tests", 10);

        std::tie(bypass_check, input_args) =
            test_args::has_command_option_and_remaining_args(input_args, "--bypass-check");

        test_args::validate_remaining_args(input_args);
    } catch (const std::exception& e) {
        log_error(tt::LogTest, "Command line arguments found exception", e.what());
    }

    if (num_tiles % tiles_per_transfer != 0) {
        log_error(
            LogTest,
            "Total number of tiles each core transfers ({}) must be the multiple "
            "of number of tiles for each transfer ({})",
            num_tiles,
            tiles_per_transfer);
    }

    if (num_tiles < tiles_per_transfer) {
        log_error(
            LogTest,
            "Total number of tiles each core transfers ({}) must be bigger "
            "than or equal to the number of tiles for each transfer ({})",
            num_tiles,
            tiles_per_transfer);
    }

    if (use_device_profiler) {
        bool device_profiler = tt::tt_metal::MetalContext::instance().rtoptions().get_profiler_enabled();
        TT_FATAL(
            device_profiler,
            "Before running the program, do one of the following in a shell: "
            "either export the environment variable by executing export TT_METAL_DEVICE_PROFILER=1, "
            "or run the program with TT_METAL_DEVICE_PROFILER=1 prefixed to the command");
    }

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        auto device = tt_metal::distributed::MeshDevice::create_unit_mesh(device_id);

        int clock_freq_mhz = get_tt_npu_clock(device->get_devices()[0]);
        if (clock_freq_mhz <= 0) {
            const char* sim_clock_mhz_env = std::getenv("TT_METAL_SIM_CLOCK_MHZ");
            clock_freq_mhz = sim_clock_mhz_env != nullptr ? std::stoi(sim_clock_mhz_env) : 1000;
            if (clock_freq_mhz <= 0) {
                log_error(LogTest, "TT_METAL_SIM_CLOCK_MHZ must be positive when the device clock is unavailable");
                return 1;
            }
            log_warning(
                LogTest,
                "Device clock query returned 0 MHz; using {} MHz for simulator bandwidth estimate",
                clock_freq_mhz);
        }
        auto grid_coord = device->compute_with_storage_grid_size();
        num_cores_c = (num_cores_c == 0) ? grid_coord.x : num_cores_c;
        num_cores_r = (num_cores_r == 0) ? grid_coord.y : num_cores_r;

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::Program();

        CoreCoord start_core = {0, 0};
        CoreCoord end_core = {(std::size_t)num_cores_c - 1, (std::size_t)num_cores_r - 1};
        CoreRange all_cores(start_core, end_core);

        uint32_t cb_tiles = 32;
        uint32_t single_tile_size = 2 * 1024;

        uint32_t cb_src0_index = 0;
        uint32_t cb_src0_addr = device->allocator()->get_base_allocator_addr(HalMemType::L1);
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{cb_src0_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_src0_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

        uint32_t cb_src1_index = 1;
        uint32_t cb_src1_addr = cb_src0_addr + (cb_tiles * single_tile_size);
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(cb_tiles * single_tile_size, {{cb_src1_index, tt::DataFormat::Float16_b}})
                .set_page_size(cb_src1_index, single_tile_size);
        tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

        auto noc_kernel = tt_metal::CreateKernel(
            program,
            (access_type == 0) ? "tests/tt_metal/tt_metal/perf_microbenchmark/"
                                 "2_noc_adjacent/kernels/noc_read.cpp"
                               : "tests/tt_metal/tt_metal/perf_microbenchmark/"
                                 "2_noc_adjacent/kernels/noc_write.cpp",
            all_cores,
            tt_metal::DataMovementConfig{
                .processor = (noc_index == 0) ? tt_metal::DataMovementProcessor::RISCV_0
                                              : tt_metal::DataMovementProcessor::RISCV_1,
                .noc = (noc_index == 0) ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default});

        for (int i = 0; i < num_cores_r; i++) {
            for (int j = 0; j < num_cores_c; j++) {
                CoreCoord logical_core = {(std::size_t)j, (std::size_t)i};

                CoreCoord adjacent_core_logical = {(std::size_t)j, (std::size_t)i};
                if (noc_direction == 0) {
                    // right (+x direction)
                    adjacent_core_logical.x = (adjacent_core_logical.x + 1) % num_cores_c;
                } else if (noc_direction == 1) {
                    // down (-y direction)
                    adjacent_core_logical.y = (adjacent_core_logical.y + num_cores_r - 1) % num_cores_r;
                } else if (noc_direction == 2) {
                    // left (-x direction)
                    adjacent_core_logical.x = (adjacent_core_logical.x + num_cores_c - 1) % num_cores_c;
                } else {
                    // up (+y direction)
                    adjacent_core_logical.y = (adjacent_core_logical.y + 1) % num_cores_r;
                }

                CoreCoord adjacent_core_noc = device->worker_core_from_logical_core(adjacent_core_logical);

                const std::array noc_runtime_args = {
                    (uint32_t)adjacent_core_noc.x,
                    (uint32_t)adjacent_core_noc.y,
                    cb_src1_addr,
                    num_tiles / tiles_per_transfer,
                    tiles_per_transfer};
                SetRuntimeArgs(program, noc_kernel, logical_core, noc_runtime_args);
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        //                      Execute Application
        ////////////////////////////////////////////////////////////////////////////
        log_info(LogTest, "Num tests {}", num_tests);
        auto mesh_workload = tt_metal::distributed::MeshWorkload();
        mesh_workload.add_program(tt::tt_metal::distributed::MeshCoordinateRange{{0, 0}, {0, 0}}, std::move(program));

        for (uint32_t i = 0; i < num_tests; ++i) {
            auto t_begin = std::chrono::steady_clock::now();
            tt_metal::distributed::EnqueueMeshWorkload(device->mesh_command_queue(), mesh_workload, false);
            tt_metal::distributed::Finish(device->mesh_command_queue());
            auto t_end = std::chrono::steady_clock::now();
            unsigned long host_elapsed_us = duration_cast<microseconds>(t_end - t_begin).count();
            unsigned long host_elapsed_cc = clock_freq_mhz * host_elapsed_us;
            unsigned long device_elapsed_cc = 0;

            log_info(
                LogTest,
                "Host wall time for NOC transfers: {}us ({}cycles at {} MHz reference)",
                host_elapsed_us,
                host_elapsed_cc,
                clock_freq_mhz);

            if (use_device_profiler) {
                device_elapsed_cc = get_t0_to_any_riscfw_end_cycle(
                    device->get_devices()[0], mesh_workload.get_programs().begin()->second);
                const double device_elapsed_us = (double)device_elapsed_cc / clock_freq_mhz;
                log_info(
                    LogTest,
                    "Device profiler time for NOC transfers: {:.3f}us ({}cycles)",
                    device_elapsed_us,
                    device_elapsed_cc);
            }

            // Total transfer amount per core = tile size * number of tiles.
            // Device bandwidth is the primary metric. Host-wall bandwidth is
            // only a simulator/reference metric and includes host/gem5 runtime.
            if (host_elapsed_cc == 0) {
                log_error(LogTest, "Host elapsed cycle count is zero; cannot compute host-wall NOC bandwidth");
                measured_host_wall_bandwidth.push_back(0.0);
                pass = false;
            } else {
                measured_host_wall_bandwidth.push_back((double)single_tile_size * num_tiles / host_elapsed_cc);
            }
            if (use_device_profiler) {
                if (device_elapsed_cc == 0) {
                    log_error(LogTest, "Device elapsed cycle count is zero; cannot compute device NOC bandwidth");
                    measured_device_bandwidth.push_back(0.0);
                    pass = false;
                } else {
                    measured_device_bandwidth.push_back((double)single_tile_size * num_tiles / device_elapsed_cc);
                }
            }

            if (use_device_profiler) {
                log_info(LogTest, "Measured device NOC bandwidth: {:.3f}B/cc", measured_device_bandwidth[i]);
            }
            log_info(LogTest, "Measured host-wall NOC bandwidth: {:.6f}B/cc", measured_host_wall_bandwidth[i]);
        }

        pass &= device->close();
    } catch (const std::exception& e) {
        pass = false;
        log_error(LogTest, "{}", e.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    // Determine if it passes performance goal
    auto avg_measured_device_bandwidth = calculate_average(measured_device_bandwidth);
    auto avg_measured_host_wall_bandwidth = calculate_average(measured_host_wall_bandwidth);
    if (pass && !bypass_check) {
        // goal is 95% of theoretical peak using a single NOC channel
        // theoretical peak: 32bytes per clock cycle
        double target_bandwidth = 32.0 * 0.9;
        if (!use_device_profiler) {
            pass = false;
            log_error(LogTest, "Performance check requires --use-device-profiler for device-time bandwidth");
        } else if (avg_measured_device_bandwidth < target_bandwidth) {
            pass = false;
            log_error(
                LogTest,
                "The NOC bandwidth does not meet the criteria. "
                "Current: {:.3f}B/cc, goal: >={:.3f}B/cc",
                avg_measured_device_bandwidth,
                target_bandwidth);
        }
    }

    // for csv
    log_info(tt::LogTest, "CSV_MICROBENCHMARK:title:test_noc_adjacent");
    log_info(
        tt::LogTest,
        "CSV_INPUT:num-cores-r:{}:num-cores-c:{}:num-tiles:{}:tiles-per-transfer:{}:noc-index:{}:noc-direction:{}:"
        "access-type:{}:use-device-profiler:{}",
        num_cores_r,
        num_cores_c,
        num_tiles,
        tiles_per_transfer,
        NOC_INDEXToString(static_cast<NOC_INDEX>(noc_index)),
        NOC_DIRECTIONToString(static_cast<NOC_DIRECTION>(noc_direction)),
        ACCESS_TYPEToString(static_cast<ACCESS_TYPE>(access_type)),
        use_device_profiler);
    log_info(tt::LogTest, "CSV_OUTPUT:DeviceBandwidth(B/cc):{:.3f}", avg_measured_device_bandwidth);
    log_info(tt::LogTest, "CSV_OUTPUT:HostWallBandwidth(B/cc):{:.6f}", avg_measured_host_wall_bandwidth);
    log_info(tt::LogTest, "CSV_RESULT:pass:{}", pass);

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        log_error(LogTest, "Test Failed");
    }

    return 0;
}
