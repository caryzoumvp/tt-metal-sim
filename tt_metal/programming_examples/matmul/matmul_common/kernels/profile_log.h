// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Lightweight per-phase profile marker for the wormhole_sim gem5 simulator.
// Each call performs a single 32-bit MMIO store to 0xFFB80200. gem5's
// TensixMMIO intercepts the write and appends a row
//   tick,core,phase,payload
// to m5out/profile_<X>_<Y>.csv (per tile). No NoC traffic involved.
// On real silicon this address is unused, so the writes are harmless;
// but the markers are intended for sim only.
#pragma once

#include <cstdint>

#define PROFILE_LOG_ADDR  0xFFB80200u
#define PROFILE_WAYPOINT_ADDR 0xFFB80204u
#define PROFILE_NOC_LO_ADDR   0xFFB80208u
#define PROFILE_NOC_HI_ADDR   0xFFB8020Cu

// 8-bit phase id space (0x00 reserved). Keep groups by initiator so that
// post-processing can route rows to the right core lane.
//
// Reader — non-mcast variant (NC core, noc0)
#define PHASE_READER_BLOCK_START                0x10
#define PHASE_READER_AFTER_READS_ISSUED         0x11
#define PHASE_READER_AFTER_BARRIER              0x12
#define PHASE_READER_AFTER_PUSH                 0x13

// Reader — mcast variants (NC core)
// in0 sender path: DRAM read → barrier → wait receivers → multicast → push
#define PHASE_READER_IN0_AFTER_READS_ISSUED     0x14
#define PHASE_READER_IN0_AFTER_BARRIER          0x15
#define PHASE_READER_IN0_WAIT_RECEIVERS         0x16  // after noc_semaphore_wait(num_dests)
#define PHASE_READER_IN0_AFTER_MCAST            0x17  // after write_multicast + semaphore_set_multicast
#define PHASE_READER_IN0_AFTER_PUSH             0x18
#define PHASE_READER_IN0_WAIT_MCAST             0x19  // receiver: after noc_semaphore_wait(VALID)
// in1 sender path (same structure, separate phase IDs for per-operand visibility)
#define PHASE_READER_IN1_AFTER_READS_ISSUED     0x1A
#define PHASE_READER_IN1_AFTER_BARRIER          0x1B
#define PHASE_READER_IN1_WAIT_RECEIVERS         0x1C
#define PHASE_READER_IN1_AFTER_MCAST            0x1D
#define PHASE_READER_IN1_AFTER_PUSH             0x1E
#define PHASE_READER_IN1_WAIT_MCAST             0x1F  // receiver: after noc_semaphore_wait(VALID)
// Compute (T0/T1/T2)
#define PHASE_COMPUTE_BLOCK_START               0x20
#define PHASE_COMPUTE_AFTER_WAIT_FRONT          0x21
#define PHASE_COMPUTE_SUBBLOCK_START            0x22
#define PHASE_COMPUTE_AFTER_RELOAD              0x23
#define PHASE_COMPUTE_AFTER_MATMUL              0x24
#define PHASE_COMPUTE_AFTER_PACK                0x25
#define PHASE_COMPUTE_BLOCK_END                 0x26
// Writer (BR core, noc1)
#define PHASE_WRITER_SUBBLOCK_START             0x30
#define PHASE_WRITER_AFTER_WRITES_ISSUED        0x31
#define PHASE_WRITER_AFTER_BARRIER              0x32
#define PHASE_WRITER_AFTER_POP                  0x33

static inline void profile_mark(uint32_t phase, uint32_t payload) {
    *(volatile uint32_t*)PROFILE_LOG_ADDR =
        ((phase & 0xff) << 24) | (payload & 0x00ffffff);
}

// Full 32-bit waypoint code (4-char ASCII packed by waypoint.h's helper()).
static inline void profile_waypoint(uint32_t code) {
    *(volatile uint32_t*)PROFILE_WAYPOINT_ADDR = code;
}

// 64-bit NoC event metadata (matches KernelProfilerNocEventMetadata::asU64()).
// gem5 latches the LO half per-core and flushes on the HI write.
static inline void profile_noc_event_64(uint64_t metadata) {
    *(volatile uint32_t*)PROFILE_NOC_LO_ADDR =
        (uint32_t)(metadata & 0xffffffffu);
    *(volatile uint32_t*)PROFILE_NOC_HI_ADDR =
        (uint32_t)(metadata >> 32);
}
