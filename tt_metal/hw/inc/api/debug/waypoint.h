// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

//
// debug/waypoint.h
//
// This file implements a method to log "waypoints" in device code
// Each riscv processor has a 4 byte debug status mailbox in L1
// Use the macro WATCHER_WAYPOINT(...) to log up to 4 characters in the mailbox
// The host watcher thread prints these periodically
// All functionality gated behind defined WATCHER_ENABLED
//
#pragma once

#include "hostdev/dev_msgs.h"
#include "internal/hw_thread.h"

// SIM_PROFILE_LOG: when defined (typically via kernel config defines map for
// the gem5 wormhole_sim build), waypoint calls additionally emit a 32-bit
// MMIO store that TensixMMIO traps into a per-tile waypoint_<x>_<y>.csv file.
// Active independently of WATCHER_ENABLED so that gem5 runs can capture every
// waypoint+timestamp without needing the host watcher poller.
#if defined(SIM_PROFILE_LOG)
static inline void sim_profile_waypoint_mmio(uint32_t code) {
    *(volatile uint32_t*)0xFFB80204u = code;
}
#endif

#if defined(WATCHER_ENABLED) && !defined(WATCHER_DISABLE_WAYPOINT) && !defined(FORCE_WATCHER_OFF)
#include <cstddef>
#include <utility>

template <size_t N, size_t... Is>
constexpr uint32_t fold(const char (&s)[N], std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) <= 4, "Up to 4 characters allowed in WATCHER_WAYPOINT");
    return ((static_cast<uint32_t>(s[Is]) << (8 * Is)) | ...);
}

template <size_t N>
constexpr uint32_t helper(const char (&s)[N]) {
    return fold(s, std::make_index_sequence<N - 1>{});
}

template <uint32_t x>
inline void write_debug_waypoint(volatile tt_l1_ptr uint32_t* debug_waypoint) {
    debug_waypoint[internal_::get_hw_thread_idx()] = x;
}

#define WATCHER_WAYPOINT_MAILBOX (volatile tt_l1_ptr uint32_t*)&((*GET_MAILBOX_ADDRESS_DEV(watcher.debug_waypoint)))

#if defined(SIM_PROFILE_LOG)
#define WAYPOINT(x) do {                                                    \
    write_debug_waypoint<helper(x)>(WATCHER_WAYPOINT_MAILBOX);              \
    sim_profile_waypoint_mmio(helper(x));                                   \
} while (0)
#else
#define WAYPOINT(x) write_debug_waypoint<helper(x)>(WATCHER_WAYPOINT_MAILBOX)
#endif

#elif defined(SIM_PROFILE_LOG)
// WATCHER is off but we still want the sim profile log of waypoints.
// Provide a minimal helper() that just packs 4 ASCII chars.
#include <cstddef>
#include <utility>
template <size_t N, size_t... Is>
constexpr uint32_t fold(const char (&s)[N], std::index_sequence<Is...>) {
    static_assert(sizeof...(Is) <= 4, "Up to 4 characters allowed in WATCHER_WAYPOINT");
    return ((static_cast<uint32_t>(s[Is]) << (8 * Is)) | ...);
}
template <size_t N>
constexpr uint32_t helper(const char (&s)[N]) {
    return fold(s, std::make_index_sequence<N - 1>{});
}
#define WAYPOINT(x) sim_profile_waypoint_mmio(helper(x))

#else  // !WATCHER_ENABLED && !SIM_PROFILE_LOG

#define WAYPOINT(x)

#endif
