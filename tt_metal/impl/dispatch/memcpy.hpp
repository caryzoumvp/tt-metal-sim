// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <tt_stl/assert.hpp>
#include <tt_stl/aligned_allocator.hpp>
#include <umd/device/driver_atomics.hpp>

#include <tt-metalium/vector_aligned.hpp>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__AVX2__)
#define LOAD_STREAM_32()                                                               \
    do {                                                                               \
        _mm256_stream_si256((__m256i*)dst8, _mm256_loadu_si256((const __m256i*)src8)); \
        src8 += sizeof(__m256i);                                                       \
        dst8 += sizeof(__m256i);                                                       \
    } while (0)
#endif

#define LOAD_STREAM_16()                                                         \
    do {                                                                         \
        _mm_stream_si128((__m128i*)dst8, _mm_loadu_si128((const __m128i*)src8)); \
        src8 += sizeof(__m128i);                                                 \
        dst8 += sizeof(__m128i);                                                 \
    } while (0)

#define LOAD_STREAM_4()                                   \
    do {                                                  \
        _mm_stream_si32((int32_t*)dst8, *(int32_t*)src8); \
        src8 += sizeof(int32_t);                          \
        dst8 += sizeof(int32_t);                          \
    } while (0)

#define LOAD_STREAM_4_UNALIGNED()                 \
    do {                                          \
        int32_t val = 0;                          \
        std::memcpy(&val, src8, sizeof(int32_t)); \
        _mm_stream_si32((int32_t*)dst8, val);     \
        src8 += sizeof(int32_t);                  \
        dst8 += sizeof(int32_t);                  \
    } while (0)

namespace tt::tt_metal {

// Ideally would work by cachelines, but the min size is less than that
// Benchmarked to be approximately 1.4x - 1.8x faster than std::memcpy
// TODO: Revisit this w/ regard to possibly eliminating min sizes and orphan writes at the end
// TODO: ditto alignment issues
#if defined(__x86_64__) || defined(__i386__)
template <bool debug_sync = false>
void memcpy_to_device(void* __restrict dst, const void* __restrict src, size_t n) {
    // Ensure destination is properly aligned for optimal SIMD performance
    TT_ASSERT((uintptr_t)dst % MEMCPY_ALIGNMENT == 0);

    const auto* src8 = static_cast<const uint8_t*>(src);
    auto* dst8 = static_cast<uint8_t*>(dst);

#if defined(__AVX2__)
    // Configuration for bulk processing: inner loop processes 8 x 32-byte operations
    // This creates 256-byte blocks (8 * 32 = 256 bytes) for maximum throughput
    constexpr uint32_t inner_loop = 8;
    constexpr uint32_t inner_blk_size = inner_loop * sizeof(__m256i);  // 256 bytes

    size_t num_lines = n / inner_blk_size;  // Number of 256-byte blocks to process

    // PHASE 1: Process 256-byte blocks 32 bytes at a time
    if (num_lines > 0) {
        // Handle potential misalignment by processing a single 16-byte chunk first
        // WARNING: This does not cover the case where dst is not 16-byte aligned
        if ((uintptr_t)dst8 % sizeof(__m256i) != 0) {
            LOAD_STREAM_16();
            n -= sizeof(__m128i);
            num_lines = n / inner_blk_size;
        }

        for (size_t i = 0; i < num_lines; ++i) {
            for (size_t j = 0; j < inner_loop; ++j) {
                LOAD_STREAM_32();
            }
            n -= inner_blk_size;
        }
    }

    // PHASE 2: Process remaining data that doesn't fill a complete 256-byte block
    if (n > 0) {
        // Phase 2.1: Process remaining 32-byte chunks
        num_lines = n / sizeof(__m256i);
        if (num_lines > 0) {
            if ((uintptr_t)dst8 % sizeof(__m256i) != 0) {
                LOAD_STREAM_16();
                n -= sizeof(__m128i);
                num_lines = n / sizeof(__m256i);
            }

            for (size_t i = 0; i < num_lines; ++i) {
                LOAD_STREAM_32();
            }
            n -= num_lines * sizeof(__m256i);
        }

        // Phase 2.2: Process remaining 16-byte chunks
        num_lines = n / sizeof(__m128i);
        if (num_lines > 0) {
            for (size_t i = 0; i < num_lines; ++i) {
                LOAD_STREAM_16();
            }
            n -= num_lines * sizeof(__m128i);
        }

        // Phase 2.3: Process remaining 4-byte chunks
        num_lines = n / sizeof(int32_t);
        if (num_lines > 0) {
            if ((uintptr_t)src8 % sizeof(int32_t) != 0) {
                for (size_t i = 0; i < num_lines; ++i) {
                    LOAD_STREAM_4_UNALIGNED();
                }
            } else {
                for (size_t i = 0; i < num_lines; ++i) {
                    LOAD_STREAM_4();
                }
            }
            n -= num_lines * sizeof(int32_t);
        }

        // Phase 2.4: Handle the final few bytes (< 4 bytes)
        if (n > 0) {
            int32_t val = 0;
            std::memcpy(&val, src8, n);
            _mm_stream_si32((int32_t*)dst8, val);
        }
    }
#else
    // SSE2-only path (no AVX/AVX2 available)
    size_t num_lines = n / sizeof(__m128i);
    if (num_lines > 0) {
        for (size_t i = 0; i < num_lines; ++i) {
            LOAD_STREAM_16();
        }
        n -= num_lines * sizeof(__m128i);
    }

    num_lines = n / sizeof(int32_t);
    if (num_lines > 0) {
        if ((uintptr_t)src8 % sizeof(int32_t) != 0) {
            for (size_t i = 0; i < num_lines; ++i) {
                LOAD_STREAM_4_UNALIGNED();
            }
        } else {
            for (size_t i = 0; i < num_lines; ++i) {
                LOAD_STREAM_4();
            }
        }
        n -= num_lines * sizeof(int32_t);
    }

    if (n > 0) {
        int32_t val = 0;
        std::memcpy(&val, src8, n);
        _mm_stream_si32((int32_t*)dst8, val);
    }
#endif

    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}
#else
// Fallback implementation for non-x86 architectures
// Uses standard memcpy since SIMD optimizations aren't available
template <bool debug_sync = false>
__attribute((nonnull(1, 2))) static inline void memcpy_to_device(
    void* __restrict dst, const void* __restrict src, size_t n) {
    memcpy(dst, src, n);
    if constexpr (debug_sync) {
        tt_driver_atomics::sfence();
    }
}
#endif

}  // namespace tt::tt_metal

#undef LOAD_STREAM_32
#undef LOAD_STREAM_16
#undef LOAD_STREAM_4
#undef LOAD_STREAM_4_UNALIGNED
