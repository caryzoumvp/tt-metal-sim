#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Lightweight op-level profiler shared across backbone and denoising-loop components."""

from __future__ import annotations

from time import perf_counter_ns


class TtOpProfiler:
    """
    Records (label, elapsed_ms, in_shapes, out_shape) entries and prints them in the
    same [op-perf] format used by TtGr00tActionHeadRuntime._dump_op_timings().

    Typical use — time a block of code manually:
        prof = TtOpProfiler()
        t0 = perf_counter_ns()
        result = some_ttnn_call(...)
        prof.record("my_op", (perf_counter_ns() - t0) / 1e6, in_shape, out_shape)
        prof.dump("siglip2")

    Or wrap a callable (captures all tensor arg shapes automatically):
        result = prof.timed("matmul", ttnn.matmul, a, b)
        # prints:  (1, N, 2048), (2048, 2048) -> (1, N, 2048)
    """

    def __init__(self) -> None:
        self._entries: list[tuple[str, float, list[tuple], tuple | None]] = []

    def record(
        self,
        label: str,
        ms: float,
        in_shape: tuple | None = None,
        out_shape: tuple | None = None,
    ) -> None:
        """Record a manually-timed block with a single in_shape (or None)."""
        in_shapes = [in_shape] if in_shape is not None else []
        self._entries.append((label, ms, in_shapes, out_shape))

    def timed(self, label: str, fn, *args, **kwargs):
        """Call fn(*args, **kwargs), record elapsed time and all tensor arg shapes."""
        in_shapes = [tuple(a.shape) for a in args if hasattr(a, "shape")]
        t0 = perf_counter_ns()
        result = fn(*args, **kwargs)
        ms = (perf_counter_ns() - t0) / 1e6
        out_shape: tuple | None = tuple(result.shape) if hasattr(result, "shape") else None
        self._entries.append((label, ms, in_shapes, out_shape))
        return result

    def dump(self, prefix: str = "") -> None:
        """Print accumulated entries in [op-perf] format and clear the list."""
        if not self._entries:
            return
        hdr = f"[op-perf]{f' {prefix}' if prefix else ''}"
        total = sum(ms for _, ms, _, _ in self._entries)
        print(f"{hdr} --- {len(self._entries)} ops  total={total:.1f} ms ---")
        for label, ms, in_shapes, out_shape in self._entries:
            if in_shapes and out_shape is not None:
                shape_str = f"  {', '.join(str(s) for s in in_shapes)} -> {out_shape}"
            elif in_shapes:
                shape_str = f"  {', '.join(str(s) for s in in_shapes)}"
            else:
                shape_str = ""
            print(f"{hdr}  {label:55s}  {ms:8.3f} ms{shape_str}")
        self._entries.clear()
