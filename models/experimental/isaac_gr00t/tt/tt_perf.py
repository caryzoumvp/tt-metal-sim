#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight op-level profiler shared across backbone and denoising-loop components.

Sim profiling
-------------
Set environment variable  TT_SIM_PROFILE=1  before running the model to emit
profile markers to gem5 via /tmp/tt_marker.sock.  The profiler connects once,
then for each timed op:

  1. Sends HOST_OP_START to gem5  → tick recorded in m5out/profile_mark_host.csv
  2. Calls the TTNN op            → kernel enqueued (async dispatch)
  3. Calls ttnn.synchronize_device() → blocks until kernel finishes on device
  4. Sends HOST_OP_END to gem5    → tick recorded

The resulting profile_mark_host.csv tick delta captures the real kernel
execution window, not just the enqueue time.  The waypoint CSV files
(profile_waypoint_<x>_<y>.csv) provide per-tile WAYPOINT("K"/"KD") events
that can be correlated with the host markers via profile_summary.py --ops.

If the sim socket is not available (hardware run), the sim profiler
degrades silently to pure wall-clock timing without any socket ops.
"""

from __future__ import annotations

import csv
import os
import socket
import struct
from time import perf_counter_ns


# ── Sim marker helper ─────────────────────────────────────────────────────────

_MARKER_SOCKET  = "/tmp/tt_marker.sock"
_CMD_WRITE_REG  = 0x02
_PROFILE_MARK   = 0xFFB80200
_HOST_OP_START  = 0xF0
_HOST_OP_END    = 0xF1
_RESP_OK        = 0x00


class _SimProfiler:
    """
    Manages a persistent connection to the gem5 marker socket.
    One shared instance per Python process; silently disabled if the sim is not running.
    """

    def __init__(self, m5out: str) -> None:
        self._sock: socket.socket | None = None
        self._seq = 0
        self._labels_fh = None
        self._labels_w  = None
        self.connected  = False

        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect(_MARKER_SOCKET)
            s.settimeout(None)
            s.sendall(struct.pack("<BII", 0xFF, 0, 0))   # CMD_PING
            resp = s.recv(1)
            if resp and resp[0] == _RESP_OK:
                self._sock = s
                self.connected = True
                os.makedirs(m5out, exist_ok=True)
                labels_path = os.path.join(m5out, "profile_mark_host_labels.csv")
                self._labels_fh = open(labels_path, "w", newline="")
                self._labels_w  = csv.writer(self._labels_fh)
                self._labels_w.writerow(
                    ["seq_num", "op_name", "test_id", "in_shapes", "out_shape"]
                )
                self._labels_fh.flush()
                print(f"[sim] TtOpProfiler: connected to {_MARKER_SOCKET}")
                print(f"[sim] TtOpProfiler: labels → {labels_path}")
            else:
                s.close()
        except OSError:
            pass   # sim not running → degrade silently

    def _send(self, phase: int, seq: int) -> None:
        if not self.connected or self._sock is None:
            return
        value = ((phase & 0xFF) << 24) | (seq & 0x00FFFFFF)
        self._sock.sendall(struct.pack("<BIII", _CMD_WRITE_REG, _PROFILE_MARK, 4, value))
        self._sock.recv(1)   # consume RESP_OK

    def start(self) -> int:
        self._seq += 1
        self._send(_HOST_OP_START, self._seq)
        return self._seq

    def end(
        self,
        seq: int,
        label: str,
        in_shapes: list[tuple],
        out_shape: tuple | None,
    ) -> None:
        self._send(_HOST_OP_END, seq)
        if self._labels_w is not None:
            self._labels_w.writerow(
                [seq, label, "model", str(in_shapes), str(out_shape or "")]
            )
            self._labels_fh.flush()   # type: ignore[union-attr]

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None
        if self._labels_fh:
            self._labels_fh.close()
            self._labels_fh = None
        self.connected = False


_SHARED_SIM_PROFILER: _SimProfiler | None = None
_SHARED_SIM_M5OUT: str | None = None


def _get_shared_sim_profiler(m5out: str) -> _SimProfiler:
    """
    Return the process-wide sim profiler.

    The full GR00T demo creates separate TtOpProfiler objects for SigLIP2, MLP1,
    Qwen3, and the action head.  Reopening the label CSV for each component would
    truncate earlier labels and reset seq_num.  Keep one socket and one labels file
    for the whole process instead.
    """
    global _SHARED_SIM_PROFILER, _SHARED_SIM_M5OUT

    if _SHARED_SIM_PROFILER is None:
        _SHARED_SIM_M5OUT = m5out
        _SHARED_SIM_PROFILER = _SimProfiler(m5out)
    elif _SHARED_SIM_M5OUT != m5out:
        print(
            f"[sim] TtOpProfiler: reusing labels in {_SHARED_SIM_M5OUT}; "
            f"ignoring requested m5out={m5out}"
        )
    return _SHARED_SIM_PROFILER


# ── Main profiler ─────────────────────────────────────────────────────────────


class TtOpProfiler:
    """
    Records (label, elapsed_ms, in_shapes, out_shape) entries and prints them in the
    same [op-perf] format used by TtGr00tActionHeadRuntime._dump_op_timings().

    Typical use — wrap a callable (captures tensor arg shapes automatically):
        result = prof.timed("matmul", ttnn.matmul, a, b)

    Sim profiling — enabled when TT_SIM_PROFILE=1 is set in the environment:
        export TT_SIM_PROFILE=1
        # Then construct with the TTNN device for accurate synchronisation:
        prof = TtOpProfiler(device=self.device)
        # Each timed() call will:
        #   1. Send HOST_OP_START to gem5
        #   2. Run the TTNN op (async enqueue)
        #   3. Call ttnn.synchronize_device(device) to wait for completion
        #   4. Send HOST_OP_END to gem5

    If the gem5 sim socket is not available the sim profiling path is skipped
    transparently; wall-clock timing works as before.
    """

    def __init__(
        self,
        device=None,
        sim: bool | None = None,
        m5out: str | None = None,
    ) -> None:
        """
        device  : ttnn.Device — required for accurate sim timing (synchronize_device).
                  If None, markers are still emitted but END comes right after the
                  async enqueue, not after kernel completion.
        sim     : True/False override; None (default) reads TT_SIM_PROFILE env var.
        m5out   : directory for gem5 output files (default: GEM5_M5OUT or m5out/).
        """
        self._entries: list[tuple[str, float, list[tuple], tuple | None]] = []
        self._device = device

        # Resolve sim flag
        if sim is None:
            sim = bool(os.environ.get("TT_SIM_PROFILE"))

        m5out_dir = m5out or os.environ.get("GEM5_M5OUT", "/workspaces/wormhole_sim/m5out")
        self._sim: _SimProfiler | None = _get_shared_sim_profiler(m5out_dir) if sim else None

    # ── Public API ────────────────────────────────────────────────────────────

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
        """
        Call fn(*args, **kwargs), record elapsed time and all tensor arg shapes.

        When sim profiling is enabled:
          - START marker → async enqueue → synchronize_device → END marker
          Wall-clock measures the complete kernel execution, not just the enqueue.
        """
        in_shapes = [tuple(a.shape) for a in args if hasattr(a, "shape")]

        # ── START marker ──────────────────────────────────────────────────────
        seq: int | None = None
        if self._sim and self._sim.connected:
            seq = self._sim.start()

        # ── Dispatch ──────────────────────────────────────────────────────────
        t0     = perf_counter_ns()
        result = fn(*args, **kwargs)

        # ── Sync (wait for kernel completion) ─────────────────────────────────
        if self._sim and self._sim.connected and self._device is not None:
            import ttnn as _ttnn   # lazy import avoids circular deps
            _ttnn.synchronize_device(self._device)

        ms = (perf_counter_ns() - t0) / 1e6

        # ── END marker ────────────────────────────────────────────────────────
        out_shape: tuple | None = tuple(result.shape) if hasattr(result, "shape") else None
        if self._sim and self._sim.connected and seq is not None:
            self._sim.end(seq, label, in_shapes, out_shape)

        self._entries.append((label, ms, in_shapes, out_shape))
        return result

    def dump(self, prefix: str = "") -> None:
        """Print accumulated entries in [op-perf] format and clear the list."""
        if not self._entries:
            return
        hdr   = f"[op-perf]{f' {prefix}' if prefix else ''}"
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

    def close(self) -> None:
        """Detach from the shared sim profiler without closing the process-wide socket."""
        self._sim = None
