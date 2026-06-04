# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
pytest configuration for GR00T op tests.

--sim flag
----------
When passed, each test sends phase-encoded profile markers to gem5 via
CMD_PROFILE_MARK before and after every TTNN op call.

Socket usage — connect-send-close per marker
--------------------------------------------
TensixDbgServer accepts only ONE client at a time.  Holding a persistent
connection would evict the tt-metal kernel-dispatch connection and hang the
sim.  Instead, each _send_mark() opens a short-lived connection:

  1. connect to /tmp/tt_sim.sock
  2. send [cmd:1][value:4 LE][0:4]
  3. recv RESP_OK (1 byte)
  4. close

This is safe because Python is single-threaded: before ttnn.op() is called,
tt-metal has no active socket connection (previous kernel already finished and
disconnected), and after ttnn.op() returns the same is true.

Wire format
-----------
  value = (phase << 24) | (seq_num & 0x00FFFFFF)
  phase 0xF0 = HOST_OP_START, 0xF1 = HOST_OP_END
  seq_num = monotone counter; matches `payload` in profile_mark_host.csv
            and `seq_num` in profile_mark_host_labels.csv

Output files
------------
  m5out/profile_mark_host.csv        tick-level markers (shared schema)
  m5out/profile_mark_host_labels.csv op-name lookup by seq_num (incremental)
  gr00t_ops_sim_baseline.csv         wall-clock summary (cwd)
"""

import csv
import os
import socket
import struct
from time import perf_counter_ns

import pytest

# ── Constants ─────────────────────────────────────────────────────────────────
HOST_US_PER_SIM_CYCLE = 681.0
# Dedicated marker socket — separate from tt-metal's kernel-dispatch socket.
# WormholeDbgServer starts a second listener on this path.
MARKER_SOCKET_PATH = "/tmp/tt_marker.sock"
# gem5 always writes m5out/ relative to its working directory (wormhole_sim).
# Override via GEM5_M5OUT env var if the sim is run from a different directory.
M5OUT_DIR = os.environ.get("GEM5_M5OUT", "/workspaces/wormhole_sim/m5out")

# CMD_WRITE_REG (0x02) to PROFILE_MARK_ADDR: same encoding as RISC-V MMIO write.
# Frame: [cmd:1][addr:4][size=4:4][value:4]
CMD_WRITE_REG     = 0x02
PROFILE_MARK_ADDR = 0xFFB80200
RESP_OK           = 0x00

HOST_OP_START = 0xF0   # docs/profile.md
HOST_OP_END   = 0xF1


# ── Marker socket — persistent, one connection per session ────────────────────

def _connect_marker_socket() -> socket.socket:
    """
    Connect to the dedicated marker socket.  Fatal if not available — the sim
    must be running before tests start when --sim is passed.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.connect(MARKER_SOCKET_PATH)
    except OSError as e:
        s.close()
        raise SystemExit(
            f"\n[sim] FATAL: cannot connect to marker socket {MARKER_SOCKET_PATH}: {e}\n"
            f"  Make sure gem5 is running (bash run_gem5_big.sh) and fully started\n"
            f"  before running pytest --sim.\n"
        )
    # Verify with a ping
    s.sendall(struct.pack('<BII', 0xFF, 0, 0))   # CMD_PING, addr=0, size=0
    resp = s.recv(1)
    if resp[0] != RESP_OK:
        s.close()
        raise SystemExit(
            f"[sim] FATAL: marker socket ping failed (response=0x{resp[0]:02x})"
        )
    return s


def _send_mark(sock: socket.socket, phase: int, seq_num: int) -> None:
    """Write one CMD_WRITE_REG(PROFILE_MARK_ADDR) on the persistent connection."""
    value = ((phase & 0xFF) << 24) | (seq_num & 0x00FFFFFF)
    sock.sendall(struct.pack('<BIII', CMD_WRITE_REG, PROFILE_MARK_ADDR, 4, value))
    sock.recv(1)   # consume RESP_OK


# ── Session fixtures ──────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--sim",
        action="store_true",
        default=False,
        help=(
            "Time each TTNN op under the gem5 simulator. "
            "Writes m5out/profile_mark_host.csv (tick markers), "
            "m5out/profile_mark_host_labels.csv (op names), "
            "and gr00t_ops_sim_baseline.csv (wall-clock summary)."
        ),
    )


@pytest.fixture(scope="session")
def sim_log(request):
    """
    Yields (records, socket_path, labels_writer, labels_fh).
    labels file is written incrementally and flushed per row.
    """
    if not request.config.getoption("--sim"):
        yield None, None, None, None
        return

    # Connect once — fatal if the sim isn't running
    sock = _connect_marker_socket()
    print(f"\n[sim] Connected to marker socket {MARKER_SOCKET_PATH}")

    records = []

    os.makedirs(M5OUT_DIR, exist_ok=True)
    labels_path = os.path.join(M5OUT_DIR, "profile_mark_host_labels.csv")
    labels_fh   = open(labels_path, "w", newline="")
    labels_w    = csv.writer(labels_fh)
    labels_w.writerow(["seq_num", "op_name", "test_id", "in_shapes", "out_shape"])
    labels_fh.flush()

    yield records, sock, labels_w, labels_fh

    sock.close()
    labels_fh.close()

    out_path = os.path.join(os.getcwd(), "gr00t_ops_sim_baseline.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seq_num", "test", "op_name",
            "in_shapes", "out_shape",
            "wall_ms", "est_sim_kcy",
        ])
        writer.writerows(records)

    print(f"\n[sim] {len(records)} ops → {out_path}")
    print(f"[sim] Tick markers  → m5out/profile_mark_host.csv")
    print(f"[sim] Op labels     → {labels_path}")


# ── Per-test SimTimer ─────────────────────────────────────────────────────────

class SimTimer:
    """
    Callable wrapper injected via the sim_timer fixture.

    sim_timer("matmul", ttnn.matmul, tt_a, tt_w)

    When --sim is active:
      1. connect → send HOST_OP_START(seq) → close
      2. call fn(...)            [blocks until sim kernel finishes]
      3. connect → send HOST_OP_END(seq)   → close
      4. append row to labels CSV (flushed immediately)
      5. append wall-clock row to records list
    """

    _seq = 0

    def __init__(self, records, sock, labels_writer, labels_fh, test_id: str):
        self._records       = records
        self._sock          = sock
        self._labels_writer = labels_writer
        self._labels_fh     = labels_fh
        self._test_id       = test_id

    def __call__(self, op_name: str, fn, *args, **kwargs):
        if self._records is None:
            return fn(*args, **kwargs)

        SimTimer._seq += 1
        seq = SimTimer._seq

        _send_mark(self._sock, HOST_OP_START, seq)

        t0     = perf_counter_ns()
        result = fn(*args, **kwargs)
        wall_ms = (perf_counter_ns() - t0) / 1e6

        _send_mark(self._sock, HOST_OP_END, seq)

        in_shapes = str([tuple(a.shape) for a in args if hasattr(a, "shape")])
        out_shape = str(tuple(result.shape)) if hasattr(result, "shape") else ""
        est_kcy   = int(wall_ms * 1_000 / HOST_US_PER_SIM_CYCLE)

        if self._labels_writer is not None:
            self._labels_writer.writerow(
                [seq, op_name, self._test_id, in_shapes, out_shape])
            self._labels_fh.flush()

        self._records.append((
            seq, self._test_id, op_name,
            in_shapes, out_shape,
            f"{wall_ms:.3f}", est_kcy,
        ))
        return result


@pytest.fixture
def sim_timer(request, sim_log):
    records, sock, labels_w, labels_fh = sim_log
    return SimTimer(records, sock, labels_w, labels_fh, request.node.name)
