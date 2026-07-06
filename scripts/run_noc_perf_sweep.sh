#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_noc_perf_sweep.sh [options]

Runs a small NoC microbenchmark sweep from tt-metal against an already running
wormhole_sim gem5 instance.

Start gem5 manually first, for example:

  cd /workspaces/wormhole_sim
  build/RISCV/gem5.opt configs/tensix/tensix_top_tlm.py --tensix-size=big

Options:
  --out DIR                    Output directory for the finalized
                               wormhole_sim/m5out copy.
  --run-single-dest            Run the single-destination unicast/multicast
                               latency sanity test.
  --run-adjacent               Run the adjacent directional NoC sweep.
  --run-rtor                   Include the RToR sweep.
  --run-stress-mcast           Run the disabled legacy NoC multicast stress
                               gtest explicitly.
  --run-one-packet             Run focused one-packet NoC data-movement tests.
  --run-core-bidirectional     Run bidirectional core-to-core NoC tests.
  --run-multicast-atomics      Run multicast atomic semaphore tests.
  --run-multicast-schemes      Run the active one-to-all multicast schemes sweep.
  --run-all-to-all             Run all-to-all NoC data-movement tests.
  --run-sdpa-reduce            Run legacy `sdpa_reduce_c` from unit_tests_legacy.
  --run-transpose-hc           Run legacy `transpose_hc` from unit_tests_legacy.
  --run-broadcast-golden       Run the one-to-all broadcast golden test.
  --run-noc-api-latency        Run focused NOC API latency tests.
  --run-noc-estimator          Run the full hardware NocEstimator* GTest sweep
                               from unit_tests_data_movement.
  --run-perf-base              Run the full baseline performance suite group.
  --run-all, all               Run every test group above.

Environment:
  TT_METAL_ROOT       Defaults to this script's tt-metal checkout.
  WORMHOLE_SIM_ROOT   Defaults to /workspaces/wormhole_sim.
  TT_SIM_SOCK         Defaults to /tmp/tt_sim.sock.
  TT_MARKER_SOCK      Defaults to /tmp/tt_marker.sock. Used for host-side
                       per-test START/END markers; set NOC_HOST_MARKERS=0
                       to disable.
  GEM5_PID            Optional gem5 PID. If unset, the script auto-detects it.
  NOC_CORES_R         Defaults to 9.
  NOC_CORES_C         Defaults to 8.
  NOC_NUM_TILES       Defaults to 1024.
  TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT
                       Defaults to 10000 for this sweep to reduce profiler
                       DRAM-buffer overflow while NoC event profiling is on.
  TT_METAL_FORCE_JIT_COMPILE
                       Defaults to 0. Set to 1 when kernel rebuild is required.
EOF
}

tt_metal_root() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${script_dir}/.." && pwd
}

wait_for_socket() {
    local sock="$1"
    local timeout_s="$2"
    local waited=0

    while [[ ! -S "${sock}" ]]; do
        if (( waited >= timeout_s )); then
            echo "Timed out waiting for ${sock}" >&2
            echo "Start wormhole_sim gem5 manually, then rerun this script." >&2
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done
}

start_host_marker_client() {
    if [[ "${NOC_HOST_MARKERS}" != "1" || ! -S "${TT_MARKER_SOCK}" ]]; then
        return 0
    fi

    HOST_MARKER_FIFO="${TMPDIR:-/tmp}/noc_marker_fifo_$$"
    rm -f "${HOST_MARKER_FIFO}"
    mkfifo "${HOST_MARKER_FIFO}"

    python3 -c '
import socket
import struct
import sys

sock_path = sys.argv[1]
cmd_write_reg = 0x02
profile_mark = 0xFFB80214

sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
sock.settimeout(2.0)
sock.connect(sock_path)
sock.settimeout(None)
try:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        phase_s, seq_s = line.split()
        phase = int(phase_s, 0)
        seq = int(seq_s, 0)
        value = ((phase & 0xff) << 24) | (seq & 0x00ffffff)
        sock.sendall(struct.pack("<BIII", cmd_write_reg, profile_mark, 4, value))
        sock.recv(1)
finally:
    sock.close()
' "${TT_MARKER_SOCK}" < "${HOST_MARKER_FIFO}" &
    HOST_MARKER_CLIENT_PID=$!
    exec {HOST_MARKER_FD}>"${HOST_MARKER_FIFO}"
}

stop_host_marker_client() {
    if [[ -n "${HOST_MARKER_FD:-}" ]]; then
        eval "exec ${HOST_MARKER_FD}>&-"
        unset HOST_MARKER_FD
    fi
    if [[ -n "${HOST_MARKER_CLIENT_PID:-}" ]]; then
        wait "${HOST_MARKER_CLIENT_PID}" 2>/dev/null || true
        unset HOST_MARKER_CLIENT_PID
    fi
    if [[ -n "${HOST_MARKER_FIFO:-}" ]]; then
        rm -f "${HOST_MARKER_FIFO}"
        unset HOST_MARKER_FIFO
    fi
}

host_marker_send() {
    local phase="$1"
    local seq="$2"

    if [[ "${NOC_HOST_MARKERS}" != "1" || -z "${HOST_MARKER_FD:-}" ]]; then
        return 0
    fi

    printf '%s %s\n' "${phase}" "${seq}" >&"${HOST_MARKER_FD}" || true
}

init_host_markers() {
    HOST_MARKER_SEQ=0
    HOST_MARKER_LABELS="${WORMHOLE_SIM_ROOT}/m5out/profile_mark_noc_host_label.csv"

    if [[ "${NOC_HOST_MARKERS}" != "1" ]]; then
        return 0
    fi

    if [[ ! -S "${TT_MARKER_SOCK}" ]]; then
        echo "WARNING: host marker socket not found: ${TT_MARKER_SOCK}; per-test host markers disabled for this run." >&2
        NOC_HOST_MARKERS=0
        return 0
    fi

    mkdir -p "$(dirname "${HOST_MARKER_LABELS}")"
    printf 'seq_num,op_name,test_id,in_shapes,out_shape\n' > "${HOST_MARKER_LABELS}"
    start_host_marker_client
    trap stop_host_marker_client EXIT
    echo "Host profile markers: ${TT_MARKER_SOCK}"
    echo "Host profile labels: ${HOST_MARKER_LABELS}"
}

host_marker_start() {
    local label="$1"
    HOST_MARKER_SEQ=$((HOST_MARKER_SEQ + 1))
    host_marker_send 0xF0 "${HOST_MARKER_SEQ}"
}

host_marker_end() {
    local seq="$1"
    local label="$2"

    host_marker_send 0xF1 "${seq}"
    if [[ "${NOC_HOST_MARKERS}" == "1" ]]; then
        printf '%s,%s,noc_perf,[],[]\n' "${seq}" "${label}" >> "${HOST_MARKER_LABELS}"
    fi
}

copy_final_m5out() {
    local root="$1"
    local out_root="$2"
    local dst="${out_root}/m5out"

    mkdir -p "${out_root}"
    if [[ -e "${dst}" ]]; then
        echo "Output snapshot already exists: ${dst}" >&2
        return 1
    fi
    if [[ -d "${root}/m5out" ]]; then
        cp -a "${root}/m5out" "${dst}"
        echo "Copied ${root}/m5out -> ${dst}"
    else
        echo "No ${root}/m5out directory found" >&2
    fi
}

find_gem5_pid() {
    if [[ -n "${GEM5_PID:-}" ]]; then
        printf '%s\n' "${GEM5_PID}"
        return 0
    fi

    pgrep -f 'gem5\.opt.*tensix_top_tlm\.py|gem5\.opt.*--tensix-size=big' | tail -n 1
}

finalize_gem5_and_copy() {
    local out_root="$1"
    local pid
    pid="$(find_gem5_pid || true)"

    if [[ -z "${pid}" ]]; then
        echo "WARNING: could not find gem5 PID; copying current m5out without forcing flush." >&2
        copy_final_m5out "${WORMHOLE_SIM_ROOT}" "${out_root}"
        return 0
    fi

    echo
    echo "Sending SIGINT to gem5 PID ${pid} to flush stats/profile files..."
    kill -INT "${pid}"

    local waited=0
    while kill -0 "${pid}" 2>/dev/null; do
        if (( waited >= 300 )); then
            echo "Timed out waiting for gem5 PID ${pid} to exit after SIGINT" >&2
            return 1
        fi
        sleep 1
        waited=$((waited + 1))
    done

    copy_final_m5out "${WORMHOLE_SIM_ROOT}" "${out_root}"
}

run_if_exists() {
    local name="$1"
    local bin="$2"
    shift 2

    if [[ ! -x "${bin}" ]]; then
        echo "SKIP ${name}: missing executable ${bin}"
        return 0
    fi

    echo
    echo "==== ${name} ===="

    local marker_seq
    local status=0
    host_marker_start "${name}"
    marker_seq="${HOST_MARKER_SEQ}"

    (
        cd "${TT_METAL_ROOT}"
        env \
            TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
            TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT}" \
            TT_METAL_FORCE_JIT_COMPILE="${TT_METAL_FORCE_JIT_COMPILE}" \
            TT_METAL_SLOW_DISPATCH_MODE=1 \
            "$bin" "$@"
    ) || status=$?

    host_marker_end "${marker_seq}" "${name}"
    return "${status}"
}

run_if_exists_fast_dispatch() {
    local name="$1"
    local bin="$2"
    shift 2

    if [[ ! -x "${bin}" ]]; then
        echo "SKIP ${name}: missing executable ${bin}"
        return 0
    fi

    echo
    echo "==== ${name} ===="

    local marker_seq
    local status=0
    host_marker_start "${name}"
    marker_seq="${HOST_MARKER_SEQ}"

    (
        cd "${TT_METAL_ROOT}"
        env \
            TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 \
            TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT}" \
            TT_METAL_FORCE_JIT_COMPILE="${TT_METAL_FORCE_JIT_COMPILE}" \
            TT_METAL_SLOW_DISPATCH_MODE=0 \
            "$bin" "$@"
    ) || status=$?

    host_marker_end "${marker_seq}" "${name}"
    return "${status}"
}

find_test_bin() {
    local rel="$1"
    local candidate
    local rel_no_config="${rel/\/Debug\//\/}"
    local rel_release="${rel/\/Debug\//\/Release\/}"
    local roots=(
        "${TT_METAL_ROOT}/build"
        "${TT_METAL_ROOT}/.build/default"
        "${TT_METAL_ROOT}/build_Debug"
        "${TT_METAL_ROOT}/build_Release"
        "${TT_METAL_ROOT}/build_RelWithDebInfo"
    )

    for root in "${roots[@]}"; do
        for candidate in "${root}/${rel}" "${root}/${rel_release}" "${root}/${rel_no_config}"; do
            if [[ -x "${candidate}" ]]; then
                printf '%s\n' "${candidate}"
                return 0
            fi
        done
    done

    printf '%s\n' "${TT_METAL_ROOT}/build/${rel}"
}

TT_METAL_ROOT="${TT_METAL_ROOT:-$(tt_metal_root)}"
WORMHOLE_SIM_ROOT="${WORMHOLE_SIM_ROOT:-/workspaces/wormhole_sim}"
TT_SIM_SOCK="${TT_SIM_SOCK:-/tmp/tt_sim.sock}"
TT_MARKER_SOCK="${TT_MARKER_SOCK:-/tmp/tt_marker.sock}"
NOC_HOST_MARKERS="${NOC_HOST_MARKERS:-1}"
OUT_ROOT="${WORMHOLE_SIM_ROOT}/m5out_noc_perf_sweep_$(date +%Y%m%d_%H%M%S)"
run_single_dest=0
run_adjacent=0
run_rtor=0
run_stress_mcast=0
run_one_packet=0
run_core_bidirectional=0
run_multicast_atomics=0
run_multicast_schemes=0
run_all_to_all=0
run_sdpa_reduce=0
run_transpose_hc=0
run_broadcast_golden=0
run_noc_api_latency=0
run_noc_estimator=0
run_perf_base=0

select_perf_base() {
    run_perf_base=1
}

select_all_tests() {
    run_single_dest=1
    run_adjacent=1
    run_rtor=1
    run_stress_mcast=1
    run_one_packet=1
    run_core_bidirectional=1
    run_multicast_atomics=1
    run_multicast_schemes=1
    run_all_to_all=1
    run_sdpa_reduce=1
    run_transpose_hc=1
    run_broadcast_golden=1
    run_noc_api_latency=1
    run_noc_estimator=1
    select_perf_base
}

while (($#)); do
    case "$1" in
        --out)
            OUT_ROOT="$2"
            shift 2
            ;;
        --run-single-dest)
            run_single_dest=1
            shift
            ;;
        --run-adjacent)
            run_adjacent=1
            shift
            ;;
        --run-rtor)
            run_rtor=1
            shift
            ;;
        --run-stress-mcast)
            run_stress_mcast=1
            shift
            ;;
        --run-one-packet)
            run_one_packet=1
            shift
            ;;
        --run-core-bidirectional)
            run_core_bidirectional=1
            shift
            ;;
        --run-multicast-atomics)
            run_multicast_atomics=1
            shift
            ;;
        --run-multicast-schemes)
            run_multicast_schemes=1
            shift
            ;;
        --run-all-to-all)
            run_all_to_all=1
            shift
            ;;
        --run-sdpa-reduce)
            run_sdpa_reduce=1
            shift
            ;;
        --run-transpose-hc)
            run_transpose_hc=1
            shift
            ;;
        --run-broadcast-golden)
            run_broadcast_golden=1
            shift
            ;;
        --run-noc-api-latency)
            run_noc_api_latency=1
            shift
            ;;
        --run-noc-estimator)
            run_noc_estimator=1
            shift
            ;;
        --run-perf-base)
            select_perf_base
            shift
            ;;
        --run-all|all)
            select_all_tests
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

NOC_CORES_R="${NOC_CORES_R:-9}"
NOC_CORES_C="${NOC_CORES_C:-8}"
NOC_NUM_TILES="${NOC_NUM_TILES:-1024}"
TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT="${TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT:-10000}"
TT_METAL_FORCE_JIT_COMPILE="${TT_METAL_FORCE_JIT_COMPILE:-0}"

if (( run_single_dest == 0 &&
      run_adjacent == 0 &&
      run_rtor == 0 &&
      run_stress_mcast == 0 &&
      run_one_packet == 0 &&
      run_core_bidirectional == 0 &&
      run_multicast_atomics == 0 &&
      run_multicast_schemes == 0 &&
      run_all_to_all == 0 &&
      run_sdpa_reduce == 0 &&
      run_transpose_hc == 0 &&
      run_broadcast_golden == 0 &&
      run_noc_api_latency == 0 &&
      run_noc_estimator == 0 &&
      run_perf_base == 0 )); then
    echo "No test group selected. Pass one or more --run-* options, --run-perf-base, or pass all." >&2
    usage
    exit 2
fi

if [[ ! -f "${TT_METAL_ROOT}/tt_env.sh" ]]; then
    echo "Cannot find ${TT_METAL_ROOT}/tt_env.sh" >&2
    exit 1
fi

if [[ ! -d "${WORMHOLE_SIM_ROOT}" ]]; then
    echo "Cannot find WORMHOLE_SIM_ROOT=${WORMHOLE_SIM_ROOT}" >&2
    exit 1
fi

echo "Waiting for simulator socket: ${TT_SIM_SOCK}"
wait_for_socket "${TT_SIM_SOCK}" 300

set +u
source "${TT_METAL_ROOT}/tt_env.sh"
set -u

mkdir -p "${OUT_ROOT}"
echo "Final output directory: ${OUT_ROOT}"
init_host_markers

NOC_SIMPLE="$(find_test_bin "test/tt_metal/perf_microbenchmark/noc/Debug/test_noc_unicast_vs_multicast_to_single_core_latency_worker")"
NOC_ADJ="$(find_test_bin "test/tt_metal/perf_microbenchmark/2_noc_adjacent/Debug/test_noc_adjacent")"
DM_TESTS="$(find_test_bin "test/tt_metal/Debug/unit_tests_data_movement")"

TEST_FAILURES=0

if [[ "${run_perf_base}" == "1" ]]; then
    if ! run_if_exists \
        "perf_base_dram_unary" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementDRAMPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementDRAMDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_one_to_one" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementOneToOnePacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementOneToOneDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_one_from_one" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementOneFromOnePacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementOneFromOneDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_loopback" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementLoopbackPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementLoopbackDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_dram_interleaved" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementDRAMInterleavedPageReadNumbersSlowDispatch:MeshDeviceFixture.TensixDataMovementDRAMInterleavedPageWriteNumbersSlowDispatch:MeshDeviceFixture.TensixDataMovementDRAMInterleavedPageDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_one_to_all" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementOneToAllUnicastPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementOneToAllMulticastPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementOneToAllMulticastLinkedPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementOneToAllUnicastDirectedIdealSlowDispatch:MeshDeviceFixture.TensixDataMovementOneToAllMulticastDirectedIdealSlowDispatch:MeshDeviceFixture.TensixDataMovementOneToAllMulticastLinkedDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_one_to_all_multicast_schemes" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementOneToAllMulticastSchemesNoLoopbackSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_one_from_all" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementOneFromAllPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementOneFromAllDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_all_to_all" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementAllToAllPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementAllToAllDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_all_from_all" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixDataMovementAllFromAllPacketSizesSlowDispatch:MeshDeviceFixture.TensixDataMovementAllFromAllDirectedIdealSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi

    if ! run_if_exists \
        "perf_base_noc_api_latency" \
        "${DM_TESTS}" \
        --gtest_filter='MeshDeviceFixture.TensixNocApiLatencyUnicastWriteSlowDispatch:MeshDeviceFixture.TensixNocApiLatencyUnicastReadSlowDispatch:MeshDeviceFixture.TensixNocApiLatencyStatefulWriteSlowDispatch:MeshDeviceFixture.TensixNocApiLatencyStatefulReadSlowDispatch:MeshDeviceFixture.TensixNocApiLatencyMulticastWriteAllSlowDispatch'; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_single_dest}" == "1" ]]; then
    if ! run_if_exists "single_dest_unicast_multicast" "${NOC_SIMPLE}"; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_adjacent}" == "1" ]]; then
    for noc in 0 1; do
        for access in 0 1; do
            for dir in 0 1 2 3; do
                if ! run_if_exists \
                    "adjacent_noc${noc}_access${access}_dir${dir}" \
                    "${NOC_ADJ}" \
                    --cores-r "${NOC_CORES_R}" \
                    --cores-c "${NOC_CORES_C}" \
                    --num-tiles "${NOC_NUM_TILES}" \
                    --tiles-per-transfer 1 \
                    --noc-index "${noc}" \
                    --noc-direction "${dir}" \
                    --access-type "${access}" \
                    --use-device-profiler \
                    --num-tests 1 \
                    --bypass-check; then
                    TEST_FAILURES=$((TEST_FAILURES + 1))
                fi
            done
        done
    done
fi

if [[ "${run_rtor}" == "1" ]]; then
    NOC_RTOR="$(find_test_bin "test/tt_metal/perf_microbenchmark/2_noc_rtor/Debug/test_noc_rtor")"
    for noc in 0 1; do
        for access in 0 1; do
            if ! run_if_exists \
                "rtor_noc${noc}_access${access}" \
                "${NOC_RTOR}" \
                --cores-r "${NOC_CORES_R}" \
                --cores-c "${NOC_CORES_C}" \
                --num-tiles "${NOC_NUM_TILES}" \
                --noc-index "${noc}" \
                --access-type "${access}" \
                --num-tests 1 \
                --bypass-check; then
                TEST_FAILURES=$((TEST_FAILURES + 1))
            fi
        done
    done
fi

if [[ "${run_stress_mcast}" == "1" ]]; then
    LEGACY_TESTS="$(find_test_bin "test/tt_metal/Debug/unit_tests_legacy")"
    if ! run_if_exists \
        "stress_noc_mcast_legacy" \
        "${LEGACY_TESTS}" \
        --gtest_filter=MeshDeviceSingleCardFixture.DISABLED_StressNocMcast \
        --gtest_also_run_disabled_tests; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_one_packet}" == "1" ]]; then
    if ! run_if_exists \
        "one_packet_directed_ideal" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.TensixDataMovementOnePacket*DirectedIdeal*; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_core_bidirectional}" == "1" ]]; then
    if ! run_if_exists \
        "core_bidirectional" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.TensixDataMovementCoreBidirectional*; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_multicast_atomics}" == "1" ]]; then
    if ! run_if_exists \
        "multicast_atomics" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.MulticastAtomic*; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_multicast_schemes}" == "1" ]]; then
    if ! run_if_exists \
        "multicast_schemes_2_0" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.TensixDataMovementOneToAllMulticastSchemesNoLoopback2_0; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_all_to_all}" == "1" ]]; then
    if ! run_if_exists \
        "all_to_all" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.TensixDataMovementAllToAll*; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_sdpa_reduce}" == "1" ]]; then
    LEGACY_TESTS="$(find_test_bin "test/tt_metal/Debug/unit_tests_legacy")"
    if ! run_if_exists \
        "sdpa_reduce_c" \
        "${LEGACY_TESTS}" \
        --gtest_filter=UnitMeshCQSingleCardSharedFixture.NIGHTLY_SdpaReduceC; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_transpose_hc}" == "1" ]]; then
    LEGACY_TESTS="$(find_test_bin "test/tt_metal/Debug/unit_tests_legacy")"
    if ! run_if_exists \
        "transpose_hc" \
        "${LEGACY_TESTS}" \
        --gtest_filter=MeshDeviceSingleCardFixture.TransposeHC; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_broadcast_golden}" == "1" ]]; then
    if ! run_if_exists \
        "broadcast_golden_one_to_all" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.TensixDataMovementOneToAllBroadcastGoldenSingle2_0; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_noc_api_latency}" == "1" ]]; then
    if ! run_if_exists \
        "noc_api_latency" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.TensixNocApiLatency*; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

if [[ "${run_noc_estimator}" == "1" ]]; then
    if ! run_if_exists \
        "noc_estimator_hardware_sweep" \
        "${DM_TESTS}" \
        --gtest_filter=GenericMeshDeviceFixture.NocEstimator*; then
        TEST_FAILURES=$((TEST_FAILURES + 1))
    fi
fi

finalize_gem5_and_copy "${OUT_ROOT}"

echo
echo "Sweep complete: ${OUT_ROOT}"
if (( TEST_FAILURES > 0 )); then
    echo "WARNING: ${TEST_FAILURES} host test command(s) failed; inspect the host log and finalized m5out."
fi
