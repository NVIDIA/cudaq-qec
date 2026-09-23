#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# gpu_roce_predecoder_test.sh
#
# Orchestration script for the AI predecoder + PyMatching pipeline over
# GPU RoCE RDMA.  Uses HOST_LOOP CPU-launched CUDA graph dispatch
# with the realtime_pipeline external ring buffer path.
#
# Modes:
#   Default (FPGA):   bridge + playback  (requires real FPGA)
#   --emulate:        emulator + bridge + playback  (no FPGA needed)
#
# Data preparation:
#   The predecoder test data (detectors.bin) is in binary format, but the
#   hsb_fpga_syndrome_playback tool expects a text syndromes file and
#   a YAML config.  This script converts them automatically.
#
# Examples:
#   # Full emulated test with d13_r104 data:
#   ./gpu_roce_predecoder_test.sh --emulate --setup-network \
#       --data-dir predecoder/test_data/d13_T104_X
#
#   # Just run (network already set up):
#   ./gpu_roce_predecoder_test.sh --emulate \
#       --data-dir predecoder/test_data/d13_T104_X
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Defaults
# ============================================================================

EMULATE=false
DO_SETUP_NETWORK=false
DO_RUN=true

CUDA_QUANTUM_DIR="${CUDA_QUANTUM_DIR:-/workspaces/cuda-quantum}"
CUDA_QX_DIR="${CUDA_QX_DIR:-/workspaces/cudaqx}"

IB_DEVICE=""
BRIDGE_IP="10.0.0.1"
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

GPU_ID=0
TIMEOUT=60
CONFIG_NAME="d13_r104"
DATA_DIR=""
PAGE_SIZE=""
NUM_PAGES=64
SPACING=""
CONTROL_PORT=8193
NUM_SHOTS=""

# Pipeline config overrides (passed through to bridge binary)
BRIDGE_OVERRIDES=()

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: gpu_roce_predecoder_test.sh [options]

Orchestration for AI predecoder + PyMatching pipeline over GPU RoCE.

Modes:
  --emulate              Use FPGA emulator (no FPGA needed)

Actions:
  --setup-network        Configure ConnectX interfaces
  --no-run               Skip running the test

Directory options:
  --cuda-quantum-dir DIR (default: /workspaces/cuda-quantum)
  --cuda-qx-dir DIR      (default: /workspaces/cuda-qx)
  --data-dir DIR         Predecoder test data (expects detectors.bin,
                         observables.bin inside; e.g. predecoder/test_data/d13_T104_X)

Network options:
  --device DEV           ConnectX IB device name (default: auto-detect)
  --bridge-ip ADDR       Bridge IP (default: 10.0.0.1)
  --emulator-ip ADDR     Emulator IP (default: 10.0.0.2)
  --fpga-ip ADDR         FPGA IP (default: 192.168.0.2)
  --mtu N                MTU size (default: 4096)

Run options:
  --config NAME          d7|d13|d13_r104|d21|d31 (default: d13_r104)
  --gpu N                GPU device ID (default: 0)
  --timeout N            Timeout in seconds (default: 60)
  --page-size N          Ring buffer slot size (default: auto from model)
  --num-pages N          Ring buffer slots (default: 64)
  --num-shots N          Limit number of shots (default: 1 for d13_r104)
  --spacing N            Inter-shot spacing in microseconds
  --control-port N       Emulator UDP control port (default: 8193)

Pipeline config overrides (applied after --config preset):
  --distance=N           QEC code distance
  --num-rounds=N         Syndrome measurement rounds
  --onnx-filename=FILE   ONNX model filename
  --num-predecoders=N    Parallel TRT instances
  --num-workers=N        Pipeline GPU workers
  --num-decode-workers=N PyMatching threads
  --label=NAME           Config label for reports

  --help, -h             Show this help

BRAM constraints (RAM_DEPTH=512):
  d7:        frame=128 B, 2 cycles/shot  -> max 256 shots
  d13:       frame=384 B, 6 cycles/shot  -> max 85 shots
  d13_r104:  frame=17536 B, 274 cycles/shot -> max 1 shot
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulate)          EMULATE=true ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --no-run)           DO_RUN=false ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --cuda-qx-dir)      CUDA_QX_DIR="$2"; shift ;;
        --data-dir)         DATA_DIR="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --config)           CONFIG_NAME="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --page-size)        PAGE_SIZE="$2"; shift ;;
        --num-pages)        NUM_PAGES="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --spacing)          SPACING="$2"; shift ;;
        --control-port)     CONTROL_PORT="$2"; shift ;;
        --distance=*|--num-rounds=*|--onnx-filename=*|--num-predecoders=*|--num-workers=*|--num-decode-workers=*|--label=*)
            BRIDGE_OVERRIDES+=("$1") ;;
        --help|-h)          print_usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
    shift
done

# ============================================================================
# Logging
# ============================================================================

_log()  { echo "==> $*"; }
_info() { echo "    $*"; }
_err()  { echo "ERROR: $*" >&2; }
_banner() {
    echo ""
    echo "========================================"
    echo "  $*"
    echo "========================================"
    echo ""
}

# ============================================================================
# Cleanup
# ============================================================================

PIDS_TO_KILL=()
TEMP_FILES=()

cleanup() {
    local pid
    for pid in "${PIDS_TO_KILL[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
            sleep 1
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    for f in "${TEMP_FILES[@]}"; do
        rm -f "$f"
    done
}
trap cleanup EXIT

# ============================================================================
# Network Setup (reused from gpu_roce_qldpc_graph_decoder_test.sh)
# ============================================================================

detect_interfaces() {
    if ! command -v ibdev2netdev &>/dev/null; then
        _err "ibdev2netdev not found. Install rdma-core or Mellanox OFED."
        return 1
    fi
    ibdev2netdev
}

ib_to_netdev() {
    local ib_dev="$1"
    local port="${2:-1}"
    ibdev2netdev | awk -v dev="$ib_dev" -v p="$port" \
        '$1 == dev && $3 == p { print $5 }'
}

netdev_to_ib() {
    local iface="$1"
    ibdev2netdev | awk -v iface="$iface" '$5 == iface { print $1 }'
}

setup_port() {
    local iface="$1"
    local ip="$2"
    local mtu="$3"
    _info "Configuring $iface: ip=$ip mtu=$mtu"

    local other
    for other in $(ip -o addr show to "${ip}/24" 2>/dev/null | awk '{print $2}' | sort -u); do
        if [[ "$other" != "$iface" ]]; then
            _info "Removing stale ${ip}/24 from $other"
            sudo ip addr del "${ip}/24" dev "$other" 2>/dev/null || true
        fi
    done

    sudo ip link set "$iface" up
    sudo ip link set "$iface" mtu "$mtu"
    sudo ip addr flush dev "$iface"
    sudo ip addr add "${ip}/24" dev "$iface"

    local ib_dev
    ib_dev=$(netdev_to_ib "$iface")
    if [[ -n "$ib_dev" ]] && command -v rdma &>/dev/null; then
        local port_count
        port_count=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
        for p in $(seq 1 "$port_count"); do
            sudo rdma link set "${ib_dev}/${p}" type eth || true
        done
        _info "  RoCEv2 mode configured for $ib_dev"
    fi

    if command -v mlnx_qos &>/dev/null; then
        sudo mlnx_qos -i "$iface" --trust=dscp 2>/dev/null || true
    fi
    if command -v ethtool &>/dev/null; then
        sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
    fi
    _info "  Done: $iface is up at $ip"
}

_add_static_arp() {
    local local_iface="$1"
    local remote_ip="$2"
    local remote_iface="$3"
    local mac
    mac=$(ip link show "$remote_iface" | awk '/ether/ {print $2}')
    if [[ -z "$mac" ]]; then
        _err "Cannot determine MAC address for $remote_iface"
        return 1
    fi
    sudo ip neigh replace "$remote_ip" lladdr "$mac" nud permanent dev "$local_iface"
    _info "  Static ARP: $remote_ip -> $mac on $local_iface"
}

do_setup_network() {
    _log "Setting up ConnectX network"

    if $EMULATE; then
        local interfaces
        interfaces=$(detect_interfaces)

        if [[ -z "$IB_DEVICE" ]]; then
            local first_dev first_iface second_iface

            first_dev=$(echo "$interfaces" | head -1 | awk '{print $1}')
            first_iface=$(echo "$interfaces" | head -1 | awk '{print $5}')
            second_iface=$(echo "$interfaces" | awk -v d="$first_dev" \
                '$1 == d && $3 == 2 {print $5}')

            if [[ -z "$second_iface" ]]; then
                second_iface=$(echo "$interfaces" | awk 'NR==2 {print $5}')
                if [[ -z "$second_iface" ]]; then
                    _err "Need two ConnectX ports for emulation mode."
                    return 1
                fi
            fi

            _info "Bridge interface:   $first_iface"
            _info "Emulator interface: $second_iface"
            setup_port "$first_iface" "$BRIDGE_IP" "$MTU"
            setup_port "$second_iface" "$EMULATOR_IP" "$MTU"

            BRIDGE_DEVICE=$(netdev_to_ib "$first_iface")
            EMULATOR_DEVICE=$(netdev_to_ib "$second_iface")

            _add_static_arp "$first_iface" "$EMULATOR_IP" "$second_iface"
            _add_static_arp "$second_iface" "$BRIDGE_IP" "$first_iface"
        else
            local iface1 iface2
            iface1=$(ib_to_netdev "$IB_DEVICE" 1)
            iface2=$(ib_to_netdev "$IB_DEVICE" 2)
            setup_port "$iface1" "$BRIDGE_IP" "$MTU"
            setup_port "$iface2" "$EMULATOR_IP" "$MTU"
            BRIDGE_DEVICE=$(netdev_to_ib "$iface1")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface2")

            _add_static_arp "$iface1" "$EMULATOR_IP" "$iface2"
            _add_static_arp "$iface2" "$BRIDGE_IP" "$iface1"
        fi

        _info "Waiting 2s for GID tables to populate..."
        sleep 2
    else
        local iface_bridge
        if [[ -n "$IB_DEVICE" ]]; then
            iface_bridge=$(ib_to_netdev "$IB_DEVICE" 1)
        else
            iface_bridge=$(detect_interfaces | head -1 | awk '{print $5}')
        fi
        if [[ -z "$iface_bridge" ]]; then
            _err "Cannot detect ConnectX interface."
            return 1
        fi
        _info "Bridge interface: $iface_bridge"
        setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
        BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")
    fi
}

# ============================================================================
# Data Preparation
# ============================================================================
# The playback tool expects a text syndromes file and a YAML config with
# syndrome_size.  The predecoder test data is in binary (detectors.bin,
# observables.bin).  This step converts N shots to the playback format.

prepare_playback_data() {
    local src_dir="$1"
    local max_shots="$2"
    local out_dir="$3"

    local detectors_bin="${src_dir}/detectors.bin"
    local observables_bin="${src_dir}/observables.bin"

    if [[ ! -f "$detectors_bin" ]]; then
        _err "detectors.bin not found in $src_dir"
        return 1
    fi

    _log "Preparing playback data from $src_dir (max_shots=$max_shots)"

    python3 - "$detectors_bin" "$observables_bin" "$max_shots" "$out_dir" <<'PYEOF'
import struct, sys, os

det_path, obs_path, max_shots_str, out_dir = sys.argv[1:5]
max_shots = int(max_shots_str)
os.makedirs(out_dir, exist_ok=True)

with open(det_path, "rb") as f:
    nrows, ncols = struct.unpack("II", f.read(8))
    n = min(nrows, max_shots)
    print(f"  detectors.bin: {nrows} samples x {ncols} detectors, extracting {n}")

    # Read corrections from observables.bin if available
    corrections = []
    if os.path.isfile(obs_path):
        with open(obs_path, "rb") as of:
            obs_rows, obs_cols = struct.unpack("II", of.read(8))
            for i in range(min(obs_rows, n)):
                row = struct.unpack(f"{obs_cols}i", of.read(obs_cols * 4))
                corrections.append(row[0])  # first observable

    # Write syndromes text file
    syn_path = os.path.join(out_dir, "syndromes_predecoder.txt")
    with open(syn_path, "w") as out:
        for shot in range(n):
            data = struct.unpack(f"{ncols}i", f.read(ncols * 4))
            out.write(f"SHOT_START {shot}\n")
            for d in data:
                out.write(f"{d}\n")
        out.write("CORRECTIONS_START\n")
        for i in range(n):
            c = corrections[i] if i < len(corrections) else 0
            out.write(f"{c}\n")
        out.write("CORRECTIONS_END\n")
    print(f"  Wrote {syn_path}")

    # Write minimal config YAML
    cfg_path = os.path.join(out_dir, "config_predecoder.yml")
    with open(cfg_path, "w") as out:
        out.write(f"syndrome_size: {ncols}\n")
    print(f"  Wrote {cfg_path} (syndrome_size={ncols})")
PYEOF

    PLAYBACK_CONFIG="${out_dir}/config_predecoder.yml"
    PLAYBACK_SYNDROMES="${out_dir}/syndromes_predecoder.txt"
}

# ============================================================================
# Tool Path Resolution
# ============================================================================

resolve_paths() {
    local cuda_qx_utils="${CUDA_QX_DIR}/build/libs/qec/unittests/utils"
    local cuda_qx_realtime="${CUDA_QX_DIR}/build/libs/qec/unittests/realtime"
    local cq_build_dir="${CUDA_QUANTUM_DIR}/realtime/build/unittests"

    BRIDGE_BIN="${cuda_qx_realtime}/gpu_roce_predecoder_bridge"
    PLAYBACK_BIN="${cuda_qx_utils}/hsb_fpga_syndrome_playback"
    EMULATOR_BIN="${cq_build_dir}/utils/hsb_fpga_emulator"

    if [[ ! -x "$BRIDGE_BIN" ]]; then
        _err "Bridge binary not found: $BRIDGE_BIN"
        return 1
    fi
    if [[ ! -x "$PLAYBACK_BIN" ]]; then
        _err "Playback binary not found: $PLAYBACK_BIN"
        return 1
    fi
    if $EMULATE && [[ ! -x "$EMULATOR_BIN" ]]; then
        _err "Emulator binary not found: $EMULATOR_BIN"
        return 1
    fi

    if [ -z "${BRIDGE_DEVICE:-}" ] && [ -n "${IB_DEVICE:-}" ]; then
        BRIDGE_DEVICE="$IB_DEVICE"
    fi
    : "${BRIDGE_DEVICE:=rocep1s0f0}"
    if $EMULATE; then
        : "${EMULATOR_DEVICE:=rocep1s0f1}"
    fi
}

# ============================================================================
# Config-dependent defaults
# ============================================================================

apply_config_defaults() {
    case "$CONFIG_NAME" in
        d7)
            : "${PAGE_SIZE:=384}"
            : "${NUM_SHOTS:=10}"
            if [[ -z "$DATA_DIR" ]]; then
                DATA_DIR="${SCRIPT_DIR}/predecoder/test_data/d7_R7_Z"
            fi
            ;;
        d13)
            : "${PAGE_SIZE:=384}"
            : "${NUM_SHOTS:=10}"
            if [[ -z "$DATA_DIR" ]]; then
                DATA_DIR="${SCRIPT_DIR}/predecoder/test_data/d13_T13_X"
            fi
            ;;
        d13_r104)
            : "${PAGE_SIZE:=32768}"
            : "${NUM_SHOTS:=1}"
            if [[ -z "$DATA_DIR" ]]; then
                DATA_DIR="${SCRIPT_DIR}/predecoder/test_data/d13_T104_X"
            fi
            ;;
        d21)
            : "${PAGE_SIZE:=2048}"
            : "${NUM_SHOTS:=1}"
            if [[ -z "$DATA_DIR" ]]; then
                DATA_DIR="${SCRIPT_DIR}/predecoder/test_data/d21_R21_Z"
            fi
            ;;
        d31)
            : "${PAGE_SIZE:=4096}"
            : "${NUM_SHOTS:=1}"
            if [[ -z "$DATA_DIR" ]]; then
                DATA_DIR="${SCRIPT_DIR}/predecoder/test_data/d31_R31_Z"
            fi
            ;;
        *)
            _err "Unknown config: $CONFIG_NAME"
            return 1
            ;;
    esac

    if [[ ! -d "$DATA_DIR" ]]; then
        _err "Data directory not found: $DATA_DIR"
        _err "Download the predecoder bundle and extract test data there."
        return 1
    fi

    _info "Config: $CONFIG_NAME"
    _info "Data dir: $DATA_DIR"
    _info "Page size: $PAGE_SIZE"
    _info "Num shots: $NUM_SHOTS"
}

# ============================================================================
# Output Parsing
# ============================================================================

wait_for_pattern() {
    local logfile="$1"
    local pattern="$2"
    local timeout_sec="$3"
    local pid_to_check="${4:-}"

    local poll_ms=500
    local waited_ms=0
    local timeout_ms=$((timeout_sec * 1000))
    while (( waited_ms < timeout_ms )); do
        if [[ -n "$pid_to_check" ]] && ! kill -0 "$pid_to_check" 2>/dev/null; then
            _err "Process $pid_to_check died unexpectedly"
            return 1
        fi
        local match
        match=$(grep -m1 "$pattern" "$logfile" 2>/dev/null || true)
        if [[ -n "$match" ]]; then
            echo "$match"
            return 0
        fi
        sleep 0.5
        waited_ms=$((waited_ms + poll_ms))
    done
    _err "Timeout waiting for pattern: $pattern"
    return 1
}

extract_hex() {
    echo "$1" | grep -oP '0x[0-9a-fA-F]+' | head -1
}

extract_decimal() {
    echo "$1" | awk -F': ' '{print $NF}' | tr -d ' '
}

# ============================================================================
# Run: Emulated Mode (3 tools)
# ============================================================================

run_emulated() {
    _banner "Predecoder Bridge Test (Emulated FPGA, $CONFIG_NAME)"

    local emu_log bridge_log
    emu_log=$(mktemp /tmp/predecoder_emulator.XXXXXX.log)
    bridge_log=$(mktemp /tmp/predecoder_bridge.XXXXXX.log)
    TEMP_FILES+=("$emu_log" "$bridge_log")

    # ---- 1. Start emulator ----
    _log "Starting FPGA emulator on port $CONTROL_PORT"
    "$EMULATOR_BIN" \
        --device="$EMULATOR_DEVICE" \
        --port="$CONTROL_PORT" \
        --bridge-ip="$BRIDGE_IP" \
        --page-size="$PAGE_SIZE" \
        > >(tee "$emu_log") 2>&1 &
    local emu_pid=$!
    PIDS_TO_KILL+=("$emu_pid")
    _info "Emulator PID: $emu_pid"

    local emu_qp_line
    emu_qp_line=$(wait_for_pattern "$emu_log" "Emulator QP:" 30 "$emu_pid") || {
        _err "Failed to get emulator QP number"
        return 1
    }
    local emu_qp
    emu_qp=$(extract_hex "$emu_qp_line")
    _info "Emulator QP: $emu_qp"

    # ---- 2. Start predecoder bridge ----
    _log "Starting predecoder bridge (remote-qp=$emu_qp)"

    local bridge_ld_path
    bridge_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDA_QX_DIR}/build/lib"

    CUDA_MODULE_LOADING=EAGER \
    LD_LIBRARY_PATH="${bridge_ld_path}:${LD_LIBRARY_PATH:-}" \
    "$BRIDGE_BIN" \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$EMULATOR_IP" \
        --remote-qp="$emu_qp" \
        --gpu="$GPU_ID" \
        --config="$CONFIG_NAME" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        "${BRIDGE_OVERRIDES[@]}" \
        > >(tee "$bridge_log") 2>&1 &
    local bridge_pid=$!
    PIDS_TO_KILL+=("$bridge_pid")
    _info "Bridge PID: $bridge_pid"

    wait_for_pattern "$bridge_log" "Bridge Ready" 120 "$bridge_pid" >/dev/null || {
        _err "Bridge did not become ready"
        _err "--- Bridge log ---"
        cat "$bridge_log" >&2
        return 1
    }

    local qp_line rkey_line addr_line
    qp_line=$(wait_for_pattern "$bridge_log" "QP Number:" 5 "$bridge_pid") || return 1
    rkey_line=$(wait_for_pattern "$bridge_log" "RKey:" 5 "$bridge_pid") || return 1
    addr_line=$(wait_for_pattern "$bridge_log" "Buffer Addr:" 5 "$bridge_pid") || return 1

    local bridge_qp bridge_rkey bridge_addr
    bridge_qp=$(extract_hex "$qp_line")
    bridge_rkey=$(extract_decimal "$rkey_line")
    bridge_addr=$(extract_hex "$addr_line")

    _info "Bridge QP:     $bridge_qp"
    _info "Bridge RKey:   $bridge_rkey"
    _info "Bridge Buffer: $bridge_addr"

    # ---- 3. Start playback tool ----
    _log "Starting syndrome playback (control-port=$CONTROL_PORT)"
    local playback_args=(
        --hsb-ip "$EMULATOR_IP"
        --control-port "$CONTROL_PORT"
        --config "$PLAYBACK_CONFIG"
        --syndromes "$PLAYBACK_SYNDROMES"
        --function-name predecode
        --qp-number "$bridge_qp"
        --rkey "$bridge_rkey"
        --buffer-addr "$bridge_addr"
        --page-size "$PAGE_SIZE"
        --num-pages "$NUM_PAGES"
    )
    if [[ -n "$NUM_SHOTS" ]]; then
        playback_args+=(--num-shots "$NUM_SHOTS")
    fi
    if [[ -n "$SPACING" ]]; then
        playback_args+=(--spacing "$SPACING")
    fi

    local playback_rc=0
    "$PLAYBACK_BIN" "${playback_args[@]}" || playback_rc=$?

    return $playback_rc
}

# ============================================================================
# Run: FPGA Mode (2 tools)
# ============================================================================

run_fpga() {
    _banner "Predecoder Bridge Test (Real FPGA, $CONFIG_NAME)"

    local bridge_log
    bridge_log=$(mktemp /tmp/predecoder_bridge.XXXXXX.log)
    TEMP_FILES+=("$bridge_log")

    # ---- 1. Start predecoder bridge ----
    _log "Starting predecoder bridge (fpga-ip=$FPGA_IP)"

    local bridge_ld_path
    bridge_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDA_QX_DIR}/build/lib"

    CUDA_MODULE_LOADING=EAGER \
    LD_LIBRARY_PATH="${bridge_ld_path}:${LD_LIBRARY_PATH:-}" \
    "$BRIDGE_BIN" \
        --device="$BRIDGE_DEVICE" \
        --peer-ip="$FPGA_IP" \
        --remote-qp=0x2 \
        --gpu="$GPU_ID" \
        --config="$CONFIG_NAME" \
        --timeout="$TIMEOUT" \
        --page-size="$PAGE_SIZE" \
        --num-pages="$NUM_PAGES" \
        "${BRIDGE_OVERRIDES[@]}" \
        > >(tee "$bridge_log") 2>&1 &
    local bridge_pid=$!
    PIDS_TO_KILL+=("$bridge_pid")
    _info "Bridge PID: $bridge_pid"

    wait_for_pattern "$bridge_log" "Bridge Ready" 120 "$bridge_pid" >/dev/null || {
        _err "Bridge did not become ready"
        return 1
    }

    local qp_line rkey_line addr_line
    qp_line=$(wait_for_pattern "$bridge_log" "QP Number:" 5 "$bridge_pid") || return 1
    rkey_line=$(wait_for_pattern "$bridge_log" "RKey:" 5 "$bridge_pid") || return 1
    addr_line=$(wait_for_pattern "$bridge_log" "Buffer Addr:" 5 "$bridge_pid") || return 1

    local bridge_qp bridge_rkey bridge_addr
    bridge_qp=$(extract_hex "$qp_line")
    bridge_rkey=$(extract_decimal "$rkey_line")
    bridge_addr=$(extract_hex "$addr_line")

    _info "Bridge QP:     $bridge_qp"
    _info "Bridge RKey:   $bridge_rkey"
    _info "Bridge Buffer: $bridge_addr"

    # ---- 2. Start playback tool ----
    _log "Starting syndrome playback (fpga=$FPGA_IP)"
    local playback_args=(
        --hsb-ip "$FPGA_IP"
        --config "$PLAYBACK_CONFIG"
        --syndromes "$PLAYBACK_SYNDROMES"
        --function-name predecode
        --qp-number "$bridge_qp"
        --rkey "$bridge_rkey"
        --buffer-addr "$bridge_addr"
        --page-size "$PAGE_SIZE"
        --num-pages "$NUM_PAGES"
    )
    if [[ -n "$NUM_SHOTS" ]]; then
        playback_args+=(--num-shots "$NUM_SHOTS")
    fi
    if [[ -n "$SPACING" ]]; then
        playback_args+=(--spacing "$SPACING")
    fi

    local playback_rc=0
    "$PLAYBACK_BIN" "${playback_args[@]}" || playback_rc=$?

    return $playback_rc
}

# ============================================================================
# Main
# ============================================================================

main() {
    _banner "GPU RoCE Predecoder + PyMatching Bridge Test"

    if $EMULATE; then
        _info "Mode: FPGA Emulation (3-tool)"
    else
        _info "Mode: Real FPGA (2-tool)"
    fi
    echo ""

    # Apply config-dependent defaults (page size, num shots, data dir)
    apply_config_defaults

    if $DO_SETUP_NETWORK; then
        do_setup_network
    fi

    if ! $DO_RUN; then
        _log "Skipping test run (--no-run)"
        return 0
    fi

    resolve_paths

    # Convert binary predecoder data to playback-compatible text format
    local prep_dir
    prep_dir=$(mktemp -d /tmp/predecoder_playback.XXXXXX)
    TEMP_FILES+=("$prep_dir/config_predecoder.yml" "$prep_dir/syndromes_predecoder.txt")
    prepare_playback_data "$DATA_DIR" "$NUM_SHOTS" "$prep_dir"

    local rc=0
    if $EMULATE; then
        run_emulated || rc=$?
    else
        run_fpga || rc=$?
    fi

    echo ""
    if [[ $rc -eq 0 ]]; then
        _banner "PREDECODER BRIDGE TEST: PASS"
    else
        _banner "PREDECODER BRIDGE TEST: FAIL"
    fi

    return $rc
}

main
