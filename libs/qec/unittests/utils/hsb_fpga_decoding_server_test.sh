#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# hsb_fpga_decoding_server_test.sh
#
# Generic orchestration script for end-to-end decoder testing over Holoscan-
# Sensor-Bridge (HSB) RDMA/RoCE, with the decode work served by the standalone
# decoding server (decoding_server) on the CPU HOST_CALL path instead of a
# GPU bridge.  Works with any CPU decoder the server's YAML config selects;
# the default profile is pymatching.
#
# Data path:
#   FPGA/emulator --RDMA WRITE--> server cpu_roce rx ring
#   server        --RDMA SEND---> FPGA SIF TX (captured by the ILA)
#
# Division of labor (identical to gpu_roce_qldpc_graph_decoder_test.sh):
#   - decoding_server (--transport=cpu_roce --qp_config=hsb_fpga) owns the
#     RDMA ring and prints its endpoint line (QEC_DECODING_SERVER_ENDPOINT
#     qp=... rkey=... buffer_addr=...).  It performs
#     NO HSB control-plane traffic.
#   - hsb_fpga_syndrome_playback is the sole FPGA control-plane writer:
#     it programs the SIF RDMA target with the server's handshake values,
#     writes the syndrome frames to BRAM, arms the ILA, enables the player,
#     and verifies the captured RPC responses.
#
# Modes:
#   Default (FPGA):   server + playback  (requires real FPGA)
#   --emulate:        emulator + server + playback  (no FPGA needed)
#
# Actions (can be combined):
#   --build            Build the surface_code-4 generator (only)
#   --setup-network    Configure ConnectX interfaces
#   (run is implicit unless only --build / --setup-network are given)
#
# Examples:
#   # Full emulated test: build, configure network, run
#   ./hsb_fpga_decoding_server_test.sh --emulate --build --setup-network
#
#   # Just run (tools already built, network already set up)
#   ./hsb_fpga_decoding_server_test.sh --emulate
#
#   # Real FPGA
#   ./hsb_fpga_decoding_server_test.sh --setup-network --device rocep1s0f0 \
#       --bridge-ip 192.168.0.1 --fpga-ip 192.168.0.2
#
# Deployment note:
#   --build builds ONLY the surface_code-4 generator (the one artifact not
#   shipped in the decoding-server image).  Everything else -- decoding_server
#   (with gpu_roce linked in), the playback tool, the HSB / cudaq-realtime
#   shared libs, and the decoder plugins -- is consumed PREBUILT: on a dev rig
#   from the /workspaces/*/build trees, and in a productized container from
#   their installed locations.  This script never builds cuda-quantum, HSB, or
#   the decoder server, and needs no proprietary .a at build time.  A clean,
#   unconfigured rig cannot bootstrap from it.
set -euo pipefail

# ============================================================================
# Defaults
# ============================================================================

EMULATE=false
DO_BUILD=false
DO_SETUP_NETWORK=false
DO_RUN=true
VERIFY=true

# Directory defaults.  HSB_DIR is used as-is (already on the correct branch);
# CUDA_QUANTUM_DIR should be checked out at the ref in CUDAQX_DIR/.cudaq_version
# (verified with a warning in do_build).
HSB_DIR="/workspaces/holoscan-sensor-bridge"
CUDA_QUANTUM_DIR="/workspaces/cuda-quantum"
CUDAQX_DIR="/workspaces/cudaqx"
DATA_DIR=""  # empty => generate data files (see resolve_data_files)

# Decoder profile.  By default the config + syndromes files are GENERATED
# fresh each run by the surface_code-4-yaml binary into GEN_DIR (they are
# derived artifacts and not checked in); --config/--syndromes or --data-dir
# switch to pre-made files and skip generation.
DECODER="pymatching"
CONFIG_FILE=""
SYNDROMES_FILE=""

# ONNX model for the trt_decoder profile (TRT predecoder + PyMatching global
# decoder).  AUTO => generate the tiny identity predecoder at run time
# (output row = [pre_L=0, syndrome untouched], so TRT preserves the syndrome
# and PyMatching performs the actual correction -- no trained model needed).
ONNX_PATH="AUTO"
ONNX_FILE=""

# Data-generation parameters (surface-code memory experiment).  The
# generator's RNG seed is fixed, so runs are reproducible.
GEN_DISTANCE=3
GEN_ROUNDS=4
GEN_P_SPAM=0.01
GEN_SHOTS=100

# Server transport.  Empty => derived from the decoder profile:
#   pymatching        -> cpu_roce  (HOST_CALL dispatch on the CPU)
#   nv-qldpc-decoder  -> gpu_roce  (self-relaunching device-graph scheduler:
#                        enqueue/get/reset run as DEVICE_CALLs on the GPU and
#                        the captured RelayBP decode graph fires device-side)
TRANSPORT=""
# GPU for the gpu_roce scheduler + decode graph.
GPU_ID=0
# Server-side GPU RoCE ring depth. "auto" uses the HSB QP's 64-entry receive
# WQE depth (rings deeper than the WQE count drop frames under load); explicit
# values above it are clamped.
GPU_ROCE_NUM_PAGES=auto

# Runtime nv-qldpc plugin for the Relay BP profile: the prebuilt
# libcudaq-qec-nv-qldpc-decoder.so, dlopen'd by both the generator (during
# syndrome generation) and the prebuilt decoder server.  Not delivered in this
# repo and has no default path -- point at it with --nv-qldpc-plugin or the
# CUDAQ_QEC_NV_QLDPC_PLUGIN env var (eventual home: a GitHub release artifact
# via all_libs_release.yml).  The pymatching profile never uses it.  The
# proprietary cudevice archive is NOT a concern of this script: it is a
# build-time input to the decoder server, which is consumed prebuilt here.
NV_QLDPC_PLUGIN="${CUDAQ_QEC_NV_QLDPC_PLUGIN:-}"

# Network defaults
IB_DEVICE=""           # auto-detect
BRIDGE_IP=""           # resolved after arg parsing (kept the qldpc script's
                       # name): emulate -> 10.0.0.1, real FPGA -> 192.168.0.1
EMULATOR_IP="10.0.0.2"
FPGA_IP="192.168.0.2"
MTU=4096

# Run defaults
TIMEOUT=60
NUM_SHOTS=""
PAGE_SIZE=384
# RX ring depth for both transports, bounded by the HSB QP's 64 receive WQEs (a
# ring with more slots than WQEs drops frames under load). The server clamps
# cpu_roce --num-slots to this; the gpu_roce ring is capped to it below.
NUM_SLOTS=64
# TX SGE bytes for the server's SEND responses.  RPCResponse (24B) + a
# bit-packed correction byte fits well inside 64, keeping every response a
# single 512-bit ILA beat.
FRAME_SIZE=64
SPACING=""
CONTROL_PORT=8193

# Build parallelism
JOBS=$(nproc 2>/dev/null || echo 8)

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: hsb_fpga_decoding_server_test.sh [options]

Generic orchestration script for decoder end-to-end testing over HSB
RDMA/RoCE with the decoding server (decoding_server) on the CPU
HOST_CALL path.  Default decoder profile: pymatching.

Modes:
  --emulate              Use FPGA emulator (3-tool mode, no FPGA needed)
                         Default: FPGA mode (2-tool, requires real FPGA)

Actions:
  --build                Build all required tools before running
  --setup-network        Configure ConnectX network interfaces
  --no-run               Skip running the test (useful with --build)

Decoder options:
  --decoder NAME         Decoder profile: pymatching (default), trt_decoder, or
                         nv-qldpc-decoder (Relay BP).  trt_decoder runs a
                         TensorRT predecoder + PyMatching global decoder in one
                         server session (requires the TRT plugin, see --build).
                         By default the config/syndromes files are generated
                         fresh each run by surface_code-4-yaml into
                         CUDAQX_DIR/build/hsb_fpga_test_data
  --onnx PATH            ONNX model for the trt_decoder profile.  Default AUTO
                         generates the identity predecoder at run time (needs
                         the python3 'onnx' module)
  --transport T          Server transport: cpu_roce or gpu_roce.  Default is
                         derived from the decoder (pymatching -> cpu_roce,
                         nv-qldpc-decoder -> gpu_roce device-graph scheduler)
  --config PATH          Use a pre-made decoding-server YAML config (skips generation)
  --syndromes PATH       Use a pre-made syndromes text file (skips generation)
  --data-dir DIR         Use pre-made DIR/config_NAME.yml + DIR/syndromes_NAME.txt
                         (skips generation)

Data-generation options (ignored when --config/--syndromes/--data-dir given):
  --distance N           Surface-code distance (default: 3)
  --num-rounds N         Measurement rounds (default: 4)
  --p-spam F             SPAM error probability (default: 0.01)
  --gen-shots N          Shots to generate in the syndromes file (default: 100)

Build options:
  --hsb-dir DIR          holoscan-sensor-bridge source directory
                         (default: /workspaces/holoscan-sensor-bridge)
  --cuda-quantum-dir DIR cuda-quantum source directory; must match the ref in
                         CUDAQX_DIR/.cudaq_version (default: /workspaces/cuda-quantum)
  --cudaqx-dir DIR       cudaqx source dir that builds the server + playback
                         (default: /workspaces/cudaqx)
  --nv-qldpc-plugin PATH Prebuilt libcudaq-qec-nv-qldpc-decoder.so, symlinked
                         into build/lib/decoder-plugins for the prebuilt server
                         + generator to dlopen.  No default; required for the
                         nv-qldpc profile (or set CUDAQ_QEC_NV_QLDPC_PLUGIN)
  --jobs N               Parallel build jobs (default: nproc)

Network options:
  --device DEV           ConnectX IB device name (default: auto-detect)
  --bridge-ip ADDR       Server-side NIC IP (default: 10.0.0.1 with
                         --emulate, 192.168.0.1 on the real FPGA; must share
                         the FPGA's /24 in FPGA mode)
  --emulator-ip ADDR     Emulator IP (default: 10.0.0.2)
  --fpga-ip ADDR         FPGA IP for non-emulate mode (default: 192.168.0.2)
  --mtu N                MTU size (default: 4096)

Run options:
  --timeout N            Server timeout in seconds (default: 60)
  --no-verify            Skip correction verification
  --num-shots N          Limit number of shots
  --page-size N          Ring buffer slot size in bytes (default: 384)
  --frame-size N         Server TX SGE bytes, cpu_roce only (default: 64;
                         gpu_roce derives payload size from page-size in YAML)
  --gpu N                GPU device id for gpu_roce (default: 0)
  --gpu-roce-num-pages N Server GPU RoCE ring slots (default: the HSB 64-WQE
                         depth; values above it are clamped, since a ring with
                         more slots than WQEs drops frames)
  --spacing N            Per-frame playback pacing in microseconds (default:
                         10; the FPGA timer fires once per frame, so a shot
                         spans frames-per-shot x spacing). The slow-decode
                         profiles trt_decoder and nv-qldpc-decoder default
                         to 100 instead, so the playback stream cannot
                         overrun the 64-slot RX ring mid-decode
  --control-port N       UDP control port for emulator (default: 8193)

  --help, -h             Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --emulate)          EMULATE=true ;;
        --build)            DO_BUILD=true ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --no-run)           DO_RUN=false ;;
        --no-verify)        VERIFY=false ;;
        --decoder)          DECODER="$2"; shift ;;
        --onnx)             ONNX_PATH="$2"; shift ;;
        --transport)        TRANSPORT="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --gpu-roce-num-pages) GPU_ROCE_NUM_PAGES="$2"; shift ;;
        --nv-qldpc-plugin)  NV_QLDPC_PLUGIN="$2"; shift ;;
        --config)           CONFIG_FILE="$2"; shift ;;
        --syndromes)        SYNDROMES_FILE="$2"; shift ;;
        --data-dir)         DATA_DIR="$2"; shift ;;
        --distance)         GEN_DISTANCE="$2"; shift ;;
        --num-rounds)       GEN_ROUNDS="$2"; shift ;;
        --p-spam)           GEN_P_SPAM="$2"; shift ;;
        --gen-shots)        GEN_SHOTS="$2"; shift ;;
        --hsb-dir)          HSB_DIR="$2"; shift ;;
        --cuda-quantum-dir) CUDA_QUANTUM_DIR="$2"; shift ;;
        --cudaqx-dir)       CUDAQX_DIR="$2"; shift ;;
        --jobs)             JOBS="$2"; shift ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --emulator-ip)      EMULATOR_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --timeout)          TIMEOUT="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --page-size)        PAGE_SIZE="$2"; shift ;;
        --frame-size)       FRAME_SIZE="$2"; shift ;;
        --spacing)          SPACING="$2"; shift ;;
        --control-port)     CONTROL_PORT="$2"; shift ;;
        --help|-h)          print_usage; exit 0 ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
    shift
done

# Resolve the server-side NIC IP by mode: the emulate loop lives on
# 10.0.0.0/24, the real FPGA on 192.168.0.0/24.  A bridge IP outside the
# FPGA's subnet does not fail loudly -- the HSB control plane just
# times out (read_uint32) as if the FPGA were dead -- so reject the
# mismatch up front instead of letting it masquerade as broken hardware.
if [[ -z "$BRIDGE_IP" ]]; then
    if $EMULATE; then BRIDGE_IP="10.0.0.1"; else BRIDGE_IP="192.168.0.1"; fi
fi
if ! $EMULATE && [[ "${BRIDGE_IP%.*}" != "${FPGA_IP%.*}" ]]; then
    echo "ERROR: --bridge-ip $BRIDGE_IP is not in the FPGA's /24 subnet" >&2
    echo "       ($FPGA_IP): the HSB control plane would time out" >&2
    echo "       (read_uint32) as if the FPGA were unreachable.  Pass a" >&2
    echo "       matching --bridge-ip/--fpga-ip pair." >&2
    exit 1
fi

# Derive the transport from the decoder profile unless explicitly chosen.
if [[ -z "$TRANSPORT" ]]; then
    case "$DECODER" in
        nv-qldpc-decoder) TRANSPORT="gpu_roce" ;;
        *)                TRANSPORT="cpu_roce" ;;
    esac
fi
if [[ "$TRANSPORT" != "cpu_roce" && "$TRANSPORT" != "gpu_roce" ]]; then
    echo "ERROR: unknown --transport $TRANSPORT (expected cpu_roce or gpu_roce)" >&2
    exit 1
fi

# Per-decoder playback pacing.  The playback engine's timer fires once per
# FRAME (one BRAM window per frame), so --spacing is per-frame pacing and a
# shot spans frames-per-shot x spacing on the wire.  A profile whose
# per-shot decode latency exceeds that arrival window overruns the 64-slot
# RX ring (dropped enqueues mis-assemble a shot's volume and the decode dies
# with an invalid-syndrome matching error; the ILA sample count comes up
# short).  For this test's profiles the slow ones are trt_decoder (fixed
# per-invocation TensorRT engine overhead at batch 1) and nv-qldpc-decoder
# (iterative relay-BP); pymatching decodes these d3 volumes in microseconds
# and keeps the unpaced default -- the tightly paced path is part of what
# its profile exercises.
#
# 100us clears every measured floor on this loop with margin: at d3/T4 the
# host-dispatch runs fail at 25us (nv-qldpc) and 10us (trt) and pass at
# 100us, and 100us is the validated value for the device-graph scheduler.
# --spacing overrides.
if [[ -z "$SPACING" ]]; then
    case "$DECODER" in
        trt_decoder|nv-qldpc-decoder) SPACING=100 ;;
    esac
fi

# The HSB QP exposes a fixed 64-entry receive WQE depth, and the receiver
# pre-posts one WQE per ring slot. A ring with MORE slots than WQEs leaves slots
# un-posted and drops frames under load: cpu_roce (whose registered MR is only
# num_slots wide) fails outright, and gpu_roce (deeper MR) drops occasionally.
# So both transports cap the RX ring at the WQE depth -- cpu_roce via
# NUM_SLOTS (the server clamps its --num-slots the same way), gpu_roce here.
readonly HSB_WQE_DEPTH=64
if (( NUM_SLOTS > HSB_WQE_DEPTH )); then
    echo "WARNING: NUM_SLOTS=$NUM_SLOTS exceeds the HSB WQE depth" \
         "($HSB_WQE_DEPTH); clamping so the playback ring modulus matches the" \
         "server's clamped RX ring" >&2
    NUM_SLOTS="$HSB_WQE_DEPTH"
fi
if [[ "$TRANSPORT" == "gpu_roce" ]]; then
    if [[ "$GPU_ROCE_NUM_PAGES" == "auto" ]]; then
        GPU_ROCE_NUM_PAGES="$HSB_WQE_DEPTH"
    elif (( GPU_ROCE_NUM_PAGES > HSB_WQE_DEPTH )); then
        echo "WARNING: --gpu-roce-num-pages=$GPU_ROCE_NUM_PAGES exceeds the HSB WQE" \
             "depth ($HSB_WQE_DEPTH); clamping to $HSB_WQE_DEPTH to avoid dropped frames" >&2
        GPU_ROCE_NUM_PAGES="$HSB_WQE_DEPTH"
    fi

    # gpu_roce requires a 128-byte-aligned server stride and a host-page-
    # aligned ring allocation. The playback frame retains the requested page
    # size, while the provider receives the aligned server stride and the
    # payload size after the 24-byte RPC header.
    SERVER_PAGE_SIZE=$(( ((PAGE_SIZE + 127) / 128) * 128 ))
    GPU_ROCE_PAYLOAD_SIZE=$((PAGE_SIZE - 24))
    if (( GPU_ROCE_PAYLOAD_SIZE <= 0 )); then
        echo "ERROR: gpu_roce page-size must exceed the 24-byte RPC header" >&2
        exit 1
    fi

    # DOCA requires the gpu_roce ring allocation (num_pages * page size) to be
    # a multiple of the HOST page size (GpuRoceTransceiver rejects it at
    # startup otherwise). 64 x 384 B = 24 KiB satisfies 4K/8K-page kernels;
    # 16K/64K-page hosts (some aarch64 configs) need a larger stride.
    HOST_PAGE_SIZE=$(getconf PAGESIZE 2>/dev/null || echo 4096)
    if (( (GPU_ROCE_NUM_PAGES * SERVER_PAGE_SIZE) % HOST_PAGE_SIZE != 0 )); then
        echo "ERROR: gpu_roce ring ($GPU_ROCE_NUM_PAGES pages x $SERVER_PAGE_SIZE B) is not a multiple of" >&2
        echo "       this host's page size ($HOST_PAGE_SIZE B). Pass --page-size so its" >&2
        echo "       128-byte-aligned server stride keeps the WQE-depth ring host-page aligned." >&2
        exit 1
    fi
fi

# ============================================================================
# Logging Helpers
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
# Network Setup (mirrors gpu_roce_qldpc_graph_decoder_test.sh)
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
    local ib_dev

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
        _info "  DSCP trust mode set"
    fi

    if command -v ethtool &>/dev/null; then
        sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
    fi

    _info "  Done: $iface is up at $ip"
}

# Pre-seed a PERMANENT neighbor entry for a real FPGA on the server interface.
# The server's QP connect resolves the FPGA's L2 (MAC) address via
# ibv_create_ah, which consults the kernel neighbor table; without priming it
# the in-call ARP resolution times out even though the link is up.
_seed_fpga_neighbor() {
    local iface="$1" fpga_ip="$2"
    ping -c 3 -W 1 -I "$iface" "$fpga_ip" >/dev/null 2>&1 || true
    local mac
    mac=$(ip neigh show "$fpga_ip" dev "$iface" 2>/dev/null \
          | awk '{for (i = 1; i <= NF; i++) if ($i == "lladdr") print $(i + 1)}' \
          | head -1)
    if [[ -n "$mac" ]]; then
        sudo ip neigh replace "$fpga_ip" lladdr "$mac" nud permanent dev "$iface"
        _info "  Static ARP: $fpga_ip -> $mac on $iface"
    else
        _err "  Could not resolve FPGA MAC for $fpga_ip on $iface."
        _err "  Check the FPGA is cabled to this NIC, powered, and reachable"
        _err "  (ping $fpga_ip); otherwise the server QP connect will time out."
    fi
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

# Convert an IPv4 address to the trailing groups of its IPv4-mapped RoCE v2
# GID, e.g. 10.0.0.1 -> "ffff:0a00:0001".
ipv4_to_gid_suffix() {
    local o1 o2 o3 o4
    IFS='.' read -r o1 o2 o3 o4 <<< "$1"
    printf "ffff:%02x%02x:%02x%02x" "$o1" "$o2" "$o3" "$o4"
}

# Poll until the IPv4-mapped RoCE v2 GID for $ip appears on $ib_dev port 1.
# The CPU RoCE transceiver requires this specific GID; it only exists while
# the netdev has the IPv4 address AND is up, and it populates asynchronously,
# so a blind sleep races the server's GID lookup.
wait_for_roce_v2_gid() {
    local ib_dev="$1" ip="$2" timeout_s="${3:-15}"
    local suffix gids_dir types_dir elapsed=0
    suffix=$(ipv4_to_gid_suffix "$ip")
    gids_dir="/sys/class/infiniband/${ib_dev}/ports/1/gids"
    types_dir="/sys/class/infiniband/${ib_dev}/ports/1/gid_attrs/types"
    if [[ ! -d "$gids_dir" ]]; then
        _info "  (no GID sysfs for $ib_dev; skipping GID wait)"
        return 0
    fi
    while (( elapsed < timeout_s * 10 )); do
        local g idx gid t
        for g in "$gids_dir"/*; do
            idx=$(basename "$g")
            gid=$(cat "$g" 2>/dev/null)
            if [[ "$gid" == *":${suffix}" ]]; then
                t=$(cat "${types_dir}/${idx}" 2>/dev/null)
                if [[ "$t" == *"RoCE v2"* ]]; then
                    _info "  RoCE v2 GID ready: ${ib_dev}[${idx}] ${gid}"
                    return 0
                fi
            fi
        done
        sleep 0.1
        elapsed=$((elapsed + 1))
    done
    _err "Timed out waiting for IPv4 RoCE v2 GID (${suffix}) on ${ib_dev}."
    _err "The server's cpu_roce bring-up will fail its GID lookup.  Verify"
    _err "${ip} is assigned to the server netdev and the interface is up."
    return 1
}

do_setup_network() {
    _log "Setting up ConnectX network"

    if $EMULATE; then
        local interfaces
        interfaces=$(detect_interfaces)

        if [[ -z "$IB_DEVICE" ]]; then
            local iface_bridge iface_emulator
            local first_dev first_iface second_iface

            first_dev=$(echo "$interfaces" | head -1 | awk '{print $1}')
            first_iface=$(echo "$interfaces" | head -1 | awk '{print $5}')

            second_iface=$(echo "$interfaces" | awk -v d="$first_dev" \
                '$1 == d && $3 == 2 {print $5}')

            if [[ -n "$second_iface" ]]; then
                iface_bridge="$first_iface"
                iface_emulator="$second_iface"
            else
                second_iface=$(echo "$interfaces" | awk 'NR==2 {print $5}')
                if [[ -z "$second_iface" ]]; then
                    _err "Need two ConnectX ports for emulation mode but only found one."
                    return 1
                fi
                iface_bridge="$first_iface"
                iface_emulator="$second_iface"
            fi

            _info "Server interface:   $iface_bridge"
            _info "Emulator interface: $iface_emulator"
            setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
            setup_port "$iface_emulator" "$EMULATOR_IP" "$MTU"

            BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface_emulator")

            _add_static_arp "$iface_bridge" "$EMULATOR_IP" "$iface_emulator"
            _add_static_arp "$iface_emulator" "$BRIDGE_IP" "$iface_bridge"
        else
            # --device accepts either one dual-port device (ports 1+2) or two
            # comma-separated single-port devices ("devA,devB": server on devA,
            # emulator on devB).
            local iface1 iface2
            if [[ "$IB_DEVICE" == *,* ]]; then
                local dev1="${IB_DEVICE%%,*}" dev2="${IB_DEVICE#*,}"
                iface1=$(ib_to_netdev "$dev1" 1)
                iface2=$(ib_to_netdev "$dev2" 1)
                if [[ -z "$iface1" || -z "$iface2" ]]; then
                    _err "Cannot resolve netdevs for devices $dev1 / $dev2"
                    return 1
                fi
            else
                iface1=$(ib_to_netdev "$IB_DEVICE" 1)
                iface2=$(ib_to_netdev "$IB_DEVICE" 2)
                if [[ -z "$iface1" || -z "$iface2" ]]; then
                    _err "Cannot find two ports on device $IB_DEVICE"
                    return 1
                fi
            fi
            setup_port "$iface1" "$BRIDGE_IP" "$MTU"
            setup_port "$iface2" "$EMULATOR_IP" "$MTU"
            BRIDGE_DEVICE=$(netdev_to_ib "$iface1")
            EMULATOR_DEVICE=$(netdev_to_ib "$iface2")

            _add_static_arp "$iface1" "$EMULATOR_IP" "$iface2"
            _add_static_arp "$iface2" "$BRIDGE_IP" "$iface1"
        fi

        # Wait for the server device's IPv4 RoCE v2 GID before proceeding so
        # the server's cpu_roce GID lookup can't race GID-table population.
        wait_for_roce_v2_gid "$BRIDGE_DEVICE" "$BRIDGE_IP" 15 || true
    else
        local iface_bridge
        if [[ -n "$IB_DEVICE" ]]; then
            iface_bridge=$(ib_to_netdev "$IB_DEVICE" 1)
        else
            iface_bridge=$(detect_interfaces | head -1 | awk '{print $5}')
        fi

        if [[ -z "$iface_bridge" ]]; then
            _err "Cannot detect ConnectX interface for the server."
            return 1
        fi

        _info "Server interface: $iface_bridge"
        setup_port "$iface_bridge" "$BRIDGE_IP" "$MTU"
        BRIDGE_DEVICE=$(netdev_to_ib "$iface_bridge")

        wait_for_roce_v2_gid "$BRIDGE_DEVICE" "$BRIDGE_IP" 15 || true

        # Pre-seed the FPGA's neighbor entry so the server QP connect can
        # resolve its MAC immediately.
        _seed_fpga_neighbor "$iface_bridge" "$FPGA_IP"
    fi
}

# ============================================================================
# Build
# ============================================================================

detect_cuda_arch() {
    local max_arch
    max_arch=$(nvcc --list-gpu-arch 2>/dev/null \
        | grep -oP 'compute_\K[0-9]+' | sort -n | tail -1)
    if [ -n "$max_arch" ]; then
        echo "$max_arch"
    fi
}

do_build() {
    _log "Building the surface_code-4 generator only (jobs=$JOBS)"

    local cudaqx_build="${CUDAQX_DIR}/build"

    # This script builds ONLY the surface_code-4 generator (the config +
    # syndrome producer that is NOT shipped in the decoding-server image).  The
    # decoder server, playback tool, HSB / cudaq-realtime shared libs, and
    # decoder plugins are consumed PREBUILT -- on a dev rig from the
    # /workspaces/*/build trees (resolve_paths points there), and in a
    # productized image from their installed locations.  So a clean,
    # unconfigured rig cannot bootstrap from this script.
    if [[ ! -f "$cudaqx_build/CMakeCache.txt" ]]; then
        _err "cudaqx build dir is not configured ($cudaqx_build/CMakeCache.txt"
        _err "missing).  This script builds only the surface_code-4 generator and"
        _err "consumes the decoder server, playback, libs, and plugins prebuilt."
        _err "Configure + build the cudaqx tree once first, or run against a"
        _err "prebuilt/installed image."
        return 1
    fi

    # cuda-quantum should be at the ref cudaqx pins (the generator links the
    # cuda-quantum realtime libs); warn on skew, as the full build did.
    local pinned_ref current_ref
    pinned_ref=$(jq -r '.cudaq.ref' "${CUDAQX_DIR}/.cudaq_version" 2>/dev/null || true)
    current_ref=$(git -C "$CUDA_QUANTUM_DIR" rev-parse HEAD 2>/dev/null || true)
    if [[ -n "$pinned_ref" && -n "$current_ref" && "$pinned_ref" != "$current_ref" ]]; then
        _err "cuda-quantum checkout ($current_ref) does not match the cudaqx pin"
        _err "($pinned_ref) from ${CUDAQX_DIR}/.cudaq_version.  Continuing, but the"
        _err "realtime libraries may be ABI-skewed against this cudaqx tree."
    elif [[ -n "$pinned_ref" ]]; then
        _info "cuda-quantum at the cudaqx pin: $pinned_ref"
    fi

    # Ensure nvcc is discoverable for the generator's device-code (nvq++)
    # compile step; best-effort (a no-op rebuild needs no compiler).
    local cuda_compiler=""
    if [[ -n "${CMAKE_CUDA_COMPILER:-}" ]]; then
        cuda_compiler="${CMAKE_CUDA_COMPILER}"
    elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
        cuda_compiler="/usr/local/cuda/bin/nvcc"
    else
        cuda_compiler="$(command -v nvcc || true)"
    fi
    if [[ -n "$cuda_compiler" && -x "$cuda_compiler" ]]; then
        local cuda_bin_dir
        cuda_bin_dir="$(dirname "$cuda_compiler")"
        case ":$PATH:" in
            *":$cuda_bin_dir:"*) ;;
            *) export PATH="$cuda_bin_dir:$PATH" ;;
        esac
    fi

    # Rebuild only the generator.  The cudaqx build dir is already configured
    # (checked above); ninja regenerates the build if CMakeLists changed
    # (reusing the cached configure) and recompiles only what is stale.  This
    # script never builds cuda-quantum, HSB, or the decoder server.
    cmake --build "$cudaqx_build" -j "$JOBS" --target surface_code-4-yaml \
        2>&1 | tail -5
    _info "surface_code-4 generator built:"
    _info "  $cudaqx_build/libs/qec/unittests/realtime/app_examples/surface_code-4-yaml"

    _banner "Build complete"
}

# ============================================================================
# Tool Path Resolution
# ============================================================================

# Decide where the config + syndromes files come from.  Pre-made files
# (--config/--syndromes or --data-dir) win and skip generation; otherwise the
# files are generated into GEN_DIR by generate_data_files().
GENERATE_DATA=false
resolve_data_files() {
    GEN_DIR="${CUDAQX_DIR}/build/hsb_fpga_test_data"

    if [[ -n "$DATA_DIR" ]]; then
        # Pre-made profile directory (e.g. checked-in data for some decoder).
        CONFIG_FILE="${CONFIG_FILE:-${DATA_DIR}/config_${DECODER}.yml}"
        SYNDROMES_FILE="${SYNDROMES_FILE:-${DATA_DIR}/syndromes_${DECODER}.txt}"
        return 0
    fi
    if [[ -n "$CONFIG_FILE" && -n "$SYNDROMES_FILE" ]]; then
        return 0
    fi
    if [[ -n "$CONFIG_FILE" || -n "$SYNDROMES_FILE" ]]; then
        _err "--config and --syndromes must be given together (or use --data-dir)."
        return 1
    fi

    # Default: generate both files fresh this run.
    GENERATE_DATA=true
    CONFIG_FILE="${GEN_DIR}/config_${DECODER}.yml"
    SYNDROMES_FILE="${GEN_DIR}/syndromes_${DECODER}.txt"
}

# Resolve ONNX_FILE for the trt_decoder profile.  AUTO generates the tiny
# identity predecoder (mirrors app_examples/surface_code-4-yaml-test.sh):
# output row = [pre_L=0, input syndrome untouched], so TRT preserves the
# syndrome and the PyMatching global decoder performs the actual correction --
# expected corrections stay computable at generation time without a trained
# model, while still exercising ONNX parse, engine build, and per-shot
# inference.
generate_identity_onnx() {
    if [[ "$ONNX_PATH" != "AUTO" ]]; then
        if [[ ! -f "$ONNX_PATH" ]]; then
            _err "--onnx file not found: $ONNX_PATH"
            return 1
        fi
        ONNX_FILE="$ONNX_PATH"
        return 0
    fi
    if ! python3 -c "import onnx" 2>/dev/null; then
        _err "python3 module 'onnx' is required to generate the identity ONNX"
        _err "model (pip install onnx), or pass a pre-made model with --onnx."
        return 1
    fi
    ONNX_FILE="${GEN_DIR}/trt_identity_predecoder.onnx"
    local syndrome_size=$(((GEN_DISTANCE * GEN_DISTANCE - 1) * GEN_ROUNDS))
    _info "Generating identity ONNX: $ONNX_FILE (syndrome_size=$syndrome_size)"
    python3 - "$ONNX_FILE" "$syndrome_size" <<'PY'
import sys

import onnx
from onnx import TensorProto, helper

output_path = sys.argv[1]
syndrome_size = int(sys.argv[2])

input_info = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, syndrome_size])
output_info = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, syndrome_size + 1])
zero = helper.make_node(
    "Constant",
    [],
    ["pre_l"],
    value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 1], [0.0]),
)
concat = helper.make_node("Concat", ["pre_l", "input"], ["output"], axis=1)
graph = helper.make_graph(
    [zero, concat], "trt_identity_predecoder", [input_info], [output_info])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
# IR 9 is sufficient for opset 19 and remains readable by the ONNX checker in
# the CUDA-QX development image.
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, output_path)
PY
    if [[ ! -f "$ONNX_FILE" ]]; then
        _err "Identity ONNX generation failed"
        return 1
    fi
}

# Generate the decoder config (DEM + decoder_custom_args) and the syndromes
# file with the surface_code-4-yaml memory-experiment binary.  Runs in GEN_DIR
# so the generator's auxiliary outputs land there too.
generate_data_files() {
    GENERATOR_BIN="${CUDAQX_DIR}/build/libs/qec/unittests/realtime/app_examples/surface_code-4-yaml"
    if [[ ! -x "$GENERATOR_BIN" ]]; then
        _err "Data generator not found: $GENERATOR_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi

    # trt_decoder profile: the generator needs an ONNX model at
    # config-generation time (--onnx_path is required with --decoder_type
    # trt_decoder).  The TRT plugin itself is consumed PREBUILT and resolved at
    # load time from the decoder-plugins dir on LD_LIBRARY_PATH, exactly like
    # the pymatching plugin -- no explicit existence check here.
    local extra_gen_flags=()
    if [[ "$DECODER" == "trt_decoder" ]]; then
        mkdir -p "$GEN_DIR"
        generate_identity_onnx || return 1
        extra_gen_flags+=(--onnx_path "$ONNX_FILE")
    fi

    _log "Generating test data (decoder=$DECODER, distance=$GEN_DISTANCE," \
         "rounds=$GEN_ROUNDS, p_spam=$GEN_P_SPAM, shots=$GEN_SHOTS)"
    mkdir -p "$GEN_DIR"

    local gen_ld_path
    gen_ld_path="${CUDA_QUANTUM_DIR}/realtime/build/lib:${CUDAQX_DIR}/build/lib"

    # nv-qldpc profile == the Relay BP test: select Relay BP custom args.
    local gen_extra_args=()
    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        gen_extra_args+=(--use-relay-bp)
    fi

    _info "$GENERATOR_BIN --distance $GEN_DISTANCE --num_rounds $GEN_ROUNDS" \
          "--p_spam $GEN_P_SPAM --decoder_type $DECODER" \
          "${extra_gen_flags[*]} ${gen_extra_args[*]:-}" \
          "--save_dem $(basename "$CONFIG_FILE")"
    (cd "$GEN_DIR" && \
     LD_LIBRARY_PATH="${gen_ld_path}:${LD_LIBRARY_PATH:-}" \
     "$GENERATOR_BIN" \
        --distance "$GEN_DISTANCE" \
        --num_rounds "$GEN_ROUNDS" \
        --p_spam "$GEN_P_SPAM" \
        --decoder_type "$DECODER" \
        ${extra_gen_flags[@]+"${extra_gen_flags[@]}"} \
        ${gen_extra_args[@]+"${gen_extra_args[@]}"} \
        --save_dem "$(basename "$CONFIG_FILE")" > gen_config.log 2>&1) || {
        _err "Config generation failed; see ${GEN_DIR}/gen_config.log"
        tail -5 "${GEN_DIR}/gen_config.log" >&2 || true
        return 1
    }

    # Configuration-identity check: the identity ONNX makes TRT->PyMatching
    # output bit-identical to plain PyMatching, so a silently-substituted
    # entry would pass verification below.  Pin the generated config to the
    # combo before anything downstream consumes it.
    if [[ "$DECODER" == "trt_decoder" ]]; then
        if ! grep -Eq "type:[[:space:]]+trt_decoder" "$CONFIG_FILE" || \
           ! grep -q "onnx_load_path" "$CONFIG_FILE"; then
            _err "Generated config lacks a trt_decoder entry with onnx_load_path: $CONFIG_FILE"
            return 1
        fi
        _info "Config carries the trt_decoder entry (onnx_load_path set)"
    fi

    _info "$GENERATOR_BIN --distance $GEN_DISTANCE --num_rounds $GEN_ROUNDS" \
          "--p_spam $GEN_P_SPAM --num_shots $GEN_SHOTS --yaml $(basename "$CONFIG_FILE")" \
          "--save_syndrome $(basename "$SYNDROMES_FILE")"
    (cd "$GEN_DIR" && \
     LD_LIBRARY_PATH="${gen_ld_path}:${LD_LIBRARY_PATH:-}" \
     "$GENERATOR_BIN" \
        --distance "$GEN_DISTANCE" \
        --num_rounds "$GEN_ROUNDS" \
        --p_spam "$GEN_P_SPAM" \
        --num_shots "$GEN_SHOTS" \
        --yaml "$(basename "$CONFIG_FILE")" \
        --save_syndrome "$(basename "$SYNDROMES_FILE")" > gen_syndromes.log 2>&1) || {
        _err "Syndrome generation failed; see ${GEN_DIR}/gen_syndromes.log"
        tail -5 "${GEN_DIR}/gen_syndromes.log" >&2 || true
        return 1
    }

    _info "Generated: $CONFIG_FILE"
    _info "Generated: $SYNDROMES_FILE"
}

# The nv-qldpc profile needs the proprietary plugin dlopen-able from the
# cudaqx decoder-plugins dir (used by both the data generator and the server).
# Symlink it opportunistically so plain runs work without --build.
ensure_nv_qldpc_plugin() {
    local plugin_dir="${CUDAQX_DIR}/build/lib/decoder-plugins"
    # Guard the empty default first: basename "" yields "" so the link path
    # would collapse to "$plugin_dir/", which -e reports as an existing
    # directory -- a false "already present".
    if [[ -z "$NV_QLDPC_PLUGIN" ]]; then
        _err "nv-qldpc profile requires the plugin path: pass --nv-qldpc-plugin PATH"
        _err "or set CUDAQ_QEC_NV_QLDPC_PLUGIN (eventual home: all_libs_release.yml artifact)."
        return 1
    fi
    local link="${plugin_dir}/$(basename "$NV_QLDPC_PLUGIN")"
    if [[ -e "$link" ]]; then
        return 0
    fi
    if [[ ! -f "$NV_QLDPC_PLUGIN" ]]; then
        _err "nv-qldpc plugin not found: $NV_QLDPC_PLUGIN"
        _err "Build it in the proprietary cuda-qx tree or pass --nv-qldpc-plugin PATH."
        return 1
    fi
    mkdir -p "$plugin_dir"
    ln -sf "$NV_QLDPC_PLUGIN" "$link"
    _info "nv-qldpc plugin symlinked: $link"
}

resolve_paths() {
    local cudaqx_utils="${CUDAQX_DIR}/build/libs/qec/unittests/utils"
    local cq_build_dir="${CUDA_QUANTUM_DIR}/realtime/build/unittests"

    SERVER_BIN="${CUDAQX_DIR}/build/bin/decoding_server"
    PLAYBACK_BIN="${cudaqx_utils}/hsb_fpga_syndrome_playback"
    EMULATOR_BIN="${cq_build_dir}/utils/hsb_fpga_emulator"

    if [[ ! -x "$SERVER_BIN" ]]; then
        _err "Decoding server binary not found: $SERVER_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi
    if [[ ! -x "$PLAYBACK_BIN" ]]; then
        _err "Playback binary not found: $PLAYBACK_BIN"
        _err "Run with --build to build the tools first."
        return 1
    fi
    if $EMULATE && [[ ! -x "$EMULATOR_BIN" ]]; then
        _err "Emulator binary not found: $EMULATOR_BIN"
        return 1
    fi
    if [[ ! -f "$CONFIG_FILE" ]]; then
        _err "Config file not found: $CONFIG_FILE"
        return 1
    fi
    if [[ ! -f "$SYNDROMES_FILE" ]]; then
        _err "Syndromes file not found: $SYNDROMES_FILE"
        return 1
    fi

    if [ -z "${BRIDGE_DEVICE:-}" ] && [ -n "${IB_DEVICE:-}" ]; then
        # Mirror do_setup_network's handling of the comma form ("devA,devB":
        # server on devA, emulator on devB) so runs without --setup-network
        # split it the same way.
        if [[ "$IB_DEVICE" == *,* ]]; then
            BRIDGE_DEVICE="${IB_DEVICE%%,*}"
            EMULATOR_DEVICE="${IB_DEVICE#*,}"
        else
            BRIDGE_DEVICE="$IB_DEVICE"
        fi
    fi
    : "${BRIDGE_DEVICE:=rocep1s0f0}"
    if $EMULATE; then
        : "${EMULATOR_DEVICE:=rocep1s0f1}"
    fi
}

# ============================================================================
# Output Parsing Helpers
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
    local line="$1"
    echo "$line" | grep -oP '0x[0-9a-fA-F]+' | head -1
}

# ============================================================================
# Server + Playback (shared by both modes)
# ============================================================================

# Start the decoding server against $1=peer_ip $2=remote_qp; scrape its
# endpoint line into SERVER_QP / SERVER_RKEY / SERVER_ADDR.
prepare_gpu_roce_config() {
    local peer_ip="$1" remote_qp="$2"
    local source_config="$CONFIG_FILE" runtime_config

    if ! python3 -c 'import yaml' 2>/dev/null; then
        _err "python3 module 'yaml' (PyYAML) is required to prepare the gpu_roce launch config"
        return 1
    fi
    runtime_config=$(mktemp /tmp/hsb_decoding_server_config.XXXXXX.yml)
    TEMP_FILES+=("$runtime_config")
    if ! python3 - "$source_config" "$runtime_config" "$GPU_ID" \
            "$BRIDGE_DEVICE" "$peer_ip" "$remote_qp" "$SERVER_PAGE_SIZE" \
            "$GPU_ROCE_NUM_PAGES" "$GPU_ROCE_PAYLOAD_SIZE" <<'PY'
import sys

import yaml

runtime_options = {"--device", "--peer-ip", "--remote-qp", "--page-size",
                   "--num-pages", "--payload-size"}


def launch_args(section, location):
    if "args" not in section:
        return []
    args = section["args"]
    if not isinstance(args, list) or not all(isinstance(arg, str)
                                              for arg in args):
        raise ValueError(f"{location}.args must be a list of strings")
    result = []
    index = 0
    while index < len(args):
        arg = args[index]
        option = arg.split("=", 1)[0]
        if option == "--gpu":
            raise ValueError(f"{location}.args must not set --gpu; "
                             "use decoder.cuda_device_id")
        if option not in runtime_options:
            result.append(arg)
            index += 1
            continue
        if arg == option:
            if index + 1 >= len(args) or args[index + 1].startswith("--"):
                raise ValueError(f"{location}.args has no value for {option}")
            index += 2
        else:
            index += 1
    return result


source_path, output_path, gpu_text, device, peer_ip, remote_qp, page_size, \
    num_pages, payload_size = sys.argv[1:]

try:
    try:
        gpu_id = int(gpu_text, 10)
    except ValueError as error:
        raise ValueError(
            f"requested cuda_device_id {gpu_text!r} is not an integer") from error
    if gpu_id < 0:
        raise ValueError("requested cuda_device_id must be non-negative")

    with open(source_path, encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a YAML mapping")

    decoders = config.get("decoders")
    if not isinstance(decoders, list) or len(decoders) != 1:
        count = len(decoders) if isinstance(decoders, list) else "non-list"
        raise ValueError(
            "gpu_roce HSB helper supports exactly one decoder "
            f"(found {count})")
    decoder = decoders[0]
    if not isinstance(decoder, dict):
        raise ValueError("decoders[0] must be a YAML mapping")

    if "dispatch" not in decoder:
        decoder["dispatch"] = "device_graph"
    elif decoder["dispatch"] != "device_graph":
        raise ValueError(
            "decoders[0].dispatch must be device_graph for gpu_roce "
            f"(found {decoder['dispatch']!r})")

    if "cuda_device_id" not in decoder:
        decoder["cuda_device_id"] = gpu_id
    elif (type(decoder["cuda_device_id"]) is not int or
          decoder["cuda_device_id"] != gpu_id):
        raise ValueError(
            "decoders[0].cuda_device_id contradicts --gpu "
            f"(YAML {decoder['cuda_device_id']!r}, requested {gpu_id})")

    transport = config.get("transport", {})
    if not isinstance(transport, dict):
        raise ValueError("transport must be a YAML mapping")
    if transport.get("provider", "gpu_roce") != "gpu_roce":
        raise ValueError(
            "transport.provider must be gpu_roce "
            f"(found {transport['provider']!r})")
    device_graph = transport.get("device_graph", {})
    if not isinstance(device_graph, dict):
        raise ValueError("transport.device_graph must be a YAML mapping")
    if device_graph.get("provider", "gpu_roce") != "gpu_roce":
        raise ValueError(
            "transport.device_graph.provider must be gpu_roce "
            f"(found {device_graph['provider']!r})")
    args = launch_args(transport, "transport")
    args += launch_args(device_graph, "transport.device_graph")
    config["transport"] = {"provider": "gpu_roce", "args": args + [
        f"--device={device}",
        f"--peer-ip={peer_ip}",
        f"--remote-qp={remote_qp}",
        f"--page-size={page_size}",
        f"--num-pages={num_pages}",
        f"--payload-size={payload_size}",
    ]}

    with open(output_path, "w", encoding="utf-8") as output:
        yaml.safe_dump(config, output, explicit_end=True, sort_keys=False)
except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
    print(f"ERROR: invalid gpu_roce launch config {source_path}: {error}",
          file=sys.stderr)
    sys.exit(1)
PY
    then
        return 1
    fi
    CONFIG_FILE="$runtime_config"
    _info "Authoritative GPU RoCE launch YAML written to $CONFIG_FILE (source unchanged: $source_config)"
}

start_server() {
    local peer_ip="$1" remote_qp="$2" server_log="$3"

    _log "Starting decoding server (decoder=$DECODER, transport=$TRANSPORT," \
         "remote-qp=$remote_qp)"

    # The server dlopens the realtime bridge providers and needs the
    # cudaq-realtime that matches .cudaq_version. Prefer an explicit
    # CUDAQ_REALTIME_LIB_DIR, then the conventional realtime build dirs.
    local server_ld_path rt_lib=""
    for d in "${CUDAQ_REALTIME_LIB_DIR:-}" \
             "${CUDA_QUANTUM_DIR}/realtime/build-bridge/lib" \
             "${CUDA_QUANTUM_DIR}/realtime/build/lib"; do
        [[ -n "$d" && -f "$d/libcudaq-realtime.so" ]] && { rt_lib="$d"; break; }
    done
    server_ld_path="${rt_lib}:${CUDAQX_DIR}/build/lib"

    local ready_pattern
    if [[ "$TRANSPORT" == "gpu_roce" ]]; then
        # Device-graph scheduler path: enqueue/get/reset run as DEVICE_CALLs
        # on the GPU and the captured RelayBP decode graph fires device-side.
        # The temporary launch YAML prepared below makes the gpu_roce provider,
        # device_graph dispatch, GPU, and runtime endpoint authoritative.
        # Eager module loading avoids lazy-load stalls inside the persistent
        # scheduler (same as the old bridge launcher).
        prepare_gpu_roce_config "$peer_ip" "$remote_qp" || return 1
        CUDA_MODULE_LOADING=EAGER \
        LD_LIBRARY_PATH="${server_ld_path}:${LD_LIBRARY_PATH:-}" \
        "$SERVER_BIN" \
            --config="$CONFIG_FILE" \
            --timeout="$TIMEOUT" \
            > >(tee "$server_log") 2>&1 &
        # The DeviceGraphTransceiver prints the QP/RKey/Buffer handshake during
        # server construction, BEFORE this READY sentinel -- so waiting for
        # READY guarantees the three lines are scrapeable.
        ready_pattern="QEC_DECODING_SERVER_READY device_graph"
    else
        LD_LIBRARY_PATH="${server_ld_path}:${LD_LIBRARY_PATH:-}" \
        "$SERVER_BIN" \
            --config="$CONFIG_FILE" \
            --transport=cpu-roce \
            --qp_config=hsb_fpga \
            --device="$BRIDGE_DEVICE" \
            --peer-ip="$peer_ip" \
            --remote-qp="$remote_qp" \
            --num-slots="$NUM_SLOTS" \
            --slot-size="$PAGE_SIZE" \
            --frame-size="$FRAME_SIZE" \
            --timeout="$TIMEOUT" \
            > >(tee "$server_log") 2>&1 &
        ready_pattern="Bridge Ready"
    fi
    SERVER_PID=$!
    PIDS_TO_KILL+=("$SERVER_PID")
    _info "Server PID: $SERVER_PID"

    wait_for_pattern "$server_log" "$ready_pattern" 60 "$SERVER_PID" >/dev/null || {
        _err "Decoding server did not become ready"
        _err "--- Server log ---"
        cat "$server_log" >&2
        return 1
    }

    # Configuration-identity check (all profiles): the server names the
    # decoder session it actually constructed.  Matters most for trt_decoder,
    # where the identity ONNX makes a silently-substituted pymatching session
    # pass the corrections verification bit-identically.
    wait_for_pattern "$server_log" "decoder 0 type: ${DECODER}" 5 "$SERVER_PID" >/dev/null || {
        _err "Server did not construct a '${DECODER}' decoder session"
        _err "--- Server log ---"
        cat "$server_log" >&2
        return 1
    }

    # The RDMA rendezvous tokens (qp=/rkey=/buffer_addr=) are published on ONE
    # line: the device-graph transceiver prints a dedicated
    # QEC_DECODING_SERVER_ENDPOINT line, while the host/cpu-roce path carries
    # the provider's endpoint info on the READY line itself. Scrape whichever
    # arrives.
    local ep_line
    ep_line=$(wait_for_pattern "$server_log" "buffer_addr=" 5 "$SERVER_PID") || return 1

    SERVER_QP=$(sed -n 's/.*[[:space:]]qp=\([0-9a-fA-FxX]*\).*/\1/p' <<<"$ep_line")
    SERVER_RKEY=$(sed -n 's/.*[[:space:]]rkey=\([0-9]*\).*/\1/p' <<<"$ep_line")
    SERVER_ADDR=$(sed -n 's/.*[[:space:]]buffer_addr=\([0-9a-fA-FxX]*\).*/\1/p' <<<"$ep_line")
    if [[ -z "$SERVER_QP" || -z "$SERVER_RKEY" || -z "$SERVER_ADDR" ]]; then
        _err "Endpoint line missing qp=/rkey=/buffer_addr= tokens: $ep_line"
        return 1
    fi

    _info "Server QP:     $SERVER_QP"
    _info "Server RKey:   $SERVER_RKEY"
    _info "Server Buffer: $SERVER_ADDR"
}

# Run playback against $1=hsb_ip; extra args appended from $2...
run_playback() {
    local hsb_ip="$1"; shift

    _log "Starting syndrome playback (hsb_ip=$hsb_ip)"
    # The FPGA/emulator writes syndrome frame rid to RDMA slot (rid % num-pages);
    # --num-pages programs that ring modulus (via the SIF RDMA target). It MUST
    # equal the server's RX ring depth, or frames past one ring land outside the
    # server's registered memory region and are silently lost -- the server then
    # starves after exactly one ring's worth of slots. The cpu_roce server ring
    # is NUM_SLOTS; the gpu_roce server ring is GPU_ROCE_NUM_PAGES (both == the 64
    # WQE depth). This also programs the emulator's SIF target, so emulated mode
    # tracks the same modulus.
    local pb_pages="$NUM_SLOTS"
    if [[ "$TRANSPORT" == "gpu_roce" ]]; then pb_pages="$GPU_ROCE_NUM_PAGES"; fi
    local playback_args=(
        --hsb-ip "$hsb_ip"
        --per-round
        --config "$CONFIG_FILE"
        --syndromes "$SYNDROMES_FILE"
        --qp-number "$SERVER_QP"
        --rkey "$SERVER_RKEY"
        --buffer-addr "$SERVER_ADDR"
        --page-size "$PAGE_SIZE"
        --num-pages "$pb_pages"
        "$@"
    )
    if $VERIFY; then
        playback_args+=(--verify)
    fi
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
# Run: Emulated Mode (3 tools)
# ============================================================================

run_emulated() {
    _banner "Decoding Server Decode Loop Test (Emulated FPGA, $DECODER)"

    local emu_log server_log
    emu_log=$(mktemp /tmp/decoding_server_emulator.XXXXXX.log)
    server_log=$(mktemp /tmp/decoding_server.XXXXXX.log)
    TEMP_FILES+=("$emu_log" "$server_log")

    # ---- 1. Start emulator ----
    # The emulator's RDMA-write ring modulus is the SIF target the playback
    # programs it with (target.max_buff + 1), not a CLI flag -- it does not parse
    # --num-pages -- so run_playback's page count drives emulated mode too.
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

    # ---- 2. Start decoding server ----
    start_server "$EMULATOR_IP" "$emu_qp" "$server_log" || return 1

    # ---- 3. Start playback tool ----
    run_playback "$EMULATOR_IP" --control-port "$CONTROL_PORT"
}

# ============================================================================
# Run: FPGA Mode (2 tools)
# ============================================================================

run_fpga() {
    _banner "Decoding Server Decode Loop Test (Real FPGA, $DECODER)"

    local server_log
    server_log=$(mktemp /tmp/decoding_server.XXXXXX.log)
    TEMP_FILES+=("$server_log")

    # ---- 1. Start decoding server (FPGA data-plane QP is fixed 0x2) ----
    start_server "$FPGA_IP" "0x2" "$server_log" || return 1

    # ---- 2. Start playback tool (BOOTP enumeration; no control port) ----
    run_playback "$FPGA_IP"
}

# ============================================================================
# Main
# ============================================================================

main() {
    _banner "HSB FPGA Decoding Server Test"

    _info "Decoder: $DECODER (decoding server, transport=$TRANSPORT)"
    if $EMULATE; then
        _info "Mode: FPGA Emulation (3-tool)"
    else
        _info "Mode: Real FPGA (2-tool)"
    fi
    echo ""

    # ---- Build ----
    if $DO_BUILD; then
        do_build
    fi

    # ---- Network setup ----
    if $DO_SETUP_NETWORK; then
        do_setup_network
    fi

    # ---- Run ----
    if ! $DO_RUN; then
        _log "Skipping test run (--no-run)"
        return 0
    fi

    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        ensure_nv_qldpc_plugin
    fi
    resolve_data_files
    if $GENERATE_DATA; then
        generate_data_files
    fi
    resolve_paths

    local rc=0
    if $EMULATE; then
        run_emulated || rc=$?
    else
        run_fpga || rc=$?
    fi

    # ---- Verdict ----
    echo ""
    if [[ $rc -eq 0 ]]; then
        _banner "DECODING SERVER DECODE LOOP ($DECODER): PASS"
    else
        _banner "DECODING SERVER DECODE LOOP ($DECODER): FAIL"
    fi

    return $rc
}

main
