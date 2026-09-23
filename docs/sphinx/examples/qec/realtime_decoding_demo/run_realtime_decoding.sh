#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# run_realtime_decoding.sh
#
# Drive the delivered decoding_server from one of two syndrome sources, both
# decoding through the SAME prebuilt server:
#
#   --source qpu-kernel   The lowered QEC kernel supplies syndromes itself over
#                         the udp wire, in software (no NIC) -- the portable,
#                         hardware-free mode -- or over the cpu_roce wire
#                         (--wire cpu_roce: real RDMA between two loopback-
#                         cabled RoCE ports; topology from the same
#                         CUDAQ_CPU_ROCE_TEST_* env vars as the in-tree
#                         cpu_roce tests).  Every decoder is served on host
#                         dispatch (a CPU thread calls the decoder; nv-qldpc
#                         still decodes on its GPU).
#
#   --source fpga         The delivered hsb_fpga_syndrome_playback tool
#                         streams pre-generated syndromes over RoCE from a REAL
#                         FPGA into the server's RDMA ring (needs a ConnectX NIC
#                         cabled to the FPGA).  CPU decoders ride the cpu_roce
#                         wire on host dispatch; nv-qldpc defaults to the
#                         gpu_roce wire on device_graph dispatch (the GPU
#                         device-call scheduler).  There is NO emulator here --
#                         emulator testing lives in the unittests
#                         hsb_fpga_decoding_server_test.sh.
#
# DELIVERABLES (consumed prebuilt from --install-prefix, never built here):
#   decoding_server, hsb_fpga_syndrome_playback, the QEC + realtime libs,
#   and the decoder plugins.
# EXAMPLE BINARIES (the only things the user compiles; from --example-build-dir):
#   surface_code_realtime_decoding       generator: writes config + syndromes
#   surface_code_realtime_decoding-cqr   lowered kernel: the live syndrome source
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Defaults
# ============================================================================

SOURCE="qpu-kernel"          # qpu-kernel | fpga
DECODER="pymatching"         # pymatching | nv-qldpc-decoder | multi_error_lut

# Where the deliverables live (bin/ + lib/).  Required unless the per-artifact
# overrides below are all given.  No dev-tree fallback -- shipped code resolves
# from one install prefix.
INSTALL_PREFIX="${CUDAQX_INSTALL_DIR:-}"
# CUDA-Q install (its runtime libs + realtime libs are needed on the load path).
CUDAQ_PREFIX="${CUDA_QUANTUM_PATH:-/usr/local/cudaq}"
# Optional separate realtime lib dir (udp transport / dispatch), if not colocated.
REALTIME_LIB_DIR="${CUDAQ_REALTIME_DIR:-}"

# The example's own build directory (holds the two compiled binaries).
EXAMPLE_BUILD_DIR="${SCRIPT_DIR}/build"

# Per-artifact overrides (power users); empty => resolved from the prefixes.
SERVER_BIN=""
PLAYBACK_BIN=""
GENERATOR_BIN=""
KERNEL_BIN=""
NV_QLDPC_PLUGIN="${CUDAQ_QEC_NV_QLDPC_PLUGIN:-}"
# Ising artifact directory for the trt_decoder profile (six locally prepared
# files: model.onnx, H_csr.bin, O_csr.bin, priors.bin, metadata.txt,
# D_sparse.txt). Nothing in it ships with CUDA-QX; see the docs recipe.
ISING_ARTIFACTS_DIR="${QEC_ISING_ARTIFACTS_DIR:-}"

# Surface-code experiment parameters. The generator applies two-qubit
# depolarizing noise on the stabilizer-extraction CNOTs (p_cnot); the fixed
# simulator seed makes every run -- and therefore the pass/fail counts --
# reproducible (pass --seed -1 for unseeded runs).
GEN_DISTANCE=3
GEN_ROUNDS=4
GEN_P_CNOT=0.001
# SPAM (single-qubit depolarizing) rate for the Ising trt_decoder profile,
# whose noise model is SPAM rather than CNOT depolarizing; 0.01 is the
# published model's trained operating point.
GEN_P_SPAM=0.01
SEED=42
# Explicit-flag trackers: the trt_decoder profile pins d=7/T=7/SPAM noise and
# must reject conflicting explicit values rather than silently override them
# (and only re-defaults --spacing when the user did not set it).
DISTANCE_EXPLICIT=false
ROUNDS_EXPLICIT=false
PCNOT_EXPLICIT=false
SPACING_EXPLICIT=false

# Shot counts differ by mode: the FPGA plays back a fixed set through a 64-slot
# RDMA RX ring; the qpu-kernel self-paces via its blocking get_corrections and
# has no ring to overrun, so it runs more shots by default.  Both overridable.
FPGA_SHOTS=85
QPU_KERNEL_SHOTS=200
NUM_SHOTS=""                 # explicit override
# The FPGA playback BRAM holds 512 frames and per-round playback spends
# (syndrome slices + 1 get_corrections) frames per shot. The Ising profile's
# d7/T7 geometry has 8 slices/shot -> 9 frames/shot -> at most 56 shots per
# playback run (504/512), just as the d3/T4 default 85 fills 510/512.
TRT_FPGA_SHOTS=56

# FPGA pacing (us): the playback engine's timer fires once per FRAME (one
# BRAM window per frame), so a shot takes frames-per-shot x SPACING on the
# wire.  Pacing keeps playback from overrunning the server's 64-slot RX
# ring.  The qpu-kernel path needs none.
SPACING="10"

# The decoding server's two knobs (both default from --source/--decoder):
#   WIRE     which bridge-provider library carries syndromes into the server
#            (loaded at runtime as libcudaq-realtime-bridge-<name>.so)
#   DISPATCH which engine consumes each decoder's ring: host (a CPU dispatcher
#            thread) or device_graph (the GPU device-call scheduler)
# The combination gate below rejects pairings this example has not wired up.
WIRE=""                      # udp | cpu_roce | gpu_roce
DISPATCH=""                  # host | device_graph
GPU_ID=0

# Network (fpga source only).  The FPGA data path is same-/24 routed (no
# gateway), so the bridge NIC's default IP lives on the FPGA's subnet.
DO_SETUP_NETWORK=false
IB_DEVICE=""                 # auto-detect first Up ConnectX
BRIDGE_IP="192.168.0.1"
FPGA_IP="192.168.0.2"
MTU=4096
PAGE_SIZE=384
# RX ring depth for both FPGA paths, bounded by the HSB QP's 64 receive WQEs
# (the FPGA writes frame rid to slot rid % NUM_SLOTS; more slots than WQEs drops
# frames under load). The server clamps the cpu_roce wire's --num-slots to
# this; the device_graph ring is capped to it in run_fpga.
NUM_SLOTS=64
FRAME_SIZE=64
# Server lifetime failsafe (seconds). decoding_server's --timeout is a TOTAL
# runtime cap (not inactivity): the server exits once elapsed time exceeds it,
# even mid-run. 300 matches the in-tree surface_code-4 external-server tests;
# the default 60 would kill long runs (large --num-shots, slow decoders).
TIMEOUT=300
VERIFY=true

# ============================================================================
# Argument parsing
# ============================================================================

print_usage() {
    cat <<'EOF'
Usage: run_realtime_decoding.sh --source {qpu-kernel|fpga} [options]

Sources:
  --source qpu-kernel     Lowered kernel streams syndromes over udp (no NIC) or,
                          with --wire cpu_roce, over real RDMA (see below).
  --source fpga           Delivered playback streams from a real FPGA over RoCE.

Common:
  --decoder NAME          pymatching (default) | nv-qldpc-decoder |
                          multi_error_lut | trt_decoder (the Ising profile:
                          TensorRT NN predecoder + PyMatching global decoder;
                          pinned to d=7, rounds=7, SPAM noise 0.01, and
                          requires the Ising artifact directory below)
  --install-prefix DIR    Deliverables prefix (decoding_server, playback, libs,
                          plugins in DIR/bin and DIR/lib).  Required (or give the
                          per-artifact overrides).
  --cudaq-prefix DIR      CUDA-Q install (default: $CUDA_QUANTUM_PATH or
                          /usr/local/cudaq)
  --realtime-lib-dir DIR  Extra realtime lib dir (udp transport / dispatch)
  --example-build-dir DIR The example's build dir with the two binaries
                          (default: <script dir>/build)
  --nv-qldpc-plugin PATH  Prebuilt libcudaq-qec-nv-qldpc-decoder.so (or set
                          CUDAQ_QEC_NV_QLDPC_PLUGIN); required for nv-qldpc
  --ising-artifacts-dir D Prepared Ising artifact directory (model.onnx,
                          H_csr.bin, O_csr.bin, priors.bin, metadata.txt,
                          D_sparse.txt); required for trt_decoder (or set
                          QEC_ISING_ARTIFACTS_DIR).  Absent/incomplete => the
                          profile SKIPS (exit 77) listing what is missing.
  --distance N            Surface-code distance (default 3; trt_decoder pins 7)
  --num-rounds N          Measurement rounds (default 4; must be >= distance;
                          trt_decoder pins 7)
  --p-cnot F              Two-qubit depolarizing rate on the CNOTs (default
                          0.001; not applicable to trt_decoder)
  --p-spam F              SPAM depolarizing rate for trt_decoder (default 0.01,
                          the published Ising model's operating point)
  --seed N                Simulator seed; -1 = unseeded (default 42)
  --num-shots N           Shots (default: 85 fpga / 200 qpu-kernel)
  --gpu N                 GPU id for nv-qldpc (default 0)
  --wire W                Bridge provider carrying syndromes into the server:
                          udp | cpu_roce | gpu_roce (default: derived --
                          qpu-kernel -> udp; fpga -> cpu_roce for host
                          dispatch, gpu_roce for device_graph)
  --dispatch D            Per-decoder ring consumer: host | device_graph
                          (default: derived -- fpga + nv-qldpc-decoder ->
                          device_graph, everything else -> host)

Per-artifact overrides:
  --server PATH  --playback PATH  --generator PATH  --kernel PATH

Network:
  --setup-network         Configure the NIC port(s) first (needs sudo): the
                          FPGA-facing port for --source fpga, or the cpu_roce
                          channel + daemon pair for --source qpu-kernel
                          --wire cpu_roce
  --mtu N                 MTU (default 4096)

qpu-kernel + --wire cpu_roce only -- RDMA topology, via the same env vars as
the in-tree cpu_roce tests (kernel side = channel, server side = daemon; two
RoCE-capable ports, e.g. loopback-cabled):
  CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE   kernel-side IB device      (required)
  CUDAQ_CPU_ROCE_TEST_CHANNEL_IP       kernel-side RoCE IPv4      (10.0.0.1)
  CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE    server-side IB device      (required)
  CUDAQ_CPU_ROCE_TEST_DAEMON_IP        server-side RoCE IPv4      (10.0.0.2)

FPGA-only:
  --device DEV            ConnectX IB device (default: auto-detect)
  --bridge-ip ADDR        Server-side NIC IP (default 192.168.0.1; must share
                          the FPGA's /24)
  --fpga-ip ADDR          FPGA IP (default 192.168.0.2)
  --spacing US            Playback pacing in us (default 10). The FPGA timer
                          fires once per FRAME (one BRAM window), so a shot
                          spans frames-per-shot x spacing; keeps the RX ring
                          from overrunning
  --no-verify             Skip playback correction verification

  --help, -h              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)           SOURCE="$2"; shift ;;
        --decoder)          DECODER="$2"; shift ;;
        --install-prefix)   INSTALL_PREFIX="$2"; shift ;;
        --cudaq-prefix)     CUDAQ_PREFIX="$2"; shift ;;
        --realtime-lib-dir) REALTIME_LIB_DIR="$2"; shift ;;
        --example-build-dir) EXAMPLE_BUILD_DIR="$2"; shift ;;
        --seed)             SEED="$2"; shift ;;
        --nv-qldpc-plugin)  NV_QLDPC_PLUGIN="$2"; shift ;;
        --distance)         GEN_DISTANCE="$2"; DISTANCE_EXPLICIT=true; shift ;;
        --num-rounds)       GEN_ROUNDS="$2"; ROUNDS_EXPLICIT=true; shift ;;
        --p-cnot)           GEN_P_CNOT="$2"; PCNOT_EXPLICIT=true; shift ;;
        --p-spam)           GEN_P_SPAM="$2"; shift ;;
        --ising-artifacts-dir) ISING_ARTIFACTS_DIR="$2"; shift ;;
        --num-shots)        NUM_SHOTS="$2"; shift ;;
        --gpu)              GPU_ID="$2"; shift ;;
        --server)           SERVER_BIN="$2"; shift ;;
        --playback)         PLAYBACK_BIN="$2"; shift ;;
        --generator)        GENERATOR_BIN="$2"; shift ;;
        --kernel)           KERNEL_BIN="$2"; shift ;;
        --setup-network)    DO_SETUP_NETWORK=true ;;
        --device)           IB_DEVICE="$2"; shift ;;
        --bridge-ip)        BRIDGE_IP="$2"; shift ;;
        --fpga-ip)          FPGA_IP="$2"; shift ;;
        --mtu)              MTU="$2"; shift ;;
        --spacing)          SPACING="$2"; SPACING_EXPLICIT=true; shift ;;
        --wire)             WIRE="$2"; shift ;;
        --dispatch)         DISPATCH="$2"; shift ;;
        --no-verify)        VERIFY=false ;;
        --help|-h)          print_usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; print_usage >&2; exit 1 ;;
    esac
    shift
done

if [[ "$SOURCE" != "qpu-kernel" && "$SOURCE" != "fpga" ]]; then
    echo "ERROR: --source must be qpu-kernel or fpga (got '$SOURCE')" >&2; exit 1
fi

# The bridge NIC and the FPGA must share a /24: there is no gateway on the
# FPGA link, and --setup-network flushes the NIC and assigns BRIDGE_IP/24, so
# a mismatched pair leaves the FPGA unreachable -- the run then dies minutes
# later as an opaque HSB read timeout. Reject it up front.
if [[ "$SOURCE" == "fpga" && "${BRIDGE_IP%.*}" != "${FPGA_IP%.*}" ]]; then
    echo "ERROR: --bridge-ip $BRIDGE_IP and --fpga-ip $FPGA_IP are on different" >&2
    echo "       /24 subnets; the FPGA data path is same-subnet routed." >&2
    exit 1
fi

# The example source also supports 'sliding_window', but that decoder is not
# yet supported over the realtime decoding path to the GPU (it needs matching
# updates to the proprietary cudevice archive), so this script does not accept
# it.
case "$DECODER" in
    pymatching|nv-qldpc-decoder|multi_error_lut|trt_decoder) ;;
    sliding_window)
       echo "ERROR: sliding_window is not yet supported over the realtime" >&2
       echo "       decoding path to the GPU." >&2; exit 1 ;;
    *) echo "ERROR: --decoder must be pymatching, nv-qldpc-decoder," >&2
       echo "       multi_error_lut, or trt_decoder (got '$DECODER')" >&2; exit 1 ;;
esac

# ---------------------------------------------------------------------------
# The Ising profile (trt_decoder = TensorRT NN predecoder + PyMatching global
# decoder) is pinned to the published model's operating point: d=7, rounds=7,
# basis Z, orientation XV, SPAM noise (--p-spam, default 0.01). Conflicting
# explicit knobs are rejected rather than silently overridden. On the FPGA the
# playback BRAM caps it at TRT_FPGA_SHOTS shots per run (see above).
# ---------------------------------------------------------------------------
if [[ "$DECODER" == "trt_decoder" ]]; then
    if { $DISTANCE_EXPLICIT && [[ "$GEN_DISTANCE" != 7 ]]; } || \
       { $ROUNDS_EXPLICIT && [[ "$GEN_ROUNDS" != 7 ]]; }; then
        echo "ERROR: the trt_decoder (Ising) profile supports distance=7," >&2
        echo "       num-rounds=7 only (the published model's operating point)." >&2
        exit 1
    fi
    if $PCNOT_EXPLICIT; then
        echo "ERROR: trt_decoder uses SPAM noise (--p-spam), not --p-cnot." >&2
        exit 1
    fi
    GEN_DISTANCE=7
    GEN_ROUNDS=7
    FPGA_SHOTS="$TRT_FPGA_SHOTS"
    # d7/T7 sends 9 frames per shot (each frame paced by SPACING) into the
    # server's 64-slot RX ring (~7 shots of buffer). At the 10us default the
    # stream outruns the served rate and the ring overruns around shot 8-18
    # (a corrupted stream then kills the decode with an invalid-syndrome
    # matching error). 100us per frame is validated on hardware.
    $SPACING_EXPLICIT || SPACING=100
    if [[ "$SOURCE" == "fpga" && -n "$NUM_SHOTS" ]] && \
       (( NUM_SHOTS > TRT_FPGA_SHOTS )); then
        echo "ERROR: the FPGA playback BRAM holds 512 frames; at d7/T7" >&2
        echo "       (9 frames/shot) at most $TRT_FPGA_SHOTS shots fit per run" >&2
        echo "       (got --num-shots $NUM_SHOTS)." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Derive the wire + dispatch defaults, then gate on the combinations this
# example has wired up and verified:
#   qpu-kernel : udp wire,      host dispatch  (any decoder incl. trt_decoder)
#   qpu-kernel : cpu_roce wire, host dispatch  (any decoder incl. trt_decoder)
#   fpga       : cpu_roce wire, host dispatch  (any decoder incl. trt_decoder)
#   fpga       : gpu_roce wire, device_graph dispatch  (nv-qldpc-decoder)
# Everything else is a real configuration of the decoding server that this
# example does not (yet) exercise, so it is rejected with the reason.
# ---------------------------------------------------------------------------
if [[ -z "$DISPATCH" ]]; then
    if [[ "$SOURCE" == "fpga" && "$DECODER" == "nv-qldpc-decoder" ]]; then
        DISPATCH="device_graph"
    else
        DISPATCH="host"
    fi
fi
case "$DISPATCH" in host|device_graph) ;;
    *) echo "ERROR: --dispatch must be host or device_graph (got '$DISPATCH')" >&2; exit 1 ;;
esac

# nv-qldpc's GPU decode (plus first-call PTX JIT on new hardware) cannot
# drain the server's 64-slot RX ring at the default 10us-per-frame playback
# pacing -- measured on-rig: 171/510 frames captured (host dispatch) and
# ~15/85 shots verified (device_graph) before the ring overruns. 100us per
# frame clears the measured floors on both dispatch paths; an explicit
# --spacing wins.
if [[ "$SOURCE" == "fpga" && "$DECODER" == "nv-qldpc-decoder" ]] && \
   ! $SPACING_EXPLICIT; then
    SPACING=100
fi

[[ "$WIRE" == "cpu-roce" ]] && WIRE="cpu_roce"   # accept both spellings
[[ "$WIRE" == "gpu-roce" ]] && WIRE="gpu_roce"
if [[ -z "$WIRE" ]]; then
    if [[ "$SOURCE" == "qpu-kernel" ]]; then WIRE="udp"
    elif [[ "$DISPATCH" == "device_graph" ]]; then WIRE="gpu_roce"
    else WIRE="cpu_roce"; fi
fi
case "$WIRE" in udp|cpu_roce|gpu_roce) ;;
    *) echo "ERROR: --wire must be udp, cpu_roce, or gpu_roce (got '$WIRE')" >&2; exit 1 ;;
esac

if [[ "$SOURCE" == "qpu-kernel" ]]; then
    if [[ "$WIRE" == "gpu_roce" ]]; then
        echo "ERROR: the qpu-kernel source is not wired to the gpu_roce wire in" >&2
        echo "       this example (use --wire udp or --wire cpu_roce)." >&2; exit 1
    fi
    if [[ "$DISPATCH" != "host" ]]; then
        echo "ERROR: device_graph dispatch on the qpu-kernel source (pinned shared" >&2
        echo "       rings) is not exposed by this example." >&2; exit 1
    fi
else
    if [[ "$DISPATCH" == "device_graph" ]]; then
        if [[ "$DECODER" != "nv-qldpc-decoder" ]]; then
            echo "ERROR: device_graph dispatch serves only nv-qldpc-decoder" >&2
            echo "       (got '$DECODER'); CPU decoders use --dispatch host." >&2; exit 1
        fi
        if [[ "$WIRE" != "gpu_roce" ]]; then
            echo "ERROR: device_graph dispatch on the FPGA requires the gpu_roce" >&2
            echo "       wire (got '$WIRE')." >&2; exit 1
        fi
    else
        if [[ "$WIRE" != "cpu_roce" ]]; then
            echo "ERROR: host dispatch on the FPGA requires the cpu_roce wire" >&2
            echo "       (got '$WIRE')." >&2; exit 1
        fi
    fi
fi

# Provider names use the YAML spelling; the server maps underscores to the
# installed providers' hyphenated sonames.
WIRE_TOKEN="$WIRE"

# ---------------------------------------------------------------------------
# qpu-kernel over cpu_roce: RDMA topology.  Same four-env-var convention as
# every in-tree cpu_roce test (CUDA-Q's CpuRoceChannelTester and the QEC
# two-process test): the lowered kernel (channel) and the server (daemon) each
# own one RoCE-capable port, cabled to the other.  The kernel binary reads the
# CHANNEL_* values itself; the DAEMON_* values feed the server's
# --device/--local-ip and double as the kernel's rendezvous target.
# ---------------------------------------------------------------------------
CHANNEL_DEVICE=""; CHANNEL_IP=""; DAEMON_DEVICE=""; DAEMON_IP=""
if [[ "$SOURCE" == "qpu-kernel" && "$WIRE" == "cpu_roce" ]]; then
    CHANNEL_DEVICE="${CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE:-}"
    CHANNEL_IP="${CUDAQ_CPU_ROCE_TEST_CHANNEL_IP:-10.0.0.1}"
    DAEMON_DEVICE="${CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE:-}"
    DAEMON_IP="${CUDAQ_CPU_ROCE_TEST_DAEMON_IP:-10.0.0.2}"
    if [[ -z "$CHANNEL_DEVICE" || -z "$DAEMON_DEVICE" ]]; then
        echo "ERROR: --wire cpu_roce needs the RDMA topology: set" >&2
        echo "       CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE (kernel-side IB device) and" >&2
        echo "       CUDAQ_CPU_ROCE_TEST_DAEMON_DEVICE (server-side IB device)." >&2
        echo "       IPs default to 10.0.0.1 / 10.0.0.2 (override with" >&2
        echo "       CUDAQ_CPU_ROCE_TEST_CHANNEL_IP / _DAEMON_IP)." >&2
        echo "       Available IB devices: $(ls /sys/class/infiniband 2>/dev/null | tr '\n' ' ')" >&2
        exit 1
    fi
fi

_log()  { echo "==> $*"; }
_info() { echo "    $*"; }
_warn() { echo "WARNING: $*" >&2; }
_err()  { echo "ERROR: $*" >&2; }
_banner() { echo; echo "========================================";
            echo "  $*"; echo "========================================"; echo; }

# ============================================================================
# Path resolution (install-prefix only; per-artifact overrides win)
# ============================================================================

resolve_paths() {
    # Canonicalize user-supplied paths first: generate_data runs the generator
    # from a temp working dir, and library paths are exported to subprocesses,
    # so a relative path that validates here would break there. Only rewrite
    # values that resolve to an existing path -- missing paths stay as-is for
    # the normal validation below.
    local _v _abs
    for _v in INSTALL_PREFIX CUDAQ_PREFIX REALTIME_LIB_DIR EXAMPLE_BUILD_DIR \
              SERVER_BIN PLAYBACK_BIN GENERATOR_BIN KERNEL_BIN NV_QLDPC_PLUGIN; do
        if [[ -n "${!_v}" ]]; then
            if _abs=$(readlink -f -- "${!_v}" 2>/dev/null) && [[ -e "$_abs" ]]; then
                printf -v "$_v" '%s' "$_abs"
            fi
        fi
    done

    if [[ -z "$SERVER_BIN"    && -n "$INSTALL_PREFIX" ]]; then SERVER_BIN="$INSTALL_PREFIX/bin/decoding_server"; fi
    if [[ -z "$PLAYBACK_BIN"  && -n "$INSTALL_PREFIX" ]]; then PLAYBACK_BIN="$INSTALL_PREFIX/bin/hsb_fpga_syndrome_playback"; fi
    # The Ising profile runs its own example pair (the surface_code-4-lineage
    # source the artifacts are bound to); everything else runs the original.
    local _example_stem="surface_code_realtime_decoding"
    [[ "$DECODER" == "trt_decoder" ]] && _example_stem="surface_code_ising_realtime_decoding"
    if [[ -z "$GENERATOR_BIN" ]]; then GENERATOR_BIN="$EXAMPLE_BUILD_DIR/${_example_stem}"; fi
    if [[ -z "$KERNEL_BIN"    ]]; then KERNEL_BIN="$EXAMPLE_BUILD_DIR/${_example_stem}-cqr"; fi

    if [[ -z "$SERVER_BIN" ]]; then
        _err "No decoding_server: pass --install-prefix DIR (or --server PATH)."; exit 1
    fi
    if [[ ! -x "$SERVER_BIN" ]]; then _err "decoding_server not found/executable: $SERVER_BIN"; exit 1; fi
    if [[ ! -x "$GENERATOR_BIN" ]]; then
        _err "generator not found: $GENERATOR_BIN"
        _err "Build the example first (cmake --build <dir>) or pass --generator PATH."; exit 1
    fi
    if [[ "$SOURCE" == "qpu-kernel" && ! -x "$KERNEL_BIN" ]]; then
        _err "lowered kernel not found: $KERNEL_BIN (build the example, or --kernel PATH)"; exit 1
    fi
    if [[ "$SOURCE" == "fpga" && ! -x "$PLAYBACK_BIN" ]]; then
        _err "playback tool not found: $PLAYBACK_BIN (an HSB-enabled deliverable)"; exit 1
    fi

    # Load path: deliverable libs + plugins, plus the CUDA-Q runtime/realtime.
    # The explicit realtime override goes FIRST: a CUDA-Q prefix may carry its
    # own libcudaq-realtime.so, and appending the override after it would let
    # the prefix's copy shadow the one the user asked for (symptom: undefined
    # symbol errors against the newer realtime ABI).
    local p=""
    [[ -n "$REALTIME_LIB_DIR" ]] && p="$REALTIME_LIB_DIR/lib:$REALTIME_LIB_DIR"
    [[ -n "$INSTALL_PREFIX"   ]] && p="$p:$INSTALL_PREFIX/lib:$INSTALL_PREFIX/lib/decoder-plugins"
    [[ -n "$CUDAQ_PREFIX"     ]] && p="$p:$CUDAQ_PREFIX/lib:$CUDAQ_PREFIX/lib/plugins"
    p="${p#:}"
    export LD_LIBRARY_PATH="${p}:${LD_LIBRARY_PATH:-}"

    # nv-qldpc plugin must be dlopen-able from the deliverable plugins dir.
    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        if [[ -z "$NV_QLDPC_PLUGIN" && -n "$INSTALL_PREFIX" \
              && -f "$INSTALL_PREFIX/lib/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so" ]]; then
            NV_QLDPC_PLUGIN="$INSTALL_PREFIX/lib/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so"
        fi
        if [[ -z "$NV_QLDPC_PLUGIN" || ! -f "$NV_QLDPC_PLUGIN" ]]; then
            _err "nv-qldpc-decoder plugin not available (pass --nv-qldpc-plugin or set"
            _err "CUDAQ_QEC_NV_QLDPC_PLUGIN).  Skipping this decoder."
            exit 77   # standard "skip" code
        fi
        local link="$INSTALL_PREFIX/lib/decoder-plugins/$(basename "$NV_QLDPC_PLUGIN")"
        if [[ -n "$INSTALL_PREFIX" && ! -e "$link" ]]; then
            # Fail fast: the server discovers plugins in this directory, so a
            # silently failed link surfaces minutes later as an opaque
            # "server did not construct an 'nv-qldpc-decoder' session".
            if ! mkdir -p "$INSTALL_PREFIX/lib/decoder-plugins" 2>/dev/null \
               || ! ln -sf "$NV_QLDPC_PLUGIN" "$link" 2>/dev/null; then
                _err "cannot place the nv-qldpc plugin in $INSTALL_PREFIX/lib/decoder-plugins"
                _err "(is the install prefix writable?)"; exit 1
            fi
        fi
    fi

    # trt_decoder needs (a) the SDK's TensorRT decoder plugin (shipped only
    # when TensorRT was available at the SDK build) and (b) the locally
    # prepared Ising artifact directory. Both absences are SKIPs, not
    # failures, mirroring the nv-qldpc plugin policy.
    if [[ "$DECODER" == "trt_decoder" ]]; then
        if [[ -z "$INSTALL_PREFIX" \
              || ! -f "$INSTALL_PREFIX/lib/decoder-plugins/libcudaq-qec-trt-decoder.so" ]]; then
            _err "TensorRT decoder plugin not found:"
            _err "  ${INSTALL_PREFIX:-<install-prefix>}/lib/decoder-plugins/libcudaq-qec-trt-decoder.so"
            _err "Skipping the trt_decoder profile."
            exit 77
        fi
        validate_ising_artifacts
    fi
}

# The Ising artifact directory holds six locally prepared files (gated Hugging
# Face weights + the NVIDIA/Ising-Decoding repository -- see the docs for the
# recipe). Missing or empty files are listed BY NAME; an unprepared setup is a
# SKIP (77), not a failure. metadata.txt's content (d7/T7/Z/XV) is enforced by
# the generator binary itself.
validate_ising_artifacts() {
    if [[ -z "$ISING_ARTIFACTS_DIR" ]]; then
        _err "trt_decoder needs the Ising artifact directory: pass"
        _err "--ising-artifacts-dir DIR or set QEC_ISING_ARTIFACTS_DIR."
        _err "See the example documentation for the preparation recipe."
        _err "Skipping the trt_decoder profile."
        exit 77
    fi
    local f missing=""
    for f in model.onnx H_csr.bin O_csr.bin priors.bin metadata.txt D_sparse.txt; do
        [[ -s "$ISING_ARTIFACTS_DIR/$f" ]] || missing="$missing $f"
    done
    if [[ -n "$missing" ]]; then
        _err "Ising artifact directory '$ISING_ARTIFACTS_DIR' is incomplete;"
        _err "missing or empty:$missing"
        _err "See the example documentation for the preparation recipe."
        _err "Skipping the trt_decoder profile."
        exit 77
    fi
    # Canonicalize: the generator embeds this path (model.onnx) into the
    # config, which the server later reads from a different working directory.
    ISING_ARTIFACTS_DIR="$(readlink -f -- "$ISING_ARTIFACTS_DIR")"
}

# ============================================================================
# Config + syndrome generation (uses the example's generator binary)
# ============================================================================

GEN_DIR=""
CONFIG_FILE=""
SYNDROMES_FILE=""

generate_data() {
    GEN_DIR="$(mktemp -d /tmp/realtime_decoding_demo.XXXXXX)"
    CONFIG_FILE="$GEN_DIR/config_${DECODER}.yml"
    SYNDROMES_FILE="$GEN_DIR/syndromes_${DECODER}.txt"

    # The Ising profile's app takes SPAM noise and the artifact directory;
    # the original app takes CNOT-depolarizing noise.
    local gen_noise_args=(--p_cnot "$GEN_P_CNOT")
    if [[ "$DECODER" == "trt_decoder" ]]; then
        gen_noise_args=(--p_spam "$GEN_P_SPAM"
                        --ising-artifacts-dir "$ISING_ARTIFACTS_DIR")
    fi

    _log "Generating decoder config (decoder=$DECODER, d=$GEN_DISTANCE, rounds=$GEN_ROUNDS)"
    ( cd "$GEN_DIR" && "$GENERATOR_BIN" \
        --distance "$GEN_DISTANCE" --num_rounds "$GEN_ROUNDS" "${gen_noise_args[@]}" \
        --decoder_type "$DECODER" \
        --save_dem "$CONFIG_FILE" ) >"$GEN_DIR/gen_config.log" 2>&1 || {
        _err "config generation failed; see $GEN_DIR/gen_config.log"; tail -5 "$GEN_DIR/gen_config.log" >&2; exit 1; }

    # nv-qldpc runs on a GPU: pin the decoder to --gpu via cuda_device_id (the
    # host worker / device-graph scheduler both honor it). device_graph
    # dispatch additionally needs the per-decoder `dispatch:` key. The
    # generator omits these optional fields, so inject them under the
    # decoder's `type:` line. pymatching/multi_error_lut need neither.
    if [[ "$DECODER" == "nv-qldpc-decoder" ]]; then
        local inject_dispatch=""
        if [[ "$DISPATCH" == "device_graph" ]]; then
            inject_dispatch="    dispatch:        device_graph"
        fi
        awk -v gpu="$GPU_ID" -v dp="$inject_dispatch" '{ print }
             /^[[:space:]]*type:/ && !d { if (dp != "") print dp;
                                          print "    cuda_device_id:  " gpu; d=1 }' \
            "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
        # Verify the injection landed: a YAML formatting change in the
        # generator would otherwise silently drop --gpu / the dispatch
        # override (pinning nv-qldpc to GPU 0, or leaving the decoder on host
        # dispatch).
        if ! grep -qE "^[[:space:]]*cuda_device_id:[[:space:]]*${GPU_ID}\$" "$CONFIG_FILE"; then
            _err "config injection failed: cuda_device_id not found in $CONFIG_FILE"; exit 1
        fi
        if [[ -n "$inject_dispatch" ]] && \
           ! grep -qE "^[[:space:]]*dispatch:[[:space:]]*device_graph\$" "$CONFIG_FILE"; then
            _err "config injection failed: dispatch not found in $CONFIG_FILE"; exit 1
        fi
    fi

    # The trt_decoder entry must carry the Ising model: a config with the
    # right type but no onnx_load_path would silently construct a bare
    # matching session on the server instead of the NN predecoder.
    if [[ "$DECODER" == "trt_decoder" ]]; then
        if ! grep -qE "type:[[:space:]]+trt_decoder" "$CONFIG_FILE" || \
           ! grep -q "onnx_load_path" "$CONFIG_FILE"; then
            _err "generated config lacks a trt_decoder entry with onnx_load_path: $CONFIG_FILE"
            exit 1
        fi
    fi

    # The FPGA source replays a pre-generated syndrome file; the qpu-kernel
    # source generates syndromes live inside the kernel, so it needs none.
    # For trt_decoder this run constructs the TensorRT decoder in-process
    # (engine build from model.onnx) to compute the expected corrections the
    # playback tool verifies per shot.
    if [[ "$SOURCE" == "fpga" ]]; then
        local shots="${NUM_SHOTS:-$FPGA_SHOTS}"
        local synd_noise_args=(--p_cnot "$GEN_P_CNOT")
        [[ "$DECODER" == "trt_decoder" ]] && synd_noise_args=(--p_spam "$GEN_P_SPAM")
        _log "Generating $shots syndromes for playback"
        ( cd "$GEN_DIR" && "$GENERATOR_BIN" \
            --distance "$GEN_DISTANCE" --num_rounds "$GEN_ROUNDS" "${synd_noise_args[@]}" \
            --seed "$SEED" --num_shots "$shots" --load_dem "$CONFIG_FILE" \
            --save_syndrome "$SYNDROMES_FILE" ) >"$GEN_DIR/gen_syndromes.log" 2>&1 || {
            _err "syndrome generation failed; see $GEN_DIR/gen_syndromes.log"; tail -5 "$GEN_DIR/gen_syndromes.log" >&2; exit 1; }
    fi
    _info "config:    $CONFIG_FILE"
    if [[ "$SOURCE" == "fpga" ]]; then _info "syndromes: $SYNDROMES_FILE"; fi
}

# ============================================================================
# Cleanup
# ============================================================================
PIDS=()
cleanup() {
    for pid in "${PIDS[@]:-}"; do
        # `|| true` on the TERM: a pid reaped between the kill -0 and the kill
        # -TERM would otherwise abort this EXIT trap under set -e AND override
        # the script's exit status (a PASS run would exit nonzero).
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && { kill -TERM "$pid" 2>/dev/null || true; sleep 0.5;
            kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true; }
    done
    [[ -n "$GEN_DIR" && -d "$GEN_DIR" ]] && rm -rf "$GEN_DIR" || true
}
trap cleanup EXIT

wait_for_pattern() {
    local logfile="$1" pattern="$2" timeout_sec="$3" pid="${4:-}"
    local waited=0
    while (( waited < timeout_sec * 10 )); do
        [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null && { _err "process $pid died"; return 1; }
        local m; m=$(grep -m1 "$pattern" "$logfile" 2>/dev/null || true)
        [[ -n "$m" ]] && { echo "$m"; return 0; }
        sleep 0.1; waited=$((waited + 1))
    done
    _err "timeout waiting for: $pattern"; return 1
}

# ============================================================================
# qpu-kernel pass/fail criteria -- the same checks the in-tree surface-code
# ctest drivers apply (surface_code-1-cqr-two-process-test.sh and
# surface_code-4-yaml-test.sh), so this example fails on the same evidence as
# the existing tests: hard decoder-failure messages, the corrections-line
# completion proof, a residual logical-error ceiling of num_shots/50 (min 1),
# and external-server evidence (the app's in-process dispatch count must be 0
# + a server dispatch-count floor).
# ============================================================================
verify_qpu_kernel_output() {
    local kernel_log="$1" server_log="$2" shots="$3"
    local rc=0

    _log "Checking realtime output (surface_code-4 test criteria)"

    # A non-graphlike DEM handed to pymatching surfaces as "Invalid column in H".
    if grep -q "Invalid column in H" "$kernel_log"; then
        _err "found 'Invalid column in H' (decoder received a non-graphlike DEM)"; rc=1
    fi
    # Hard decoder-init / dispatch failures.
    if grep -q "terminate called" "$kernel_log"; then
        _err "found 'terminate called' (the app aborted)"; rc=1
    fi
    if grep -q "Decoder 0 not found" "$kernel_log"; then
        _err "found 'Decoder 0 not found' (decoder was not registered)"; rc=1
    fi
    if grep -q "Error initializing decoders" "$kernel_log"; then
        _err "found 'Error initializing decoders'"; rc=1
    fi

    # This line proves the realtime decoding path actually ran to completion.
    if ! grep -q "Number of corrections decoder found:" "$kernel_log"; then
        _err "missing 'Number of corrections decoder found:' line (decoding did not complete)"; rc=1
    fi

    # Residual logical errors must stay under the ceiling: a decoder that is
    # wired up but decoding wrong produces ~random logicals, far above it.
    local max_non_zero=$((shots / 50))
    (( max_non_zero < 1 )) && max_non_zero=1
    local non_zero
    non_zero=$(grep "Number of non-zero values measured :" "$kernel_log" \
        | awk -F': ' '{print $2}' | tr -d '[:space:]')
    if ! [[ "$non_zero" =~ ^[0-9]+$ ]]; then
        _err "'Number of non-zero values measured' is not a number (got '$non_zero')"; rc=1
    elif (( non_zero > max_non_zero )); then
        _err "residual logical errors ($non_zero) exceed ceiling ($max_non_zero) -- decoder appears wired-but-wrong"; rc=1
    else
        _info "residual logical errors: $non_zero (ceiling $max_non_zero) -- OK"
    fi

    # External-server evidence: the app's in-process decoding service must
    # have dispatched NOTHING (decode did not silently stay local), and the
    # server must have dispatched at least shots * (rounds + 3) RPCs --
    # reset + (rounds+1) enqueues + get_corrections per shot.
    local inproc
    inproc=$(grep "CQR service dispatch count:" "$kernel_log" | awk -F': ' '{print $2}' | tr -d '[:space:]')
    if [[ "$inproc" != "0" ]]; then
        _err "in-process CQR dispatch count is '${inproc:-<missing>}' (expected 0); decode did not stay in the server"; rc=1
    fi
    local dispatches min_dispatches=$((shots * (GEN_ROUNDS + 3)))
    dispatches=$(sed -n 's/^QEC_DECODING_SERVER_DISPATCHED count=\([0-9][0-9]*\)$/\1/p' \
        "$server_log" | tail -n1)
    if ! [[ "$dispatches" =~ ^[0-9]+$ ]] || (( dispatches < min_dispatches )); then
        _err "server dispatch count '${dispatches:-<missing>}' is below $min_dispatches"; rc=1
    else
        _info "server evidence: dispatches=$dispatches (>= $min_dispatches) -- OK"
    fi

    # Per-decoder ring evidence (one ring per decoder in the YAML): decoder 0's
    # ring must have carried traffic.
    local ring0
    ring0=$(sed -n 's/^QEC_DECODING_SERVER_RING decoder=0 dispatched=\([0-9][0-9]*\).*/\1/p' \
        "$server_log" | tail -n1)
    if ! [[ "$ring0" =~ ^[0-9]+$ ]] || (( ring0 == 0 )); then
        _err "ring 0 dispatch count '${ring0:-<missing>}' (expected > 0)"; rc=1
    else
        _info "ring evidence: decoder 0 dispatched=$ring0 -- OK"
    fi

    return $rc
}

# ============================================================================
# qpu-kernel source: lowered kernel drives the server over UDP
# ============================================================================
run_qpu_kernel() {
    _banner "Realtime decoding: qpu-kernel source (wire=$WIRE), decoder=$DECODER"

    local server_log; server_log="$GEN_DIR/server.log"
    _log "Starting decoding_server on the $WIRE wire"
    # The GPU for nv-qldpc is selected via cuda_device_id in the config (injected
    # in generate_data), not a server flag.
    local server_args=(--config="$CONFIG_FILE" --transport="$WIRE_TOKEN" --port=0
                       --timeout="$TIMEOUT")
    if [[ "$WIRE" == "cpu_roce" ]]; then
        # The RDMA ring geometry (slots x slot-size) is part of the cpu_roce
        # wire contract -- the lowered kernel's channel writes straight into
        # the server's rings and hardcodes 8 x 256 -- so pin the server to it
        # explicitly rather than relying on its defaults.
        server_args+=(--device="$DAEMON_DEVICE" --local-ip="$DAEMON_IP"
                      --num-slots=8 --slot-size=256)
    fi
    "$SERVER_BIN" "${server_args[@]}" >"$server_log" 2>&1 &
    local spid=$!; PIDS+=("$spid")
    # trt_decoder builds its TensorRT engine from model.onnx while the server
    # constructs decoders, delaying READY; give it a longer allowance.
    local ready_wait=60
    [[ "$DECODER" == "trt_decoder" ]] && ready_wait=180
    wait_for_pattern "$server_log" "QEC_DECODING_SERVER_READY" "$ready_wait" "$spid" >/dev/null || {
        cat "$server_log" >&2; return 1; }
    # udp: the UDP data port; cpu_roce: the TCP rendezvous port (the RDMA wire
    # itself is negotiated via the QP/rkey exchange).
    local port
    port=$(grep -m1 "QEC_DECODING_SERVER_READY" "$server_log" \
           | grep -oE 'port=[0-9]+' | head -1 | cut -d= -f2)
    _info "server ready on the $WIRE wire, port $port (pid $spid)"

    local shots="${NUM_SHOTS:-$QPU_KERNEL_SHOTS}"
    local kernel_log; kernel_log="$GEN_DIR/kernel.log"
    _log "Running lowered kernel: $shots shots"
    # QEC_DECODING_SERVER_PORT routes every decoding device_call to the
    # external server (the app brings the channel up itself in its
    # QEC_APP_CQR build).  For cpu_roce the app additionally reads the
    # transport selector and the channel-side RDMA topology from the env.
    local kernel_env=(QEC_DECODING_SERVER_PORT="$port")
    if [[ "$WIRE" == "cpu_roce" ]]; then
        kernel_env+=(QEC_DECODING_SERVER_TRANSPORT=cpu_roce
                     CUDAQ_CPU_ROCE_TEST_CHANNEL_DEVICE="$CHANNEL_DEVICE"
                     CUDAQ_CPU_ROCE_TEST_CHANNEL_IP="$CHANNEL_IP"
                     CUDAQ_CPU_ROCE_TEST_DAEMON_IP="$DAEMON_IP")
    fi
    local kernel_noise_args=(--p_cnot "$GEN_P_CNOT")
    [[ "$DECODER" == "trt_decoder" ]] && kernel_noise_args=(--p_spam "$GEN_P_SPAM")
    local rc=0
    env "${kernel_env[@]}" \
        "$KERNEL_BIN" --load_dem "$CONFIG_FILE" \
        --distance "$GEN_DISTANCE" --num_rounds "$GEN_ROUNDS" \
        "${kernel_noise_args[@]}" --seed "$SEED" \
        --num_shots "$shots" 2>&1 | tee "$kernel_log" || rc=$?
    # Stop the server and wait for it to exit so its shutdown stats
    # (QEC_DECODING_SERVER_DISPATCHED ...) are flushed to the log.
    kill -TERM "$spid" 2>/dev/null || true
    wait "$spid" 2>/dev/null || true
    if (( rc != 0 )); then
        _err "realtime phase exited with non-zero status ($rc)"
    fi
    local vrc=0
    verify_qpu_kernel_output "$kernel_log" "$server_log" "$shots" || vrc=$?
    (( rc != 0 )) && return $rc
    return $vrc
}

# ============================================================================
# fpga source: delivered playback streams syndromes over RoCE from a real FPGA
# (network helpers ported from the unittests hsb_fpga_decoding_server_test.sh)
# ============================================================================

ib_to_netdev() { ibdev2netdev | awk -v d="$1" -v p="${2:-1}" '$1==d && $3==p {print $5}'; }
netdev_to_ib() { ibdev2netdev | awk -v i="$1" '$5==i {print $1}'; }

ipv4_to_gid_suffix() {
    local o1 o2 o3 o4; IFS='.' read -r o1 o2 o3 o4 <<< "$1"
    printf "ffff:%02x%02x:%02x%02x" "$o1" "$o2" "$o3" "$o4"
}
wait_for_roce_v2_gid() {
    local ib_dev="$1" ip="$2" timeout_s="${3:-15}" suffix gids types elapsed=0
    suffix=$(ipv4_to_gid_suffix "$ip")
    gids="/sys/class/infiniband/${ib_dev}/ports/1/gids"
    types="/sys/class/infiniband/${ib_dev}/ports/1/gid_attrs/types"
    [[ -d "$gids" ]] || return 0
    while (( elapsed < timeout_s * 10 )); do
        local g idx gid t
        for g in "$gids"/*; do
            idx=$(basename "$g"); gid=$(cat "$g" 2>/dev/null)
            if [[ "$gid" == *":${suffix}" ]]; then
                t=$(cat "${types}/${idx}" 2>/dev/null)
                [[ "$t" == *"RoCE v2"* ]] && { _info "RoCE v2 GID ready: ${ib_dev}[${idx}]"; return 0; }
            fi
        done
        sleep 0.1; elapsed=$((elapsed + 1))
    done
    _err "timed out waiting for RoCE v2 GID (${suffix}) on ${ib_dev}"; return 1
}
setup_port() {
    local iface="$1" ip="$2" mtu="$3" ib_dev
    _info "Configuring $iface: ip=$ip mtu=$mtu"
    sudo ip link set "$iface" up
    sudo ip link set "$iface" mtu "$mtu"
    sudo ip addr flush dev "$iface"
    sudo ip addr add "${ip}/24" dev "$iface"
    ib_dev=$(netdev_to_ib "$iface")
    if [[ -n "$ib_dev" ]] && command -v rdma &>/dev/null; then
        local pc; pc=$(ls -d "/sys/class/infiniband/${ib_dev}/ports/"* 2>/dev/null | wc -l)
        for p in $(seq 1 "$pc"); do sudo rdma link set "${ib_dev}/${p}" type eth || true; done
    fi
    command -v mlnx_qos &>/dev/null && sudo mlnx_qos -i "$iface" --trust=dscp 2>/dev/null || true
    command -v ethtool  &>/dev/null && sudo ethtool -C "$iface" adaptive-rx off rx-usecs 0 2>/dev/null || true
}
seed_fpga_neighbor() {
    local iface="$1" fpga_ip="$2" mac
    ping -c 3 -W 1 -I "$iface" "$fpga_ip" >/dev/null 2>&1 || true
    mac=$(ip neigh show "$fpga_ip" dev "$iface" 2>/dev/null \
          | awk '{for(i=1;i<=NF;i++) if($i=="lladdr") print $(i+1)}' | head -1)
    if [[ -n "$mac" ]]; then
        sudo ip neigh replace "$fpga_ip" lladdr "$mac" nud permanent dev "$iface"
        _info "static ARP: $fpga_ip -> $mac on $iface"
    else
        _err "could not resolve FPGA MAC for $fpga_ip on $iface (cabled? powered? ping $fpga_ip)"
    fi
}

setup_network() {
    _log "Setting up ConnectX network for the FPGA"
    local iface
    if [[ -n "$IB_DEVICE" ]]; then iface=$(ib_to_netdev "$IB_DEVICE" 1)
    else iface=$(ibdev2netdev | awk '/\(Up\)/{print $5; exit}'); fi
    [[ -n "$iface" ]] || { _err "cannot detect a ConnectX interface"; return 1; }
    _info "server interface: $iface"
    setup_port "$iface" "$BRIDGE_IP" "$MTU"
    BRIDGE_DEVICE=$(netdev_to_ib "$iface")
    wait_for_roce_v2_gid "$BRIDGE_DEVICE" "$BRIDGE_IP" 15 || true
    seed_fpga_neighbor "$iface" "$FPGA_IP"
}

# qpu-kernel/cpu_roce network setup: give the channel and daemon ports their
# RoCE IPv4s and wait for the matching RoCE v2 GIDs (the kernel populates a
# GID asynchronously after `ip addr add`, and the transceiver's GID lookup
# loses that race, notably after a cold boot).  Unlike the FPGA path there is
# no peer to ARP-seed -- the RDMA endpoints cross the TCP rendezvous -- but a
# stale copy of either IP on another interface would confuse routing, so
# sweep it off first.
_setup_roce_port() {
    local role="$1" ib_dev="$2" ip="$3" iface other
    iface=$(ib_to_netdev "$ib_dev" 1)
    [[ -n "$iface" ]] || { _err "no netdev for IB device '$ib_dev' ($role)"; return 1; }
    for other in $(ls /sys/class/net/ 2>/dev/null); do
        [[ "$other" != "$iface" ]] && \
            sudo ip addr del "${ip}/24" dev "$other" 2>/dev/null || true
    done
    _info "$role: $ib_dev ($iface) ip=$ip"
    setup_port "$iface" "$ip" "$MTU"
    wait_for_roce_v2_gid "$ib_dev" "$ip" 20
}
setup_network_cpu_roce() {
    _log "Setting up the cpu_roce channel + daemon port pair"
    _setup_roce_port "channel (kernel side)" "$CHANNEL_DEVICE" "$CHANNEL_IP"
    _setup_roce_port "daemon (server side)"  "$DAEMON_DEVICE"  "$DAEMON_IP"

    # Same-host port pairs never ARP for each other's address: the kernel
    # routes local IPs via lo and never probes the wire, so the RoCE path
    # resolution (ibv_modify_qp to RTR) times out with errno=110 unless static
    # neighbor entries pin each peer IP to the other port's MAC -- the exact
    # same-host analogue of the FPGA path's seed_fpga_neighbor. Harmless on a
    # genuine two-host topology (the entries are simply correct there too when
    # the devices are local; skipped if either netdev is not resolvable).
    local ch_if dm_if ch_mac dm_mac
    ch_if=$(ib_to_netdev "$CHANNEL_DEVICE" 1)
    dm_if=$(ib_to_netdev "$DAEMON_DEVICE" 1)
    if [[ -n "$ch_if" && -n "$dm_if" ]]; then
        ch_mac=$(cat "/sys/class/net/$ch_if/address")
        dm_mac=$(cat "/sys/class/net/$dm_if/address")
        sudo ip neigh replace "$DAEMON_IP" lladdr "$dm_mac" nud permanent dev "$ch_if"
        sudo ip neigh replace "$CHANNEL_IP" lladdr "$ch_mac" nud permanent dev "$dm_if"
        _info "static neighbors: $DAEMON_IP -> $dm_mac on $ch_if; $CHANNEL_IP -> $ch_mac on $dm_if"
    fi
}

prepare_gpu_roce_config() {
    local peer_ip="$1" remote_qp="$2" num_pages="$3" page_size="$4"
    local remote_qp_decimal=$((remote_qp))
    local payload_size=$((PAGE_SIZE - 24))
    if (( payload_size <= 0 )); then
        _err "gpu_roce page size ($PAGE_SIZE) must exceed the 24-byte RPC header"
        return 1
    fi
    if grep -Eq '^[[:space:]]*transport:' "$CONFIG_FILE"; then
        _err "generated config already contains a transport section: $CONFIG_FILE"
        return 1
    fi

    # CONFIG_FILE already lives in this run's temporary directory. Strip the
    # generator's terminal YAML document marker, then serialize the dynamic
    # endpoint once so the server sees one configuration source.
    awk '
        { lines[NR] = $0 }
        END {
            last = NR
            while (last > 0 && lines[last] ~ /^[[:space:]]*$/)
                --last
            if (last > 0 && lines[last] ~ /^[[:space:]]*\.\.\.[[:space:]]*$/)
                --last
            for (line = 1; line <= last; ++line)
                print lines[line]
        }
    ' "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"
    cat >> "$CONFIG_FILE" <<EOF
transport:
  provider: gpu_roce
  args:
    - --device=$BRIDGE_DEVICE
    - --peer-ip=$peer_ip
    - --remote-qp=$remote_qp_decimal
    - --page-size=$page_size
    - --num-pages=$num_pages
    - --payload-size=$payload_size
...
EOF
    _info "GPU RoCE launch settings written to YAML: $CONFIG_FILE"
}

start_roce_server() {
    local peer_ip="$1" remote_qp="$2" server_log="$3"
    local device_graph_num_pages="$4" device_graph_page_size="$5"
    _log "Starting decoding_server (wire=$WIRE, dispatch=$DISPATCH, remote-qp=$remote_qp)"
    local ready
    if [[ "$DISPATCH" == "device_graph" ]]; then
        # All-device_graph config: the standalone device-graph transceiver
        # brings up the GPU RoCE wire from the generated YAML; no transport
        # environment variables, CLI selector, or provider arguments are used.
        prepare_gpu_roce_config "$peer_ip" "$remote_qp" \
            "$device_graph_num_pages" "$device_graph_page_size" || return 1
        CUDA_MODULE_LOADING=EAGER \
        "$SERVER_BIN" --config="$CONFIG_FILE" --timeout="$TIMEOUT" \
            > >(tee "$server_log") 2>&1 &
        ready="QEC_DECODING_SERVER_READY device_graph"
    else
        "$SERVER_BIN" --config="$CONFIG_FILE" --transport="$WIRE_TOKEN" --qp_config=hsb_fpga \
            --device="$BRIDGE_DEVICE" --peer-ip="$peer_ip" --remote-qp="$remote_qp" \
            --num-slots="$NUM_SLOTS" --slot-size="$PAGE_SIZE" --frame-size="$FRAME_SIZE" \
            --timeout="$TIMEOUT" > >(tee "$server_log") 2>&1 &
        ready="Bridge Ready"
    fi
    local spid=$!; PIDS+=("$spid"); SERVER_PID="$spid"
    wait_for_pattern "$server_log" "$ready" 60 "$spid" >/dev/null || { cat "$server_log" >&2; return 1; }
    wait_for_pattern "$server_log" "decoder 0 type: ${DECODER}" 5 "$spid" >/dev/null || {
        _err "server did not construct a '${DECODER}' session"; cat "$server_log" >&2; return 1; }
    # The RDMA handshake is published as qp=/rkey=/buffer_addr= tokens on ONE
    # line: the READY line for host dispatch (the bridge provider's endpoint
    # info rides it), or a dedicated QEC_DECODING_SERVER_ENDPOINT line from
    # the device-graph transceiver. Scrape whichever arrives.
    local ep_line
    ep_line=$(wait_for_pattern "$server_log" "buffer_addr=" 5 "$spid") || return 1
    SERVER_QP=$(sed -n 's/.*[[:space:]]qp=\([0-9a-fA-FxX]*\).*/\1/p' <<<"$ep_line")
    SERVER_RKEY=$(sed -n 's/.*[[:space:]]rkey=\([0-9]*\).*/\1/p' <<<"$ep_line")
    SERVER_ADDR=$(sed -n 's/.*[[:space:]]buffer_addr=\([0-9a-fA-FxX]*\).*/\1/p' <<<"$ep_line")
    if [[ -z "$SERVER_QP" || -z "$SERVER_RKEY" || -z "$SERVER_ADDR" ]]; then
        _err "endpoint line missing qp=/rkey=/buffer_addr= tokens: $ep_line"; return 1
    fi
    _info "server QP=$SERVER_QP RKey=$SERVER_RKEY Buffer=$SERVER_ADDR"
}

run_fpga() {
    _banner "Realtime decoding: fpga source (RoCE), decoder=$DECODER"
    # The HSB QP has a fixed 64-entry receive WQE depth and the receiver
    # pre-posts one WQE per ring slot, so a ring with more slots than WQEs
    # leaves slots un-posted and drops frames under load. Both FPGA paths run
    # a WQE-depth ring: the cpu_roce wire via NUM_SLOTS (which the server also
    # clamps), device_graph dispatch via DEVICE_GRAPH_NUM_PAGES here.
    local HSB_WQE_DEPTH=64
    local DEVICE_GRAPH_NUM_PAGES="$HSB_WQE_DEPTH"
    # GPU RoCE ring slots are 128-byte granular. Keep PAGE_SIZE as the frame
    # budget, but give the provider and playback the same aligned slot stride.
    local DEVICE_GRAPH_PAGE_SIZE=$(( ((PAGE_SIZE + 127) / 128) * 128 ))
    if (( NUM_SLOTS > HSB_WQE_DEPTH )); then
        _warn "NUM_SLOTS=$NUM_SLOTS exceeds the HSB WQE depth ($HSB_WQE_DEPTH); clamping"
        NUM_SLOTS="$HSB_WQE_DEPTH"
    fi
    if [[ "$DISPATCH" == "device_graph" ]]; then
        # DOCA requires the device-graph ring allocation (num_pages * page
        # size) to be a multiple of the HOST page size. 64 x 384 B = 24 KiB
        # satisfies 4K/8K-page kernels; on 16K/64K-page hosts (some aarch64
        # configs) the server would reject the ring at startup, so fail fast
        # with the constraint spelled out.
        local host_page; host_page=$(getconf PAGESIZE)
        if (( (DEVICE_GRAPH_NUM_PAGES * DEVICE_GRAPH_PAGE_SIZE) % host_page != 0 )); then
            _err "device_graph ring ($DEVICE_GRAPH_NUM_PAGES slots x $DEVICE_GRAPH_PAGE_SIZE B) is not a multiple of this host's page size ($host_page B)."
            _err "The HSB frame stride must be a multiple of $(( host_page / HSB_WQE_DEPTH )) B on this host; see the unittests"
            _err "hsb_fpga_decoding_server_test.sh (--page-size) for a tunable-geometry run."
            exit 1
        fi
    fi
    : "${BRIDGE_DEVICE:=${IB_DEVICE:-rocep1s0f0}}"

    local server_log="$GEN_DIR/server.log"
    start_roce_server "$FPGA_IP" "0x2" "$server_log" \
        "$DEVICE_GRAPH_NUM_PAGES" "$DEVICE_GRAPH_PAGE_SIZE" || return 1

    _log "Streaming syndromes from the FPGA via playback (spacing=${SPACING}us)"
    # The FPGA writes syndrome frame rid to RDMA slot (rid % num-pages), so the
    # playback ring modulus MUST equal the server's RX ring depth. Otherwise
    # frames past one ring land outside the server's registered memory region
    # and are silently lost, and the server starves after exactly one ring
    # (num_slots). The cpu_roce wire's server ring is NUM_SLOTS; the
    # device_graph ring is DEVICE_GRAPH_NUM_PAGES.
    local pb_pages="$NUM_SLOTS"
    local pb_page_size="$PAGE_SIZE"
    if [[ "$DISPATCH" == "device_graph" ]]; then
        pb_pages="$DEVICE_GRAPH_NUM_PAGES"
        pb_page_size="$DEVICE_GRAPH_PAGE_SIZE"
    fi
    local args=( --hsb-ip "$FPGA_IP" --per-round --config "$CONFIG_FILE"
        --syndromes "$SYNDROMES_FILE" --qp-number "$SERVER_QP" --rkey "$SERVER_RKEY"
        --buffer-addr "$SERVER_ADDR" --page-size "$pb_page_size" --num-pages "$pb_pages" )
    $VERIFY && args+=(--verify)
    [[ -n "$NUM_SHOTS" ]] && args+=(--num-shots "$NUM_SHOTS")
    [[ -n "$SPACING"   ]] && args+=(--spacing "$SPACING")
    local rc=0
    "$PLAYBACK_BIN" "${args[@]}" || rc=$?
    kill -TERM "$SERVER_PID" 2>/dev/null || true; sleep 0.3
    return $rc
}

# ============================================================================
# Main
# ============================================================================
main() {
    _banner "Realtime Decoding Demo"
    _info "source=$SOURCE  decoder=$DECODER  wire=$WIRE  dispatch=$DISPATCH"

    resolve_paths
    if $DO_SETUP_NETWORK; then
        if [[ "$SOURCE" == "fpga" ]]; then setup_network
        elif [[ "$SOURCE" == "qpu-kernel" && "$WIRE" == "cpu_roce" ]]; then
            setup_network_cpu_roce
        fi
    fi
    generate_data

    local rc=0
    if [[ "$SOURCE" == "qpu-kernel" ]]; then run_qpu_kernel || rc=$?
    else run_fpga || rc=$?; fi

    echo
    if [[ $rc -eq 0 ]]; then _banner "REALTIME DECODING ($SOURCE / $DECODER): PASS"
    else _banner "REALTIME DECODING ($SOURCE / $DECODER): FAIL"; fi
    return $rc
}
main
