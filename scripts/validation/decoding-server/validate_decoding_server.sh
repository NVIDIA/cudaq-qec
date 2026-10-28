#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Validate a cudaq-decoding release image (docker/release/Dockerfile.decoding-server).
# Everything exercised here ships in the image; the checks run on UDP loopback and
# need no NIC or FPGA. A GPU is required for the TensorRT decoder and the
# hsb_fpga_syndrome_playback check.
#
# Usage: scripts/validation/decoding-server/validate_decoding_server.sh \
#            [--final-image <image>] [--skip-gpu]
#
# The image defaults to the NGC staging copy of the latest CUDA 13 build. Tags
# follow build_decoding_server_docker.yaml: cu<major>-latest,
# cu<major>-<date>-<sha>, and cu<major>-<version> for release builds.
#
# Checks:
#   1. shared library resolution for everything under /opt/nvidia/cudaq-qec
#   2. decoding_server + playback-emulator with the shipped pymatching config
#   3. shipped multi_error_lut config
#   4. two decoders in one server, per-decoder rings
#   5. 200 paced shots through pymatching
#   6. chromobius plugin fed by a Stim DEM
#   7. malformed config exits non-zero with a diagnostic
#   8. hsb_fpga_syndrome_playback --help                                (GPU)
#   9. trt_decoder builds an engine from the repo's test ONNX model     (GPU)
#  10. nvq++ compiles and runs a CUDA-Q program and a QEC program
#  11. the shipped examples build with the shipped build.sh
#  12. the realtime example tests from libs/qec/unittests/realtime/app_examples
#      run against those binaries, with decoding_server as the external server
#
# The ONNX model, chromobius DEM and example test scripts come from this
# checkout and are mounted read-only into the container.

set -uo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
ONNX_MODEL="$REPO_ROOT/assets/tests/surface_code_decoder.onnx"
CHROMOBIUS_DEM="$REPO_ROOT/libs/qec/test_data/chromobius/basic_reference.dem"
APP_EXAMPLE_TESTS="$REPO_ROOT/libs/qec/unittests/realtime/app_examples"

FINAL_IMAGE="nvcr.io/nvstaging/cudaq-qec/cudaq-decoding:cu13-latest"
SKIP_GPU=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --final-image) FINAL_IMAGE=$2; shift 2 ;;
        --skip-gpu) SKIP_GPU=1; shift ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--final-image <image>] [--skip-gpu]"
            exit 1 ;;
    esac
done
for f in "$ONNX_MODEL" "$CHROMOBIUS_DEM" "$APP_EXAMPLE_TESTS/surface_code-1-test.sh"; do
    [[ -f "$f" ]] || { echo "Missing test input: $f"; exit 1; }
done

PREFIX=/opt/nvidia/cudaq-qec
BIN=$PREFIX/bin
CONTAINER="cudaq-decoding-validate-$(date +%s)"
num_failures=0

GPU_ARGS=()
[[ $SKIP_GPU -eq 0 ]] && GPU_ARGS=(--gpus all)

# The image's ENTRYPOINT is a login shell; keep the container alive and run
# each check with docker exec. Test inputs are mounted read-only at /inputs.
docker run -d --name "$CONTAINER" "${GPU_ARGS[@]}" \
    -v "$ONNX_MODEL":/inputs/model.onnx:ro \
    -v "$CHROMOBIUS_DEM":/inputs/basic_reference.dem:ro \
    -v "$APP_EXAMPLE_TESTS":/inputs/app_examples:ro \
    --entrypoint bash "$FINAL_IMAGE" -c "sleep infinity" > /dev/null
cleanup() { docker rm -f "$CONTAINER" > /dev/null 2>&1 || true; }
trap cleanup EXIT

# in_container <label> <script>: run <script> in a login shell inside the
# container and record PASS/FAIL from its exit code.
in_container() {
    local label=$1 script=$2
    echo
    echo "=== $label"
    if docker exec "$CONTAINER" bash -l -c "set -uo pipefail; BIN=$BIN; PREFIX=$PREFIX; $script"; then
        echo "PASS: $label"
    else
        echo "FAIL: $label (exit $?)"
        num_failures=$((num_failures + 1))
    fi
}

# Shared shell functions for the server-based checks, prepended to each script.
# start_server <config> sets PORT, READY_LINE and SERVER_PID; run_emulator
# <schedule> <csv> [extra args] drives the server with --source=0:rounds.txt.
SERVER_HELPERS='
start_server() {
    QEC_DECODING_SERVER_STATS=1 $BIN/decoding_server --config="$1" --transport=udp --port=0 --timeout=120 > server.log 2>&1 &
    SERVER_PID=$!; PORT=""
    for _ in $(seq 1 1200); do
        READY_LINE=$(grep -m1 QEC_DECODING_SERVER_READY server.log 2>/dev/null || true)
        PORT=$(echo "$READY_LINE" | sed -n "s/.*port=\([0-9]\+\).*/\1/p")
        [[ -n "$PORT" ]] && return 0
        kill -0 $SERVER_PID 2>/dev/null || break
        sleep 0.1
    done
    echo "server did not become READY:"; head -20 server.log; return 1
}
stop_server() {
    kill -TERM $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
    grep -h DECODER_STATS server.log || true
}
run_emulator() {
    local schedule=$1 out=$2; shift 2
    $BIN/playback-emulator --schedule="$schedule" --backend=udp --udp-endpoint=0:127.0.0.1:$PORT \
        --source=0:rounds.txt --out="$out" "$@" > emulator.log 2>&1 || { cat emulator.log; return 1; }
}
correction() { awk -F, "\$3==\"get_corrections\"{print \$11}" "$1"; }
status() { awk -F, "\$3==\"get_corrections\"{print \$7}" "$1"; }
cd "$(mktemp -d)"
'

echo "Validating cudaq-decoding image $FINAL_IMAGE"
echo "Container: $CONTAINER"

# libcuda.so.1 comes from the NVIDIA container runtime and is absent without --gpus.
IGNORE_LIBS="^$"
[[ $SKIP_GPU -eq 1 ]] && IGNORE_LIBS="libcuda\.so\.1"
in_container "1. shared libraries resolve" '
    missing=$(find $PREFIX/bin $PREFIX/lib -type f \( -perm -u+x -o -name "*.so*" \) \
        -exec sh -c "ldd \"\$1\" 2>/dev/null | grep \"not found\" | grep -vE \"'"$IGNORE_LIBS"'\" | sed \"s|^|\$1: |\"" _ {} \;)
    if [[ -n "$missing" ]]; then echo "$missing"; exit 1; fi
    echo "all binaries and libraries under $PREFIX resolve"
'

in_container "2. pymatching: shipped config" "$SERVER_HELPERS"'
    start_server $BIN/decoding_server_config.yaml || exit 1
    printf "010\n" > rounds.txt
    printf "0 reset\n1 stream source=0 rounds=1\n2 get_corrections return_size=3\n" > schedule.txt
    run_emulator schedule.txt out.csv || exit 1
    stop_server
    echo "correction=$(correction out.csv) status=$(status out.csv)"
    [[ "$(correction out.csv)" == "010" && "$(status out.csv)" == "0" ]]
'

in_container "3. multi_error_lut: shipped config" "$SERVER_HELPERS"'
    start_server $BIN/decoding_server_config_multi_error_lut.yaml || exit 1
    printf "001\n" > rounds.txt
    printf "0 reset\n1 stream source=0 rounds=1\n2 get_corrections return_size=3\n" > schedule.txt
    run_emulator schedule.txt out.csv || exit 1
    stop_server
    echo "correction=$(correction out.csv) status=$(status out.csv)"
    [[ "$(correction out.csv)" == "001" && "$(status out.csv)" == "0" ]]
'

in_container "4. two decoders, per-decoder rings" "$SERVER_HELPERS"'
    cat > two.yaml <<EOF
decoders:
  - id: 0
    type: pymatching
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args: {error_rate_vec: [0.1, 0.1, 0.1], merge_strategy: smallest_weight}
  - id: 1
    type: multi_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args: {lut_error_depth: 1}
EOF
    start_server two.yaml || exit 1
    PORT1=$(echo "$READY_LINE" | sed -n "s/.*ring1=\([0-9]\+\).*/\1/p")
    echo "ring0=$PORT ring1=${PORT1:-MISSING}"
    [[ -n "$PORT1" ]] || { stop_server; exit 1; }
    printf "100\n" > rounds.txt; printf "001\n" > r1.txt
    printf "0 reset session=0\n0 reset session=1\n1 stream source=0 rounds=1 session=0\n1 stream source=1 rounds=1 session=1\n2 get_corrections return_size=3 session=0\n2 get_corrections return_size=3 session=1\n" > schedule.txt
    run_emulator schedule.txt out.csv --udp-endpoint=1:127.0.0.1:$PORT1 --source=1:r1.txt || exit 1
    stop_server
    c0=$(awk -F, "\$3==\"get_corrections\" && \$2==0{print \$11}" out.csv)
    c1=$(awk -F, "\$3==\"get_corrections\" && \$2==1{print \$11}" out.csv)
    echo "decoder0 100 -> $c0 (expect 100); decoder1 001 -> $c1 (expect 001)"
    [[ "$c0" == "100" && "$c1" == "001" ]]
'

in_container "5. pymatching: 200 paced shots" "$SERVER_HELPERS"'
    # Paced at 1 ms per event: the emulator does not wait for replies, and an
    # unpaced burst overruns the UDP transport slot ring.
    start_server $BIN/decoding_server_config.yaml || exit 1
    for _ in $(seq 1 600); do echo 010; done > rounds.txt
    : > schedule.txt; t=0
    for _ in $(seq 1 200); do
        printf "%d reset\n%d stream source=0 rounds=3\n%d get_corrections return_size=3\n" $t $((t+1)) $((t+2)) >> schedule.txt
        t=$((t+3))
    done
    run_emulator schedule.txt out.csv --tick=1ms || exit 1
    stop_server
    ok=$(awk -F, "\$3==\"get_corrections\" && \$7==0 && \$11==\"010\"" out.csv | wc -l)
    bad=$(awk -F, "NR>1 && \$7!=0 && \$7!=100" out.csv | wc -l)
    echo "good get_corrections=$ok/200, non-OK events=$bad"
    [[ "$ok" == "200" && "$bad" == "0" ]]
'

in_container "6. chromobius via stim_dem_path" "$SERVER_HELPERS"'
    cat > dem.yaml <<EOF
decoders:
  - id: 0
    type: chromobius
    stim_dem_path: /inputs/basic_reference.dem
    D_sparse: [0, -1, 1, -1]
EOF
    start_server dem.yaml || exit 1
    printf "10\n" > rounds.txt
    printf "0 reset\n1 stream source=0 rounds=1\n2 get_corrections return_size=3\n" > schedule.txt
    run_emulator schedule.txt out.csv || exit 1
    stop_server
    echo "D0 fired -> observables $(correction out.csv) (expect 100) status=$(status out.csv)"
    [[ "$(correction out.csv)" == "100" && "$(status out.csv)" == "0" ]]
'

in_container "7. malformed config fails cleanly" '
    cd "$(mktemp -d)"
    printf "decoders:\n  - id: 0\n    type: pymatching\n    block_size: 3\n    syndrome_size: 3\n    H_sparse: [0, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1]\n" > bad.yaml
    timeout 20 $BIN/decoding_server --config=bad.yaml --transport=udp --port=0 --timeout=5 > out.log 2>&1; rc=$?
    echo "rc=$rc: $(grep -m1 -iE "error|not equal" out.log | head -c 120)"
    [[ $rc -ne 0 && -s out.log ]]
'

if [[ $SKIP_GPU -eq 0 ]]; then
    in_container "8. hsb_fpga_syndrome_playback --help" '
        $BIN/hsb_fpga_syndrome_playback --help > out.log 2>&1 || true
        head -3 out.log
        [[ -s out.log ]] && ! grep -q "error while loading" out.log
    '

    in_container "9. trt_decoder: engine build from ONNX" "$SERVER_HELPERS"'
        # 24-bit syndrome; H and D are identities and O is one row, since the
        # model decodes straight to the observable and these only set sizes.
        ident=""; for i in $(seq 0 23); do ident="${ident:+$ident, }$i, -1"; done
        cat > trt.yaml <<EOF
decoders:
  - id: 0
    type: trt_decoder
    block_size: 24
    syndrome_size: 24
    H_sparse: [$ident]
    O_sparse: [0, -1]
    D_sparse: [$ident]
    decoder_custom_args:
      onnx_load_path: /inputs/model.onnx
      engine_save_path: $PWD/model.engine
      engine_output_format: observables
      precision: best
EOF
        start_server trt.yaml || exit 1
        printf "010000000000000000000000\n" > rounds.txt
        printf "0 reset\n1 stream source=0 rounds=1\n2 get_corrections return_size=1\n" > schedule.txt
        run_emulator schedule.txt out.csv || exit 1
        stop_server
        grep -iE "TensorRT|engine" server.log | head -3
        echo "observable=$(correction out.csv) status=$(status out.csv)"
        [[ -s model.engine && "$(status out.csv)" == "0" ]]
    '
else
    echo
    echo "Skipping GPU checks 8 and 9 (--skip-gpu)"
fi

in_container "10. nvq++: CUDA-Q and QEC programs" '
    cd "$(mktemp -d)"
    cat > t1.cpp <<EOF
#include <cudaq.h>
#include <cstdio>
struct bell { void operator()() __qpu__ { cudaq::qvector q(2); h(q[0]); x<cudaq::ctrl>(q[0], q[1]); mz(q); } };
int main() { auto r = cudaq::sample(200, bell{}); printf("outcomes=%zu\n", r.size()); return r.size() == 2 ? 0 : 1; }
EOF
    nvq++ --target stim t1.cpp -o t1 2>&1 | grep -v "^warning" || true
    [[ -x t1 ]] && ./t1 || exit 1
    cat > t2.cpp <<EOF
#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"
#include <cstdio>
int main() {
  auto code = cudaq::qec::get_code("surface_code", cudaqx::heterogeneous_map{{"distance", 3}});
  auto H = code->get_parity();
  printf("surface_code d=3 parity check: %zu x %zu\n", H.shape()[0], H.shape()[1]);
  return H.shape()[0] > 0 ? 0 : 1;
}
EOF
    nvq++ --target stim t2.cpp -o t2 -lcudaq-qec 2>&1 | grep -v "^warning" || true
    [[ -x t2 ]] && ./t2
'

in_container "11. examples build" '
    cd ~/examples
    rm -rf /root/validate-build
    BUILD_DIR=/root/validate-build ./build.sh > /root/validate-build.log 2>&1 || { tail -30 /root/validate-build.log; exit 1; }
    ls -1 /root/validate-build
'

# Each case mirrors an add_test in app_examples/CMakeLists.txt. Arguments the
# scripts take but the image cannot supply (quantinuum lib dir) are given a
# nonexistent path, as the in-tree tests pass NULL for an unused server.
# Thresholds are the in-tree values: at the fixed seed the runs are
# deterministic and these leave margin while still catching a broken decode.
run_example_test() { # <label> <command...>
    local label=$1; shift
    in_container "12. app_examples: $label" "
        cd \"\$(mktemp -d)\"; B=/root/validate-build; SRV=\$BIN/decoding_server; T=/inputs/app_examples
        $* > test.log 2>&1 || { tail -20 test.log; exit 1; }
        grep -iE 'Number of|dispatch count|passed|completed|succeeded' test.log | tail -6 || true
    "
}
run_example_test "surface_code-1 local, multi_error_lut" \
    'bash $T/surface_code-1-test.sh $B/surface_code-1-local $B/surface_code-1-local 3 60 30 NULL 12 /nonexistent'
run_example_test "surface_code-1 local, sliding_window" \
    'bash $T/surface_code-1-test.sh $B/surface_code-1-local $B/surface_code-1-local 3 60 30 NULL 12 /nonexistent sliding_window 6 1'
run_example_test "surface_code-1 local, sliding_window with pymatching inner decoder" \
    'CUDAQ_DEFAULT_SIMULATOR=stim $B/surface_code-1-local --distance 3 --num_rounds 3 --num_shots 50 --decoder_type sliding_window --sw_window_size 3 --sw_inner_decoder pymatching'
run_example_test "surface_code-1 local, pymatching task graph" \
    'EXTRA_CLI_ARGS=--task_graph bash $T/surface_code-1-test.sh $B/surface_code-1-local $B/surface_code-1-local 3 60 30 NULL 12 /nonexistent pymatching'
run_example_test "surface_code-1 local, pymatching task graph, async streaming" \
    'EXTRA_CLI_ARGS="--task_graph --async_streaming" bash $T/surface_code-1-test.sh $B/surface_code-1-local $B/surface_code-1-local 3 60 30 NULL 12 /nonexistent pymatching'
run_example_test "surface_code-1 cqr, in-process host dispatch" \
    'CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch bash $T/surface_code-1-test.sh $B/surface_code-1-cqr $B/surface_code-1-cqr 3 60 30 NULL 12 /nonexistent'
run_example_test "surface_code-1 cqr, two-process" \
    'bash $T/surface_code-1-cqr-two-process-test.sh $B/surface_code-1-cqr 3 60 30 $SRV 12'
run_example_test "surface_code-1 cqr, two-process, 2 logical qubits" \
    'bash $T/surface_code-1-cqr-two-process-test.sh $B/surface_code-1-cqr 3 120 60 $SRV 12 multi_error_lut 2'
run_example_test "surface_code-1 cqr, two-process, unified host loop" \
    'QEC_DECODING_SERVER_EXTRA_ARGS=--host-loop=unified bash $T/surface_code-1-cqr-two-process-test.sh $B/surface_code-1-cqr 3 60 30 $SRV 12'
run_example_test "surface_code-2 local" \
    'bash $T/surface_code-2-test.sh $B/surface_code-2-local $B/surface_code-2-local 3 10 10 NULL /nonexistent'
run_example_test "surface_code-3 local, prep0" \
    'bash $T/surface_code-3-test.sh $B/surface_code-3-local $B/surface_code-3-local 3 NULL 0.010 prep0 /nonexistent'
run_example_test "surface_code-3 local, prepp" \
    'bash $T/surface_code-3-test.sh $B/surface_code-3-local $B/surface_code-3-local 3 NULL 0.010 prepp /nonexistent'
run_example_test "surface_code-4-yaml, pymatching d3" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 3 3 pymatching 200'
run_example_test "surface_code-4-yaml, pymatching d5" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 5 5 pymatching 200'
run_example_test "surface_code-4-yaml, pymatching task graph d3" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 3 3 pymatching 200 --task_graph'
run_example_test "surface_code-4-yaml, parallel task graph, streaming" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 3 3 pymatching 200 --task_graph --parallel_task_graph'
run_example_test "surface_code-4-yaml, parallel task graph, async streaming" \
    'VERIFY_SYNC_ASYNC_REPLAY=1 bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 3 3 pymatching 200 --task_graph --parallel_task_graph --async_streaming'
run_example_test "surface_code-4-yaml, d5 with 6 rounds" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 5 6 pymatching 20'
run_example_test "surface_code-4-yaml, 2 logical qubits" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 3 3 pymatching 20 --num_logical 2'
run_example_test "surface_code-4-yaml, per-patch decoder list" \
    'bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml 3 3 pymatching,pymatching 20 --num_logical 2'
run_example_test "surface_code-4-yaml cqr, external server, 3 logical qubits" \
    'QEC_DECODING_SERVER=$SRV bash $T/surface_code-4-yaml-test.sh $B/surface_code-4-yaml-cqr 3 3 pymatching 200 --num_logical 3'
run_example_test "surface_code-5 per-decoder rings" \
    'bash $T/surface_code-5-per-decoder-rings-test.sh $B/surface_code-5-per-decoder-rings'
run_example_test "surface_code-6 mapped enqueue, in-process" \
    'bash $T/surface_code-6-mapped-enqueue-test.sh $B/surface_code-6-mapped-enqueue NULL /nonexistent'

echo
if [[ $num_failures -eq 0 ]]; then
    echo "Validation complete successfully for $FINAL_IMAGE"
else
    echo "Validation failed: $num_failures check(s) failed for $FINAL_IMAGE"
fi
exit $num_failures
