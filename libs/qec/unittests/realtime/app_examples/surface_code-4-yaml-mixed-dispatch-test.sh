#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Two-process MIXED-DISPATCH test: one decoding server hosting a host-ring
# CPU decoder (multi_error_lut) AND a device_graph nv-qldpc decoder (the GPU
# device-graph scheduler as a ring consumer) side by side, each on its own
# per-decoder ring.  The wire comes ONLY from the YAML transport section
# (provider udp; the device_graph ring adds --pinned-rings so the GPU
# scheduler can poll the rings directly) -- no --transport on the command
# line.
#
# Asserts:
#   1. the server publishes one ring per decoder on the READY line;
#   2. the sc4 app decodes over both rings (per-decoder logical_errors);
#   3. the device-graph scheduler genuinely fired decode graphs: trigger
#      debug rc=0 with fires == tail_relaunches > 0;
#   4. both rings dispatched traffic.
#
# Requirements (the CMake registration gates on the build-time ones):
#   - decoding_server linked with the device-graph component (proprietary
#     cudevice archive);
#   - the nv-qldpc decoder plugin;
#   - a CUDA GPU at runtime (skipped otherwise, exit 77).
#
# Usage: surface_code-4-yaml-mixed-dispatch-test.sh <sc4-cqr-binary> <server>

set -uo pipefail

APP="${1:?path to surface_code-4-yaml-cqr}"
SERVER="${2:?path to decoding_server}"
[[ -x "$APP" ]] || { echo "FAIL: app not found: $APP"; exit 1; }
[[ -x "$SERVER" ]] || { echo "FAIL: server not found: $SERVER"; exit 1; }

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1 || {
  echo "SKIP: no CUDA GPU available"; exit 77; }

# Device-side graph launch (the trigger path this test asserts on) requires
# compute capability >= 9.0; the dispatch kernel compiles the trigger block
# out below that (#if __CUDA_ARCH__ >= 900), so on e.g. an A100 the scheduler
# runs but the decode graph can never fire and the trigger-debug check would
# false-FAIL with rc=-1000 fires=0.
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
     | head -1)
cc_major="${cc%%.*}"
[[ -n "$cc_major" && "$cc_major" =~ ^[0-9]+$ && "$cc_major" -ge 9 ]] || {
  echo "SKIP: device-graph dispatch requires compute capability >= 9.0" \
       "(found ${cc:-unknown})"; exit 77; }

plugin_dir="$(dirname "$SERVER")/../lib/decoder-plugins"
[[ -e "$plugin_dir/libcudaq-qec-nv-qldpc-decoder.so" ]] || {
  echo "SKIP: nv-qldpc decoder plugin not present in $plugin_dir"; exit 77; }

workdir=$(mktemp -d)
server_log="$workdir/server.log"
app_log="$workdir/app.log"
config="$workdir/mixed.yaml"
server_pid=""
cleanup() {
  [[ -n "$server_pid" ]] && kill -KILL "$server_pid" 2>/dev/null
  rm -rf "$workdir"
}
trap cleanup EXIT

# distance-3 / 3-round surface-code Z decoders (matrices as produced by the
# sc4 app's own DEM save for --distance 3 --num_rounds 3): decoder 0 is a
# CPU LUT on a host ring; decoder 1 is nv-qldpc RelayBP behind the GPU
# device-graph scheduler.
H_SPARSE='[ -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, -1, 7, 8, 9, 10, -1, 1, 2, 11, 12, -1, 3, 4, 8, 9, 13, 14, 15, -1, 5, 6, 16, 17, -1, 4, 5, 9, 10, 16, 18, 19, -1, 2, 4, 12, 14, 19, 20, 21, -1, 14, 15, 20, 22, -1, 23, 24, 25, 26, 27, 28, 29, -1, 30, 31, 32, 33, -1, 24, 25, 34, 35, -1, 26, 27, 31, 32, 36, 37, 38, -1, 28, 29, 39, 40, -1, 27, 28, 32, 33, 39, 41, 42, -1, 25, 27, 35, 37, 42, 43, 44, -1, 37, 38, 43, 45, -1, -1, -1, -1, -1 ]'
O_SPARSE='[ 2, 6, 12, 17, 21, 25, 29, 35, 40, 44, -1 ]'
D_SPARSE='[ 4, -1, 5, -1, 6, -1, 7, -1, 0, 8, -1, 1, 9, -1, 2, 10, -1, 3, 11, -1, 4, 12, -1, 5, 13, -1, 6, 14, -1, 7, 15, -1, 8, 16, -1, 9, 17, -1, 10, 18, -1, 11, 19, -1, 12, 20, -1, 13, 21, -1, 14, 22, -1, 15, 23, -1, 20, 24, 25, -1, 21, 25, 26, 28, 29, -1, 22, 27, 28, 30, 31, -1, 23, 31, 32, -1 ]'
ERR_VEC='[ 0.00666667, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00666667, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00666667, 0.00334452, 0.00334452, 0.00666667, 0.00334452, 0.00666667, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00666667, 0.00334452, 0.00334452, 0.00334452, 0.00334452, 0.00666667, 0.00334452, 0.00334452, 0.00666667, 0.00334452 ]'

cat > "$config" <<EOF
transport:
  provider:        udp
  device_graph:
    args:          [--pinned-rings]
decoders:
  - id:              0
    type:            multi_error_lut
    block_size:      46
    syndrome_size:   24
    H_sparse:        $H_SPARSE
    O_sparse:        $O_SPARSE
    D_sparse:        $D_SPARSE
    decoder_custom_args:
      lut_error_depth: 1
  - id:              1
    type:            nv-qldpc-decoder
    dispatch:        device_graph
    block_size:      46
    syndrome_size:   24
    H_sparse:        $H_SPARSE
    O_sparse:        $O_SPARSE
    D_sparse:        $D_SPARSE
    error_rate_vec:  $ERR_VEC
    decoder_custom_args:
      use_sparsity:    true
      max_iterations:  50
      clip_value:      200
      bp_method:       3
      gamma0:          0
      gamma_dist:      [ 0.1, 0.2 ]
      srelay_config:
        pre_iter:        5
        num_sets:        10
        stopping_criterion: All
        stop_nconv:      1
      composition:     1
      repeatable:      true
EOF

# --- server: wire named by the YAML only (no --transport) ------------------
CUDAQ_LOG_LEVEL=info "$SERVER" --config="$config" --port=0 --timeout=120 \
    > "$server_log" 2>&1 &
server_pid=$!

ready=""
for _ in $(seq 1 60); do
  ready=$(grep -m1 "QEC_DECODING_SERVER_READY" "$server_log" || true)
  [[ -n "$ready" ]] && break
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 0.5
done
[[ -n "$ready" ]] || {
  echo "FAIL: server never printed READY"; tail -15 "$server_log"; exit 1; }
P0=$(sed -n 's/.*ring0=\([0-9]*\).*/\1/p' <<< "$ready")
P1=$(sed -n 's/.*ring1=\([0-9]*\).*/\1/p' <<< "$ready")
[[ -n "$P0" && -n "$P1" ]] || {
  echo "FAIL: READY line lacks per-decoder ring ports: $ready"; exit 1; }
echo "PASS: per-decoder rings published (ring0=$P0 ring1=$P1)"

# --- app: decode over both rings -------------------------------------------
QEC_DECODING_SERVER_PORT="$P0" QEC_DECODING_SERVER_PORT_1="$P1" \
    timeout 100 "$APP" --distance 3 --num_rounds 3 --num_shots 5 \
    --p_spam 0.05 --yaml "$config" --num_logical 2 --use-relay-bp \
    > "$app_log" 2>&1
rc=$?
[[ $rc -eq 0 ]] || {
  echo "FAIL: app exited $rc"; tail -15 "$app_log"; exit 1; }
grep -q "decoder\[0\] (multi_error_lut).*logical_errors" "$app_log" || {
  echo "FAIL: no host-ring decoder result"; tail -15 "$app_log"; exit 1; }
grep -q "decoder\[1\] (nv-qldpc-decoder).*logical_errors" "$app_log" || {
  echo "FAIL: no device-ring decoder result"; tail -15 "$app_log"; exit 1; }
echo "PASS: both decoders returned results over their own rings"

# --- teardown: scheduler health + per-ring traffic --------------------------
kill -TERM "$server_pid" 2>/dev/null
for _ in $(seq 1 40); do
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 0.5
done
kill -0 "$server_pid" 2>/dev/null && {
  echo "FAIL: server did not exit on SIGTERM"; exit 1; }
server_pid=""

debug=$(grep -m1 "trigger debug" "$server_log" || true)
[[ -n "$debug" ]] || {
  echo "FAIL: no trigger-debug line (device-graph consumer never ran)"
  tail -15 "$server_log"; exit 1; }
fires=$(sed -n 's/.*fires=\([0-9]*\).*/\1/p' <<< "$debug")
tails=$(sed -n 's/.*tail_relaunches=\([0-9]*\).*/\1/p' <<< "$debug")
grep -q "trigger debug rc=0" <<< "$debug" || {
  echo "FAIL: device-side trigger launch failed: $debug"; exit 1; }
[[ -n "$fires" && "$fires" -gt 0 && "$fires" -eq "$tails" ]] || {
  echo "FAIL: unhealthy scheduler (fires=$fires tails=$tails): $debug"
  exit 1; }
echo "PASS: decode graph fired $fires times, rc=0, tails match"

for ring in 0 1; do
  n=$(sed -n "s/.*QEC_DECODING_SERVER_RING decoder=$ring dispatched=\([0-9]*\).*/\1/p" \
      "$server_log" | head -1)
  [[ -n "$n" && "$n" -gt 0 ]] || {
    echo "FAIL: ring $ring dispatched nothing"; tail -10 "$server_log"
    exit 1; }
done
echo "PASS: both rings carried traffic"

echo "surface_code-4-yaml-mixed-dispatch-test PASSED"
