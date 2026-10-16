#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Test driver for surface_code-5-per-decoder-rings: one ring buffer (and one
# dispatcher) per decoder via cudaq::device_call(device_id == decoder_id).
#
# Asserts:
#   1. the example decodes correctly and self-verifies its dispatch count;
#   2. the runtime created TWO device sessions (two rings, two dispatchers),
#      observed via the CUDA-Q device-call runtime log;
#   3. the per-device channel override spec is parsed and routes: pointing
#      decoder 1 at device_dispatch reaches the service plugin with a Gpu-mode
#      session request (which the Host-only QEC plugin declines), while
#      decoder 0 still comes up on its host ring.
#
# Usage: surface_code-5-per-decoder-rings-test.sh [path-to-binary]

set -uo pipefail

BIN="${1:-$(dirname "$0")/surface_code-5-per-decoder-rings}"
[[ -x "$BIN" ]] || { echo "FAIL: binary not found: $BIN"; exit 1; }

log=$(mktemp)
trap 'rm -f "$log"' EXIT

# --- 1+2: two decoders, two rings, correct corrections --------------------
CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch CUDAQ_LOG_LEVEL=info \
    "$BIN" >"$log" 2>&1
rc=$?
grep -q "PER-DECODER-RINGS PASSED" "$log" || {
  echo "FAIL: example did not pass"; tail -5 "$log"; exit 1; }
[[ $rc -eq 0 ]] || { echo "FAIL: nonzero exit $rc"; exit 1; }
grep -q "service initialized for device 0" "$log" || {
  echo "FAIL: no session for device 0"; exit 1; }
grep -q "service initialized for device 1" "$log" || {
  echo "FAIL: no session for device 1 (rings were not per-decoder)"; exit 1; }
echo "PASS: two decoders decoded over two per-decoder rings"

# --- 3: per-device channel override routes device 1 elsewhere -------------
CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch,1=device_dispatch CUDAQ_LOG_LEVEL=info \
    "$BIN" >"$log" 2>&1
grep -q "service initialized for device 0" "$log" || {
  echo "FAIL: override run: device 0 host ring missing"; exit 1; }
# Device 1 must NOT have come up as a host session: the override steers it to
# the GPU channel, and the (Host-only) QEC service plugin declines Gpu-mode
# sessions today, so the run must fail rather than silently fall back.
if grep -q "service initialized for device 1" "$log"; then
  echo "FAIL: override ignored; device 1 initialized on the default channel"
  exit 1
fi
echo "PASS: per-device channel override routed device 1 to device_dispatch"

echo "surface_code-5-per-decoder-rings-test PASSED"
