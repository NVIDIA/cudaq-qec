#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Prove the decoders-only build environment works: configure and build the
# decoder stack (CUDAQ_QEC_DECODERS_ONLY=ON) with Python bindings, then assert
# the invariants that make it "decoders only". Meant to run *inside* the
# docker/build_env/cudaqx.decoders.Dockerfile image, with the repo mounted at
# the current working directory.
#
# Usage: verify_build_env.sh [install_tensorrt]
#   install_tensorrt: on|off (default off) -- whether the image has TensorRT,
#                     which decides whether the TRT decoder plugin must build.

set -euo pipefail

TRT="${1:-off}"
BUILD_DIR="${BUILD_DIR:-/tmp/build_decoders_only}"
CONFIGURE_LOG="${CONFIGURE_LOG:-/tmp/decoders_configure.log}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

trt_enabled=false
case "$TRT" in
  on|ON|1|true|TRUE|yes|YES) trt_enabled=true ;;
esac

cmake_args=(
  -S libs/qec -B "$BUILD_DIR" -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DCUDAQ_QEC_DECODERS_ONLY=ON
  -DCUDAQX_QEC_BINDINGS_PYTHON=ON
)
# When TensorRT is present, hard-require the plugin so a silent AUTO skip cannot
# hide a regression; when it is absent, leave the default (AUTO) so it is
# skipped and we can assert its absence below.
if $trt_enabled; then
  cmake_args+=(-DCUDAQ_QEC_BUILD_TRT_DECODER=ON)
fi

log "Configuring decoders-only build (tensorrt=$TRT)"
cmake "${cmake_args[@]}" 2>&1 | tee "$CONFIGURE_LOG"

log "Building"
cmake --build "$BUILD_DIR" -j"$(nproc)"

# 1. This image has no cudaq-realtime, so configuring must WARN (not fail) that
#    the decoding server is skipped, and no decoding_server may be produced --
#    covers the not-found half of the decoding-server gate.
log "Checking the decoding server was skipped, not fatal"
grep -q "cudaq-realtime not found" "$CONFIGURE_LOG" \
  || die "expected a warning that cudaq-realtime was not found"
[ ! -e "$BUILD_DIR/bin/decoding_server" ] \
  || die "decoding_server was built without cudaq-realtime"
lib="$BUILD_DIR/lib/libcudaq-qec-decoders.so"
[ -f "$lib" ] || die "expected decoder library not found: $lib"

# 2. The standalone Python module must import and decode without ever pulling in
#    cudaq (it is importable as the bare _qec_decoders_standalone module).
log "Python smoke test (pymatching decode, no cudaq import)"
PYTHONPATH="$BUILD_DIR/python" python3 - <<'PY'
import sys
assert "cudaq" not in sys.modules
import numpy as np
import _qec_decoders_standalone
qec = _qec_decoders_standalone.qecrt
# Distance-4 repetition code (pymatching needs a matching graph: every H column
# with at most 2 ones).
H = np.array([[1, 1, 0, 0],
              [0, 1, 1, 0],
              [0, 0, 1, 1]], dtype=np.uint8)
decoder = qec.get_decoder("pymatching", H)
result = decoder.decode([1, 1, 0])
assert result.converged
correction = (np.asarray(result.result) >= 0.5).astype(np.uint8)
np.testing.assert_array_equal((H @ correction) % 2, [1, 1, 0])
assert "cudaq" not in sys.modules
print("pymatching decode OK without CUDA-Q:", correction.tolist())
PY

# 3. libcudaq-qec-decoders.so must not link CUDA-Q, LLVM or MLIR -- that is the
#    whole point of the decoders-only build.
needed=$(readelf -d "$lib" | awk '/NEEDED/ {print $NF}' | tr -d '[]')
log "libcudaq-qec-decoders.so NEEDED:"
echo "$needed" | sed 's/^/    /'
if echo "$needed" | grep -Eiq 'libcudaq|libLLVM|libMLIR|libnvqir'; then
  die "libcudaq-qec-decoders.so links an unexpected CUDA-Q/LLVM/MLIR library"
fi

# 4. The TRT decoder plugin must be built iff TensorRT was provided.
trt_plugin="$BUILD_DIR/lib/decoder-plugins/libcudaq-qec-trt-decoder.so"
if $trt_enabled; then
  [ -f "$trt_plugin" ] || die "TensorRT present but the TRT decoder plugin was not built"
  readelf -d "$trt_plugin" | grep -q 'libnvinfer' \
    || die "TRT decoder plugin is not linked against libnvinfer"
  log "TRT decoder plugin built and linked to TensorRT"
else
  [ ! -f "$trt_plugin" ] \
    || die "TensorRT absent but the TRT decoder plugin was built"
  log "TRT decoder plugin correctly auto-disabled"
fi

log "Decoders-only build environment verified (tensorrt=$TRT)"
