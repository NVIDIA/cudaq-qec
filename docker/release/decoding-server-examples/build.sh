#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# Build the realtime decoding examples with the nvq++ shipped in this image.
# Two variants exist, where an example supports them:
#
#   <name>-local  decoders run in-process, inside the CUDA-Q program.
#   <name>-cqr    the kernel is lowered with -frealtime-lowering so each
#                 decoding call is dispatched over the realtime channel to a
#                 separately running decoding_server.
#
# surface_code-1 and surface_code-4-yaml are built both ways; surface_code-2
# and -3 are local only; surface_code-5 exists only in lowered form.
#
# Usage:  ./build.sh [example ...]        (default: all)
#         BUILD_DIR=<dir> ./build.sh       (default: ./build)
#
# The flags below follow libs/qec/unittests/realtime/app_examples/CMakeLists.txt
# in the cudaqx repository, which builds the same sources in-tree. The QEC and
# cudaq-realtime headers and libraries are found through CPLUS_INCLUDE_PATH,
# LIBRARY_PATH and LD_LIBRARY_PATH, which the image sets.
set -euo pipefail

cd "$(dirname "$0")"
BUILD_DIR=${BUILD_DIR:-./build}
mkdir -p "$BUILD_DIR"

# QEC libraries every example links. --no-as-needed is required: linked with
# --as-needed the programs build but fail at startup ("Could not load the
# requested plugin").
QEC_LIBS=(-Wl,--no-as-needed -lcudaq-qec -lcudaq-qec-decoders -lcudaq-qec-realtime-decoding)
# Shared by every realtime app (see qec_realtime_app_link_options in the CMake).
LINK_OPTS=(-Wl,--allow-shlib-undefined -Wl,--export-dynamic -lcudaq-realtime-dispatch)
# Libraries for the lowered (cqr) variant: the simulated-QPU client whose
# device_calls go over the realtime channel, the in-process decoding-server
# plugin, and CUDA-Q's device-call runtime.
CQR_LIBS=(-lcudaq-qec-realtime-decoding-simulation-cqr -lcudaq-qec-realtime-decoding-server-cqr -lcudaq-device-call-runtime)

# Compile to an object first, then link, as the in-tree CMake build does
# (cudaqx_add_device_code compiles with `-c -fPIC`).
build() { # build <output> <source> [compile flags...] -- [link args...]
  local out=$1 src=$2; shift 2
  local cflags=() largs=()
  while [[ $# -gt 0 && $1 != -- ]]; do cflags+=("$1"); shift; done
  [[ $# -gt 0 ]] && shift
  largs=("$@")
  echo "== $out"
  nvq++ --target stim -c -fPIC "${cflags[@]}" "$src" -o "$BUILD_DIR/$out.o" 2>&1 \
    | grep -v "^warning: The CUDA-Q\|cc.alloca\|^ *%\|^ *\^" || true
  nvq++ --target stim "$BUILD_DIR/$out.o" -o "$BUILD_DIR/$out" "${largs[@]}" 2>&1 \
    | grep -v "^warning: The CUDA-Q" || true
  rm -f "$BUILD_DIR/$out.o"
  test -x "$BUILD_DIR/$out"
}

SELECTED=("$@")
sel() { [[ ${#SELECTED[@]} -eq 0 ]] || [[ " ${SELECTED[*]} " == *" $1 "* ]]; }

if sel surface_code-1; then
  build surface_code-1-local surface_code-1.cpp -- "${QEC_LIBS[@]}" -lcudaq-qec-realtime-decoding-simulation "${LINK_OPTS[@]}"
  build surface_code-1-cqr   surface_code-1.cpp -frealtime-lowering -DQEC_APP_CQR -- "${QEC_LIBS[@]}" "${CQR_LIBS[@]}" "${LINK_OPTS[@]}"
fi
if sel surface_code-2; then
  build surface_code-2-local surface_code-2.cpp -- "${QEC_LIBS[@]}" -lcudaq-qec-realtime-decoding-simulation "${LINK_OPTS[@]}"
fi
if sel surface_code-3; then
  build surface_code-3-local surface_code-3.cpp -- "${QEC_LIBS[@]}" -lcudaq-qec-realtime-decoding-simulation "${LINK_OPTS[@]}"
fi
if sel surface_code-4-yaml; then
  build surface_code-4-yaml     surface_code-4-yaml.cpp -- "${QEC_LIBS[@]}" -lcudaq-qec-realtime-decoding-simulation "${LINK_OPTS[@]}"
  build surface_code-4-yaml-cqr surface_code-4-yaml.cpp -frealtime-lowering -DQEC_APP_EXTERNAL_DECODING_SERVER -- "${QEC_LIBS[@]}" -lcudaq-qec-realtime-decoding-simulation-cqr -lcudaq-device-call-runtime "${LINK_OPTS[@]}"
fi
if sel surface_code-5-per-decoder-rings; then
  build surface_code-5-per-decoder-rings surface_code-5-per-decoder-rings.cpp -frealtime-lowering -- "${QEC_LIBS[@]}" "${CQR_LIBS[@]}" "${LINK_OPTS[@]}"
fi

echo; echo "Built into $BUILD_DIR:"; ls -1 "$BUILD_DIR"
