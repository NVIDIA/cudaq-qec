#!/bin/bash
set -euo pipefail

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build and install cudaq-realtime, then build CUDA-Q against that install so
# device_call support is enabled in the final CUDA-Q prefix.
#
# Positional arguments:
#   1. CMAKE_BUILD_TYPE
#   2. C compiler
#   3. C++ compiler
#
# Environment:
#   CUDAQ_SRC             CUDA-Q source checkout, default: ./cudaq
#   CUDAQ_INSTALL_PREFIX  CUDA-Q install prefix, default: /usr/local/cudaq

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$script_dir/cudaq_realtime_cmake_flags.sh"

BUILD_TYPE=${1:-"Release"}
CC=${2:-${CC:-"gcc"}}
CXX=${3:-${CXX:-"g++"}}

export CC
export CXX

: "${CUDAQ_SRC:=cudaq}"
: "${CUDAQ_INSTALL_PREFIX:=/usr/local/cudaq}"

CUDAQ_SRC=$(cd "$CUDAQ_SRC" && pwd)

log "Building cudaq-realtime from $CUDAQ_SRC/realtime"
cmake -G Ninja -S "$CUDAQ_SRC/realtime" -B "$CUDAQ_SRC/realtime/build" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
  -DCMAKE_CUDA_FLAGS="$(cudaq_realtime_cmake_cuda_flags)" \
  -DCUDAQ_REALTIME_BUILD_TESTS=OFF \
  -DCUDAQ_REALTIME_BUILD_EXAMPLES=OFF \
  -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=OFF
cmake --build "$CUDAQ_SRC/realtime/build" --target install --parallel

log "Building CUDA-Q with realtime support"
(
  cd "$CUDAQ_SRC"
  bash scripts/build_cudaq.sh -v -c "$BUILD_TYPE" -- \
    "-DCUDAQ_REALTIME_DIR=$CUDAQ_INSTALL_PREFIX"
)
