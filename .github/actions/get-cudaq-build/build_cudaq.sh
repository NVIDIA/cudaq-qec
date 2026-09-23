#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Thin wrapper that preserves the positional-args contract used by
# .github/actions/get-cudaq-build/action.yaml while delegating the actual
# realtime + CUDA-Q build to scripts/build_cudaq_with_realtime.sh.
BUILD_TYPE=${1:-"Release"}
LAUNCHER=${2:-""}  # accepted for backward compat; upstream auto-detects ccache via PATH
CC=${3:-"gcc"}
CXX=${4:-"g++"}

export CC
export CXX

# Keep cudaqx CI's previous defaults (these differ from upstream defaults).
export LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/local/llvm}
export CUDAQ_REQUIRE_OPENMP=${CUDAQ_REQUIRE_OPENMP:-TRUE}
export CUDAQ_BUILD_TESTS=${CUDAQ_BUILD_TESTS:-FALSE}
export CUDAQ_WERROR=${CUDAQ_WERROR:-OFF}
# CUDAQ_INSTALL_PREFIX / CUQUANTUM_INSTALL_PREFIX / CUTENSOR_INSTALL_PREFIX /
# CCACHE_DIR are expected to be set by the calling action (see action.yaml).

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/../../.." && pwd)

bash "$repo_root/scripts/build_cudaq_with_realtime.sh" "$BUILD_TYPE" "$CC" "$CXX"
