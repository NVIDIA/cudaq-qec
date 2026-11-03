#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

. "$(dirname "$0")/setup_custabilizer.sh"

build_dir=$1
install_prefix=$2
cudaq_prefix=$3

_rt_flag=""
if [ -n "$CUDAQ_REALTIME_ROOT" ]; then
  _rt_flag="-DCUDAQ_REALTIME_ROOT=$CUDAQ_REALTIME_ROOT"
fi
_prop_archive_flag=""
if [ -n "$CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE" ]; then
  _prop_archive_flag="-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=$CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE"
fi

cmake -S . -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-12 \
  -DCMAKE_CXX_COMPILER=g++-12 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCUDAQ_DIR="$cudaq_prefix/lib/cmake/cudaq/" \
  -DCUDAQX_ENABLE_LIBS="all" \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$install_prefix" \
  $_rt_flag \
  $_prop_archive_flag

cmake --build "$build_dir" --target install -j 4
