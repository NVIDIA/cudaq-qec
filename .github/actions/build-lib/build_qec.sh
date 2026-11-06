#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e

. "$(dirname "$0")/setup_custabilizer.sh"
. "$(dirname "$0")/../../../scripts/cudaq_realtime_cmake_flags.sh"

build_dir=$1
install_prefix=$2
cudaq_prefix=$3

# Build cuda-quantum realtime library + HSB tools (if CUDAQ_REALTIME_ROOT not set)
if [ -z "$CUDAQ_REALTIME_ROOT" ]; then
  CUDAQ_REALTIME_ROOT=/tmp/cudaq-realtime
  CUDAQ_REPO=${CUDAQ_REPO:-$(jq -r '.cudaq.repository' .cudaq_version)}
  CUDAQ_REF=${CUDAQ_REF:-$(jq -r '.cudaq.ref' .cudaq_version)}
  echo "Using CUDA-Q realtime source: ${CUDAQ_REPO}@${CUDAQ_REF}"
  _build_cwd=$(pwd)

  cd /tmp
  rm -rf cudaq-realtime-src $CUDAQ_REALTIME_ROOT
  git clone --filter=blob:none --no-checkout "https://github.com/${CUDAQ_REPO}.git" cudaq-realtime-src
  cd cudaq-realtime-src
  git sparse-checkout init --cone
  git sparse-checkout set realtime cmake
  git checkout "$CUDAQ_REF"

  # Install build tools and DOCA/Holoscan SDK for HSB.
  export HSB_ROOT=/tmp/holoscan-sensor-bridge
  HSB_BUILD=${HSB_ROOT}/build
  bash /tmp/cudaq-realtime-src/realtime/scripts/install_devdeps.sh

  # The script keeps the devcontainer's rdma-core, which is DOCA-OFED 3.1.0:
  # too old for the DOCA 3.4.0 libdoca_common it installs (MLX5_1.27 needed,
  # 1.25 present), so RoCE targets fail to link. Install rdma-core from the
  # DOCA repo the script registered, and check libmlx5 covers libdoca_common's
  # MLX5 symbol versions. Mirrors docker/release/Dockerfile.decoding-server.
  apt-get install -y --no-install-recommends \
    libibverbs1 libibverbs-dev ibverbs-providers librdmacm1 librdmacm-dev
  libmlx5=$(ldconfig -p | awk '/libmlx5\.so\.1 /{print $NF; exit}')
  libdoca=$(ls /opt/mellanox/doca/lib/*/libdoca_common.so | head -1)
  for v in $(objdump -p "$libdoca" | grep -o 'MLX5_1\.[0-9]*' | sort -u); do
    objdump -T "$libmlx5" | grep -q "$v" \
      || { echo "ERROR: $libmlx5 lacks $v, required by $libdoca"; exit 1; }
  done
  echo "== libmlx5 satisfies libdoca_common ($libmlx5)"

  # Build cuda-quantum realtime with HSB tools enabled,
  # which produces libcudaq-realtime-bridge-gpu-roce.so needed by the bridge.
  cd /tmp/cudaq-realtime-src/realtime
  mkdir -p build && cd build

  cmake -G Ninja -DCMAKE_INSTALL_PREFIX="$CUDAQ_REALTIME_ROOT" \
    -DCMAKE_CUDA_FLAGS="$(cudaq_realtime_cmake_cuda_flags)" \
    -DCUDAQ_REALTIME_ENABLE_HSB_TOOLS=ON \
    -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT \
    -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_BUILD \
    ..
  ninja
  ninja install

  cd "$_build_cwd"
fi

HSB_ROOT=/tmp/holoscan-sensor-bridge
HSB_BUILD=${HSB_ROOT}/build

_prop_archive_flag=""
if [ -n "$CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE" ]; then
  _prop_archive_flag="-DCUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE=$CUDAQ_QEC_REALTIME_CUDEVICE_PROPRIETARY_ARCHIVE"
fi

cmake -S libs/qec -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-12 \
  -DCMAKE_CXX_COMPILER=g++-12 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCUDAQ_DIR="$cudaq_prefix/lib/cmake/cudaq/" \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON \
  -DCMAKE_INSTALL_PREFIX="$install_prefix" \
  -DCUDAQ_REALTIME_ROOT=$CUDAQ_REALTIME_ROOT \
  -DCUDAQX_QEC_ENABLE_REALTIME_PIPELINE=ON \
  -DCUDAQX_QEC_ENABLE_HSB_TOOLS=ON \
  -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$HSB_ROOT \
  -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$HSB_BUILD \
  $_prop_archive_flag

cmake --build "$build_dir" --target install -j 4
