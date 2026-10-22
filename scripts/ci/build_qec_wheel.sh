#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2022 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# /!\ This script must be run inside an docker container /!\

python_version=3.12
python=python${python_version}

# Delete previous wheels
rm wheels/cudaq_qec*.whl

# Exit immediately if any command returns a non-zero status
set -e

git config --global --add safe.directory /cuda-qx

cd /cuda-qx/libs/qec

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-12/enable

export CC=gcc
export CXX=g++

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$HOME/.cudaq/lib/cmake/cudaq;-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-12/root/usr/lib/gcc/x86_64-redhat-linux/12/" \
$python -m build --wheel

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/*.whl \
  --exclude libcudaq-em-default.so \
  --exclude libcudaq-python-interop.so \
  --exclude libcudaq-ensmallen.so \
  --exclude libcudaq-common.so \
  --exclude libcudaq-platform-default.so \
  --exclude libnvqir-qpp.so \
  --exclude libnvqir.so \
  --exclude libcudaq.so \
  --exclude libcudaq-operator.so \
  --exclude libcudaq-nlopt.so \
  --wheel-dir /wheels

