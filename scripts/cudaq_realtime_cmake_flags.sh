#!/bin/sh

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# CMake flags required when CUDA-QX builds CUDA-Q's realtime dependency.
cudaq_realtime_cmake_cuda_flags() {
  _cudaq_realtime_cuda_flags="${CMAKE_CUDA_FLAGS:-}"
  _cudaq_realtime_host_cxx="${CXX:-c++}"
  _cudaq_realtime_host_cxx_major=$("$_cudaq_realtime_host_cxx" -dumpversion 2>/dev/null | cut -d. -f1)
  _cudaq_realtime_cuda_major=$(nvcc --version 2>/dev/null |
    sed -n 's/^.*release \([0-9][0-9]*\)\..*$/\1/p')

  # CUDA 12's nvcc frontend cannot parse GCC 12's AVX-512 BF16 intrinsic
  # headers. CUDA-Q realtime does not use those intrinsics, so defining the
  # headers' include guards keeps them out of the nvcc parse.
  if [ "$(uname -m)" = x86_64 ] && [ "$_cudaq_realtime_cuda_major" = 12 ] && \
    [ "$_cudaq_realtime_host_cxx_major" = 12 ]; then
    _cudaq_realtime_cuda_flags="${_cudaq_realtime_cuda_flags} -D_AVX512BF16INTRIN_H_INCLUDED -D_AVX512BF16VLINTRIN_H_INCLUDED"
  fi

  printf '%s' "$_cudaq_realtime_cuda_flags"
}
