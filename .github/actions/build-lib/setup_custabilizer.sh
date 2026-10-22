#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared helper used by the CI build scripts to make sure the cuStabilizer
# library (shipped inside the cuquantum-python pip wheel) is installed in the
# active Python environment before CMake configures the QEC library.
#
# CMake's FindcuStabilizer module (cmake/Modules/FindcuStabilizer.cmake) then
# auto-discovers the wheel layout and bakes its lib directory into the
# generated binaries' RPATH, so no LD_LIBRARY_PATH wrangling is required at
# either build time or test time.
#
# Honors a pre-set CUSTABILIZER_ROOT (e.g. for system installs) by skipping
# the pip step.  Must be *sourced*, not executed.

if [ -z "$CUSTABILIZER_ROOT" ] && [ -x "$(command -v pip)" ]; then
  NVCC_BIN=${CUDACXX:-$(command -v nvcc)}
  CUDA_MAJOR=""
  if [ -n "$NVCC_BIN" ] && [ -x "$NVCC_BIN" ]; then
    CUDA_MAJOR=$("$NVCC_BIN" --version | sed -nE 's/.*release ([0-9]+)\..*/\1/p' | head -n 1)
  fi
  pip install "cuquantum-python-cu${CUDA_MAJOR:-12}>=26.3.0"
fi
