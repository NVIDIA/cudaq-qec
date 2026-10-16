#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# ==============================================================================
# Handling options
# ==============================================================================

set -eo pipefail

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --python-version  Python version to build wheel for (e.g. 3.10)"
    echo "  --cuda-version    CUDA version to build wheel for (e.g. 12.6 or 13.0)"
    echo "  -j                Number of parallel jobs to build CUDA-Q with"
    echo "                    (e.g. 8)"
}

parse_options() {
    while (( $# > 0 )); do
        case "$1" in
            --python-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    python_version=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --cuda-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    cuda_version=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            -j)
                if [[ -n "$2" && "$2" != -* ]]; then
                    num_par_jobs=("$2")
                    cudaq_ninja_jobs_arg="-j $num_par_jobs"
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            -*)
                echo "Error: Unknown option $1" >&2
                show_help
                exit 1
                ;;
            *)
                echo "Error: Unknown argument $1" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

# Defaults
python_version=3.10
cudaq_ninja_jobs_arg=""
cuda_version=12.6

# Parse options
parse_options "$@"


export CUDA_VERSION=${cuda_version}
export CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-12/enable

export CC=gcc
export CXX=g++

python=python${python_version}
${python} -m pip install --no-cache-dir numpy auditwheel

echo "Building CUDA-Q."
cd cudaq

# ==============================================================================
# Building MLIR bindings
# ==============================================================================

echo "Building MLIR bindings for ${python}" && \
    rm -rf "$LLVM_INSTALL_PREFIX/src" "$LLVM_INSTALL_PREFIX/python_packages" && \
    Python3_EXECUTABLE="$(which ${python})" \
    LLVM_PROJECTS='clang;mlir;python-bindings' \
    LLVM_CMAKE_CACHE=/cmake/caches/LLVM.cmake LLVM_SOURCE=/llvm-project \
    bash scripts/build_llvm.sh -c Release -v

# ==============================================================================
# Building CUDA-Q
# ==============================================================================

# Link the Python bindings against Python3::Module rather than
# Python3::Python, so the wheel does not hard-link libpython.
#
# These are line-local substitutions rather than a `git apply` patch on
# purpose: a diff also has to match the surrounding context, which drifts
# independently of the lines we care about.  It broke once already when
# cuda-quantum dropped `cudaq-py-utils` from the cudaq-pyscf link line
# (NVIDIA/cuda-quantum "Use shared libcudaqMLIR dependency everywhere",
# #4928), which sat in the trailing context of the second hunk -- neither
# -C1 nor --3way recovers from that.  Do NOT use `patch` either, which can
# hang on a "File to patch" prompt in CI.
apply_sed() {
  local file="$1" expr="$2" description="$3"
  if [ ! -f "$file" ]; then
    echo "build_cudaq: $file not found; cannot apply '$description'" >&2
    return 1
  fi
  sed -i "$expr" "$file"
}

apply_sed CMakeLists.txt \
  's/find_package(Python 3 COMPONENTS Interpreter Development)/find_package(Python 3 COMPONENTS Interpreter Development.Module)/;
   s/find_package(Python3 COMPONENTS Interpreter Development)/find_package(Python3 COMPONENTS Interpreter Development.Module)/' \
  'Python Development -> Development.Module'

apply_sed python/runtime/cudaq/domains/plugins/CMakeLists.txt \
  's/nanobind-static Python3::Python/nanobind-static Python3::Module/' \
  'cudaq-pyscf Python3::Python -> Python3::Module'

# Fail loudly unless the substituted forms are actually present now and the
# old forms are gone.  Requiring the NEW forms (not just the absence of the
# old ones) matters: if upstream reshapes these lines, sed matches nothing,
# the old-form greps also match nothing, and the build would otherwise
# silently produce a wheel that hard-links libpython.
plugins_cmake=python/runtime/cudaq/domains/plugins/CMakeLists.txt
if ! grep -qF 'find_package(Python 3 COMPONENTS Interpreter Development.Module)' CMakeLists.txt ||
   ! grep -qF 'find_package(Python3 COMPONENTS Interpreter Development.Module)' CMakeLists.txt ||
   ! grep -qF 'nanobind-static Python3::Module' "$plugins_cmake" ||
   grep -qF 'COMPONENTS Interpreter Development)' CMakeLists.txt ||
   grep -qF 'nanobind-static Python3::Python' "$plugins_cmake"; then
  echo "build_cudaq: Python component substitution did not take effect; the" >&2
  echo "  cuda-quantum CMake files have changed shape and this script needs" >&2
  echo "  updating." >&2
  exit 1
fi

$python -m venv --system-site-packages .venv
source .venv/bin/activate
CUDAQ_BUILD_TESTS=FALSE bash scripts/build_cudaq.sh -v ${cudaq_ninja_jobs_arg}
