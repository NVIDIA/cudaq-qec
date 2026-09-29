#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

# ==============================================================================
# Handling options
# ==============================================================================

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --build-type      Build type (e.g., Release)"
    echo "  --cuda-version    CUDA version to build wheel for (e.g. 12 or 13)"
    echo "  --cudaq-prefix    Path to CUDA-Q's install prefix"
    echo "                    (default: \$HOME/.cudaq)"
    echo "  --python-version  Python version to build wheel for (e.g. 3.11)"
    echo "  --tensorrt-path   Path to TensorRT installation directory"
    echo "                    (default: /trt_download/TensorRT-10.13.3.9)"
    echo "  --devdeps         Build wheels suitable for internal testing"
    echo "                    (not suitable for distribution but sometimes"
    echo "                    helpful for debugging)"
    echo "  --version         Specify version of wheels to produce"
    echo "                    (default: 0.0.0)"
}

parse_options() {
    while (( $# > 0 )); do
        case "$1" in
            --build-type)
                if [[ -n "$2" && "$2" != -* ]]; then
                    build_type=("$2")
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
            --cudaq-prefix)
                if [[ -n "$2" && "$2" != -* ]]; then
                    cudaq_prefix=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --python-version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    python_version=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --tensorrt-path)
                if [[ -n "$2" && "$2" != -* ]]; then
                    tensorrt_path=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --devdeps)
                devdeps=true
                shift 1
                ;;
            --version)
                if [[ -n "$2" && "$2" != -* ]]; then
                    wheels_version=("$2")
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
cudaq_prefix=$HOME/.cudaq
build_type=Release
python_version=3.11
tensorrt_path=/trt_download/TensorRT-10.13.3.9
devdeps=false
wheels_version=0.0.0
cuda_version=12

# Parse options
parse_options "$@"

echo "Building in $build_type mode for Python $python_version, version $wheels_version, CUDA version $cuda_version"

# ==============================================================================
# Helpers
# ==============================================================================

python=python${python_version}
ARCH=$(uname -m)
PLAT_STR=""

if $devdeps; then
  PLAT_STR="--plat manylinux_2_34_x86_64"
else
  # We need to use a newer toolchain because CUDA-QX libraries rely on c++20
  source /opt/rh/gcc-toolset-12/enable
fi

export CC=gcc
export CXX=g++
export SETUPTOOLS_SCM_PRETEND_VERSION=$wheels_version
export CUDAQX_QEC_VERSION=$wheels_version

# ==============================================================================
# cuStabilizer / cuQuantum SDK location
# ==============================================================================
# Install the cuquantum-python pip wheel into the outer interpreter (it bundles
# the cuStabilizer headers and library).  `python -m build` later spawns an
# isolated venv whose Python_EXECUTABLE has no cuquantum-python wheel
# installed, so the Python probe in cmake/Modules/FindcuStabilizer.cmake cannot
# find the library on its own.  Resolve the wheel's install prefix here and
# export CUQUANTUM_ROOT so the isolated build env's CMake invocation picks it
# up.  Honors a pre-set CUQUANTUM_ROOT (e.g. for system installs).
if [ -z "$CUQUANTUM_ROOT" ]; then
  $python -m pip install --upgrade "cuquantum-python-cu${cuda_version}>=26.3.0"
  CUQUANTUM_ROOT=$($python -m pip show "custabilizer-cu${cuda_version}" 2>/dev/null \
                   | sed -nE 's|^Location: (.*)|\1/cuquantum|p')
fi
if [ -z "$CUQUANTUM_ROOT" ] || [ ! -f "$CUQUANTUM_ROOT/include/custabilizer.h" ]; then
  echo "ERROR: could not locate cuStabilizer headers via custabilizer-cu${cuda_version}." >&2
  echo "       Make sure cuquantum-python-cu${cuda_version} is installed for $python." >&2
  exit 1
fi
export CUQUANTUM_ROOT
echo "Using CUQUANTUM_ROOT=$CUQUANTUM_ROOT"

# ==============================================================================
# QEC library
# ==============================================================================

cd libs/qec
cp pyproject.toml.cu${cuda_version} pyproject.toml

SKBUILD_CMAKE_ARGS="-DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq;-DTENSORRT_ROOT=$tensorrt_path"
if ! $devdeps; then
  SKBUILD_CMAKE_ARGS+=";-DCMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN=/opt/rh/gcc-toolset-12/root/usr/lib/gcc/${ARCH}-redhat-linux/12/"
fi
SKBUILD_CMAKE_ARGS+=";-DCMAKE_BUILD_TYPE=$build_type"
export SKBUILD_CMAKE_ARGS
$python -m build --wheel

CUDAQ_EXCLUDE_LIST=$(for f in $(find $cudaq_prefix/lib -name "*.so" -printf "%P\n" | sort); do echo "--exclude $f"; done | tr '\n' ' ')

# We need to exclude a few libraries to prevent auditwheel from mistakenly grafting them into the wheel.
LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib:$tensorrt_path/lib" \
$python -m auditwheel -v repair dist/*.whl $CUDAQ_EXCLUDE_LIST \
  --wheel-dir /wheels \
  --exclude libcudart.so.${cuda_version} \
  --exclude libcustabilizer.so.0 \
  --exclude libnvinfer.so.10 \
  --exclude libnvonnxparser.so.10 \
  --exclude libcudaq-qec.so \
  --exclude libcudaq-qec-realtime-decoding.so \
  ${PLAT_STR}


echo "Wheel builds are complete: "
ls -la /wheels
