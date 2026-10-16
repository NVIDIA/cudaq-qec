#!/bin/sh

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# /!\ This script must be run inside an docker container /!\

mkdir /wheels

git clone --filter=tree:0 https://github.com/NVIDIA/cuda-quantum.git /cuda-quantum
cd /cuda-quantum

export CUDA_VERSION=12.6

# We need to use a newer toolchain because CUDA-QX libraries rely on c++20
source /opt/rh/gcc-toolset-12/enable

export CC=gcc
export CXX=g++

# ==============================================================================
# Installing dependencies
# ==============================================================================

python_version=3.12
python=python${python_version}
${python} -m pip install --no-cache-dir numpy auditwheel

echo "Building MLIR bindings for ${python}" && \
    rm -rf "$LLVM_INSTALL_PREFIX/src" "$LLVM_INSTALL_PREFIX/python_packages" && \
    Python3_EXECUTABLE="$(which ${python})" \
    LLVM_PROJECTS='clang;mlir;python-bindings' \
    LLVM_CMAKE_CACHE=/cmake/caches/LLVM.cmake LLVM_SOURCE=/llvm-project \
    bash /scripts/build_llvm.sh -c Release -v 

# ==============================================================================
# Building CUDA-Q wheel
# ==============================================================================

echo "Building CUDA-Q wheel for ${python}."
cd /cuda-quantum

# Select the correct pyproject.toml file.
rm -f pyproject.toml # remove the symlink if it exists
cp pyproject.toml.cu${CUDA_VERSION} pyproject.toml

# Build the wheel
echo "Building wheel for python${python_version}."

# Find any external NVQIR simulator assets to be pulled in during wheel packaging.
export CUDAQ_EXTERNAL_NVQIR_SIMS=$(bash scripts/find_wheel_assets.sh assets)
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/assets"
export CUQUANTUM_INSTALL_PREFIX=/usr/local/cuquantum
export CUTENSOR_INSTALL_PREFIX=/usr/local/cutensor

bash scripts/configure_build.sh install-cuquantum
bash scripts/configure_build.sh install-cutensor

SETUPTOOLS_SCM_PRETEND_VERSION=${CUDA_QUANTUM_VERSION:-0.0.0} \
CUDACXX="$CUDA_INSTALL_PREFIX/bin/nvcc" CUDAHOSTCXX=$CXX \
$python -m build --wheel

cudaq_major=$(echo ${CUDA_VERSION} | cut -d . -f1)

LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/_skbuild/lib" \
$python -m auditwheel -v repair dist/cuda_quantum*linux_*.whl \
  --exclude libcustatevec.so.1 \
  --exclude libcutensornet.so.2 \
  --exclude libcublas.so.$cudaq_major \
  --exclude libcublasLt.so.$cudaq_major \
  --exclude libcusolver.so.$cudaq_major \
  --exclude libcutensor.so.2 \
  --exclude libnvToolsExt.so.1 \
  --exclude libcudart.so.$cudaq_major.0 \
  --wheel-dir /wheels

# ==============================================================================
# Building CUDA-Q
# ==============================================================================

echo "Building CUDA-Q."
cd /cuda-quantum

# Link the Python bindings against Python3::Module rather than
# Python3::Python, so the wheel does not hard-link libpython.  Line-local
# substitutions rather than a `git apply` patch, for the reasons documented
# in .github/workflows/scripts/build_cudaq.sh.
sed -i \
  's/find_package(Python 3 COMPONENTS Interpreter Development)/find_package(Python 3 COMPONENTS Interpreter Development.Module)/;
   s/find_package(Python3 COMPONENTS Interpreter Development)/find_package(Python3 COMPONENTS Interpreter Development.Module)/' \
  CMakeLists.txt

sed -i 's/nanobind-static Python3::Python/nanobind-static Python3::Module/' \
  python/runtime/cudaq/domains/plugins/CMakeLists.txt

# Require the substituted forms to be present (not just the old forms gone):
# if upstream reshapes these lines, sed matches nothing and the build would
# otherwise silently produce a wheel that hard-links libpython.
plugins_cmake=python/runtime/cudaq/domains/plugins/CMakeLists.txt
if ! grep -qF 'find_package(Python 3 COMPONENTS Interpreter Development.Module)' CMakeLists.txt ||
   ! grep -qF 'find_package(Python3 COMPONENTS Interpreter Development.Module)' CMakeLists.txt ||
   ! grep -qF 'nanobind-static Python3::Module' "$plugins_cmake" ||
   grep -qF 'COMPONENTS Interpreter Development)' CMakeLists.txt ||
   grep -qF 'nanobind-static Python3::Python' "$plugins_cmake"; then
  echo "build_cudaq_wheel: Python component substitution did not take effect;" >&2
  echo "  the cuda-quantum CMake files have changed shape and this script" >&2
  echo "  needs updating." >&2
  exit 1
fi

$python -m venv --system-site-packages .venv
source .venv/bin/activate
CUDAQ_BUILD_TESTS=FALSE bash scripts/build_cudaq.sh -v
