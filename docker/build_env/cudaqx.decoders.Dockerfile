# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Minimal build environment for the CUDA-QX QEC *decoders-only* build
# (CUDAQ_QEC_DECODERS_ONLY=ON), built on a plain ubuntu:24.04 base -- no
# CUDA-Q and no LLVM/MLIR/Clang required.
#
# What the decoders-only build actually needs (see Building.md, "Decoders-only
# Build"):
#   * a C/C++ toolchain + CMake (>= 3.28, which ubuntu:24.04 ships) + Ninja
#   * git/wget/curl + CA certs: the build FetchContent's Stim, fmt, nanobind,
#     and xtl/xtensor/xtensor-blas from GitHub at configure time
#   * a CUDA toolkit for cudart: libcudaq-qec-decoders.so links CUDA::cudart
#     because decoder::get() validates and pins the "cuda_device_id"
#     construction parameter (cudaGetDeviceCount / cudaSetDevice). This is a
#     real dependency, not just NVTX. nvcc is not used (no .cu is compiled in
#     decoders-only mode) but it is installed so find_package(CUDAToolkit)
#     resolves the toolkit robustly.
#   * BLAS/LAPACK: libcudaqx-core (whole-archived into the decoders .so) uses
#     xtensor-blas (xlinalg.hpp), which links BLAS + LAPACK. libopenblas-dev
#     supplies both (it provides the libblas.so and liblapack.so alternatives),
#     so a separate liblapack-dev is not needed.
#   * Python dev headers + nanobind (fetched) for the optional
#     CUDAQX_QEC_BINDINGS_PYTHON module; numpy is only for the smoke test.
#
# LLVM/MLIR/Clang are deliberately absent: libs/qec/CMakeLists.txt gates them
# out entirely under CUDAQ_QEC_DECODERS_ONLY. (LLVMSupport re-enters only when
# the decoding server is built, which needs a cudaq-realtime install and is not
# part of this image.)
#
# Build the image:
#   docker build -f docker/build_env/cudaqx.decoders.Dockerfile \
#     --build-arg cuda_version=12.6 -t cudaqx-decoders-build .
#
# Then, with the repo mounted at /workspaces/cudaqx:
#   cmake -S libs/qec -B build_decoders_only -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCUDAQ_QEC_DECODERS_ONLY=ON \
#     -DCUDAQX_QEC_BINDINGS_PYTHON=ON
#   cmake --build build_decoders_only -j$(nproc)

FROM ubuntu:24.04

# CUDA toolkit version to install (cudart + nvcc). Any version present in the
# NVIDIA ubuntu2404 repo works; 12.6 matches the repo's primary target.
ARG cuda_version=12.6

LABEL org.opencontainers.image.description="Ubuntu 24.04 build env for the CUDA-QX QEC decoders-only build (no CUDA-Q, no LLVM)"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudaqx"
LABEL org.opencontainers.image.title="cudaq-decoders-build"
LABEL org.opencontainers.image.url="https://github.com/NVIDIA/cudaqx"

ENV DEBIAN_FRONTEND=noninteractive

# Base build tooling and the decoder stack's link-time deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
      git \
      wget \
      curl \
      gnupg \
      build-essential \
      gcc \
      g++ \
      ninja-build \
      cmake \
      patchelf \
      python3 \
      python3-dev \
      python3-pip \
      python3-numpy \
      libopenblas-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# CUDA cudart (+ nvcc for robust toolkit discovery) from the NVIDIA apt repo.
RUN CUDA_DASH=$(echo "${cuda_version}" | tr '.' '-') \
  && arch=$(dpkg --print-architecture) \
  && case "${arch}" in \
       amd64) repo_arch=x86_64 ;; \
       arm64) repo_arch=sbsa ;; \
       *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;; \
     esac \
  && wget -q -O /tmp/cuda-keyring.deb \
       "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/${repo_arch}/cuda-keyring_1.1-1_all.deb" \
  && dpkg -i /tmp/cuda-keyring.deb \
  && rm -f /tmp/cuda-keyring.deb \
  && apt-get update && apt-get install -y --no-install-recommends \
       cuda-nvcc-${CUDA_DASH} \
       cuda-cudart-dev-${CUDA_DASH} \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# Optional TensorRT for the cudaq-qec-trt-decoder plugin (opt-in; default off).
# The plugin's CUDAQ_QEC_BUILD_TRT_DECODER option is AUTO by default, so simply
# having libnvinfer + libnvonnxparser installed (in /usr/{include,lib}/<triple>,
# where these dev packages land and where the plugin searches) makes the build
# pick it up; pass -DCUDAQ_QEC_BUILD_TRT_DECODER=ON at configure time to
# hard-require it. TensorRT comes from the same NVIDIA CUDA apt repo added
# above; apt resolves the latest build, whose CUDA tag must line up with
# cuda_version for the plugin to load at runtime.
ARG install_tensorrt=off
RUN case "${install_tensorrt}" in \
      on|ON|1|true|TRUE|yes|YES) \
        apt-get update && apt-get install -y --no-install-recommends \
          libnvinfer-dev libnvonnxparsers-dev \
        && apt-get clean && rm -rf /var/lib/apt/lists/* ;; \
      *) echo "install_tensorrt=${install_tensorrt}: skipping TensorRT; the cudaq-qec-trt-decoder plugin will be auto-disabled." ;; \
    esac

# Make the toolkit discoverable to CMake's find_package(CUDAToolkit).
ENV PATH=/usr/local/cuda-${cuda_version}/bin:${PATH}
ENV CUDAToolkit_ROOT=/usr/local/cuda-${cuda_version}

WORKDIR /workspaces/cudaqx
