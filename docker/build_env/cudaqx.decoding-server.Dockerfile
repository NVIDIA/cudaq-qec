# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Build environment for the CUDA-QX QEC *decoding server*, built on a plain
# ubuntu:24.04 base.  This is the decoders-only build
# (CUDAQ_QEC_DECODERS_ONLY=ON) plus the one extra ingredient the server needs:
# a cudaq-realtime install (libcudaq-realtime.so + the cudaq/realtime headers).
# It still needs no CUDA-Q runtime, no simulators, and no MLIR/Clang.
#
# Where cudaq-realtime comes from
# -------------------------------
# cudaq-realtime is a CUDA-Q build artifact with no apt package, so we pluck
# just its pieces out of the CUDA-Q nightly runtime image (first stage below)
# and drop them into a small prefix at /opt/cudaq-realtime.  Only these files
# make up the "cudaq-realtime installation" the server links against:
#   * include/cudaq/realtime/**  -- the dispatcher, bridge and device_call
#     headers (self-contained: they include nothing outside cudaq/realtime)
#   * lib/libcudaq-realtime.so   -- the bridge loader + dispatcher C API
#     (NEEDED: only libstdc++/libc; it does not even pull in cudart)
#   * lib/libcudaq-realtime-dispatch.a -- device-graph scheduler shims, located
#     (not linked here) so device_graph dispatch compiles
#   * lib/libcudaq-realtime-bridge-{udp,cpu-roce}.so -- transport providers the
#     server dlopen()s at runtime for --transport=udp / cpu_roce
#   * lib/libcudaq-realtime-{host-dispatch,udp-transport,cpu-roce-transport}.a
#     -- internal transport archives, copied to keep the prefix self-consistent
# We deliberately do NOT copy the rest of the CUDA-Q prefix (~767M): no
# simulators, no compiler, no cudaq runtime.
#
# LLVM
# ----
# The server's only LLVM use is llvm::yaml / llvm::json in the realtime config
# parser (LLVMSupport) -- no MLIR, no Clang.  But config.cpp calls
# llvm::yaml::IO::error(), which the base IO class only gained in LLVM 21+, so
# Ubuntu's own llvm-dev (18/19/20) is too old.  We therefore install the
# prebuilt llvm-22-dev from apt.llvm.org: still just the LLVMSupport library +
# headers (not a from-source toolchain, not the NVIDIA LLVM fork), matching the
# LLVM 22 the code was written against.  This relies on the relaxed (non-exact)
# LLVM version requirement in libs/qec/lib/realtime/CMakeLists.txt.
#
# CUDA
# ----
# Unlike the decoders-only build, the server build compiles CUDA device code
# (lib/realtime/gpu_kernels.cu), so it needs nvcc + cudart, not just cudart.
#
# Build the image:
#   docker build -f docker/build_env/cudaqx.decoding-server.Dockerfile \
#     -t cudaqx-decoding-server-build .
#
# Then, with the repo mounted at /workspaces/cudaqx:
#   cmake -S libs/qec -B build_server -G Ninja \
#     -DCMAKE_BUILD_TYPE=Release \
#     -DCUDAQ_QEC_DECODERS_ONLY=ON \
#     -DCUDAQ_REALTIME_ROOT=/opt/cudaq-realtime \
#     -DLLVM_DIR="$LLVM_DIR" \
#     -DLLVM_VERSION_MAJOR="$LLVM_VERSION_MAJOR" -DLLVM_VERSION_MINOR="$LLVM_VERSION_MINOR"
#   cmake --build build_server -j$(nproc) --target decoding_server

# Stage 1: source of the cudaq-realtime artifacts (COPY --from only).
ARG cudaq_image=nvcr.io/nvidia/nightly/cuda-quantum:cu13-latest
FROM ${cudaq_image} AS cudaq_realtime

# Stage 2: the ubuntu:24.04 build environment.
FROM ubuntu:24.04

# CUDA toolkit version (cudart + nvcc). The realtime .so is CUDA-version
# agnostic, so this only governs the device code we compile ourselves.
ARG cuda_version=12.6
# LLVM (from apt.llvm.org) used only for LLVMSupport (llvm::yaml / llvm::json).
# Must be >= 21 for llvm::yaml::IO::error(); 22 matches the pinned version the
# realtime config parser was written against.
ARG llvm_version=22

LABEL org.opencontainers.image.description="Ubuntu 24.04 build env for the CUDA-QX QEC decoding server (plucked cudaq-realtime, distro LLVM, no CUDA-Q)"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudaqx"
LABEL org.opencontainers.image.title="cudaqx-decoding-server-build"
LABEL org.opencontainers.image.url="https://github.com/NVIDIA/cudaqx"

ENV DEBIAN_FRONTEND=noninteractive

# Base build tooling, the decoder stack's link-time deps (BLAS/LAPACK for
# xtensor-blas in core), Python for the optional bindings, and llvm-dev for
# LLVMSupport.
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
      libzstd-dev \
      zlib1g-dev \
      libz3-dev \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# LLVMSupport from apt.llvm.org (prebuilt; provides llvm::yaml::IO::error(),
# absent from Ubuntu's own llvm <= 20). No MLIR/Clang -- just the dev package.
#
# llvm-<v>-dev is huge (~450MB of static component libs + a 143MB shared
# libLLVM + 91MB of tools), but the server links only libLLVMSupport.a (~6MB)
# and needs the headers + CMake package files.  We reclaim the rest by
# TRUNCATING the unused files to 0 bytes -- not deleting them: LLVM's
# LLVMExports*.cmake runs an existence check over every imported target's file
# and FATAL_ERRORs on a missing one, even for components we never link, so the
# files must still exist.  Truncating keeps them present but empty.  This is
# done in the same RUN as the install so the layer only ever stores the
# shrunken tree (a later-layer delete would leave the 450MB in the lower
# layer).  Side effect: dpkg still believes those files are full-size; harmless
# for a build image.  Net LLVM footprint drops from ~590MB to ~48MB (mostly the
# 40MB of headers, which a build environment must keep).
RUN wget -q -O /etc/apt/trusted.gpg.d/apt.llvm.org.asc \
       https://apt.llvm.org/llvm-snapshot.gpg.key \
  && . /etc/os-release \
  && echo "deb http://apt.llvm.org/${VERSION_CODENAME}/ llvm-toolchain-${VERSION_CODENAME}-${llvm_version} main" \
       > /etc/apt/sources.list.d/llvm.list \
  && apt-get update && apt-get install -y --no-install-recommends \
       llvm-${llvm_version}-dev \
  && find /usr/lib/llvm-${llvm_version}/lib -type f ! -path '*/cmake/*' \
       ! -name 'libLLVMSupport.a' ! -name 'libLLVMDemangle.a' \
       -exec truncate -s 0 {} + \
  && find /usr/lib/llvm-${llvm_version}/bin -type f -exec truncate -s 0 {} + \
  && find /usr/lib -maxdepth 2 -type f -name 'libLLVM*.so*' \
       -exec truncate -s 0 {} + \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# CUDA cudart + nvcc from the NVIDIA apt repo (nvcc is needed here: the server
# build compiles lib/realtime/gpu_kernels.cu).
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

# Pluck the cudaq-realtime install (headers + libs only) into a small prefix.
COPY --from=cudaq_realtime \
       /opt/nvidia/cudaq/include/cudaq/realtime \
       /opt/cudaq-realtime/include/cudaq/realtime
COPY --from=cudaq_realtime \
       /opt/nvidia/cudaq/lib/libcudaq-realtime.so \
       /opt/nvidia/cudaq/lib/libcudaq-realtime-dispatch.a \
       /opt/nvidia/cudaq/lib/libcudaq-realtime-bridge-udp.so \
       /opt/nvidia/cudaq/lib/libcudaq-realtime-bridge-cpu-roce.so \
       /opt/nvidia/cudaq/lib/libcudaq-realtime-host-dispatch.a \
       /opt/nvidia/cudaq/lib/libcudaq-realtime-udp-transport.a \
       /opt/nvidia/cudaq/lib/libcudaq-realtime-cpu-roce-transport.a \
       /opt/cudaq-realtime/lib/

# Make the toolkit, the plucked realtime install, and the distro LLVM
# discoverable to CMake.
ENV PATH=/usr/local/cuda-${cuda_version}/bin:${PATH}
ENV CUDAToolkit_ROOT=/usr/local/cuda-${cuda_version}
ENV CUDAQ_REALTIME_ROOT=/opt/cudaq-realtime
ENV LLVM_DIR=/usr/lib/llvm-${llvm_version}/lib/cmake/llvm
ENV LLVM_VERSION_MAJOR=${llvm_version}
ENV LLVM_VERSION_MINOR=1
ENV PATH=/usr/lib/llvm-${llvm_version}/bin:${PATH}

WORKDIR /workspaces/cudaqx
