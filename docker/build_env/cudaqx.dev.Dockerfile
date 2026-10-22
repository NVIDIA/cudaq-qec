# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

ARG base_image=ghcr.io/nvidia/cuda-quantum-devcontainer:amd64-cu12.6-gcc12-main
# Selects the final stage below; must be lowercase, since stage names are.
ARG build_cudaq=on
FROM $base_image AS deps

ARG cuda_version=12.6

LABEL org.opencontainers.image.description="Dev tools for building and testing CUDA-QX libraries"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudaqx"
LABEL org.opencontainers.image.title="cudaqx-dev"
LABEL org.opencontainers.image.url="https://github.com/NVIDIA/cudaqx"

RUN apt-get update && CUDA_DASH=$(echo $cuda_version | tr '.' '-') \
  && apt-get install -y jq cuda-nvtx-${CUDA_DASH} \
  && apt-get install -y git-lfs \
  && apt-get autoremove -y --purge && apt-get clean && rm -rf /var/lib/apt/lists/*

# build_cudaq=off: dev tools only, with no CUDA-Q source, build scripts, or
# install prefix baked in, so CUDA-Q can be built anywhere later.
FROM deps AS cudaq-off

FROM deps AS cudaq-on

COPY .cudaq_version /cudaq_version
COPY scripts/build_cudaq_with_realtime.sh /usr/local/bin/build_cudaq_with_realtime.sh
COPY scripts/cudaq_realtime_cmake_flags.sh /usr/local/bin/cudaq_realtime_cmake_flags.sh

ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

RUN mkdir -p /workspaces/cudaq && cd /workspaces/cudaq \
  && git init \
  && CUDAQ_REPO=$(jq -r '.cudaq.repository' /cudaq_version) \
  && CUDAQ_COMMIT=$(jq -r '.cudaq.ref' /cudaq_version) \
  && git remote add origin https://github.com/${CUDAQ_REPO} \
  && git fetch -q --depth=1 origin ${CUDAQ_COMMIT} \
  && git reset --hard FETCH_HEAD

# CUDA-Q realtime's cpu_roce transport requires HSB's HololinkRoce package.
# Build that CUDA-free leaf at the ref CUDA-Q pins and install it under
# /usr/local. It needs a newer libibverbs than the base image's OFED one,
# taken from the DOCA repo CUDA-Q pins.
RUN set -e; \
  deps=/workspaces/cudaq/realtime/scripts/deps_common.sh; \
  DOCA_VERSION=$(sed -n 's/^DOCA_VERSION=\${DOCA_VERSION:-\(.*\)}/\1/p' "$deps"); \
  HSB_REPO=$(sed -n 's/^CUDAQ_REALTIME_HSB_REPO=\${CUDAQ_REALTIME_HSB_REPO:-\(.*\)}/\1/p' "$deps"); \
  HSB_REF=$(sed -n 's/^CUDAQ_REALTIME_HSB_REF=\${CUDAQ_REALTIME_HSB_REF:-\(.*\)}/\1/p' "$deps"); \
  test -n "$DOCA_VERSION" && test -n "$HSB_REPO" && test -n "$HSB_REF"; \
  DOCA_ARCH=$(uname -m); case "$DOCA_ARCH" in aarch64|arm64) DOCA_ARCH="arm64-sbsa" ;; esac; \
  echo "== rdma-core from DOCA $DOCA_VERSION; HololinkRoce from $HSB_REPO @ $HSB_REF"; \
  wget -qO /usr/share/keyrings/nvidia-doca.asc \
    https://linux.mellanox.com/public/repo/doca/public_keys/nvidia-doca-rpm-gpg-public-key.asc; \
  echo "deb [signed-by=/usr/share/keyrings/nvidia-doca.asc] https://linux.mellanox.com/public/repo/doca/$DOCA_VERSION/ubuntu24.04/$DOCA_ARCH /" \
    > /etc/apt/sources.list.d/doca.list; \
  apt-get update; \
  apt-get install -y --no-install-recommends \
    libibverbs-dev libibverbs1 ibverbs-providers ibverbs-utils rdma-core librdmacm1 libibumad3 libibmad5; \
  apt-get clean && rm -rf /var/lib/apt/lists/*; \
  git clone -q --depth 1 --branch "$HSB_REF" "$HSB_REPO" /tmp/hsb; \
  cmake -G Ninja -S /tmp/hsb/src/hololink/transport/roce -B /tmp/hsb/build-roce \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local; \
  cmake --build /tmp/hsb/build-roce --target install; \
  test -f /usr/local/lib/cmake/HololinkRoce/HololinkRoceConfig.cmake; \
  rm -rf /tmp/hsb /etc/apt/sources.list.d/doca.list /usr/share/keyrings/nvidia-doca.asc

RUN cd /workspaces/cudaq \
  && CUDAQ_SRC=/workspaces/cudaq bash /usr/local/bin/build_cudaq_with_realtime.sh \
  && rm -rf build realtime/build

#RUN mkdir -p /workspaces/cudaqx && cd /workspaces/cudaqx \
#  && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq .. \
#  && ninja install

FROM cudaq-${build_cudaq}
