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
  && git reset --hard FETCH_HEAD \
  && CUDAQ_SRC=/workspaces/cudaq bash /usr/local/bin/build_cudaq_with_realtime.sh \
  && rm -rf build realtime/build

#RUN mkdir -p /workspaces/cudaqx && cd /workspaces/cudaqx \
#  && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDAQ_DIR=/usr/local/cudaq/lib/cmake/cudaq .. \
#  && ninja install

FROM cudaq-${build_cudaq}
