# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

ARG base_image=ghcr.io/nvidia/cuda-quantum-devdeps:manylinux-amd64-cu12.6-gcc12-main
FROM ${base_image}

ARG python_version=3.12
ARG cuda_version=12.6

LABEL org.opencontainers.image.description="Dev tools for building and testing CUDA-QX libraries"
LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudaqx"
LABEL org.opencontainers.image.title="cudaqx-dev"
LABEL org.opencontainers.image.url="https://github.com/NVIDIA/cudaqx"

ENV CUDAQ_INSTALL_PREFIX=/usr/local/cudaq

RUN CUDA_DASH=$(echo $cuda_version | tr '.' '-') \
  && dnf install -y jq cuda-nvtx-${CUDA_DASH}
RUN mkdir -p /workspaces/cudaqx
COPY .cudaq_version /workspaces/cudaqx
COPY .github/workflows/scripts/build_cudaq.sh /workspaces/cudaqx
RUN mkdir /cudaq-wheels
COPY cudaq-wheels/ /cudaq-wheels/

RUN mkdir -p /workspaces/cudaqx/cudaq && cd /workspaces/cudaqx/cudaq \
  && git init \
  && CUDAQ_REPO=$(jq -r '.cudaq.repository' /workspaces/cudaqx/.cudaq_version) \
  && CUDAQ_COMMIT=$(jq -r '.cudaq.ref' /workspaces/cudaqx/.cudaq_version) \
  && git remote add origin https://github.com/${CUDAQ_REPO} \
  && git fetch -q --depth=1 origin ${CUDAQ_COMMIT} \
  && git reset --hard FETCH_HEAD \
  && cd .. \
  && bash build_cudaq.sh --python-version ${python_version} --cuda-version ${cuda_version} \
  && rm -rf cudaq
