#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Upgrade the rdma-core runtime packages (libibverbs and friends) from the DOCA
# apt repository.
#
# Workaround: the cuda-quantum-devcontainer ships a Mellanox OFED libibverbs
# that predates ibv_reg_dmabuf_mr, which HSB's RoCE transport (used by CUDA-Q
# realtime's cpu_roce) requires. Builds get a newer libibverbs as a side effect
# of CUDA-Q's install_devdeps.sh; test containers that only consume build
# artifacts need the same runtime libraries or the Python module fails to load
# with "version `IBVERBS_1.12' not found". Any rdma-core providing that symbol
# version works, so DOCA_VERSION only needs to be new enough, not an exact
# match. Drop this once the devcontainer ships a dmabuf-capable rdma-core.

set -euo pipefail

DOCA_VERSION=${DOCA_VERSION:-3.4.0}
DOCA_ARCH=$(uname -m)
case "$DOCA_ARCH" in aarch64|arm64) DOCA_ARCH="arm64-sbsa" ;; esac
echo "Installing rdma-core runtime from DOCA $DOCA_VERSION ($DOCA_ARCH)"

wget -qO /usr/share/keyrings/nvidia-doca.asc \
  https://linux.mellanox.com/public/repo/doca/public_keys/nvidia-doca-rpm-gpg-public-key.asc
echo "deb [signed-by=/usr/share/keyrings/nvidia-doca.asc] https://linux.mellanox.com/public/repo/doca/$DOCA_VERSION/ubuntu24.04/$DOCA_ARCH /" \
  > /etc/apt/sources.list.d/doca.list
apt-get update
apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers librdmacm1
rm -f /etc/apt/sources.list.d/doca.list /usr/share/keyrings/nvidia-doca.asc
