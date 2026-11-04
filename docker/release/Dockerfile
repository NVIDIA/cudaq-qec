# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# To build the release image run the following from the top-level of the repo (must clone the cudaq-qec repo)
#
# Download the Docker overlay files, may be on a different repo, so adjust this
# command as necessary.
# gh release download -R NVIDIA/cudaq-qec docker-files-12345
#
# docker build -t ghcr.io/nvidia/cudaq-qec:latest . -f docker/release/Dockerfile
#
# Test the image with
#
# scripts/validation/container/validate_container.sh

ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
ARG cuda_version=12.6
# DOCA release to take libibverbs from. Keep in step with DOCA_VERSION in
# cuda-quantum's realtime/scripts/deps_common.sh at the .cudaq_version pin.
ARG doca_version=3.4.0
FROM $base_image

# Set build-time variables for architecture-specific files
ARG TARGETARCH

ARG cuda_version

USER root

# Update the copyright notification:
RUN echo -e "This container also includes CUDA-Q QEC.\n"\
"Use of this container implies consent to the NVIDIA Software License agreement at\n"\
"https://github.com/NVIDIA/cudaqx/blob/main/libs/qec/LICENSE\n" >> "$CUDA_QUANTUM_PATH/Copyright.txt"

# Determine the appropriate zip file
COPY installed_files-${TARGETARCH}-cu${cuda_version}.zip /tmp/

# Overlay installed files
RUN apt-get update && apt-get install -y unzip && \
    unzip /tmp/installed_files-${TARGETARCH}-cu${cuda_version}.zip -d /opt/nvidia/cudaq && \
    rm /tmp/installed_files-${TARGETARCH}-cu${cuda_version}.zip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# The QEC Python extension links the playback emulator's cpu_roce transport
# against libibverbs and needs IBVERBS_1.12. The CUDA-Q base image carries the
# Mellanox OFED 5.3 libibverbs (IBVERBS_1.11), so upgrade rdma-core from the
# DOCA release CUDA-Q pins, as the dev and decoding-server images do.
ARG doca_version
RUN set -e; \
    DOCA_ARCH=$(uname -m); case "$DOCA_ARCH" in aarch64|arm64) DOCA_ARCH="arm64-sbsa" ;; esac; \
    curl -fsSL https://linux.mellanox.com/public/repo/doca/public_keys/nvidia-doca-rpm-gpg-public-key.asc \
        -o /usr/share/keyrings/nvidia-doca.asc; \
    echo "deb [signed-by=/usr/share/keyrings/nvidia-doca.asc] https://linux.mellanox.com/public/repo/doca/${doca_version}/ubuntu24.04/${DOCA_ARCH} /" \
        > /etc/apt/sources.list.d/doca.list; \
    apt-get update && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers; \
    rm -f /etc/apt/sources.list.d/doca.list /usr/share/keyrings/nvidia-doca.asc; \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy CUDA-QX examples and tests
COPY docs/sphinx/examples /home/cudaq/cudaqx-examples
COPY libs/qec/python/tests /home/cudaq/cudaqx_pytests/qec
# Reference data the Python tests resolve relative to their own location
# (tests/../../test_data), e.g. test_chromobius.py.
COPY libs/qec/test_data /home/cudaq/cudaqx_pytests/test_data

# Bake in CUDA-Q import check helper (used by QA validation scripts)
RUN mkdir -p /home/cudaq/cudaqx-scripts/validation
COPY scripts/validation/container/check_cudaq_import.py \
     /home/cudaq/cudaqx-scripts/validation/check_cudaq_import.py
RUN chown -R cudaq:cudaq /home/cudaq/cudaqx-scripts

# Set permissions after copy (important)
RUN chown -R cudaq /home/cudaq && chgrp -R cudaq /home/cudaq

USER cudaq

WORKDIR /home/cudaq
ENTRYPOINT ["bash", "-l"]
