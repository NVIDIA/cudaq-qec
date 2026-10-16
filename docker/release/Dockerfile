# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


# To build the release image run the following from the top-level of the repo (must clone the cudaqx repo)
#
# Download the Docker overlay files, may be on a different repo, so adjust this
# command as necessary.
# gh release download -R NVIDIA/cudaqx docker-files-12345
#
# docker build -t ghcr.io/nvidia/cudaqx:latest . -f docker/release/Dockerfile
#
# Test the image with
#
# scripts/validation/container/validate_container.sh

ARG base_image=nvcr.io/nvidia/nightly/cuda-quantum:cu12-latest
ARG cuda_version=12.6
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

# Copy CUDA-QX examples and tests
COPY docs/sphinx/examples /home/cudaq/cudaqx-examples
COPY libs/qec/python/tests /home/cudaq/cudaqx_pytests/qec

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
