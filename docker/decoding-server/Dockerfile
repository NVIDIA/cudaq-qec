# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is a temporary, minimal image used to validate the decoding-server
# multi-platform publication workflow while getting the CI wiring in place.
FROM ubuntu:24.04

ARG TARGETARCH
ARG CUDA_VERSION

RUN printf '%s\n' \
      'CUDA-QX decoding-server image wiring test' \
      "architecture=${TARGETARCH}" \
      "cuda=${CUDA_VERSION}" \
      > /etc/cudaqx-decoding-server-image

CMD ["cat", "/etc/cudaqx-decoding-server-image"]
