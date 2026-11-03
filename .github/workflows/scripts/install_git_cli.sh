#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2022 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if grep -q AlmaLinux /etc/os-release; then
    ARCH=$(uname -m)
    # Determine the correct download URL based on architecture
    case "$ARCH" in
        x86_64)
            GH_ARCH="linux_amd64"
            ;;
        aarch64)
            GH_ARCH="linux_arm64"
            ;;
        *)
            echo "Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    # Download and install gh
    GH_VERSION="2.67.0"
    GH_URL="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_${GH_ARCH}.rpm"

    echo "Downloading GitHub CLI ($GH_ARCH)..."
    curl -LO "$GH_URL"

    echo "Installing GitHub CLI..."
    rpm -i --nodeps "gh_${GH_VERSION}_${GH_ARCH}.rpm"
else
    (type -p wget >/dev/null || (apt update && apt-get install wget -y)) \
        && mkdir -p -m 755 /etc/apt/keyrings \
        && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
        && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
        && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
        && apt update \
        && apt install -y --no-install-recommends gh
fi
