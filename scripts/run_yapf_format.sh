#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage:
# bash scripts/run_yapf_format.sh
#
# This script will use the yapf executable in your PATH.

cd $(git rev-parse --show-toplevel)

# Run Clang Format
git ls-files -- '*.py' | xargs yapf -i

# Take us back to where we were
cd -
