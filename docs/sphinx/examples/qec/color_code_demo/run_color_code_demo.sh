#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# run_color_code_demo.sh [options]
#
# Runs the Ising color-code decoder demo. Fetches Ising-Decoding into
# ${COLOR_CODE_DEMO_DEPS} (default ./deps), then runs color_code_demo.py with
# the given options (see --help). Pass the model weights with --weights.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="${COLOR_CODE_DEMO_DEPS:-${PWD}/deps}"
ISING_REPOSITORY="https://github.com/NVIDIA/Ising-Decoding.git"
ISING_COMMIT="33acb152e403bc189f2effdb07f1a87b34c745f1"

if ! python3 -c 'import cudaq_qec' 2>/dev/null; then
    echo "ERROR: cudaq_qec is not importable; set up the CUDA-Q QEC environment first." >&2
    exit 1
fi

case " $* " in
    *" -h "* | *" --help "*) exec python3 "${SCRIPT_DIR}/color_code_demo.py" --help ;;
esac

ISING="${DEPS}/Ising-Decoding"
if [ ! -d "${ISING}/.git" ]; then
    echo "Fetching Ising-Decoding..."
    git clone -q "${ISING_REPOSITORY}" "${ISING}"
    git -C "${ISING}" checkout -q "${ISING_COMMIT}"
fi

python3 "${SCRIPT_DIR}/color_code_demo.py" --ising "${ISING}" "$@"
