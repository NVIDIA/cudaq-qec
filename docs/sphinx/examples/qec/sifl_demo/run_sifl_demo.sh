#!/bin/bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
#
# run_sifl_demo.sh [options]
#
# Runs the Streaming Interleaved Feed-forward Latency (SIFL) demo. Builds the
# per_round_decoder plugin next to sifl_demo.py, runs the demo with the given
# options (see --help), and removes the plugin on exit.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if ! INSTALL_PREFIX="$(python3 -c \
    'import cudaq_qec, os; print(os.path.dirname(os.path.dirname(cudaq_qec.__file__)))' \
    2>/dev/null)"; then
    echo "ERROR: cudaq_qec is not importable; set up the CUDA-Q QEC environment first." >&2
    exit 1
fi

PLUGIN="${SCRIPT_DIR}/libper_round_decoder.so"
trap 'rm -f "${PLUGIN}"' EXIT

echo "Building per_round_decoder..."
g++ -std=c++17 -shared -fPIC "${SCRIPT_DIR}/per_round_decoder.cpp" \
    -I"${INSTALL_PREFIX}/include" -L"${INSTALL_PREFIX}/lib" \
    -lcudaq-qec-decoders -o "${PLUGIN}"

python3 "${SCRIPT_DIR}/sifl_demo.py" "$@"
