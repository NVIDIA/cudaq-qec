#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Exit immediately if any command returns a non-zero status
set -e

# ==============================================================================
# Handling options
# ==============================================================================

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo "  -i, --install     Install libs"
    echo "  --cudaq-prefix    Path to CUDA-Q's install prefix"
    echo "                    (default: \$HOME/.cudaq)"
    echo "  --install-prefix  Path to install prefix"
    echo "                    (default: cudaq-prefix)"
}

parse_options() {
    while (( $# > 0 )); do
        case "$1" in
            -h|--help)
                show_help
                exit 0
                ;;
            -i|--install)
                install=1
                shift 1
                ;;
            --cudaq-prefix)
                if [[ -n "$2" && "$2" != -* ]]; then
                    cudaq_prefix=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            --install-prefix)
                if [[ -n "$2" && "$2" != -* ]]; then
                    install_prefix=("$2")
                    shift 2
                else
                    echo "Error: Argument for $1 is missing" >&2
                    exit 1
                fi
                ;;
            -*)
                echo "Error: Unknown option $1" >&2
                show_help
                exit 1
                ;;
            *)
                echo "Error: Unknown argument $1" >&2
                show_help
                exit 1
                ;;
        esac
    done
}

# Initialize an empty array to store libs names
libs=()
install=0
cudaq_prefix=$HOME/.cudaq

# Parse options
parse_options "$@"

install_prefix=${install_prefix:-$cudaq_prefix}


# ==============================================================================
# Test top-level build
# ==============================================================================

cmake -S . -B "build" \
  -DCUDAQ_DIR=$cudaq_prefix/lib/cmake/cudaq/ \
  -DCUDAQX_ENABLE_LIBS="all" \
  -DCMAKE_INSTALL_PREFIX=$install_prefix \
  -DCUDAQX_INCLUDE_TESTS=ON \
  -DCUDAQX_BINDINGS_PYTHON=ON 
cmake --build "build" -j
cmake --build "build" --target run_tests
cmake --build "build" --target run_python_tests
if [ $install -eq 1 ]; then
  cmake --build "build" --target install
fi

