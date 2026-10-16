#!/bin/sh

# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Set PATH
export PATH="/cudaq-install/bin:$PATH"
export PYTHONPATH="/cudaq-install:$HOME/.cudaqx"
echo "Setting PYTHONPATH=$PYTHONPATH"

# Set CUDA-QX paths for nvq++
CUDAQX_INCLUDE="$HOME/.cudaqx/include"
CUDAQX_LIB="$HOME/.cudaqx/lib"

LIB=$1  # Accepts "qec" or "all"

echo "Running example tests for $LIB..."
echo "----------------------------------"

# List to track failed tests
FAILED_TESTS=()

run_python_test() {
    local file=$1
    echo "Running Python example: $file"
    echo "------------------------------"
    python3 "$file"
    if [ $? -ne 0 ]; then
        echo "Python test failed: $file"
        FAILED_TESTS+=("$file")
    fi
    echo ""
}

run_cpp_test() {
    local file=$1
    local lib_flag=$2
    echo "Compiling and running C++ example: $file"
    echo "-----------------------------------------"
    
    nvq++ $lib_flag \
        -I"$CUDAQX_INCLUDE" -L"$CUDAQX_LIB" -Wl,-rpath,"$CUDAQX_LIB" \
        "$file"
    
    if [ $? -ne 0 ]; then
        echo "Compilation failed: $file"
        FAILED_TESTS+=("$file")
        return
    fi
    
    ./a.out
    if [ $? -ne 0 ]; then
        echo "Execution failed: $file"
        FAILED_TESTS+=("$file")
    fi
    echo ""
}

if [[ "$LIB" == "qec" || "$LIB" == "all" ]]; then
    echo "Running QEC examples..."
    echo "------------------------"
    
    for file in examples/qec/python/*.py; do
        run_python_test "$file"
    done
    
    for file in examples/qec/cpp/*.cpp; do
        # Get the filename without the path.
        filename=$(basename $file)
        # If the cpp file contains an nvq++ command, fetch the command line
        # options from it and use them here. If there is no nvq++ command, use
        # the default options.
        nvqpp_options=$(grep nvq++ $file | sed -re "s/.*nvq\+\+ //" | sed -re "s/ $filename//")
        if [ -n "$nvqpp_options" ]; then
            run_cpp_test "$file" "$nvqpp_options"
        else
            run_cpp_test "$file" "--target=stim -lcudaq-qec"
        fi
    done
fi

# Final summary
if [ ${#FAILED_TESTS[@]} -ne 0 ]; then
    echo "========================================"
    echo "Some tests failed:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "- $test"
    done
    echo "========================================"
    exit 1
else
    echo "All tests passed successfully!"
    exit 0
fi
