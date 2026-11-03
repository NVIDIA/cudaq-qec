#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

# The offline configuration validator must work on a machine that has the
# installed CUDA-Q QEC libraries but no Python package: it loads
# libcudaq-qec-realtime-decoding.so from --lib-dir, reads the JSON Schema
# through its C entry point, and the schema must list every decoder plugin
# that installation has. python3 -I ignores PYTHONPATH, so the only
# requirements are the pure-Python pyyaml and jsonschema packages; the step
# installs them if the image lacks them and skips if that is impossible.
run_config_validator_smoke() {
    local validator=libs/qec/tools/config-validator/validate_decoder_config.py
    local config=libs/qec/tools/decoding-server/decoding_server_config.yaml
    local schema=/tmp/decoder_config.schema.json
    local plugin="$CUDAQX_LIB/decoder-plugins/libcudaq-qec-nv-qldpc-decoder.so"
    echo "Config validator: schema from the installed libraries in $CUDAQX_LIB"
    echo "----------------------------------------------------------------------"
    if ! python3 -I -c "import yaml, jsonschema" 2>/dev/null; then
        python3 -m pip install -q pyyaml jsonschema 2>/dev/null || true
    fi
    if ! python3 -I -c "import yaml, jsonschema" 2>/dev/null; then
        echo "pyyaml/jsonschema unavailable; config validator smoke test skipped"
        return
    fi
    rm -f "$schema"
    if ! LD_LIBRARY_PATH="$CUDAQX_LIB:${CUDAQ_REALTIME_ROOT:-/tmp/cudaq-realtime}/lib:$LD_LIBRARY_PATH"          python3 -I "$validator" --lib-dir "$CUDAQX_LIB" --export-schema "$schema"; then
        echo "Schema export with --lib-dir failed"
        FAILED_TESTS+=("config-validator --lib-dir export")
        return
    fi
    if [ -e "$plugin" ]; then
        # The plugin sits next to the libraries, so the schema must know it.
        if ! grep -q '"nv-qldpc-decoder"' "$schema"; then
            echo "nv-qldpc-decoder plugin is installed but missing from the schema"
            FAILED_TESTS+=("config-validator --lib-dir plugin listing")
            return
        fi
        echo "nv-qldpc-decoder plugin listed in the schema"
    fi
    if ! python3 -I "$validator" --schema "$schema" "$config"; then
        echo "Example configuration failed validation"
        FAILED_TESTS+=("config-validator example config")
        return
    fi
    echo ""
}

if [[ "$LIB" == "qec" || "$LIB" == "all" ]]; then
    echo "Running QEC examples..."
    echo "------------------------"

    run_config_validator_smoke
    
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
