#!/bin/bash

# ============================================================================ #
# Copyright (c) 2024 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

CUDAQX_INSTALL_PREFIX=${CUDAQX_INSTALL_PREFIX:-"$HOME/.cudaqx"}
DOCS_INSTALL_PREFIX=${DOCS_INSTALL_PREFIX:-"$CUDAQX_INSTALL_PREFIX/docs"}
export PYTHONPATH="$CUDAQX_INSTALL_PREFIX:${PYTHONPATH}"

# Process command line arguments
force_update=""

__optind__=$OPTIND
OPTIND=1
while getopts ":u:" opt; do
  case $opt in
    u) force_update="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

# Need to know the top-level of the repo
working_dir=`pwd`
repo_root=$(git rev-parse --show-toplevel)
docs_exit_code=0 # updated in each step

# Make sure these are full path so that it doesn't matter where we use them
docs_build_output="$repo_root/build/docs"
sphinx_output_dir="$docs_build_output/sphinx"
doxygen_output_dir="$docs_build_output/doxygen"
dialect_output_dir="$docs_build_output/Dialects"

# Clean up any previous build output and then recreate the build directory
rm -rf "$docs_build_output"
mkdir -p "$docs_build_output"

doxygen_exe=doxygen

# Create the conf.py file needed by Sphinx
echo "Generating conf.py ..."
sphinx_conf_in="$repo_root/docs/sphinx/conf.py.in"
sphinx_conf="$repo_root/docs/sphinx/conf.py"

# Verify that the input file exists before proceeding
if [ ! -f "$sphinx_conf_in" ]; then
  echo "Error: Sphinx configuration template '$sphinx_conf_in' does not exist." >&2
  exit 1
fi

# Replace placeholders of the form @VAR@ in the template file with their variable values.
if [ -z "$CUDAQ_INSTALL_DIR" ]; then
  if [ -d "$HOME/.cudaq" ]; then
    CUDAQ_INSTALL_DIR="$HOME/.cudaq"
  elif [ -d "/usr/local/cudaq" ]; then
    CUDAQ_INSTALL_DIR="/usr/local/cudaq"
  else
    echo "Error: CUDAQ not found. Set CUDAQ_INSTALL_DIR to the CUDAQ install prefix." >&2
    (return 0 2>/dev/null) && return 1 || exit 1
  fi
fi
export LD_LIBRARY_PATH="$CUDAQ_INSTALL_DIR/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="$CUDAQ_INSTALL_DIR:${PYTHONPATH}"
CMAKE_BINARY_DIR="$repo_root/build"
SPHINX_SOURCE="$repo_root/docs/sphinx"

sed -e "s|@CUDAQ_INSTALL_DIR@|${CUDAQ_INSTALL_DIR}|g" \
    -e "s|@CMAKE_BINARY_DIR@|${CMAKE_BINARY_DIR}|g" \
    -e "s|@SPHINX_SOURCE@|${SPHINX_SOURCE}|g" \
    "$sphinx_conf_in" > "$sphinx_conf"

echo "Configuration file created at: $sphinx_conf"

# Generate API documentation using Doxygen
echo "Generating XML documentation using Doxygen..."
mkdir -p "${doxygen_output_dir}"
doxygen_input="$repo_root/docs/Doxyfile.in"

# Get all the headers
CUDAQX_ALL_LIBS="qec"
lib_headers=""
lib_headers="$lib_headers $(find "$repo_root/libs/core/include" -name "*.h")"
# Add headers from each library
for lib in $CUDAQX_ALL_LIBS; do
    lib_headers="$lib_headers $(find "$repo_root/libs/${lib}/include" -name "*.h")"
done
doxygen_input=$(echo "$lib_headers" | tr '\n' ' ')

sed -e "s|@DOXYGEN_OUTPUT_DIR@|${doxygen_output_dir}|g" \
    -e "s|@DOXYGEN_INPUT@|${doxygen_input}|g" \
    "$repo_root/docs/Doxyfile.in" > "${doxygen_output_dir}/Doxyfile"

"$doxygen_exe" "${doxygen_output_dir}/Doxyfile" 2> "$logs_dir/doxygen_error.txt" 1> "$logs_dir/doxygen_output.txt"
doxygen_exit_code=$?
if [ ! "$doxygen_exit_code" -eq "0" ]; then
    cat "$logs_dir/doxygen_output.txt" "$logs_dir/doxygen_error.txt"
    echo "Failed to generate documentation using doxygen."
    echo "Doxygen exit code: $doxygen_exit_code"
    docs_exit_code=11
fi

echo "Building CUDA-QX documentation using Sphinx..."
cd "$repo_root/docs"
# The docs build so far is fast such that we do not care about the cached outputs.
# Revisit this when caching becomes necessary.

rm -rf sphinx/_doxygen/
rm -rf sphinx/_mdgen/
cp -r "$doxygen_output_dir" sphinx/_doxygen/
# cp -r "$dialect_output_dir" sphinx/_mdgen/ # uncomment once we use the content from those files

rm -rf "$sphinx_output_dir"
sphinx-build -v -n  --keep-going -b html \
  -Dbreathe_projects.cudaqx="${doxygen_output_dir}/xml" \
  sphinx "$sphinx_output_dir" -j auto #2> "$logs_dir/sphinx_error.txt" 1> "$logs_dir/sphinx_output.txt"

sphinx_exit_code=$?
if [ ! "$sphinx_exit_code" -eq "0" ]; then
    echo "Failed to generate documentation using sphinx-build."
    echo "Sphinx exit code: $sphinx_exit_code"
    echo "======== logs ========"
    cat "$logs_dir/sphinx_output.txt" "$logs_dir/sphinx_error.txt"
    echo "======================"
    docs_exit_code=12
fi

rm -rf sphinx/_doxygen/
rm -rf sphinx/_mdgen/

# Verify that none of the following strings appear in the generated HTML.
# Any match is a documentation bug that must be fixed in the RST or conf.py.in:
#   MagicMock                    — autodoc targeted a MagicMock symbol in docs-gen mode
#   "alias of"                   — autoclass on a symbol whose __module__ differs from
#                                  the documenting module; applies in all build modes
#   _pycudaqx_qec_the_suffix_*   — internal pybind11 module name that doc_replacements
#                                  in conf.py.in should have stripped from all output
#
# Exit codes: 13 = MagicMock, 14 = "alias of", 15 = internal pybind11 module name.
invalid_html_checks=(
    "13|MagicMock|MagicMock references"
    "14|alias of|'alias of' references"
    "15|_pycudaqx_qec_the_suffix_matters|internal pybind11 module name"
)

for check in "${invalid_html_checks[@]}"; do
    IFS='|' read -r check_exit pattern label <<< "$check"
    if grep -rql "$pattern" "$sphinx_output_dir" --include="*.html" 2>/dev/null; then
        echo "ERROR: ${label} found in generated documentation:"
        grep -rn "$pattern" "$sphinx_output_dir" --include="*.html" | \
            sed "s|${sphinx_output_dir}/||"
        docs_exit_code=$check_exit
    fi
done

mkdir -p "$DOCS_INSTALL_PREFIX"
if [ "$docs_exit_code" -eq "0" ]; then
    cp -r "$sphinx_output_dir"/* "$DOCS_INSTALL_PREFIX"
    touch "$DOCS_INSTALL_PREFIX/.nojekyll"
    echo "Documentation was generated in $DOCS_INSTALL_PREFIX."
    echo "To browse it, open this url in a browser: file://$DOCS_INSTALL_PREFIX/index.html"
else
    echo "Documentation generation failed with exit code $docs_exit_code."
    echo "Check the logs in $logs_dir, and the documentation build output in $docs_build_output."
fi

cd "$working_dir" && (return 0 2>/dev/null) && return $docs_exit_code || exit $docs_exit_code
