#!/bin/sh

# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Exit immediately if any command returns a non-zero status
set -e

# Uncomment these lines to enable core files
#set +e
#ulimit -c unlimited

# Installing dependencies
python_version=$1
python=python${python_version}
platform=$2
cuda_version=$3
cudaq_version=$4
cudaqx_version=$5

# Verify the input arguments aren't empty, one at a time.
if [ -z "$python_version" ] ; then
  echo "Error: python_version is empty"
  exit 1
fi
if [ -z "$platform" ] ; then
  echo "Error: platform is empty"
  exit 1
fi
if [ -z "$cuda_version" ] ; then
  echo "Error: cuda_version is empty"
  exit 1
fi
if [ -z "$cudaq_version" ] ; then
  echo "Error: cudaq_version is empty"
  exit 1
fi
if [ -z "$cudaqx_version" ] ; then
  echo "Error: cudaqx_version is empty"
  exit 1
fi

# Validate that $cuda_version has exactly 1 dot (i.e. 12.6, not 12.6.0)
if [[ $(echo $cuda_version | grep -o '\.' | wc -l) -ne 1 ]]; then
  echo "Error: cuda_version must have exactly 1 dot"
  exit 1
fi
cuda_major=$(echo $cuda_version | cut -d '.' -f 1)

${python} -m pip install --no-cache-dir pytest

# The following packages are needed for our tests. They are not true
# dependencies for our delivered package.
${python} -m pip install onnxscript # for trt decoder tests
${python} -m pip install matplotlib # for pseudo_threshold.py

FIND_LINKS="--find-links /wheels/ --find-links /metapackages/"

# If special CUDA-Q wheels have been built for this test, install them here.
if [ -d /cudaq-wheels ]; then
  echo "Custom CUDA-Q wheels directory found. Will add to the --find-links list."
  FIND_LINKS="${FIND_LINKS} --find-links /cudaq-wheels/"
  ${python} -m pip install $FIND_LINKS "cuda-quantum-cu${cuda_major}==${cudaq_version}"
fi

# Install TensorRT.
if [[ "$cuda_major" == "13" ]]; then
  ${python} -m pip install "tensorrt-cu13==10.13.*" "cuda_toolkit[cudart]==${cuda_version}.*"
elif [[ "$cuda_major" == "12" ]]; then
  # If x86_64, then install tensorrt-cu12. If arm64, then do not install tensorrt-cu12.
  if [ "$(uname -m)" == "x86_64" ]; then
    ${python} -m pip install "tensorrt-cu12==10.13.*" "cuda_toolkit[cudart]==${cuda_version}.*"
  fi
fi

# QEC library
# ======================================

# Pre-install torch with the correct variant so cudaq-qec[all]'s "torch>=2.9.0"
# dependency is already satisfied and pip doesn't pull the default (CPU) build.
cuda_no_dot=$(echo $cuda_version | sed 's/\.//')
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  echo "CUDA detected — installing CUDA-matched torch"
  ${python} -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu${cuda_no_dot}
else
  ${python} -m pip install torch==2.9.0
fi

echo "Installing QEC library with all optional dependencies"
${python} -m pip install ${FIND_LINKS} --extra-index-url https://pypi.nvidia.com/ "cudaq-qec[all]==${cudaqx_version}"

if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  ${python} -m pytest -v -s libs/qec/python/tests/
else
  ${python} -m pytest -v -s libs/qec/python/tests/ --ignore=libs/qec/python/tests/test_trt_decoder.py
fi

# Verify that the correct version of cuda-quantum and cudaq-qec were installed.
package_installed=$(python${python_version} -m pip list | grep cudaq-qec-cu | cut -d ' ' -f1)
package_expected=cudaq-qec-cu${cuda_major}
if [ "$package_installed" != "$package_expected" ]; then
  echo "::error Expected installation of $package_expected package, but got $package_installed."
  exit 1
fi

package_installed=$(python${python_version} -m pip list | grep cuda-quantum-cu | cut -d ' ' -f1)
package_expected=cuda-quantum-cu${cuda_major}
if [ "$package_installed" != "$package_expected" ]; then
  echo "::error Expected installation of $package_expected package, but got $package_installed."
  exit 1
fi

# Smoke test the (undocumented) standalone decoders module shipped in the wheel:
# it is importable as the bare top-level `_qec_decoders_standalone` and must
# decode without ever pulling in cudaq. Run in a fresh interpreter so the
# sys.modules check is meaningful.
#
# The module links libcudart. Normally `import cudaq` puts the CUDA runtime on
# the loader path, but this test (and any cudaq-free user) skips cudaq, so
# provide cudart from the CUDA runtime wheel -- what a cudaq-free user must do.
# The package name differs by CUDA major (cu12 keeps the -cu12 suffix; cu13+
# uses the unsuffixed nvidia-cuda-runtime), and installing is best-effort since
# some images already ship a system cudart.
# ======================================
echo "Smoke testing standalone decoders module (_qec_decoders_standalone)"
if [ "$cuda_major" = "12" ]; then
  ${python} -m pip install "nvidia-cuda-runtime-cu12" || true
else
  ${python} -m pip install "nvidia-cuda-runtime" || true
fi
# libcudart's location in the CUDA runtime wheel differs by CUDA major
# (nvidia/cuda_runtime/lib for cu12, nvidia/cu13/lib for cu13), so discover it
# rather than hard-coding, and put it on LD_LIBRARY_PATH. Images that already
# ship a system cudart leave this empty and rely on the default loader path.
cudart_so=$(${python} -c "import glob, os, nvidia; print(next(iter(glob.glob(os.path.join(list(nvidia.__path__)[0], '**', 'libcudart.so*'), recursive=True)), ''))" 2>/dev/null || true)
cudart_dir=""
if [ -n "$cudart_so" ]; then cudart_dir=$(dirname "$cudart_so"); fi
LD_LIBRARY_PATH="${cudart_dir}${cudart_dir:+:}${LD_LIBRARY_PATH}" ${python} - <<'EOF'
import sys
import numpy as np
import _qec_decoders_standalone as m
assert "cudaq" not in sys.modules, "importing _qec_decoders_standalone pulled in cudaq"
H = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
assert m.qecrt.get_decoder("pymatching", H).decode([1, 1, 0]).converged
assert "cudaq" not in sys.modules, "using the standalone decoder pulled in cudaq"
print("standalone decoders module OK (no cudaq import)")
EOF

# Test the libraries with examples
# ======================================
echo "Testing libraries with examples"

# Install stim for AMD platform for tensor network decoder examples
if echo $platform | grep -qi "amd64"; then
  echo "Installing stim and beliefmatching for AMD64 platform"
  ${python} -m pip install stim beliefmatching
fi

for domain in "qec"; do
    echo "Testing ${domain} Python examples with Python ${python_version} ..."
    cd examples/${domain}/python
    shopt -s nullglob # don't throw errors if no Python files exist
    for f in *.py; do
        echo Testing $f...
        ${python} $f
        res=$?
        if [ $res -ne 0 ]; then
            echo "Python tests failed for ${domain} with Python ${python_version}: $res"
        fi
    done
    shopt -u nullglob  # reset setting, just for cleanliness
    cd - # back to the original directory
done
