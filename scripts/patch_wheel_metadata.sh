#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is used to patch the wheel metadata for a PyPI package. For now,
# it is anticipated that one would use this script as a reference and update the
# MODIFY_ME1 and MODIFY_ME2 sections before using it.

# Note: you may need to run "python3 -m pip install -U wheel" first.

# Review and modify the following variables
PACKAGE_BASE_DASH=cudaq-qec
PACKAGE_BASE_UNDER=cudaq_qec
CUDA_VERSIONS="cu12 cu13"
ORIG_VER=0.5.0
NEW_VER=0.5.0.post1

# Make sure that curl, jq, python3, and wget are installed.
for cmd in curl jq python3 wget; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "$cmd could not be found"
    exit 1
  fi
done

mkdir -p wheels_orig wheels_new

for CUDA_VERSION in $CUDA_VERSIONS; do
  PACKAGE_NAME_DASH="${PACKAGE_BASE_DASH}-${CUDA_VERSION}"
  PACKAGE_NAME_UNDER="${PACKAGE_BASE_UNDER}_${CUDA_VERSION}"

  # Make a temporary directory to work in
  TMP_DIR=$(mktemp -d)
  echo "Using temporary directory: $TMP_DIR"
  # Be sure to clean up the temporary directory on exit
  trap "rm -rf $TMP_DIR" EXIT

  echo "============================================"
  echo "Processing ${PACKAGE_NAME_DASH} ..."
  echo "============================================"

  echo "Downloading the original wheels into wheels_orig..."
  curl -fsSL "https://pypi.org/pypi/${PACKAGE_NAME_DASH}/${ORIG_VER}/json" \
  | jq -r '.urls[] | select(.packagetype=="bdist_wheel") | .url' \
  | xargs -n1 -P4 -I{} wget -c -P wheels_orig {}

  echo "Placing the patched wheels into wheels_new..."
  for f in wheels_orig/${PACKAGE_NAME_UNDER}-${ORIG_VER}-*.whl; do
    [[ -f "$f" ]] || continue
    echo "Patching wheel: $f"
    python3 -m wheel unpack "$f" -d "$TMP_DIR"

    # --- Begin modifications
    FILE2CHANGE="$TMP_DIR/${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info/METADATA"

    # Update the version
    sed -i "s/^Version: ${ORIG_VER}/Version: ${NEW_VER}/" "$FILE2CHANGE"

    # MODIFY_ME2 - review and modify the METADATA file here
    # Update FROM:
    # Requires-Dist: cuda-quantum-cu##>=0.13
    # TO:
    # Requires-Dist: cuda-quantum-cu##==0.13.*
    sed -i "s/^Requires-Dist: cuda-quantum-${CUDA_VERSION}>=0\.13/Requires-Dist: cuda-quantum-${CUDA_VERSION}==0.13.*/" "$FILE2CHANGE"
    # --- End modifications

    # Re-package into a new whl file now
    cd "$TMP_DIR"
    mv "${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${ORIG_VER}.dist-info" \
       "${PACKAGE_NAME_UNDER}-${ORIG_VER}/${PACKAGE_NAME_UNDER}-${NEW_VER}.dist-info"
    python3 -m wheel pack "${PACKAGE_NAME_UNDER}-${ORIG_VER}" -d .
    cd -
    mv "$TMP_DIR/${PACKAGE_NAME_UNDER}-${NEW_VER}"*.whl wheels_new/
    rm -rf "$TMP_DIR"
  done
done

echo "Done!"
echo "Your original wheels are in wheels_orig, and your patched wheels are in wheels_new."
echo "You can now upload the patched wheels to PyPI."
