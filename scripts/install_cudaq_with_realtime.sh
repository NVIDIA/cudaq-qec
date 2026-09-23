#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Rebuild and install CUDA-Q *with realtime + device_call support*, pinned to
# the SHA recorded in cudaqx's .cudaq_version, into CUDAQ_INSTALL_PREFIX.
#
# device_call (the host-dispatch service the QEC realtime libraries link
# against) is gated in CUDA-Q on CUDAQ_ENABLE_REALTIME. The actual build/install
# is delegated to the canonical CI recipe,
# .github/actions/get-cudaq-build/build_cudaq.sh (it installs cudaq-realtime and
# then configures the top-level build with -DCUDAQ_REALTIME_DIR, which flips
# CUDAQ_ENABLE_REALTIME on). This wrapper just adds the .cudaq_version SHA
# checkout up front and a post-build check that the device_call artifacts landed.
#
# Configuration (all overridable via environment):
#   CUDAQ_SRC             cuda-quantum checkout dir         [/workspaces/cudaq]
#   CUDAQ_INSTALL_PREFIX  install prefix                    [/usr/local/cudaq]
#   LLVM_INSTALL_PREFIX   LLVM used to build CUDA-Q         [/usr/local/llvm]
#   BUILD_TYPE            CMAKE_BUILD_TYPE                  [Release]
#   CC / CXX              host compilers                    [gcc / g++]
#   CUDAQ_REPO/CUDAQ_REF  override the .cudaq_version pin   [from .cudaq_version]
#   FORCE_CHECKOUT=1      allow checkout over a dirty tree  (DISCARDS changes)
#
# Usage:
#   scripts/install_cudaq_with_realtime.sh

set -euo pipefail

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
version_file="$repo_root/.cudaq_version"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CUDAQ_SRC=${CUDAQ_SRC:-/workspaces/cudaq}
# Resolve to absolute path so later cd's don't confuse relative paths.
# The parent directory must already exist (we're about to clone into it or it's
# already a git checkout).
_cudaq_parent=$(cd "$(dirname "$CUDAQ_SRC")" 2>/dev/null && pwd) || \
  die "parent directory of CUDAQ_SRC does not exist: $(dirname "$CUDAQ_SRC")"
CUDAQ_SRC="$_cudaq_parent/$(basename "$CUDAQ_SRC")"
unset _cudaq_parent
CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-/usr/local/cudaq}
LLVM_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX:-/usr/local/llvm}
BUILD_TYPE=${BUILD_TYPE:-Release}
CC=${CC:-gcc}
CXX=${CXX:-g++}
FORCE_CHECKOUT=${FORCE_CHECKOUT:-0}

command -v jq  >/dev/null 2>&1 || die "jq is required (parses .cudaq_version)"
command -v git >/dev/null 2>&1 || die "git is required"
command -v cmake >/dev/null 2>&1 || die "cmake is required"
[ -f "$version_file" ] || die "$version_file not found"

# SHA (and repo) come from .cudaq_version unless overridden.
CUDAQ_REPO=${CUDAQ_REPO:-$(jq -r '.cudaq.repository' "$version_file")}
CUDAQ_REF=${CUDAQ_REF:-$(jq -r '.cudaq.ref' "$version_file")}
[ -n "$CUDAQ_REPO" ] && [ "$CUDAQ_REPO" != "null" ] || die "could not read .cudaq.repository from $version_file"
[ -n "$CUDAQ_REF" ]  && [ "$CUDAQ_REF"  != "null" ] || die "could not read .cudaq.ref from $version_file"

log "CUDA-Q ${CUDAQ_REPO}@${CUDAQ_REF}"
log "  source : $CUDAQ_SRC"
log "  install: $CUDAQ_INSTALL_PREFIX"
log "  llvm   : $LLVM_INSTALL_PREFIX"
log "  build  : $BUILD_TYPE (CC=$CC CXX=$CXX)"

# ---------------------------------------------------------------------------
# 1. Get the source at the pinned SHA
# ---------------------------------------------------------------------------
if [ -d "$CUDAQ_SRC/.git" ]; then
  cd "$CUDAQ_SRC"
  if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$CUDAQ_REF" 2>/dev/null || echo none)" ]; then
    log "Source already at ${CUDAQ_REF}; skipping checkout"
  else
    if [ -n "$(git status --porcelain)" ] && [ "$FORCE_CHECKOUT" != "1" ]; then
      die "$CUDAQ_SRC has local changes; commit/stash them or set FORCE_CHECKOUT=1 to discard."
    fi
    log "Fetching + checking out ${CUDAQ_REF}"
    git fetch origin "$CUDAQ_REF" || git fetch origin
    git checkout --force "$CUDAQ_REF"
  fi
else
  # No submodules: CUDA-Q uses the prebuilt LLVM_INSTALL_PREFIX / NANOBIND_INSTALL_PREFIX
  # and FetchContent for tpls, so initializing submodules would needlessly clone
  # the huge tpls/llvm tree. (Matches the CI checkout, which inits no submodules.)
  log "Cloning ${CUDAQ_REPO} into $CUDAQ_SRC (shallow)"
  # --no-checkout: skip checking out the default branch; we only want CUDAQ_REF.
  git clone --no-checkout "https://github.com/${CUDAQ_REPO}.git" "$CUDAQ_SRC"
  cd "$CUDAQ_SRC"
  # Fetch only the pinned SHA with a shallow window so we don't pull full history.
  git fetch --depth=10 origin "$CUDAQ_REF"
  git checkout --force FETCH_HEAD
fi

[ -x "$LLVM_INSTALL_PREFIX/bin/llvm-config" ] || \
  log "WARNING: no llvm-config at $LLVM_INSTALL_PREFIX/bin; set LLVM_INSTALL_PREFIX if the build can't find LLVM."

# ---------------------------------------------------------------------------
# 2. Clean the build dirs, then build + install realtime AND the full CUDA-Q
#    (with device_call) by delegating to the canonical CI recipe instead of
#    duplicating it. That script uses a positional contract (BUILD_TYPE,
#    LAUNCHER, CC, CXX) and reads CUDAQ_SRC / CUDAQ_INSTALL_PREFIX /
#    LLVM_INSTALL_PREFIX from the env.
# ---------------------------------------------------------------------------
build_script="$repo_root/.github/actions/get-cudaq-build/build_cudaq.sh"
[ -f "$build_script" ] || die "delegate build script not found: $build_script"

# Force a fresh build. ($CUDAQ_SRC/build is also wiped by the inner build script,
# but $CUDAQ_SRC/realtime/build is incremental, so clean both explicitly.)
log "Cleaning build directories"
rm -rf "$CUDAQ_SRC/realtime/build" "$CUDAQ_SRC/build"

log "Building realtime + CUDA-Q via $build_script"
CUDAQ_SRC="$CUDAQ_SRC" \
CUDAQ_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
LLVM_INSTALL_PREFIX="$LLVM_INSTALL_PREFIX" \
  bash "$build_script" "$BUILD_TYPE" "" "$CC" "$CXX"

# ---------------------------------------------------------------------------
# 3. Verify device_call support actually installed (the markers from manual
#    inspection: runtime library + public header).
# ---------------------------------------------------------------------------
log "Verifying device_call support in $CUDAQ_INSTALL_PREFIX"
dc_lib="$CUDAQ_INSTALL_PREFIX/lib/libcudaq-device-call-runtime.so"
dc_hdr="$CUDAQ_INSTALL_PREFIX/include/cudaq/realtime/device_call_service.h"
ok=1
if [ -f "$dc_lib" ]; then echo "  [ok]      $dc_lib"; else echo "  [MISSING] $dc_lib"; ok=0; fi
if [ -f "$dc_hdr" ]; then echo "  [ok]      $dc_hdr"; else echo "  [MISSING] $dc_hdr"; ok=0; fi
[ "$ok" = "1" ] || die "build finished but device_call artifacts are missing -- check that CUDA was found and CUDAQ_ENABLE_REALTIME was TRUE."

log "Done. CUDA-Q with device_call support installed at $CUDAQ_INSTALL_PREFIX"
log "Configure cudaqx with: -DCUDAQ_DIR=$CUDAQ_INSTALL_PREFIX/lib/cmake/cudaq -DCUDAQ_REALTIME_ROOT=$CUDAQ_INSTALL_PREFIX"
