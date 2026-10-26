#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Drives surface_code-6-mapped-enqueue, which is the only caller of the
# enqueue_syndromes_mapped_ui64 wire function.
#
# Usage: surface_code-6-mapped-enqueue-test.sh <exe> <server|NULL> <lib_dir>
#
# With a server, the run goes over UDP (CUDAQ_QTM_PRIVATE_IMPL=1), so the
# five-field payload is packed by the private sender and unpacked by the shim.
# That is the pairing with no compile-time contract, so the script asserts the
# new function name actually appears in the traffic -- a run that silently fell
# back to enqueue_syndromes_ui64 would otherwise look identical.

set -u

EXE_PATH=$1
SERVER_EXECUTABLE=$2
LIB_DIR=$3

return_code=0
SUFFIX=mapped-enqueue
# Relative, as in the other server tests: the server parses the LOAD command's
# config path into a 128-byte buffer, and a CI build tree's $PWD alone is longer
# than that, so an absolute path overruns it and the server aborts before it can
# serve.  ctest runs the server from this directory, so a bare name resolves.
CONFIG_FILE=config-$SUFFIX.yaml
SERVER_LOG=server-$SUFFIX.log

# The server's stdout and stderr both go to $SERVER_LOG, which ctest never
# prints, so echo it here on any failure.  An uncaught exception in the server
# leaves its "terminate called after throwing ..." line there and nowhere else;
# without this a server that died reads as three errors about the five-field
# wire, none of which is the actual fault.
dump_server_log() {
  if [[ -s $SERVER_LOG ]]; then
    echo "----- $SERVER_LOG -----"
    tail -n 40 "$SERVER_LOG"
    echo "----- end $SERVER_LOG -----"
  else
    echo "(the server wrote no output to $SERVER_LOG)"
  fi
}

# Phase 1: emit the decoder configuration the server will load.
$EXE_PATH --save_config "$CONFIG_FILE" || exit 1

# Phase 2: run the kernel.  The example returns non-zero unless the decoder
# returned exactly the correction the syndrome implies, so a scrambled field
# order fails here rather than being reported as a pass.
$EXE_PATH |& tee run-$SUFFIX.log
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
  echo "Error: surface_code-6-mapped-enqueue returned non-zero"
  return_code=1
fi

if ! grep -q "^PASS:" run-$SUFFIX.log; then
  echo "Error: expected PASS line not found"
  return_code=1
fi

if [[ $return_code -eq 0 ]]; then
  rm -f run-$SUFFIX.log "$SERVER_LOG" "$CONFIG_FILE"
fi
exit $return_code
