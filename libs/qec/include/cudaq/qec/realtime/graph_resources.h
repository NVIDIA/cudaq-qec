/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace cudaq::qec::realtime {

/// Resources returned by decoder::capture_decode_graph().
///
/// The decoder plugin captures a CUDA graph internally and populates this
/// struct.  The host dispatcher (libcudaq-realtime-host-dispatch) uses
/// graph_exec / stream to launch the graph, and writes per-slot I/O
/// addresses into h_mailbox before each launch.  function_id is used by
/// the host dispatcher to route RPC requests to the correct graph worker.
struct graph_resources {
  cudaGraphExec_t graph_exec = nullptr;
  cudaStream_t stream = nullptr;
  void **d_mailbox = nullptr; ///< device-mapped pinned pointer
  void **h_mailbox = nullptr; ///< host pointer to same pinned memory
  uint32_t function_id = 0;
};

} // namespace cudaq::qec::realtime
