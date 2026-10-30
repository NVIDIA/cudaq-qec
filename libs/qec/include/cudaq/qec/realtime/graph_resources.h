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
/// struct.  The device-side graph scheduler (DeviceGraphRingConsumer /
/// DeviceGraphTransceiver) launches graph_exec from the GPU dispatch kernel;
/// function_id names the RPC the graph serves.
struct graph_resources {
  cudaGraphExec_t graph_exec = nullptr;
  cudaStream_t stream = nullptr;
  uint32_t function_id = 0;
};

} // namespace cudaq::qec::realtime
