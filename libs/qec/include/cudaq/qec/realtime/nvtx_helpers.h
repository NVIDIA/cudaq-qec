/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifdef ENABLE_NVTX

#include <cstdint>
#include <nvtx3/nvToolsExt.h>

namespace cudaq::qec::realtime::experimental {

struct NvtxRange {
  explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange &) = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;
};

/// A range carrying an index, such as a window's, as its payload. A
/// negative index means none, and the range carries no payload.
struct nvtx_indexed_range {
  nvtx_indexed_range(const char *name, std::int64_t index) {
    nvtxEventAttributes_t attributes{};
    attributes.version = NVTX_VERSION;
    attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attributes.message.ascii = name;
    if (index >= 0) {
      attributes.payloadType = NVTX_PAYLOAD_TYPE_INT64;
      attributes.payload.llValue = index;
    }
    nvtxRangePushEx(&attributes);
  }
  ~nvtx_indexed_range() { nvtxRangePop(); }
  nvtx_indexed_range(const nvtx_indexed_range &) = delete;
  nvtx_indexed_range &operator=(const nvtx_indexed_range &) = delete;
};

// Two levels, so __LINE__ expands before it is pasted: each range in a
// scope gets a variable of its own.
#define NVTX_CONCAT_INNER(a, b) a##b
#define NVTX_CONCAT(a, b) NVTX_CONCAT_INNER(a, b)
#define NVTX_RANGE(name)                                                       \
  cudaq::qec::realtime::experimental::NvtxRange NVTX_CONCAT(_nvtx_range_,      \
                                                            __LINE__)(name)
#define NVTX_RANGE_INDEXED(name, index)                                        \
  cudaq::qec::realtime::experimental::nvtx_indexed_range NVTX_CONCAT(          \
      _nvtx_indexed_range_, __LINE__)(name, index)
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)

} // namespace cudaq::qec::realtime::experimental
#else

#define NVTX_RANGE(name) (void)0
#define NVTX_RANGE_INDEXED(name, index) (void)0
#define NVTX_PUSH(name) (void)0
#define NVTX_POP() (void)0
#define NVTX_MARK(name) (void)0

#endif
