/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#ifdef ENABLE_NVTX

#include <nvtx3/nvToolsExt.h>

namespace cudaq::qec::realtime::experimental {

struct NvtxRange {
  explicit NvtxRange(const char *name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange &) = delete;
  NvtxRange &operator=(const NvtxRange &) = delete;
};

#define NVTX_RANGE(name)                                                       \
  cudaq::qec::realtime::experimental::NvtxRange _nvtx_range_##__LINE__(name)
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#define NVTX_MARK(name) nvtxMarkA(name)

} // namespace cudaq::qec::realtime::experimental
#else

#define NVTX_RANGE(name) (void)0
#define NVTX_PUSH(name) (void)0
#define NVTX_POP() (void)0
#define NVTX_MARK(name) (void)0

#endif
