/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cuda_runtime_api.h>
#include <vector>

// Controllable CUDA --wrap state. Defaults match the original always-success
// pass: thread-local current device, eight visible devices, every call OK.
struct qec_test_cuda_tls_current {
  operator int() const;
  qec_test_cuda_tls_current &operator=(int device);
};

struct qec_test_cuda_ctl {
  qec_test_cuda_tls_current current;
  int count = 8;
  cudaError_t get_status = cudaSuccess;
  cudaError_t set_status = cudaSuccess;
  cudaError_t count_status = cudaSuccess;
  std::vector<int> sets;

  void reset();
};

qec_test_cuda_ctl &qec_test_cuda();
