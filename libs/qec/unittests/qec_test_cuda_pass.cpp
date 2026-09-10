/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cuda_runtime_api.h>

namespace {
thread_local int g_qec_test_cuda_device = 0;
} // namespace

extern "C" {

cudaError_t __wrap_cudaGetDevice(int *device) {
  if (device)
    *device = g_qec_test_cuda_device;
  return cudaSuccess;
}

cudaError_t __wrap_cudaSetDevice(int device) {
  g_qec_test_cuda_device = device;
  return cudaSuccess;
}

cudaError_t __wrap_cudaGetDeviceCount(int *count) {
  if (count)
    *count = 8;
  return cudaSuccess;
}

const char *__wrap_cudaGetErrorString(cudaError_t) { return "qec-test-cuda"; }

} // extern "C"
