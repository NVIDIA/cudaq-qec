/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "qec_test_cuda_pass.h"

namespace {
thread_local int g_qec_test_cuda_device = 0;
qec_test_cuda_ctl g_qec_test_cuda;
} // namespace

qec_test_cuda_tls_current::operator int() const {
  return g_qec_test_cuda_device;
}

qec_test_cuda_tls_current &qec_test_cuda_tls_current::operator=(int device) {
  g_qec_test_cuda_device = device;
  return *this;
}

void qec_test_cuda_ctl::reset() {
  g_qec_test_cuda_device = 0;
  count = 8;
  get_status = cudaSuccess;
  set_status = cudaSuccess;
  count_status = cudaSuccess;
  sets.clear();
}

qec_test_cuda_ctl &qec_test_cuda() { return g_qec_test_cuda; }

extern "C" {

cudaError_t __wrap_cudaGetDevice(int *device) {
  if (g_qec_test_cuda.get_status != cudaSuccess)
    return g_qec_test_cuda.get_status;
  if (device)
    *device = g_qec_test_cuda_device;
  return cudaSuccess;
}

cudaError_t __wrap_cudaSetDevice(int device) {
  if (g_qec_test_cuda.set_status != cudaSuccess)
    return g_qec_test_cuda.set_status;
  g_qec_test_cuda.sets.push_back(device);
  g_qec_test_cuda_device = device;
  return cudaSuccess;
}

cudaError_t __wrap_cudaGetDeviceCount(int *count) {
  if (g_qec_test_cuda.count_status != cudaSuccess)
    return g_qec_test_cuda.count_status;
  if (count)
    *count = g_qec_test_cuda.count;
  return cudaSuccess;
}

const char *__wrap_cudaGetErrorString(cudaError_t) { return "qec-test-cuda"; }

} // extern "C"
