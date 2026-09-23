/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/dem_sampling.h"

#include "dem_sampling_utils.h"
#include <cuda_runtime.h>
#include <custabilizer.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void cuda_check(cudaError_t err, const char *msg) {
  if (err != cudaSuccess)
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
}

void custab_check(custabilizerStatus_t s, const char *msg) {
  if (s != CUSTABILIZER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(msg) + ": " +
                             custabilizerGetErrorString(s));
}

struct StreamCudaDeleter {
  cudaStream_t stream = nullptr;
  void operator()(void *p) const {
    if (p)
      cudaFreeAsync(p, stream);
  }
};

template <typename T>
using DevicePtr = std::unique_ptr<T, StreamCudaDeleter>;

template <typename T>
DevicePtr<T> device_alloc(size_t count, cudaStream_t stream) {
  T *p = nullptr;
  cuda_check(cudaMallocAsync(&p, count * sizeof(T), stream), "cudaMallocAsync");
  return DevicePtr<T>(p, StreamCudaDeleter{stream});
}

struct HandleRAII {
  custabilizerHandle_t h = nullptr;
  HandleRAII() { custab_check(custabilizerCreate(&h), "custabilizerCreate"); }
  ~HandleRAII() {
    if (h)
      custabilizerDestroy(h);
  }
  HandleRAII(const HandleRAII &) = delete;
  HandleRAII &operator=(const HandleRAII &) = delete;
};

} // namespace

namespace cudaq::qec::dem_sampler::gpu {

bool sample_dem(const uint8_t *d_check_matrix, size_t num_checks,
                size_t num_error_mechanisms,
                const double *d_error_probabilities, size_t num_shots,
                unsigned seed, uint8_t *d_checks_out, uint8_t *d_errors_out,
                std::uintptr_t stream_handle) {
  if (num_checks == 0 || num_error_mechanisms == 0 || num_shots == 0)
    return true;

  using namespace cudaq::qec::dem_sampling_utils;

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_handle);

  HandleRAII handle;

  // ── 1. Pack H^T from H directly on device ────────────────────────────────
  // For GF2SparseDenseMatrixMultiply:
  //   C = A @ B where A is CSR [shots × errors], and B is dense packed
  //   [errors × n_checks_padded/32]. Here B = H^T packed row-wise.
  //
  // We avoid a host round-trip by packing H^T directly from H in a single
  // kernel.
  uint64_t n_checks_padded = ((num_checks + 31) / 32) * 32;
  uint64_t check_words = n_checks_padded / 32;

  // Pack H^T row-wise: [num_error_mechanisms × num_checks] -> [num_err × words]
  auto d_ht_packed =
      device_alloc<uint32_t>(num_error_mechanisms * check_words, stream);
  pack_check_matrix_transposed_rowwise(d_check_matrix, d_ht_packed.get(),
                                       num_checks, num_error_mechanisms,
                                       stream);

  // ── 2. Sparse Bernoulli sampling ──────────────────────────────────────────
  size_t ws_size = 0;
  custab_check(custabilizerSampleProbArraySparsePrepare(
                   handle.h, static_cast<int64_t>(num_shots),
                   static_cast<int64_t>(num_error_mechanisms), &ws_size),
               "SparsePrepare");

  auto d_workspace = device_alloc<uint8_t>(ws_size, stream);

  // Initial capacity estimate: E[nnz] * 1.25 + 1024
  // Copy probabilities to host via the caller's stream to respect stream
  // ordering (the data may have been produced by a kernel on this stream).
  std::vector<double> h_probs(num_error_mechanisms);
  cuda_check(cudaMemcpyAsync(h_probs.data(), d_error_probabilities,
                             num_error_mechanisms * sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             "copy probs");
  cuda_check(cudaStreamSynchronize(stream), "sync probs");
  double sum_p = 0.0;
  for (auto p : h_probs)
    sum_p += p;
  uint64_t capacity = static_cast<uint64_t>(num_shots * sum_p * 1.25 + 1024);
  uint64_t max_cap = num_shots * num_error_mechanisms;
  if (capacity > max_cap)
    capacity = max_cap;
  if (capacity < num_shots)
    capacity = num_shots;

  auto d_row_offsets = device_alloc<uint64_t>(num_shots + 1, stream);
  auto d_col_indices = device_alloc<uint64_t>(capacity, stream);

  uint64_t nnz = capacity;
  custabilizerStatus_t status = custabilizerSampleProbArraySparseCompute(
      handle.h, static_cast<int64_t>(num_shots),
      static_cast<int64_t>(num_error_mechanisms), d_error_probabilities,
      static_cast<uint64_t>(seed), &nnz, d_col_indices.get(),
      d_row_offsets.get(), d_workspace.get(), ws_size, stream);

  if (status == CUSTABILIZER_STATUS_INSUFFICIENT_SPARSE_STORAGE) {
    // nnz now contains the required capacity
    capacity = nnz;
    d_col_indices = device_alloc<uint64_t>(capacity, stream);
    nnz = capacity;
    custab_check(custabilizerSampleProbArraySparseCompute(
                     handle.h, static_cast<int64_t>(num_shots),
                     static_cast<int64_t>(num_error_mechanisms),
                     d_error_probabilities, static_cast<uint64_t>(seed), &nnz,
                     d_col_indices.get(), d_row_offsets.get(),
                     d_workspace.get(), ws_size, stream),
                 "SparseCompute retry");
  } else {
    custab_check(status, "SparseCompute");
  }

  // ── 3. GF(2) sparse-dense matmul: syndromes = errors * H^T ──────────────
  // A = CSR errors [num_shots × num_error_mechanisms]
  // B = H^T packed [num_error_mechanisms × n_checks_padded/32]
  // C = packed syndromes [num_shots × n_checks_padded/32]
  auto d_syndromes_packed =
      device_alloc<uint32_t>(num_shots * check_words, stream);

  custab_check(custabilizerGF2SparseDenseMatrixMultiply(
                   handle.h, static_cast<uint64_t>(num_shots),
                   static_cast<uint64_t>(n_checks_padded),
                   static_cast<uint64_t>(num_error_mechanisms), nnz,
                   d_col_indices.get(), d_row_offsets.get(), d_ht_packed.get(),
                   0, // beta=0: assign
                   d_syndromes_packed.get(), stream),
               "GF2SparseDenseMatrixMultiply");

  // ── 4. Unpack syndromes: bitpacked → uint8 ──────────────────────────────
  unpack_syndromes_gpu(d_syndromes_packed.get(), d_checks_out, num_shots,
                       num_checks, stream);

  // ── 5. CSR errors → dense uint8 ─────────────────────────────────────────
  csr_to_dense_fused(d_row_offsets.get(), d_col_indices.get(), num_shots,
                     num_error_mechanisms, d_errors_out, stream);

  cuda_check(cudaStreamSynchronize(stream), "sync");

  return true;
}

} // namespace cudaq::qec::dem_sampler::gpu
