/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Implementation file for shared predecoder pipeline utilities.
//
// Provides the runtime behavior behind the helper types declared in
// predecoder_pipeline_common.h, including the TensorRT pre-launch staging
// callback, the deferred PyMatching job queue, and binary dataset loaders.

#include "predecoder_pipeline_common.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

// =============================================================================
// PipelineConfig::from_name
// =============================================================================

std::optional<PipelineConfig>
PipelineConfig::from_name(const std::string &name) {
  if (name == "d7")
    return d7_r7();
  if (name == "d13")
    return d13_r13();
  if (name == "d13_r104")
    return d13_r104();
  if (name == "d21")
    return d21_r21();
  if (name == "d21_r42")
    return d21_r42();
  if (name == "d31")
    return d31_r31();
  return std::nullopt;
}

// =============================================================================
// PipelineConfig::apply_cli_overrides
// =============================================================================

void PipelineConfig::apply_cli_overrides(int argc, char *argv[]) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto val_of = [&](const std::string &prefix) -> std::string {
      return arg.substr(prefix.size());
    };
    if (arg.find("--distance=") == 0)
      distance = std::stoi(val_of("--distance="));
    else if (arg.find("--num-rounds=") == 0)
      num_rounds = std::stoi(val_of("--num-rounds="));
    else if (arg.find("--onnx-filename=") == 0)
      onnx_filename = val_of("--onnx-filename=");
    else if (arg.find("--num-predecoders=") == 0)
      num_predecoders = std::stoi(val_of("--num-predecoders="));
    else if (arg.find("--num-workers=") == 0)
      num_workers = std::stoi(val_of("--num-workers="));
    else if (arg.find("--num-decode-workers=") == 0)
      num_decode_workers = std::stoi(val_of("--num-decode-workers="));
    else if (arg.find("--label=") == 0)
      label = val_of("--label=");
  }
}

// =============================================================================
// Pre-launch DMA copy callback
// =============================================================================

// Store the most recent slot pointer in the mailbox bank so the CPU polling
// path can map TensorRT completion back to the original ring slot, then issue
// an asynchronous host-to-device copy of the detector payload located
// immediately after the RPC header.
void pre_launch_input_copy(void *user_data, void *slot_dev,
                           cudaStream_t stream) {
  NVTX_PUSH("PreLaunchCopy");
  auto *ctx = static_cast<PreLaunchCopyCtx *>(user_data);
  ctx->h_ring_ptrs[0] = slot_dev;
  ptrdiff_t offset = static_cast<uint8_t *>(slot_dev) - ctx->rx_data_dev_base;
  const void *slot_host = ctx->rx_data_host_base + offset;
  cudaMemcpyAsync(ctx->d_trt_input,
                  static_cast<const uint8_t *>(slot_host) +
                      CUDAQ_RPC_HEADER_SIZE,
                  ctx->input_size, cudaMemcpyHostToDevice, stream);
  NVTX_POP();
}

// =============================================================================
// PyMatchQueue
// =============================================================================

// Keep the critical section short so predecoder workers can hand off jobs to
// the PyMatching pool without stalling the realtime pipeline.
void PyMatchQueue::push(PyMatchJob &&j) {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    jobs_.push(std::move(j));
  }
  cv_.notify_one();
}

// Workers sleep until a new job arrives or shutdown() is called. Returning
// false signals that the worker should exit.
bool PyMatchQueue::pop(PyMatchJob &out) {
  std::unique_lock<std::mutex> lk(mtx_);
  cv_.wait(lk, [&] { return !jobs_.empty() || stop_; });
  if (stop_ && jobs_.empty())
    return false;
  out = std::move(jobs_.front());
  jobs_.pop();
  return true;
}

// Wake all waiting workers so they can observe stop_ and exit once the queue
// has drained.
void PyMatchQueue::shutdown() {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    stop_ = true;
  }
  cv_.notify_all();
}

// =============================================================================
// SparseCSR
// =============================================================================

// Expand the CSR representation into a dense row-major tensor so helper code
// can build decoders using the regular tensor-based APIs.
cudaqx::tensor<uint8_t> SparseCSR::to_dense() const {
  cudaqx::tensor<uint8_t> T;
  std::vector<uint8_t> data(static_cast<size_t>(nrows) * ncols, 0);
  for (uint32_t r = 0; r < nrows; ++r)
    for (int32_t j = indptr[r]; j < indptr[r + 1]; ++j)
      data[static_cast<size_t>(r) * ncols + indices[j]] = 1;
  T.copy(data.data(), {static_cast<size_t>(nrows), static_cast<size_t>(ncols)});
  return T;
}

// Primarily used to fetch the first observable row for projecting residual
// corrections onto logical parity bits.
std::vector<uint8_t> SparseCSR::row_dense(uint32_t r) const {
  std::vector<uint8_t> row(ncols, 0);
  for (int32_t j = indptr[r]; j < indptr[r + 1]; ++j)
    row[indices[j]] = 1;
  return row;
}

// =============================================================================
// Test data loaders
// =============================================================================

// The binary format starts with a uint32 row/column header followed by
// row-major int32 payload data.
bool load_binary_file(const std::string &path, uint32_t &out_rows,
                      uint32_t &out_cols, std::vector<int32_t> &data) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good())
    return false;
  f.read(reinterpret_cast<char *>(&out_rows), sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(&out_cols), sizeof(uint32_t));
  size_t count = static_cast<size_t>(out_rows) * out_cols;
  data.resize(count);
  f.read(reinterpret_cast<char *>(data.data()), count * sizeof(int32_t));
  return f.good();
}

// Return a default-constructed TestData on any load or consistency failure so
// callers can use TestData::loaded() as the success check.
TestData load_test_data(const std::string &data_dir) {
  TestData td;
  std::string det_path = data_dir + "/detectors.bin";
  std::string obs_path = data_dir + "/observables.bin";

  if (!load_binary_file(det_path, td.num_samples, td.num_detectors,
                        td.detectors)) {
    std::cerr << "ERROR: Failed to load " << det_path << "\n";
    return td;
  }
  uint32_t obs_samples = 0;
  if (!load_binary_file(obs_path, obs_samples, td.num_observables,
                        td.observables)) {
    std::cerr << "ERROR: Failed to load " << obs_path << "\n";
    td.num_samples = 0;
    return td;
  }
  if (obs_samples != td.num_samples) {
    std::cerr << "ERROR: sample count mismatch: detectors=" << td.num_samples
              << " observables=" << obs_samples << "\n";
    td.num_samples = 0;
    return td;
  }
  std::cout << "[Data] Loaded " << td.num_samples << " samples, "
            << td.num_detectors << " detectors, " << td.num_observables
            << " observables from " << data_dir << "\n";
  return td;
}

// The file layout is [nrows, ncols, nnz, indptr..., indices...].
bool load_csr(const std::string &path, SparseCSR &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good())
    return false;
  f.read(reinterpret_cast<char *>(&out.nrows), sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(&out.ncols), sizeof(uint32_t));
  f.read(reinterpret_cast<char *>(&out.nnz), sizeof(uint32_t));
  out.indptr.resize(out.nrows + 1);
  out.indices.resize(out.nnz);
  f.read(reinterpret_cast<char *>(out.indptr.data()),
         (out.nrows + 1) * sizeof(int32_t));
  f.read(reinterpret_cast<char *>(out.indices.data()),
         out.nnz * sizeof(int32_t));
  return f.good();
}

// Missing optional files leave the corresponding members empty while still
// allowing the benchmark to proceed with the available data.
StimData load_stim_data(const std::string &data_dir) {
  StimData sd;

  if (!load_csr(data_dir + "/H_csr.bin", sd.H)) {
    std::cerr << "[Data] No H_csr.bin found in " << data_dir << "\n";
    return sd;
  }
  std::cout << "[Data] Loaded H_csr " << sd.H.nrows << "x" << sd.H.ncols << " ("
            << sd.H.nnz << " nnz)\n";

  if (load_csr(data_dir + "/O_csr.bin", sd.O))
    std::cout << "[Data] Loaded O_csr " << sd.O.nrows << "x" << sd.O.ncols
              << " (" << sd.O.nnz << " nnz)\n";

  std::string priors_path = data_dir + "/priors.bin";
  std::ifstream pf(priors_path, std::ios::binary);
  if (pf.good()) {
    uint32_t nedges = 0;
    pf.read(reinterpret_cast<char *>(&nedges), sizeof(uint32_t));
    sd.priors.resize(nedges);
    pf.read(reinterpret_cast<char *>(sd.priors.data()),
            nedges * sizeof(double));
    std::cout << "[Data] Loaded " << sd.priors.size() << " priors\n";
  }
  return sd;
}
