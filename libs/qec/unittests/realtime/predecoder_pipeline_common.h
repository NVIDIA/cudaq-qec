/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file predecoder_pipeline_common.h
/// @brief Shared types and helpers for the AI predecoder + PyMatching pipeline.
///
/// Used by both the software-only benchmark
/// (test_realtime_predecoder_w_pymatching.cpp) and the FPGA bridge
/// (gpu_roce_predecoder_bridge.cpp). These helpers are example and test support
/// code rather than part of the stable library API, but documenting them keeps
/// the benchmark and bridge configuration visible from the generated docs.

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#ifndef CUDA_VERSION
#define CUDA_VERSION 13000
#endif

#include "cudaq/qec/realtime/pipeline.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include "cudaq/qec/code.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/ai_predecoder_service.h"
#include "cudaq/qec/realtime/nvtx_helpers.h"

using namespace cudaq::qec;
using namespace cudaq::qec::realtime::experimental;
namespace rt_sdk = cudaq::realtime;
namespace rt_pipeline = cudaq::qec::realtime::experimental;

#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  do {                                                                         \
  } while (0)
#endif
#endif

// =============================================================================
// Pipeline Configuration
// =============================================================================

/// @brief Maximum number of pipeline ring buffer slots.
/// @details Shared by the benchmark and bridge so both data sources use the
/// same ring depth assumptions when staging requests into the realtime
/// pipeline.
constexpr size_t NUM_SLOTS = 16;

/// @brief Named configuration for a predecoder pipeline instance.
///
/// Each factory method returns a preset combining QEC code parameters
/// (distance, rounds) with an ONNX model filename and thread/worker counts
/// tuned for that model size.
struct PipelineConfig {
  /// @brief Human-readable label (e.g. "d13_r104_X").
  std::string label;
  /// @brief QEC surface code distance.
  int distance;
  /// @brief Number of QEC syndrome measurement rounds.
  int num_rounds;
  /// @brief ONNX model filename (looked up under ONNX_MODEL_DIR).
  std::string onnx_filename;
  /// @brief Number of parallel TensorRT predecoder instances.
  int num_predecoders;
  /// @brief Number of pipeline GPU worker threads.
  int num_workers;
  /// @brief Number of PyMatching decode worker threads.
  int num_decode_workers;

  /// @brief Full path to the ONNX model file.
  /// @return Concatenation of ONNX_MODEL_DIR and @p onnx_filename.
  std::string onnx_path() const {
    return std::string(ONNX_MODEL_DIR) + "/" + onnx_filename;
  }

  /// @brief Full path to the cached TensorRT engine file.
  /// @return Same as onnx_path() but with a .engine extension.
  std::string engine_path() const {
    std::string name = onnx_filename;
    auto dot = name.rfind('.');
    if (dot != std::string::npos)
      name = name.substr(0, dot);
    return std::string(ONNX_MODEL_DIR) + "/" + name + ".engine";
  }

  /// @brief Distance-7, 7-round Z-basis config.
  static PipelineConfig d7_r7() {
    return {"d7_r7_Z", 7, 7, "model1_d7_r7_unified_Z_batch1.onnx", 16, 16, 32};
  }

  /// @brief Distance-13, 13-round X-basis config.
  static PipelineConfig d13_r13() {
    return {"d13_r13_X", 13, 13, "predecoder_memory_d13_T13_X.onnx",
            16,          16, 32};
  }

  /// @brief Distance-13, 104-round X-basis config.
  static PipelineConfig d13_r104() {
    return {
        "d13_r104_X", 13, 104, "predecoder_memory_d13_T104_X.onnx", 8, 8, 16};
  }

  /// @brief Distance-21, 21-round Z-basis config.
  static PipelineConfig d21_r21() {
    return {"d21_r21_Z", 21, 21, "model1_d21_r21_unified_X_batch1.onnx",
            16,          16, 32};
  }

  /// @brief Distance-21, 42-round X-basis config.
  static PipelineConfig d21_r42() {
    return {"d21_r42_X", 21, 42, "predecoder_memory_model_1_d21_T42_X.onnx",
            8,           8,  16};
  }

  /// @brief Distance-31, 31-round Z-basis config.
  static PipelineConfig d31_r31() {
    return {"d31_r31_Z", 31, 31, "model1_d31_r31_unified_Z_batch1.onnx",
            16,          16, 32};
  }

  /// @brief Construct a PipelineConfig from a JSON string.
  ///
  /// Required fields: distance, num_rounds, onnx_filename, num_predecoders,
  /// num_workers, num_decode_workers.  Optional: label (defaults to "custom").
  ///
  /// @param json_str JSON object as a string.
  /// @return Populated PipelineConfig.
  /// @throws std::runtime_error if parsing fails or required fields are
  /// missing.
  static PipelineConfig from_json(const std::string &json_str);

  /// @brief Apply command-line overrides to this config.
  ///
  /// Scans argv for --distance=, --num-rounds=, --onnx-filename=,
  /// --num-predecoders=, --num-workers=, --num-decode-workers=, and
  /// --label= flags.  Any flag found overrides the corresponding field.
  /// Unknown flags are silently ignored (handled by the caller).
  ///
  /// @param argc Argument count.
  /// @param argv Argument vector.
  void apply_cli_overrides(int argc, char *argv[]);

  /// @brief Select a named preset config.
  /// @param name One of: d7, d13, d13_r104, d21, d21_r42, d31.
  /// @return The matching config, or std::nullopt if name is unrecognized.
  static std::optional<PipelineConfig> from_name(const std::string &name);
};

/// @brief Round a value up to the next power of two.
/// @param v Input value (must be > 0).
/// @return Smallest power of two >= @p v.
/// @details Used when sizing buffers that must satisfy alignment or TensorRT
/// engine requirements.
inline size_t round_up_pow2(size_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v |= v >> 32;
  return v + 1;
}

// =============================================================================
// Decoder Context (PyMatching worker pool)
// =============================================================================

/// @brief Shared state for a pool of PyMatching MWPM decoder instances.
///
/// Each decode worker thread acquires its own decoder via acquire_decoder()
/// and accumulates timing and syndrome density statistics atomically. The
/// context also carries shape information needed to interpret the predecoder
/// residual syndrome output.
struct DecoderContext {
  /// @brief Pool of decoder instances (one per decode worker thread).
  std::vector<std::unique_ptr<cudaq::qec::decoder>> decoders;
  /// @brief Atomic counter for round-robin decoder assignment.
  std::atomic<int> next_decoder_idx{0};

  /// @brief Number of Z stabilizers (for spatial slice decoding).
  int z_stabilizers = 0;
  /// @brief Number of spatial slices for per-slice decoding.
  int spatial_slices = 0;
  /// @brief Number of residual detectors in the predecoder output.
  int num_residual_detectors = 0;

  /// @brief If true, use the full H matrix for decoding (not per-slice).
  bool use_full_H = false;

  /// @brief Acquire a thread-local decoder from the pool.
  /// @return Pointer to the decoder assigned to the calling thread.
  /// @details The first call on each thread claims a stable decoder index via
  /// @c next_decoder_idx so repeated calls from that thread reuse the same
  /// decoder instance.
  cudaq::qec::decoder *acquire_decoder() {
    thread_local int my_idx =
        next_decoder_idx.fetch_add(1, std::memory_order_relaxed);
    return decoders[my_idx % decoders.size()].get();
  }

  /// @brief Cumulative PyMatching decode time in microseconds.
  std::atomic<int64_t> total_decode_us{0};
  /// @brief Cumulative total worker time in microseconds.
  std::atomic<int64_t> total_worker_us{0};
  /// @brief Number of decode operations completed (for averaging).
  std::atomic<int> decode_count{0};

  /// @brief Number of input detectors (before predecoder).
  int num_input_detectors = 0;
  /// @brief Cumulative nonzero count in input syndromes.
  std::atomic<int64_t> total_input_nonzero{0};
  /// @brief Cumulative nonzero count in output (residual) syndromes.
  std::atomic<int64_t> total_output_nonzero{0};
};

// =============================================================================
// Pre-launch DMA copy callback
// =============================================================================

/// @brief Context for the pre-launch callback that copies ring buffer data
/// into the TensorRT input buffer before graph launch.
struct PreLaunchCopyCtx {
  /// @brief Device pointer to the TensorRT input tensor.
  void *d_trt_input;
  /// @brief Size of the input tensor in bytes.
  size_t input_size;
  /// @brief Host mailbox bank written by the dispatcher callback.
  /// @details The current slot device pointer is published here so the CPU
  /// polling path can recover the originating ring slot after TensorRT
  /// inference completes.
  void **h_ring_ptrs;
  /// @brief Device-mapped base of the RX data ring.
  uint8_t *rx_data_dev_base;
  /// @brief Host-mapped base of the RX data ring.
  uint8_t *rx_data_host_base;
};

/// @brief Pre-launch callback: async-copies detector data from the ring buffer
/// slot into the TensorRT input tensor.
/// @param user_data Pointer to a PreLaunchCopyCtx.
/// @param slot_dev Device pointer to the ring buffer slot.
/// @param stream CUDA stream for the async copy.
void pre_launch_input_copy(void *user_data, void *slot_dev,
                           cudaStream_t stream);

// =============================================================================
// Worker context
// =============================================================================

/// @brief Per-worker context passed through gpu_worker_resources::user_context.
/// @details Bridges the GPU predecoder worker with the downstream PyMatching
/// decode pool and correctness-tracking arrays used by the benchmark.
struct WorkerCtx {
  /// @brief Pointer to the AI predecoder service for this worker.
  ai_predecoder_service *predecoder;
  /// @brief Shared decoder context (PyMatching pool and statistics).
  DecoderContext *decoder_ctx;
  /// @brief Array to store per-request total correction counts.
  /// Indexed by request_id when correctness checking is enabled.
  int32_t *decode_corrections = nullptr;
  /// @brief Array to store per-request logical prediction parity.
  /// Indexed by request_id when correctness checking is enabled.
  int32_t *decode_logical_pred = nullptr;
  /// @brief Maximum number of requests to track.
  int max_requests = 0;
  /// @brief Observable row from the O matrix for logical projection.
  const uint8_t *obs_row = nullptr;
  /// @brief Length of the observable row.
  size_t obs_row_size = 0;
};

/// @brief Packed RPC response containing decode results.
/// @details This payload is written directly into the pipeline response buffer
/// after the RPC header, so the struct remains packed to match the transport
/// framing exactly.
struct __attribute__((packed)) DecodeResponse {
  /// @brief Total number of corrections applied (predecoder + PyMatching).
  int32_t total_corrections;
  /// @brief Whether the PyMatching decoder converged (1 = yes).
  int32_t converged;
};

// =============================================================================
// PyMatching work queue
// =============================================================================

/// @brief A single PyMatching decode job queued from the CPU stage.
struct PyMatchJob {
  /// @brief Pipeline slot that originated this job.
  int origin_slot;
  /// @brief RPC request ID from the header.
  uint64_t request_id;
  /// @brief Pointer to the ring buffer data for this slot.
  void *ring_buffer_ptr;
};

/// @brief Thread-safe queue for dispatching PyMatching decode jobs.
/// @details The queue connects the pipeline CPU stage, which only harvests
/// completed predecoder outputs, to a separate pool of PyMatching workers that
/// can finish decoding asynchronously and later call
/// @ref
/// cudaq::qec::realtime::experimental::realtime_pipeline::complete_deferred.
class PyMatchQueue {
public:
  /// @brief Enqueue a job and wake one waiting worker.
  /// @param j The job to enqueue (moved).
  void push(PyMatchJob &&j);

  /// @brief Dequeue a job, blocking until one is available or shutdown.
  /// @param out Output: the dequeued job.
  /// @return True if a job was dequeued, false if the queue is shut down.
  bool pop(PyMatchJob &out);

  /// @brief Signal all waiting workers to drain and exit.
  void shutdown();

private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::queue<PyMatchJob> jobs_;
  bool stop_ = false;
};

// =============================================================================
// Test data (pre-generated from Stim, or random)
// =============================================================================

/// @brief Container for pre-generated syndrome test data.
///
/// Loaded from binary files (detectors.bin, observables.bin) and accessed
/// by sample index with automatic wraparound. This lets the benchmark replay a
/// fixed corpus indefinitely at a target request rate.
struct TestData {
  /// @brief Flattened detector samples (num_samples * num_detectors).
  std::vector<int32_t> detectors;
  /// @brief Flattened observable labels (num_samples * num_observables).
  std::vector<int32_t> observables;
  /// @brief Number of syndrome samples in the dataset.
  uint32_t num_samples = 0;
  /// @brief Number of detectors per sample.
  uint32_t num_detectors = 0;
  /// @brief Number of observables per sample.
  uint32_t num_observables = 0;

  /// @brief Check whether data was successfully loaded.
  /// @return True if samples and detectors are present.
  bool loaded() const { return num_samples > 0 && num_detectors > 0; }

  /// @brief Return a pointer to the detector array for a given sample.
  /// @param idx Sample index (wraps modulo num_samples).
  /// @return Pointer to num_detectors int32 values.
  const int32_t *sample(int idx) const {
    return detectors.data() +
           (static_cast<size_t>(idx % num_samples) * num_detectors);
  }

  /// @brief Return a single observable value for a given sample.
  /// @param idx Sample index (wraps modulo num_samples).
  /// @param obs Observable index within the sample.
  /// @return The observable value (0 or 1).
  int32_t observable(int idx, int obs = 0) const {
    return observables[static_cast<size_t>(idx % num_samples) *
                           num_observables +
                       obs];
  }
};

/// @brief Load a binary file with a (rows, cols) header and int32 data.
/// @param path File path.
/// @param out_rows Output: number of rows.
/// @param out_cols Output: number of columns.
/// @param data Output: flattened row-major data.
/// @return True on success.
bool load_binary_file(const std::string &path, uint32_t &out_rows,
                      uint32_t &out_cols, std::vector<int32_t> &data);

/// @brief Load test data (detectors + observables) from a directory.
/// @param data_dir Directory containing detectors.bin and observables.bin.
/// @return Populated TestData (check loaded() for success).
TestData load_test_data(const std::string &data_dir);

// =============================================================================
// Stim-derived parity check matrix loader (CSR sparse -> dense tensor)
// =============================================================================

/// @brief Sparse matrix in Compressed Sparse Row (CSR) format.
///
/// Used to store parity check matrices (H) and observable matrices (O)
/// loaded from binary files before converting them into the dense tensor form
/// expected by the decoder construction helpers.
struct SparseCSR {
  /// @brief Number of rows.
  uint32_t nrows = 0;
  /// @brief Number of columns.
  uint32_t ncols = 0;
  /// @brief Number of nonzero entries.
  uint32_t nnz = 0;
  /// @brief Row pointer array (size nrows+1).
  std::vector<int32_t> indptr;
  /// @brief Column index array (size nnz).
  std::vector<int32_t> indices;

  /// @brief Check whether the matrix was loaded.
  /// @return True if dimensions are nonzero.
  bool loaded() const { return nrows > 0 && ncols > 0; }

  /// @brief Convert to a dense uint8 tensor.
  /// @return Dense tensor of shape [nrows, ncols] with 0/1 entries.
  cudaqx::tensor<uint8_t> to_dense() const;

  /// @brief Extract a single row as a dense vector.
  /// @param r Row index.
  /// @return Dense vector of length ncols with 0/1 entries.
  std::vector<uint8_t> row_dense(uint32_t r) const;
};

/// @brief Collection of Stim-derived data for configuring PyMatching.
/// @details Bundles the parity check matrix, observable projection matrix, and
/// optional edge priors emitted by the offline Stim export step.
struct StimData {
  /// @brief Parity check matrix (H) in CSR format.
  SparseCSR H;
  /// @brief Observables matrix (O) in CSR format.
  SparseCSR O;
  /// @brief Per-edge error probabilities.
  std::vector<double> priors;
};

/// @brief Load a sparse CSR matrix from a binary file.
/// @param path File path (expects nrows, ncols, nnz header + indptr + indices).
/// @param out Output SparseCSR struct.
/// @return True on success.
bool load_csr(const std::string &path, SparseCSR &out);

/// @brief Load Stim-derived data (H, O, priors) from a directory.
/// @param data_dir Directory containing H_csr.bin, O_csr.bin, priors.bin.
/// @return Populated StimData (check H.loaded() for success).
StimData load_stim_data(const std::string &data_dir);
