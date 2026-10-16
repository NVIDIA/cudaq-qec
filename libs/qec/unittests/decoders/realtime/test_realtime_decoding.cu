/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_realtime_decoding.cu
/// @brief End-to-end test for realtime decoding pipeline with mock decoder.
///
/// This test verifies the complete realtime decoding flow using the
/// cuda-quantum host API and the dispatch kernel linked from
/// libcudaq-realtime.so:
/// 1. GpuRoceTransceiver-style ring buffer communication (cudaHostAllocMapped)
/// 2. Host API wires dispatcher and launches persistent kernel
/// 3. Mock decoder that returns pre-recorded expected corrections
/// 4. Data loaded from config_multi_err_lut.yml and syndromes_multi_err_lut.txt

#include <algorithm>
#include <cstdint>
#include <cuda_runtime.h>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <unistd.h>
#include <vector>

// cuda-quantum host API
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

// cuda-quantum RPC types/hash helper
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

// cuda-quantum kernel types for graph-aware dispatch
#include "cudaq/realtime/daemon/dispatcher/kernel_types.h"

// cudaqx mock decoder (sources in unittests/realtime/)
#include "mock_decode_handler.cuh"
#include "mock_decode_setup.h"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

//==============================================================================
// GpuRoceTransceiver-Style Ring Buffer
//==============================================================================

/// @brief Allocate GpuRoceTransceiver-style ring buffer with mapped memory.
bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                          volatile uint64_t **host_flags_out,
                          volatile uint64_t **device_flags_out,
                          uint8_t **host_data_out, uint8_t **device_data_out) {

  void *host_flags_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_flags_ptr, num_slots * sizeof(uint64_t),
                                  cudaHostAllocMapped);
  if (err != cudaSuccess)
    return false;

  void *device_flags_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_flags_ptr, host_flags_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void *host_data_ptr = nullptr;
  err =
      cudaHostAlloc(&host_data_ptr, num_slots * slot_size, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void *device_data_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_data_ptr, host_data_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    cudaFreeHost(host_data_ptr);
    return false;
  }

  memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));

  *host_flags_out = static_cast<volatile uint64_t *>(host_flags_ptr);
  *device_flags_out = static_cast<volatile uint64_t *>(device_flags_ptr);
  *host_data_out = static_cast<uint8_t *>(host_data_ptr);
  *device_data_out = static_cast<uint8_t *>(device_data_ptr);

  return true;
}

void free_ring_buffer(volatile uint64_t *host_flags, uint8_t *host_data) {
  if (host_flags) {
    cudaFreeHost(const_cast<uint64_t *>(host_flags));
  }
  if (host_data) {
    cudaFreeHost(host_data);
  }
}

// Helper function to check if a GPU is available
bool isGpuAvailable() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);
  return (err == cudaSuccess && deviceCount > 0);
}

} // namespace

//==============================================================================
// Helper Kernels
//==============================================================================

/// @brief Initialize mock_decoder instance on device using placement new
__global__ void
init_mock_decoder_kernel(cudaq::qec::realtime::mock_decoder *d_decoder,
                         cudaq::qec::realtime::mock_decoder_context *d_ctx) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    new (d_decoder) cudaq::qec::realtime::mock_decoder(*d_ctx);
  }
}

//==============================================================================
// Test Fixture
//==============================================================================

class RealtimeDecodingTest : public ::testing::Test {
protected:
  // Convenience aliases for types from the shared header
  using SyndromeEntry = cudaq::qec::realtime::SyndromeEntry;

  void SetUp() override {
    // Skip all tests if no GPU is available
    if (!isGpuAvailable()) {
      GTEST_SKIP() << "No GPU available, skipping realtime decoding tests";
    }

    // Enable host-mapped memory before any CUDA context creation.
    cudaError_t flags_err = cudaSetDeviceFlags(cudaDeviceMapHost);
    ASSERT_TRUE(flags_err == cudaSuccess ||
                flags_err == cudaErrorSetOnActiveProcess);

    data_dir_ = std::string(TEST_DATA_DIR);

    // Load config
    std::ifstream config_file(data_dir_ + "/config_multi_err_lut.yml");
    ASSERT_TRUE(config_file.good()) << "Could not open config file";
    std::string config_content((std::istreambuf_iterator<char>(config_file)),
                               std::istreambuf_iterator<char>());

    syndrome_size_ =
        cudaq::qec::realtime::parse_scalar(config_content, "syndrome_size");
    block_size_ =
        cudaq::qec::realtime::parse_scalar(config_content, "block_size");
    ASSERT_GT(syndrome_size_, 0u);
    ASSERT_GT(block_size_, 0u);

    // Load syndrome test data
    syndromes_ = cudaq::qec::realtime::load_syndromes(
        data_dir_ + "/syndromes_multi_err_lut.txt", syndrome_size_);
    ASSERT_FALSE(syndromes_.empty()) << "No syndrome data loaded";

    // Build lookup table for mock decoder
    num_lookup_entries_ = syndromes_.size();
    lookup_measurements_.resize(num_lookup_entries_ * syndrome_size_);
    lookup_corrections_.resize(num_lookup_entries_);

    for (std::size_t i = 0; i < num_lookup_entries_; ++i) {
      memcpy(lookup_measurements_.data() + i * syndrome_size_,
             syndromes_[i].measurements.data(), syndrome_size_);
      lookup_corrections_[i] = syndromes_[i].expected_correction;
    }

    // Allocate device memory for lookup tables
    CUDA_CHECK(
        cudaMalloc(&d_lookup_measurements_, lookup_measurements_.size()));
    CUDA_CHECK(cudaMalloc(&d_lookup_corrections_, lookup_corrections_.size()));
    CUDA_CHECK(cudaMemcpy(d_lookup_measurements_, lookup_measurements_.data(),
                          lookup_measurements_.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lookup_corrections_, lookup_corrections_.data(),
                          lookup_corrections_.size(), cudaMemcpyHostToDevice));

    // Set up mock decoder context
    ctx_.num_measurements = syndrome_size_;
    ctx_.num_observables = 1;
    ctx_.lookup_measurements = d_lookup_measurements_;
    ctx_.lookup_corrections = d_lookup_corrections_;
    ctx_.num_lookup_entries = num_lookup_entries_;

    // Allocate device context
    CUDA_CHECK(cudaMalloc(&d_ctx_,
                          sizeof(cudaq::qec::realtime::mock_decoder_context)));
    CUDA_CHECK(cudaMemcpy(d_ctx_, &ctx_, sizeof(ctx_), cudaMemcpyHostToDevice));

    // Allocate and initialize CRTP decoder instance
    CUDA_CHECK(
        cudaMalloc(&d_decoder_, sizeof(cudaq::qec::realtime::mock_decoder)));

    // Initialize decoder using placement new kernel
    init_mock_decoder_kernel<<<1, 1>>>(d_decoder_, d_ctx_);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Set global decoder for RPC-style calls
    cudaq::qec::realtime::set_mock_decoder(d_decoder_);

    // Allocate ring buffers (with space for RPCHeader)
    slot_size_ = sizeof(cudaq::realtime::RPCHeader) +
                 std::max(syndrome_size_, static_cast<std::size_t>(256));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host_,
                                     &rx_flags_, &rx_data_host_, &rx_data_));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host_,
                                     &tx_flags_, &tx_data_host_, &tx_data_));

    // Allocate control variables
    void *tmp_shutdown = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    shutdown_flag_ = static_cast<volatile int *>(tmp_shutdown);
    void *tmp_d_shutdown = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    d_shutdown_flag_ = static_cast<volatile int *>(tmp_d_shutdown);
    *shutdown_flag_ = 0;
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(const_cast<int *>(d_shutdown_flag_), &zero,
                          sizeof(int), cudaMemcpyHostToDevice));

    // Allocate stats
    CUDA_CHECK(cudaMalloc(&d_stats_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats_, 0, sizeof(uint64_t)));

    // Set up function table using library helper
    CUDA_CHECK(
        cudaMalloc(&d_function_entries_, sizeof(cudaq_function_entry_t)));
    cudaq::qec::realtime::setup_mock_decode_function_table(d_function_entries_);

    // Host API wiring
    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.device_id = 0;
    config.num_blocks = 1;
    config.threads_per_block = 32;
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.vp_id = 0;
    config.kernel_type = CUDAQ_KERNEL_REGULAR;
    config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

    ASSERT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_),
              CUDAQ_OK);

    cudaq_ringbuffer_t ringbuffer{};
    ringbuffer.rx_flags = rx_flags_;
    ringbuffer.tx_flags = tx_flags_;
    ringbuffer.rx_data = rx_data_;
    ringbuffer.tx_data = tx_data_;
    ringbuffer.rx_stride_sz = slot_size_;
    ringbuffer.tx_stride_sz = slot_size_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer),
              CUDAQ_OK);

    cudaq_function_table_t table{};
    table.entries = d_function_entries_;
    table.count = 1;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table),
              CUDAQ_OK);

    ASSERT_EQ(
        cudaq_dispatcher_set_control(dispatcher_, d_shutdown_flag_, d_stats_),
        CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_launch_fn(
                  dispatcher_,
                  &cudaq::qec::realtime::mock_decode_launch_dispatch_kernel),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_start(dispatcher_), CUDAQ_OK);
  }

  void TearDown() override {
    if (shutdown_flag_) {
      *shutdown_flag_ = 1;
      __sync_synchronize();
    }
    if (dispatcher_) {
      cudaq_dispatcher_stop(dispatcher_);
      cudaq_dispatcher_destroy(dispatcher_);
      dispatcher_ = nullptr;
    }

    if (manager_) {
      cudaq_dispatch_manager_destroy(manager_);
      manager_ = nullptr;
    }

    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);

    if (shutdown_flag_)
      cudaFreeHost(const_cast<int *>(shutdown_flag_));
    if (d_lookup_measurements_)
      cudaFree(d_lookup_measurements_);
    if (d_lookup_corrections_)
      cudaFree(d_lookup_corrections_);
    if (d_decoder_)
      cudaFree(d_decoder_);
    if (d_ctx_)
      cudaFree(d_ctx_);
    if (d_stats_)
      cudaFree(d_stats_);
    if (d_function_entries_)
      cudaFree(d_function_entries_);
    if (graph_exec_)
      cudaGraphExecDestroy(graph_exec_);
    if (graph_)
      cudaGraphDestroy(graph_);
    if (d_graph_io_ctx_)
      cudaFree(d_graph_io_ctx_);
  }

  void setup_function_table_graph() {
    // Allocate GraphIOContext on device (dispatch kernel fills it before
    // each fire-and-forget graph launch)
    CUDA_CHECK(
        cudaMalloc(&d_graph_io_ctx_, sizeof(cudaq::realtime::GraphIOContext)));
    CUDA_CHECK(cudaMemset(d_graph_io_ctx_, 0,
                          sizeof(cudaq::realtime::GraphIOContext)));

    // Create CUDA graph with decoder kernel
    cudaStream_t capture_stream;
    CUDA_CHECK(cudaStreamCreate(&capture_stream));

    CUDA_CHECK(
        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal));
    cudaq::qec::realtime::mock_decode_graph_kernel<<<1, 1, 0, capture_stream>>>(
        d_graph_io_ctx_);
    CUDA_CHECK(cudaStreamEndCapture(capture_stream, &graph_));

    // Instantiate for device launch with device launch flag
    CUDA_CHECK(cudaGraphInstantiateWithFlags(
        &graph_exec_, graph_, cudaGraphInstantiateFlagDeviceLaunch));

    // Upload graph to device
    CUDA_CHECK(cudaGraphUpload(graph_exec_, capture_stream));
    CUDA_CHECK(cudaStreamSynchronize(capture_stream));
    CUDA_CHECK(cudaStreamDestroy(capture_stream));

    // Set up function table entry with graph
    CUDA_CHECK(
        cudaMalloc(&d_function_entries_, sizeof(cudaq_function_entry_t)));

    cudaq_function_entry_t host_entry{};
    host_entry.handler.graph_exec = graph_exec_;
    host_entry.function_id = cudaq::qec::realtime::MOCK_DECODE_FUNCTION_ID;
    host_entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    host_entry.reserved[0] = 0;
    host_entry.reserved[1] = 0;
    host_entry.reserved[2] = 0;

    // Schema: same as device call mode
    host_entry.schema.num_args = 1;
    host_entry.schema.num_results = 1;
    host_entry.schema.reserved = 0;
    host_entry.schema.args[0].type_id = CUDAQ_TYPE_BIT_PACKED;
    host_entry.schema.args[0].reserved[0] = 0;
    host_entry.schema.args[0].reserved[1] = 0;
    host_entry.schema.args[0].reserved[2] = 0;
    host_entry.schema.args[0].size_bytes = 16;
    host_entry.schema.args[0].num_elements = 128;
    host_entry.schema.results[0].type_id = CUDAQ_TYPE_UINT8;
    host_entry.schema.results[0].reserved[0] = 0;
    host_entry.schema.results[0].reserved[1] = 0;
    host_entry.schema.results[0].reserved[2] = 0;
    host_entry.schema.results[0].size_bytes = 1;
    host_entry.schema.results[0].num_elements = 1;

    CUDA_CHECK(cudaMemcpy(d_function_entries_, &host_entry,
                          sizeof(cudaq_function_entry_t),
                          cudaMemcpyHostToDevice));

    func_count_ = 1;
  }

  /// @brief Write syndrome to RX buffer in RPC format.
  void write_rpc_request(std::size_t slot,
                         const std::vector<uint8_t> &measurements) {
    uint8_t *slot_data =
        const_cast<uint8_t *>(rx_data_host_) + slot * slot_size_;

    // Write RPCHeader
    cudaq::realtime::RPCHeader *header =
        reinterpret_cast<cudaq::realtime::RPCHeader *>(slot_data);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = cudaq::qec::realtime::MOCK_DECODE_FUNCTION_ID;
    header->arg_len = static_cast<std::uint32_t>(measurements.size());
    header->request_id = static_cast<std::uint32_t>(slot);
    header->ptp_timestamp = 0;

    memcpy(slot_data + sizeof(cudaq::realtime::RPCHeader), measurements.data(),
           measurements.size());
  }

  /// @brief Read response from TX buffer (symmetric: dispatch kernel writes to
  /// TX).
  bool read_rpc_response(std::size_t slot, uint8_t &correction,
                         std::int32_t *status_out = nullptr,
                         std::uint32_t *result_len_out = nullptr) {
    __sync_synchronize();
    const uint8_t *slot_data =
        const_cast<uint8_t *>(tx_data_host_) + slot * slot_size_;

    // Read RPCResponse
    const cudaq::realtime::RPCResponse *response =
        reinterpret_cast<const cudaq::realtime::RPCResponse *>(slot_data);

    if (response->magic != cudaq::realtime::RPC_MAGIC_RESPONSE) {
      return false;
    }
    if (status_out)
      *status_out = response->status;
    if (result_len_out)
      *result_len_out = response->result_len;

    if (response->status != 0) {
      return false;
    }

    // Read correction data after response header
    correction = *(slot_data + sizeof(cudaq::realtime::RPCResponse));
    return true;
  }

  std::string data_dir_;
  std::size_t syndrome_size_ = 0;
  std::size_t block_size_ = 0;
  std::vector<SyndromeEntry> syndromes_;

  std::vector<uint8_t> lookup_measurements_;
  std::vector<uint8_t> lookup_corrections_;
  std::size_t num_lookup_entries_ = 0;
  uint8_t *d_lookup_measurements_ = nullptr;
  uint8_t *d_lookup_corrections_ = nullptr;

  cudaq::qec::realtime::mock_decoder_context ctx_;
  cudaq::qec::realtime::mock_decoder_context *d_ctx_ = nullptr;
  cudaq::qec::realtime::mock_decoder *d_decoder_ = nullptr;

  static constexpr std::size_t num_slots_ = 4;
  std::size_t slot_size_ = 256;
  volatile uint64_t *rx_flags_host_ = nullptr;
  volatile uint64_t *tx_flags_host_ = nullptr;
  volatile uint64_t *rx_flags_ = nullptr;
  volatile uint64_t *tx_flags_ = nullptr;
  uint8_t *rx_data_host_ = nullptr;
  uint8_t *tx_data_host_ = nullptr;
  uint8_t *rx_data_ = nullptr;
  uint8_t *tx_data_ = nullptr;

  volatile int *shutdown_flag_ = nullptr;
  volatile int *d_shutdown_flag_ = nullptr;
  uint64_t *d_stats_ = nullptr;

  // Function table for dispatch kernel
  cudaq_function_entry_t *d_function_entries_ = nullptr;
  std::size_t func_count_ = 1;

  // Graph launch support
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaq::realtime::GraphIOContext *d_graph_io_ctx_ =
      nullptr; // IO context for graph

  // Host API handles
  cudaq_dispatch_manager_t *manager_ = nullptr;
  cudaq_dispatcher_t *dispatcher_ = nullptr;
};

//==============================================================================
// Tests
//==============================================================================

/// @brief End-to-end test over the full syndromes file.
/// This verifies the integration between cudaqx and cuda-quantum.
TEST_F(RealtimeDecodingTest, DispatchKernelAllShots) {
  const std::size_t num_test_shots = syndromes_.size();
  std::size_t correct_count = 0;

  for (std::size_t i = 0; i < num_test_shots; ++i) {
    std::size_t slot = i % num_slots_;
    const auto &entry = syndromes_[i];

    // Wait for slot to be free
    int timeout = 50;
    while (rx_flags_host_[slot] != 0 && timeout-- > 0) {
      usleep(100);
    }
    ASSERT_GT(timeout, 0) << "Timeout waiting for RX slot " << slot;

    // Send request
    write_rpc_request(slot, entry.measurements);
    __sync_synchronize();
    const_cast<volatile uint64_t *>(rx_flags_host_)[slot] =
        reinterpret_cast<uint64_t>(rx_data_ + slot * slot_size_);

    // Wait for response
    timeout = 50;
    while (tx_flags_host_[slot] == 0 && timeout-- > 0) {
      usleep(100);
    }
    if (timeout <= 0) {
      std::cerr << "DispatchKernelAllShots timeout diagnostics:\n"
                << "  slot = " << slot << "\n"
                << "  rx_flags_host[slot] = " << rx_flags_host_[slot] << "\n"
                << "  tx_flags_host_[slot] = " << tx_flags_host_[slot] << "\n"
                << std::flush;
    }
    ASSERT_GT(timeout, 0) << "Timeout waiting for TX slot " << slot;

    // Check result
    uint8_t correction = 0;
    std::int32_t status = 0;
    std::uint32_t result_len = 0;
    if (read_rpc_response(slot, correction, &status, &result_len)) {
      if (correction == entry.expected_correction) {
        correct_count++;
      } else {
        std::cerr << "RPC mismatch slot " << slot << " status=" << status
                  << " result_len=" << result_len
                  << " expected=" << static_cast<int>(entry.expected_correction)
                  << " got=" << static_cast<int>(correction) << "\n";
      }
    } else {
      std::cerr << "RPC failure slot " << slot << " status=" << status
                << " result_len=" << result_len
                << " expected=" << static_cast<int>(entry.expected_correction)
                << " got=" << static_cast<int>(correction) << "\n";
    }

    // Clear TX flag
    const_cast<volatile uint64_t *>(tx_flags_host_)[slot] = 0;
  }

  double accuracy = 100.0 * correct_count / num_test_shots;
  std::cout << "Dispatch kernel mock decoder accuracy: " << correct_count << "/"
            << num_test_shots << " (" << accuracy << "%)" << std::endl;

  EXPECT_EQ(correct_count, num_test_shots)
      << "All shots should decode correctly";
}

/// @brief Full end-to-end test for graph launch mode using proper graph-based
/// dispatch. This test bypasses the host API to use the new
/// cudaq_create_dispatch_graph_* API which properly wraps the dispatch kernel
/// in a graph, enabling device-side graph launch.
TEST_F(RealtimeDecodingTest, DispatchKernelAllShotsGraphLaunch) {
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  if (prop.major < 9) {
    GTEST_SKIP()
        << "Graph device launch requires compute capability 9.0+, found "
        << prop.major << "." << prop.minor;
  }

  // Shutdown existing dispatcher (we'll manage dispatch directly)
  *shutdown_flag_ = 1;
  __sync_synchronize();
  ASSERT_EQ(cudaq_dispatcher_stop(dispatcher_), CUDAQ_OK);

  // Clean up device call function table
  if (d_function_entries_) {
    cudaFree(d_function_entries_);
    d_function_entries_ = nullptr;
  }

  // Set up graph-based function table
  setup_function_table_graph();

  // Create stream for dispatch
  cudaStream_t dispatch_stream;
  CUDA_CHECK(cudaStreamCreate(&dispatch_stream));

  // Reset shutdown flag for the new dispatch kernel
  *shutdown_flag_ = 0;
  __sync_synchronize();

  // Create dispatch graph context using the NEW API
  // This wraps the dispatch kernel in a graph so device-side cudaGraphLaunch
  // works
  cudaq_dispatch_graph_context *dispatch_ctx = nullptr;
  cudaError_t create_err = cudaq_create_dispatch_graph_regular(
      rx_flags_, tx_flags_, rx_data_, tx_data_, slot_size_, slot_size_,
      d_function_entries_, func_count_, d_graph_io_ctx_, d_shutdown_flag_,
      d_stats_, num_slots_, 1, 32, /*triggered_graph_exec=*/nullptr,
      dispatch_stream, &dispatch_ctx);

  if (create_err != cudaSuccess) {
    cudaStreamDestroy(dispatch_stream);
    GTEST_SKIP() << "Failed to create dispatch graph: "
                 << cudaGetErrorString(create_err);
  }

  // Launch dispatch graph - now device-side cudaGraphLaunch will work!
  CUDA_CHECK(cudaq_launch_dispatch_graph(dispatch_ctx, dispatch_stream));

  const std::size_t num_test_shots = syndromes_.size();
  std::size_t correct_count = 0;

  // Create separate stream for polling to avoid blocking
  cudaStream_t poll_stream;
  CUDA_CHECK(cudaStreamCreate(&poll_stream));

  for (std::size_t i = 0; i < num_test_shots; ++i) {
    std::size_t slot = i % num_slots_;
    const auto &entry = syndromes_[i];

    // Wait for slot to be free
    int timeout = 100;
    while (rx_flags_host_[slot] != 0 && timeout-- > 0) {
      usleep(1000);
    }
    ASSERT_GT(timeout, 0) << "Timeout waiting for RX slot " << slot;

    // Send request
    write_rpc_request(slot, entry.measurements);
    __sync_synchronize();
    const_cast<volatile uint64_t *>(rx_flags_host_)[slot] =
        reinterpret_cast<uint64_t>(rx_data_ + slot * slot_size_);

    // Wait for response (poll using separate stream)
    timeout = 100;
    while (timeout-- > 0) {
      __sync_synchronize();
      if (tx_flags_host_[slot] != 0)
        break;
      usleep(1000);
    }
    if (timeout <= 0) {
      std::cerr << "DispatchKernelAllShotsGraphLaunch timeout diagnostics:\n"
                << "  slot = " << slot << "\n"
                << "  rx_flags_host[slot] = " << rx_flags_host_[slot] << "\n"
                << "  tx_flags_host_[slot] = " << tx_flags_host_[slot] << "\n"
                << std::flush;
    }
    ASSERT_GT(timeout, 0) << "Timeout waiting for TX slot " << slot;

    // Check result
    uint8_t correction = 0;
    std::int32_t status = 0;
    std::uint32_t result_len = 0;
    if (read_rpc_response(slot, correction, &status, &result_len)) {
      if (correction == entry.expected_correction) {
        correct_count++;
      } else {
        std::cerr << "Graph RPC mismatch slot " << slot << " status=" << status
                  << " result_len=" << result_len
                  << " expected=" << static_cast<int>(entry.expected_correction)
                  << " got=" << static_cast<int>(correction) << "\n";
      }
    } else {
      std::cerr << "Graph RPC failure slot " << slot << " status=" << status
                << " result_len=" << result_len
                << " expected=" << static_cast<int>(entry.expected_correction)
                << " got=" << static_cast<int>(correction) << "\n";
    }

    // Clear TX flag
    const_cast<volatile uint64_t *>(tx_flags_host_)[slot] = 0;
  }

  // Signal shutdown
  *shutdown_flag_ = 1;
  __sync_synchronize();
  usleep(100000); // Give kernel time to exit

  // Cleanup
  CUDA_CHECK(cudaStreamSynchronize(dispatch_stream));
  CUDA_CHECK(cudaq_destroy_dispatch_graph(dispatch_ctx));
  CUDA_CHECK(cudaStreamDestroy(poll_stream));
  CUDA_CHECK(cudaStreamDestroy(dispatch_stream));

  double accuracy = 100.0 * correct_count / num_test_shots;
  std::cout << "Graph dispatch mock decoder accuracy: " << correct_count << "/"
            << num_test_shots << " (" << accuracy << "%)" << std::endl;

  EXPECT_EQ(correct_count, num_test_shots)
      << "All shots should decode correctly with graph launch";
}
