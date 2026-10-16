/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq.h"
#include "cudaq/qec/realtime/decoding.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include "cudaq/realtime.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

extern "C" void cudaqx_qec_realtime_device_call_service_force_link();
// Self-verification hook (see assertion below): proves the device_call actually
// traversed the cudaq-realtime host-dispatch ring to the service instead of
// silently bypassing to the direct trampoline.
extern "C" std::uint64_t cudaqx_qec_device_call_dispatch_count();

namespace {

namespace config = cudaq::qec::decoding::config;

constexpr std::uint64_t kDecoderId = 0;
constexpr std::uint64_t kBlockSize = 3;
constexpr std::uint64_t kSyndromeSize = 3;
constexpr std::uint64_t kSyndromeTag = 1;
constexpr std::uint64_t kRunShots = 1;
constexpr std::size_t kActiveSyndromeIndex = 1;
constexpr std::size_t kSparseEntriesPerRow = 2;
constexpr std::int64_t kSparseRowEnd = -1;
constexpr double kUniformErrorRate = 0.1;
constexpr std::int64_t kExpectedCorrection = 1;

std::vector<std::int64_t> make_identity_sparse_matrix() {
  std::vector<std::int64_t> sparse_matrix;
  sparse_matrix.reserve(kSparseEntriesPerRow * kSyndromeSize);
  for (std::uint64_t column = {}; column < kSyndromeSize; ++column) {
    sparse_matrix.push_back(static_cast<std::int64_t>(column));
    sparse_matrix.push_back(kSparseRowEnd);
  }
  return sparse_matrix;
}

config::multi_decoder_config make_config() {
  config::decoder_config decoder_config;
  decoder_config.id = kDecoderId;
  decoder_config.type = "pymatching";
  decoder_config.block_size = kBlockSize;
  decoder_config.syndrome_size = kSyndromeSize;

  const auto identity_sparse_matrix = make_identity_sparse_matrix();
  decoder_config.H_sparse = identity_sparse_matrix;
  decoder_config.O_sparse = identity_sparse_matrix;
  decoder_config.D_sparse = identity_sparse_matrix;
  decoder_config.error_rate_vec =
      std::vector<double>(kBlockSize, kUniformErrorRate);

  cudaqx::heterogeneous_map pymatching_args;
  pymatching_args.insert("merge_strategy", "smallest_weight");
  decoder_config.decoder_custom_args = pymatching_args;

  config::multi_decoder_config multi_config;
  multi_config.decoders.push_back(decoder_config);
  return multi_config;
}

struct DecoderGuard {
  bool armed = false;
  ~DecoderGuard() {
    if (armed)
      config::finalize_decoders();
  }
};

struct RealtimeGuard {
  bool armed = false;
  ~RealtimeGuard() {
    if (armed)
      cudaq::realtime::finalize();
  }
};

__qpu__ std::int64_t pymatching_device_call_kernel() {
  constexpr std::uint64_t kKernelDecoderId = 0;
  constexpr std::uint64_t kKernelBlockSize = 3;
  constexpr std::uint64_t kKernelSyndromeSize = 3;
  constexpr std::uint64_t kKernelSyndromeTag = 1;
  constexpr std::size_t kKernelActiveSyndromeIndex = 1;

  cudaq::qec::decoding::reset_decoder(/*decoder_id=*/kKernelDecoderId);

  std::vector<bool> syndrome(kKernelSyndromeSize);
  for (std::size_t i = 0; i < kKernelSyndromeSize; ++i)
    syndrome[i] = false;
  syndrome[kKernelActiveSyndromeIndex] = true;
  cudaq::qec::decoding::enqueue_syndromes_test(
      /*decoder_id=*/kKernelDecoderId, syndrome, /*tag=*/kKernelSyndromeTag);

  auto corrections = cudaq::qec::decoding::get_corrections(
      /*decoder_id=*/kKernelDecoderId, /*return_size=*/kKernelBlockSize,
      /*reset=*/true);
  return corrections[kKernelActiveSyndromeIndex] ? std::int64_t{1}
                                                 : std::int64_t{0};
}

bool is_gpu_available() {
  int device_count = 0;
  return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
}

void initialize_realtime() {
  int argc = 1;
  char program[] = "test_pymatching_device_call_realtime";
  char *argv[] = {program, nullptr};
  cudaq::realtime::initialize(argc, argv);
}

} // namespace

TEST(PyMatchingDeviceCallRealtime, HostDispatch) {
  // FIXME: Remove this guard once cudaq-realtime host_dispatch no longer
  // requires a visible CUDA device before dispatch reaches the host service.
  if (!is_gpu_available())
    GTEST_SKIP() << "No GPU available; cudaq-realtime host_dispatch "
                    "currently requires a visible CUDA device.";

  // Keep the service library loaded so CUDA-Q can discover its
  // cudaqGetDeviceCallServicePluginInfo symbol via dlsym(RTLD_DEFAULT).
  cudaqx_qec_realtime_device_call_service_force_link();

  auto decoder_config = make_config();
  ASSERT_EQ(config::configure_decoders(decoder_config), 0);
  DecoderGuard decoder_guard{true};

  initialize_realtime();
  RealtimeGuard realtime_guard{true};

  const auto results = cudaq::run(kRunShots, pymatching_device_call_kernel);
  ASSERT_EQ(results.size(), kRunShots);
  EXPECT_EQ(results[0], kExpectedCorrection);

  // Self-verify the device_call actually went over the cudaq-realtime
  // host-dispatch ring to the server (a correct result alone does not prove
  // this
  // -- the direct trampoline would also produce it).
  EXPECT_GT(cudaqx_qec_device_call_dispatch_count(), 0u)
      << "device_call did not reach the host-dispatch service; it likely "
         "bypassed to the direct trampoline (missing -frealtime-lowering on "
         "the "
         "device wrappers, or the wrong simulation library was linked).";
}
