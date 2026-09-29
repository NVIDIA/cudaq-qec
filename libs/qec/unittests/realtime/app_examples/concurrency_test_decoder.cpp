/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/decoder.h"

#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudaq::qec {
namespace {

class reusable_decode_barrier {
public:
  explicit reusable_decode_barrier(std::size_t participants)
      : participants_(participants) {
    if (participants_ < 2)
      throw std::invalid_argument(
          "QEC_CONCURRENCY_TEST_DECODERS must be at least 2");
  }

  void arrive_and_wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    const auto generation = generation_;
    if (++arrived_ == participants_) {
      arrived_ = 0;
      ++generation_;
      std::cout << "QEC_CONCURRENCY_TEST_BARRIER generation=" << generation_
                << " participants=" << participants_ << std::endl;
      cv_.notify_all();
      return;
    }

    if (!cv_.wait_for(lock, std::chrono::seconds(30),
                      [&] { return generation_ != generation; }))
      throw std::runtime_error(
          "timed out waiting for concurrent decoder workers");
  }

private:
  const std::size_t participants_;
  std::size_t arrived_ = 0;
  std::size_t generation_ = 0;
  std::mutex mutex_;
  std::condition_variable cv_;
};

std::size_t barrier_participants() {
  const char *value = std::getenv("QEC_CONCURRENCY_TEST_DECODERS");
  if (!value || value[0] == '\0')
    throw std::runtime_error("QEC_CONCURRENCY_TEST_DECODERS is required by "
                             "concurrency_test_decoder");

  std::size_t parsed = 0;
  try {
    const auto participants = std::stoull(value, &parsed);
    if (parsed != std::string(value).size())
      throw std::invalid_argument("trailing characters");
    return participants;
  } catch (const std::exception &) {
    throw std::runtime_error(
        "QEC_CONCURRENCY_TEST_DECODERS must be an integer >= 2");
  }
}

reusable_decode_barrier &decode_barrier() {
  static reusable_decode_barrier barrier(barrier_participants());
  return barrier;
}

} // namespace

/// Test-only decoder used to prove that independent decoding-server workers
/// enter decode concurrently. The factory performs one initialization decode
/// per instance, which is deliberately excluded from the barrier. Every
/// subsequent decode rendezvous with all configured instances before returning.
class concurrency_test_decoder : public decoder {
public:
  concurrency_test_decoder(decoder_init inputs,
                           decode_result_type requested_output,
                           const cudaqx::heterogeneous_map &)
      : decoder(std::move(inputs), requested_output) {
    std::cout << "QEC_CONCURRENCY_TEST_DECODER_CONSTRUCTED" << std::endl;
  }

  decoder_result decode(const std::vector<float_t> &) override {
    decoder_result result{
        true, std::vector<float_t>(get_num_observables(), 0.0), std::nullopt};

    if (initialization_probe_) {
      initialization_probe_ = false;
      return result;
    }

    decode_barrier().arrive_and_wait();

    // Give decoder 1 a distinct correction so the application can verify that
    // per-decoder responses were routed back to the corresponding patch.
    if (!result.result.empty() && get_decoder_id() == 1)
      result.result[0] = 1.0;
    return result;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      concurrency_test_decoder,
      static std::unique_ptr<decoder> create(
          decoder_init inputs, std::optional<decode_result_type> output,
          const cudaqx::heterogeneous_map &params) {
        return std::make_unique<concurrency_test_decoder>(
            std::move(inputs), output.value_or(decode_result_type::observables),
            params);
      })

private:
  bool initialization_probe_ = true;
};

CUDAQ_EXT_PT_REGISTER_TYPE(concurrency_test_decoder)

} // namespace cudaq::qec
