/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/environment.h"
#include "cudaq/qec/logger.h"

#include "quantinuum_decoding.h"
#include "../realtime_decoding.h"
#include <dlfcn.h>
#include <vector>

// Get an environment variable to determine if we should use the simulated
// implementation or the real implementation, using call_once.
static bool z_use_private_impl =
    cudaq::qec::get_env_bool("CUDAQ_QTM_PRIVATE_IMPL", false);

// This class dynamically loads a private library for special test
// configurations. It is not needed for normal usage.
class quantinuum_private_handler {
public:
  /// Dynamic library handle.
  void *handle = nullptr;

  /// Function pointers for special test mode (internal usage only).
  void (*enqueue_syndromes_ui64_private)(std::uint64_t, std::uint64_t,
                                         std::uint64_t,
                                         std::uint64_t) = nullptr;
  std::uint64_t (*get_corrections_ui64_private)(std::uint64_t, std::uint64_t,
                                                std::uint64_t) = nullptr;
  void (*reset_decoder_ui64_private)(std::uint64_t) = nullptr;

  /// Constructor
  quantinuum_private_handler() {
    // Don't throw errors if this doesn't exist unless z_use_private_impl is
    // true.
    handle = dlopen("libcudaq-qec-realtime-decoding-quantinuum-private.so",
                    RTLD_LAZY | RTLD_GLOBAL);
    if (handle) {
      enqueue_syndromes_ui64_private = reinterpret_cast<void (*)(
          std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t)>(
          dlsym(handle, "enqueue_syndromes_ui64_private"));
      get_corrections_ui64_private = reinterpret_cast<std::uint64_t (*)(
          std::uint64_t, std::uint64_t, std::uint64_t)>(
          dlsym(handle, "get_corrections_ui64_private"));
      reset_decoder_ui64_private = reinterpret_cast<void (*)(std::uint64_t)>(
          dlsym(handle, "reset_decoder_ui64_private"));
    }
    if (z_use_private_impl) {
      // Verify that the functions exist if this flag is set.
      if (!handle) {
        throw std::runtime_error(
            "libcudaq-qec-realtime-decoding-quantinuum-private.so not found. "
            "Cannot run with CUDAQ_QTM_PRIVATE_IMPL=1.");
      }
      if (!enqueue_syndromes_ui64_private || !get_corrections_ui64_private ||
          !reset_decoder_ui64_private) {
        throw std::runtime_error(
            "Private implementation of quantinuum decoding functions not "
            "found in libcudaq-qec-realtime-decoding-quantinuum-private.so. "
            "Cannot run with CUDAQ_QTM_PRIVATE_IMPL=1.");
      }
    }
  }

  /// Destructor
  ~quantinuum_private_handler() {
    // Clear these so that this library can be dlclosed and dlopened again.
    if (handle)
      dlclose(handle);
    handle = nullptr;
    enqueue_syndromes_ui64_private = nullptr;
    get_corrections_ui64_private = nullptr;
    reset_decoder_ui64_private = nullptr;
  }
};

static quantinuum_private_handler z_quantinuum_private_handler;

void enqueue_syndromes_ui64(std::uint64_t decoder_id,
                            std::uint64_t syndrome_size, std::uint64_t syndrome,
                            std::uint64_t tag) {
  CUDA_QEC_INFO(
      "Entering enqueue_syndromes_ui64 for decoder id: {} and tag: {}",
      decoder_id, tag);
  if (z_use_private_impl) {
    z_quantinuum_private_handler.enqueue_syndromes_ui64_private(
        decoder_id, syndrome_size, syndrome, tag);
  } else {
    std::vector<uint8_t> syndrome_u8(syndrome_size);
    for (std::size_t i = 0; i < syndrome_size; i++)
      syndrome_u8[i] = (syndrome >> i) & 1;
    cudaq::qec::decoding::host::enqueue_syndromes(
        decoder_id, syndrome_u8.data(), syndrome_u8.size(), tag);
  }
}

std::uint64_t get_corrections_ui64(std::uint64_t decoder_id,
                                   std::uint64_t return_size,
                                   std::uint64_t reset) {
  CUDA_QEC_INFO("Entering get_corrections_ui64 for decoder id: {}", decoder_id);
  if (z_use_private_impl) {
    return z_quantinuum_private_handler.get_corrections_ui64_private(
        decoder_id, return_size, reset);
  } else {
    std::vector<uint8_t> corrections(return_size);
    cudaq::qec::decoding::host::get_corrections(decoder_id, corrections.data(),
                                                corrections.size(),
                                                static_cast<bool>(reset));
    std::uint64_t corrections_int = 0;
    for (std::size_t i = 0; i < return_size; i++)
      corrections_int |= static_cast<std::uint64_t>(corrections[i]) << i;
    return corrections_int;
  }
  return 0; // should never get here
}

void reset_decoder_ui64(std::uint64_t decoder_id) {
  CUDA_QEC_INFO("Entering reset_decoder_ui64 for decoder id: {}", decoder_id);
  if (z_use_private_impl) {
    z_quantinuum_private_handler.reset_decoder_ui64_private(decoder_id);
  } else {
    cudaq::qec::decoding::host::reset_decoder(decoder_id);
  }
}
