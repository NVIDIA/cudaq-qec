/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace cudaq::qec::realtime {

/// @brief CRTP base class for autonomous decoders.
///
/// This provides a common interface for GPU-based decoders that can be
/// integrated with the real-time dispatch system. Derived decoders implement
/// the actual decoding logic while inheriting RPC integration capabilities.
///
/// Design Philosophy:
/// - Zero-CPU data path: Decoders run entirely on GPU without CPU involvement
/// - Event-driven: Decoders respond to incoming data events via RPC protocol
/// - Composable: CRTP pattern allows compile-time polymorphism
///
/// Usage:
/// @code
/// class MyDecoder : public autonomous_decoder<MyDecoder> {
/// public:
///   __device__ void decode_impl(const uint8_t* measurements,
///                               uint8_t* corrections,
///                               std::size_t num_measurements,
///                               std::size_t num_observables);
/// };
/// @endcode
///
/// @tparam Derived The derived decoder class (CRTP pattern)
template <typename Derived>
class autonomous_decoder {
public:
  /// @brief Decode measurements into corrections (CRTP dispatch).
  ///
  /// This is the main entry point that delegates to the derived class's
  /// implementation via CRTP.
  ///
  /// @param measurements Input raw measurements from quantum circuit
  /// @param corrections Output buffer for computed corrections
  /// @param num_measurements Number of input measurements
  /// @param num_observables Number of observable corrections to output
  __device__ void decode(const uint8_t *__restrict__ measurements,
                         uint8_t *__restrict__ corrections,
                         std::size_t num_measurements,
                         std::size_t num_observables) {
    static_cast<Derived *>(this)->decode_impl(
        measurements, corrections, num_measurements, num_observables);
  }

  /// @brief Get the derived decoder instance.
  ///
  /// Provides access to derived-class-specific methods and data.
  ///
  /// @return Reference to the derived decoder
  __device__ __host__ Derived &derived() {
    return *static_cast<Derived *>(this);
  }

  /// @brief Get the derived decoder instance (const version).
  __device__ __host__ const Derived &derived() const {
    return *static_cast<const Derived *>(this);
  }

protected:
  // Protected constructor to prevent direct instantiation
  autonomous_decoder() = default;
  ~autonomous_decoder() = default;

  // Non-copyable, non-movable (decoders are typically global device objects)
  autonomous_decoder(const autonomous_decoder &) = delete;
  autonomous_decoder &operator=(const autonomous_decoder &) = delete;
  autonomous_decoder(autonomous_decoder &&) = delete;
  autonomous_decoder &operator=(autonomous_decoder &&) = delete;
};

} // namespace cudaq::qec::realtime
