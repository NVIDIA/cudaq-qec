/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/realtime/graph_resources.h"

#include <atomic>
#include <memory>

namespace cudaq::qec {

// Fake graph-capable decoder for isolated lifecycle / CLI tests. Capture
// returns a heap graph_resources whose graph_exec is a non-null dummy token
// that must never reach a real CUDA API.
class cc_test_graph_decoder : public decoder {
public:
  inline static std::atomic<int> last_reserved_sms{-1};
  inline static std::atomic<int> release_count{0};

  cc_test_graph_decoder(const sparse_binary_matrix &H,
                        const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    (void)params;
    set_O_sparse(std::vector<std::vector<uint32_t>>{{0}});
    set_D_sparse(std::vector<std::vector<uint32_t>>{{0}});
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    decoder_result result;
    result.converged = true;
    result.result = {syndrome.empty() ? float_t{0} : syndrome.front()};
    return result;
  }

  bool supports_graph_dispatch() const override { return true; }

  void *capture_decode_graph(int reserved_sms = 0) override {
    last_reserved_sms.store(reserved_sms);
    auto *gr = new cudaq::qec::realtime::graph_resources();
    gr->graph_exec = reinterpret_cast<cudaGraphExec_t>(this);
    return gr;
  }

  void release_decode_graph(void *graph_resources) override {
    ++release_count;
    delete static_cast<cudaq::qec::realtime::graph_resources *>(
        graph_resources);
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      cc_test_graph_decoder,
      static std::unique_ptr<decoder> create(
          const decoder_init &init, const cudaqx::heterogeneous_map &params) {
        return make_pcm_decoder<cc_test_graph_decoder>(init, params);
      })
};

// Same as above, but capture_decode_graph() returns nullptr so DecodingServer
// hits the "requires graph dispatch" throw.
class cc_test_null_graph_decoder : public decoder {
public:
  cc_test_null_graph_decoder(const sparse_binary_matrix &H,
                             const cudaqx::heterogeneous_map &params)
      : decoder(H) {
    (void)params;
    set_O_sparse(std::vector<std::vector<uint32_t>>{{0}});
    set_D_sparse(std::vector<std::vector<uint32_t>>{{0}});
  }

  decoder_result decode(const std::vector<float_t> &) override { return {}; }

  bool supports_graph_dispatch() const override { return true; }

  void *capture_decode_graph(int reserved_sms = 0) override {
    (void)reserved_sms;
    return nullptr;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      cc_test_null_graph_decoder,
      static std::unique_ptr<decoder> create(
          const decoder_init &init, const cudaqx::heterogeneous_map &params) {
        return make_pcm_decoder<cc_test_null_graph_decoder>(init, params);
      })
};

} // namespace cudaq::qec
