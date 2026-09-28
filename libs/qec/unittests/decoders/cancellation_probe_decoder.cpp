/*******************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/decoder.h"

namespace cudaq::qec {

/// @brief Test-only decoder that reports std::nullopt once `tok` reports a
/// stop at the level it polls for (`stop_level` param: "soft" by default, or
/// "hard"), and a converged result otherwise, so composite decoders can be
/// checked for forwarding the token they were given unchanged.
class cancellation_probe_decoder : public decoder {
  cancellation_level stop_level_ = cancellation_level::soft;

public:
  cancellation_probe_decoder(cudaq::qec::decoder_init inputs,
                             decode_result_type requested_output,
                             const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    if (params.get<std::string>("stop_level", "soft") == "hard")
      stop_level_ = cancellation_level::hard;
  }

  using decoder::decode;
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    // A default-constructed token never requests a stop, so this always has
    // a value.
    return decode(syndrome, cancellation_token{}).value();
  }

  std::optional<decoder_result> decode(const std::vector<float_t> &syndrome,
                                       cancellation_token tok) override {
    if (tok.stop_requested(stop_level_))
      return std::nullopt;
    decoder_result result;
    result.converged = true;
    result.result.assign(block_size, 0.0);
    if (get_result_type() == decode_result_type::observables)
      result.result.assign(get_num_observables(), 0.0);
    return result;
  }

  using decoder::decode_batch;
  std::optional<std::vector<decoder_result>>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes,
               cancellation_token tok) override {
    std::vector<decoder_result> results;
    for (const auto &s : syndromes) {
      auto r = decode(s, tok);
      if (!r)
        return std::nullopt;
      results.push_back(std::move(*r));
    }
    return results;
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      cancellation_probe_decoder, static std::unique_ptr<decoder> create(
                                      cudaq::qec::decoder_init inputs,
                                      std::optional<decode_result_type> output,
                                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<cancellation_probe_decoder>(
            std::move(inputs), output.value_or(decode_result_type::errors),
            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(cancellation_probe_decoder)

} // namespace cudaq::qec
