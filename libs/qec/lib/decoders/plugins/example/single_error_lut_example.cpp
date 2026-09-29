/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include <cassert>
#include <map>
#include <vector>

namespace cudaq::qec {

/// @brief This is a simple LUT (LookUp Table) decoder that demonstrates how to
/// build a simple decoder that can correctly decode errors during a single bit
/// flip in the block.
class single_error_lut_example : public decoder {
private:
  std::map<std::string, std::size_t> single_qubit_err_signatures;

public:
  single_error_lut_example(cudaq::qec::decoder_init inputs,
                           decode_result_type requested_output,
                           const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    // The requested result form is validated here, at construction, so an
    // unsupported request fails at setup rather than on the first decode. This
    // example produces an error frame only; a decoder that can also project to
    // observables would instead call project_errors_to_observables() before
    // returning.
    if (requested_output != decode_result_type::errors)
      throw std::invalid_argument(
          "single_error_lut_example produces an error frame only; construct it "
          "for error output");
    const auto &H = get_inputs().detector_error_matrix();
    // Decoder-specific constructor arguments can be placed in `params`.

    // The loop below sets err_sig[r] = '1' (not XOR-toggle), so canonicalize
    // to drop GF(2)-duplicate row indices first.
    std::vector<std::vector<std::uint32_t>> H_e2d =
        H.canonicalize().to_nested_csc();

    for (std::size_t qErr = 0; qErr < block_size; qErr++) {
      std::string err_sig(syndrome_size, '0');
      for (std::uint32_t r : H_e2d[qErr])
        err_sig[r] = '1';
      // printf("Adding err_sig=%s for qErr=%lu\n", err_sig.c_str(), qErr);
      single_qubit_err_signatures.insert({err_sig, qErr});
    }
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    // This is a simple decoder that simply results
    decoder_result result{false, std::vector<float_t>(block_size, 0.0)};

    // Convert syndrome to a string
    std::string syndrome_str(syndrome.size(), '0');
    assert(syndrome_str.length() == syndrome_size);
    bool anyErrors = false;
    for (std::size_t i = 0; i < syndrome_size; i++) {
      if (cudaq::qec::convert_soft_to_hard(syndrome[i])) {
        syndrome_str[i] = '1';
        anyErrors = true;
      }
    }

    if (!anyErrors) {
      result.converged = true;
      return result;
    }

    auto it = single_qubit_err_signatures.find(syndrome_str);
    if (it != single_qubit_err_signatures.end()) {
      assert(it->second < block_size);
      result.converged = true;
      result.result[it->second] = 1.0;
    } else {
      // Leave result.converged set to false.
    }

    return result;
  }

  virtual ~single_error_lut_example() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      single_error_lut_example, static std::unique_ptr<decoder> create(
                                    cudaq::qec::decoder_init inputs,
                                    std::optional<decode_result_type> output,
                                    const cudaqx::heterogeneous_map &params) {
        return std::make_unique<single_error_lut_example>(
            std::move(inputs), output.value_or(decode_result_type::errors),
            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(single_error_lut_example)

// Registering a parameter schema alongside the decoder makes the decoder
// configurable through the realtime decoding YAML (`decoder_custom_args`)
// without any changes to the cudaq-qec libraries: the YAML layer converts the
// section to the heterogeneous_map this decoder's constructor receives, using
// only this description. This decoder takes no parameters, so the schema is
// empty; list each accepted key and its param_kind here otherwise.
namespace {
struct schema_registrar {
  schema_registrar() {
    cudaq::qec::decoding::config::register_decoder_schema(
        {"single_error_lut_example", {}});
  }
};
schema_registrar register_schema;
} // namespace

} // namespace cudaq::qec
