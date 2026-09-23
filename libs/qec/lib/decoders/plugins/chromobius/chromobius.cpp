/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "chromobius/decode/decoder.h"
#include "stim.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

namespace {

struct chromobius_init_data {
  stim::DetectorErrorModel dem;
};

chromobius_init_data make_chromobius_init_data(const decoder_init &inputs) {
  if (!inputs.has_stim_dem()) {
    throw std::runtime_error(
        "Chromobius decoder requires a Stim detector error model string as "
        "decoder input. Use get_decoder(\"chromobius\", dem_text, params).");
  }

  stim::DetectorErrorModel dem;
  try {
    dem = stim::DetectorErrorModel(inputs.stim_dem());
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("Chromobius Stim DEM parse failed: ") +
                             e.what());
  }

  const auto num_observables =
      static_cast<std::size_t>(dem.count_observables());
  if (num_observables > 64) {
    throw std::runtime_error(
        "Chromobius currently returns observable flips as a 64-bit mask; "
        "CUDA-Q QEC wrapper supports at most 64 observables.");
  }

  return chromobius_init_data{std::move(dem)};
}

bool get_bool_param(const cudaqx::heterogeneous_map &params,
                    const std::string &key, bool default_value) {
  return params.contains(key) ? params.get<bool>(key) : default_value;
}

} // namespace

/// @brief Wrapper around the Chromobius Mobius decoder for color-code detector
/// error models.
class chromobius : public decoder {
private:
  stim::DetectorErrorModel dem;
  ::chromobius::Decoder chromobius_decoder;
  std::size_t num_detector_bytes = 0;
  std::size_t num_observables = 0;
  bool return_weight = false;

  std::vector<uint8_t> hard_syndrome;
  std::vector<uint8_t> packed_detection_events;

public:
  chromobius(decoder_init inputs, chromobius_init_data init_data,
             decode_result_type requested_output,
             const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output),
        dem(std::move(init_data.dem)) {
    // Chromobius predicts observable flips directly and cannot be inverted to
    // an error frame. Reject the request at construction rather than on the
    // first live shot.
    if (requested_output != decode_result_type::observables)
      throw std::invalid_argument(
          "Chromobius cannot return an error frame; construct it for "
          "observable output");
    ::chromobius::DecoderConfigOptions options;
    options.drop_mobius_errors_involving_remnant_errors =
        get_bool_param(params, "drop_mobius_errors_involving_remnant_errors",
                       options.drop_mobius_errors_involving_remnant_errors);
    options.ignore_decomposition_failures =
        get_bool_param(params, "ignore_decomposition_failures",
                       options.ignore_decomposition_failures);
    options.include_coords_in_mobius_dem =
        get_bool_param(params, "include_coords_in_mobius_dem",
                       options.include_coords_in_mobius_dem);

    return_weight = get_bool_param(params, "return_weight", false);

    chromobius_decoder =
        ::chromobius::Decoder::from_dem(dem, std::move(options));
    chromobius_decoder.write_mobius_match_to_std_err =
        get_bool_param(params, "write_mobius_match_to_stderr", false);

    syndrome_size = static_cast<std::size_t>(dem.count_detectors());
    num_observables = static_cast<std::size_t>(dem.count_observables());
    if (num_observables > 64) {
      throw std::runtime_error(
          "Chromobius currently returns observable flips as a 64-bit mask; "
          "CUDA-Q QEC wrapper supports at most 64 observables.");
    }

    num_detector_bytes = (syndrome_size + 7) / 8;
    hard_syndrome.resize(syndrome_size);
    packed_detection_events.resize(num_detector_bytes);
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    if (syndrome.size() != syndrome_size) {
      throw std::runtime_error(
          "Chromobius syndrome length must match the number of detectors");
    }

    std::fill(packed_detection_events.begin(), packed_detection_events.end(),
              uint8_t{0});
    cudaq::qec::convert_vec_soft_to_hard(syndrome, hard_syndrome);
    for (std::size_t i = 0; i < syndrome.size(); ++i) {
      if (hard_syndrome[i])
        packed_detection_events[i >> 3] |= static_cast<uint8_t>(1u << (i & 7));
    }

    float weight = 0.0f;
    auto prediction = chromobius_decoder.decode_detection_events(
        std::span<const uint8_t>(packed_detection_events.data(),
                                 packed_detection_events.size()),
        return_weight ? &weight : nullptr);

    decoder_result result{.converged = true};
    result.result.resize(num_observables);
    for (std::size_t i = 0; i < num_observables; ++i)
      result.result[i] = static_cast<float_t>((prediction >> i) & 1);

    if (return_weight) {
      cudaqx::heterogeneous_map opt_results;
      opt_results.insert("weight", static_cast<double>(weight));
      result.opt_results = std::move(opt_results);
    }
    return result;
  }

  std::string get_version() const override {
    return "CUDA-Q QEC Chromobius Decoder wrapper";
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      chromobius, static std::unique_ptr<decoder> create(
                      cudaq::qec::decoder_init inputs,
                      std::optional<decode_result_type> output,
                      const cudaqx::heterogeneous_map &params) {
        auto init_data = make_chromobius_init_data(inputs);
        return std::make_unique<chromobius>(
            std::move(inputs), std::move(init_data),
            output.value_or(decode_result_type::observables), params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(chromobius)

// Parameter schema for the realtime decoding YAML (`decoder_custom_args` for
// `type: chromobius`, and the trt_decoder `global_decoder_params` section when
// `global_decoder: chromobius`). Registered here so the schema ships with the
// decoder itself.
namespace {
struct chromobius_schema_registrar {
  chromobius_schema_registrar() {
    using k = cudaq::qec::decoding::config::param_kind;
    cudaq::qec::decoding::config::register_decoder_schema(
        {"chromobius",
         {
             {"drop_mobius_errors_involving_remnant_errors", k::boolean},
             {"ignore_decomposition_failures", k::boolean},
             {"include_coords_in_mobius_dem", k::boolean},
             {"return_weight", k::boolean},
             {"write_mobius_match_to_stderr", k::boolean},
         }});
  }
};
chromobius_schema_registrar register_chromobius_schema;
} // namespace

} // namespace cudaq::qec
