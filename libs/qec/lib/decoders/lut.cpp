/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/logger.h"
#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

namespace cudaq::qec {

/// @brief This is a simple LUT (LookUp Table) decoder that demonstrates how to
/// build a simple decoder that can decode errors with a small number of errors
/// in the block.
class multi_error_lut : public decoder {
private:
  /// @brief This is a struct that contains the list of errors for a LUT entry
  /// and the probability of the list of errors.
  struct lut_entry {
    /// List of errors for this LUT entry
    std::vector<std::size_t> error_list;
    /// Probability of the list of errors in error_list.
    double p;
  };
  std::map<std::string, lut_entry> error_signatures;

  // Input parameters
  int lut_error_depth = 1;
  std::vector<double> error_rate_vec;

  // List of available result types for this decoder
  const std::vector<std::string> available_result_types = {
      "error_probability", // Probability of the detected error (bool)
      "syndrome_weight",   // Number of non-zero syndrome measurements (bool)
      "decoding_time"      // Time taken to perform the decoding (bool)
  };

  // Output parameters
  bool has_opt_results = false;
  bool error_probability = false;
  bool syndrome_weight = false;
  bool decoding_time = false;

public:
  multi_error_lut(cudaq::qec::decoder_init inputs,
                  decode_result_type requested_output,
                  const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    // This decoder computes an error frame. Producing observables requires an
    // observable mapping to project through; reject at construction rather
    // than on the first decode.
    if (requested_output == decode_result_type::observables &&
        !get_inputs().has_observable_model())
      throw std::invalid_argument(
          "lut decoder was constructed for observable output but its model "
          "supplies no observable mapping");
    const auto &H = get_inputs().detector_error_matrix();
    error_rate_vec = get_inputs().error_rates();
    if (params.contains("lut_error_depth")) {
      lut_error_depth = params.get<int>("lut_error_depth");
      if (lut_error_depth < 1) {
        throw std::runtime_error("lut_error_depth must be >= 1");
      }
      if (lut_error_depth > block_size) {
        throw std::runtime_error("lut_error_depth must be <= block_size");
      }
    }
    if (!error_rate_vec.empty()) {
      if (error_rate_vec.size() != block_size) {
        throw std::runtime_error("error_rate_vec must be of size block_size");
      }
      // Validate that the values in the error_rate_vec are between 0 and 1.
      for (auto error_rate : error_rate_vec) {
        if (error_rate < 0.0 || error_rate > 1.0) {
          throw std::runtime_error(
              "error_rate_vec value is out of range [0, 1]");
        }
      }
    }
    // Binomial coefficient to check if lut_error_depth is too large
    auto binom = [](int n, int k) {
      return 1 / ((n + 1) * std::beta(n - k + 1, k + 1));
    };
    if (binom(block_size, lut_error_depth) > 1e9) {
      throw std::runtime_error("lut_error_depth is too large for LUT decoder");
    }
    // Decoder-specific constructor arguments can be placed in `params`.
    // Check if opt_results was requested
    if (params.contains("opt_results")) {
      try {
        auto requested_results =
            params.get<cudaqx::heterogeneous_map>("opt_results");

        // Validate requested result types
        auto invalid_types = validate_config_parameters(requested_results,
                                                        available_result_types);

        if (!invalid_types.empty()) {
          std::string error_msg =
              "Requested result types not available in LUT decoder: ";
          for (size_t i = 0; i < invalid_types.size(); ++i) {
            error_msg += invalid_types[i];
            if (i < invalid_types.size() - 1) {
              error_msg += ", ";
            }
          }
          throw std::runtime_error(error_msg);
        } else {
          has_opt_results = true;
          error_probability = requested_results.get<bool>("error_probability",
                                                          error_probability);
          syndrome_weight =
              requested_results.get<bool>("syndrome_weight", syndrome_weight);
          decoding_time =
              requested_results.get<bool>("decoding_time", decoding_time);
        }
      } catch (const std::runtime_error &e) {
        throw; // Re-throw if it's our error
      } catch (...) {
        throw std::runtime_error("opt_results must be a heterogeneous_map");
      }
    }

    // No canonicalize: toggleSynForError XORs per row index, so duplicate
    // indices in a column GF(2)-cancel naturally.
    std::vector<std::vector<std::uint32_t>> H_e2d = H.to_nested_csc();
    auto toggleSynForError = [&H_e2d](std::string &err_sig, std::size_t qErr) {
      for (std::size_t r : H_e2d[qErr])
        err_sig[r] = err_sig[r] == '1' ? '0' : '1';
    };

    // For each qubit with a possible error, calculate an error signature.
    for (std::size_t k = 1; k <= lut_error_depth; k++) {
      std::string bitmask(block_size, 0);
      // Initialize the leading "k" values to 1.
      std::fill(bitmask.begin(), bitmask.begin() + k, 1);
      // Now loop over all the permutations of the bitmask.
      do {
        std::string err_sig(syndrome_size, '0');
        std::vector<std::size_t> error_list;
        error_list.reserve(lut_error_depth);
        double p = 1.0;
        for (std::size_t qErr = 0; qErr < block_size; qErr++) {
          if (bitmask[qErr]) {
            toggleSynForError(err_sig, qErr);
            error_list.push_back(qErr);
            if (qErr < error_rate_vec.size()) {
              p *= error_rate_vec[qErr];
            }
          }
        }
        auto it = error_signatures.find(err_sig);
        if (it != error_signatures.end()) {
          // Syndrome already found, so we have ambiguous errors. Defer to the
          // one that was previously added, so don't add it again.
          CUDA_QEC_INFO(
              "Ambiguous error signature: err_sig={} for error_list={}",
              err_sig, error_list);
        } else {
          CUDA_QEC_INFO("Adding err_sig={} for error_list={}", err_sig,
                        error_list);
          error_signatures.insert(
              {std::move(err_sig), lut_entry{std::move(error_list), p}});
        }
      } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    }
  }

  decoder_result decode(const std::vector<float_t> &syndrome) override {
    // This is a simple decoder with trivial results
    auto t0 = std::chrono::high_resolution_clock::now();
    decoder_result result{.result = std::vector<float_t>(block_size, 0.0)};

    // This decoder computes an error frame. Whether that frame is projected is
    // fixed at construction, so the decision is read from immutable instance
    // state rather than negotiated per call.
    const bool project = get_result_type() == decode_result_type::observables;
    auto finish = [&](decoder_result &r) {
      if (project) {
        std::vector<float_t> observables(get_num_observables(), 0.0);
        project_errors_to_observables(r.result.data(), observables.data(),
                                      observables.size());
        r.result = std::move(observables);
      }
    };

    // Convert syndrome to a string
    std::string syndrome_str(syndrome.size(), '0');
    int syndrome_weight = 0;
    assert(syndrome_str.length() == syndrome_size);
    bool anyErrors = false;
    for (std::size_t i = 0; i < syndrome_size; i++) {
      if (cudaq::qec::convert_soft_to_hard(syndrome[i])) {
        syndrome_str[i] = '1';
        anyErrors = true;
        syndrome_weight++;
      }
    }

    if (!anyErrors) {
      result.converged = true;
      finish(result);
      return result;
    }

    auto it = error_signatures.find(syndrome_str);
    if (it != error_signatures.end()) {
      result.converged = true;
      for (auto qErr : it->second.error_list)
        result.result[qErr] = 1.0 - result.result[qErr];
    } else {
      // Leave result.converged set to false.
    }

    // Add opt_results if requested
    /*
     * Example opt_results map:
     * {
     *   "error_probability": true,    // Include error probability in results
     *   "syndrome_weight": true,      // Include syndrome weight in results
     *   "decoding_time": false,       // Don't include decoding time
     * }
     */
    if (has_opt_results) {
      result.opt_results =
          cudaqx::heterogeneous_map(); // Initialize the optional map
      if (error_probability) {
        if (it != error_signatures.end())
          result.opt_results->insert("error_probability", it->second.p);
        else
          result.opt_results->insert("error_probability", 0.0);
      }
      if (syndrome_weight) {
        result.opt_results->insert("syndrome_weight", syndrome_weight);
      }
      if (decoding_time) {
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = t1 - t0;
        result.opt_results->insert("decoding_time", duration.count());
      }
    }

    finish(result);
    return result;
  }

  virtual ~multi_error_lut() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      multi_error_lut, static std::unique_ptr<decoder> create(
                           cudaq::qec::decoder_init inputs,
                           std::optional<decode_result_type> output,
                           const cudaqx::heterogeneous_map &params) {
        return std::make_unique<multi_error_lut>(
            std::move(inputs), output.value_or(decode_result_type::errors),
            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(multi_error_lut)

class single_error_lut : public multi_error_lut {
public:
  single_error_lut(cudaq::qec::decoder_init inputs,
                   decode_result_type requested_output,
                   const cudaqx::heterogeneous_map &params)
      : multi_error_lut(std::move(inputs), requested_output, params) {}

  virtual ~single_error_lut() {}

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      single_error_lut, static std::unique_ptr<decoder> create(
                            cudaq::qec::decoder_init inputs,
                            std::optional<decode_result_type> output,
                            const cudaqx::heterogeneous_map &params) {
        return std::make_unique<single_error_lut>(
            std::move(inputs), output.value_or(decode_result_type::errors),
            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(single_error_lut)

// Parameter schemas for the realtime decoding YAML (`decoder_custom_args`),
// registered alongside the decoders they describe.
namespace {
struct lut_schema_registrar {
  lut_schema_registrar() {
    using k = decoding::config::param_kind;
    decoding::config::register_decoder_schema({"single_error_lut", {}});
    decoding::config::register_decoder_schema(
        {"multi_error_lut",
         {
             {"lut_error_depth", k::int32},
         }});
  }
};
lut_schema_registrar register_lut_schemas;
} // namespace

} // namespace cudaq::qec
