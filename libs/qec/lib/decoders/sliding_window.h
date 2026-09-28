/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "round_layout.h"
#include "cudaq/qec/decoder.h"
#include <vector>

namespace cudaq::qec {

/// @brief A sliding window decoder that processes syndromes in overlapping
/// windows
///
/// This decoder divides the syndrome stream into overlapping windows and
/// decodes each window independently using an inner decoder. It's designed for
/// low-latency decoding of streaming syndrome data.
class sliding_window : public decoder {
private:
  /// Canonical materialized matrix view retained by this matrix-family decoder.
  const sparse_binary_matrix &H;

  // --- Input parameters ---

  /// The number of rounds of syndrome data in each window.
  std::size_t window_size = 1;
  /// The number of rounds to advance the window by each time.
  std::size_t step_size = 1;
  /// The number of syndromes per round (the interior-layer width).
  std::size_t num_syndromes_per_round = 0;
  /// The width of the first/last (boundary) rounds. A caller-supplied 0 means
  /// "uniform layout"; the constructor normalizes it to num_syndromes_per_round
  /// so the rest of the class always sees a concrete boundary width.
  std::size_t num_boundary_syndromes = 0;
  /// When forming a window, should error mechanisms that span the start round
  /// and any preceding rounds be included?
  bool straddle_start_round = false;
  /// When forming a window, should error mechanisms that span the end round and
  /// any subsequent rounds be included?
  bool straddle_end_round = true;
  /// The vector of error rates for the error mechanisms.
  std::vector<double> error_rate_vec;
  /// The name of the inner decoder to use.
  std::string inner_decoder_name;
  /// The parameters to pass to the inner decoder.
  cudaqx::heterogeneous_map inner_decoder_params;

  // Derived parameters.
  std::size_t num_windows = 0;
  /// Detector-row layers in the [B | S...S | B] layout (minimum 2 for a memory
  /// circuit with no interior rounds)
  std::size_t num_detector_layers = 0;
  std::vector<std::unique_ptr<decoder>> inner_decoders;
  std::vector<std::size_t> first_columns;

  // Boundary-aware detector-layer layout ([B | S...S | B]); maps between layers
  // and detector rows. Shared with the get_pcm_for_rounds machinery.
  details::round_layout layout;
  // Enum type for timing data.
  enum WindowProcTimes {
    INITIALIZE_WINDOW,     // 0
    SLIDE_WINDOW,          // 1
    COPY_DATA,             // 2
    INDEX_CALCULATION,     // 3
    MODIFY_SYNDROME_SLICE, // 4
    INNER_DECODE,          // 5
    CONVERT_TO_HARD,       // 6
    COMMIT_TO_RESULT,      // 7
    NUM_WINDOW_PROC_TIMES  // 8
  };

  // State data
  std::vector<std::vector<std::vector<cudaq::qec::float_t>>> window_rounds;
  std::size_t rounds_since_last_reset = 0;
  std::size_t batch_size = 1;
  std::size_t num_windows_decoded = 0;
  std::vector<std::vector<bool>> syndrome_mods; // [batch_size, syndrome_size]
  std::vector<decoder_result> rw_results;       // [batch_size]
  std::vector<double> window_proc_times;
  std::array<double, WindowProcTimes::NUM_WINDOW_PROC_TIMES>
      window_proc_times_arr = {};

  /// @brief Validate constructor inputs
  void validate_inputs();

  /// @brief Reset per-syndrome streaming state and size the result/mod buffers.
  /// @param batch_size The number of independent syndromes to decode together
  /// (1 for non-batched mode).
  void initialize_window(std::size_t batch_size);

  /// @brief Decode the active window from the rolling buffer, commit, and back
  /// out committed errors into the next window's syndrome mods.
  /// @return false if the inner decoder abandoned the window.
  bool decode_window(cancellation_token tok);

  /// @brief Drop any partially streamed block so the next round starts a new
  /// one.
  void reset_stream();

public:
  /// @brief Constructor
  /// @param H The full parity check matrix for all rounds
  /// @param params A heterogeneous map containing required parameters:
  ///   - window_size: Size of each decoding window (in rounds)
  ///   - step_size: Step size between consecutive windows (in rounds)
  ///   - num_syndromes_per_round: Number of syndromes per (interior) round
  ///   - num_boundary_syndromes: Boundary-layer width (0 if uniform)
  ///   - inner_decoder_name: Name of the inner decoder to use
  ///   - inner_decoder_params: Parameters for the inner decoder (optional)
  sliding_window(cudaq::qec::decoder_init inputs,
                 decode_result_type requested_output,
                 const cudaqx::heterogeneous_map &params);
  sliding_window(cudaq::qec::decoder_init inputs,
                 decoder_output_request request,
                 const cudaqx::heterogeneous_map &params);

  /// @brief Decode a syndrome vector
  /// @param syndrome The syndrome measurements to decode
  /// @return The decoded error correction
  decoder_result decode(const std::vector<float_t> &syndrome) override;

  /// @brief Decode a syndrome vector, forwarding `tok` to the inner decoders.
  /// @param syndromes Syndrome measurements to decode
  /// @param tok The cancellation token to use
  /// @return std::nullopt if an inner decoder honored a stop (the partial
  /// block is dropped), a result with an empty `result` vector until the
  /// final window is complete, otherwise the decoded result.
  std::optional<decoder_result> decode(const std::vector<float_t> &syndrome,
                                       cancellation_token tok) override;

  /// @brief Decode multiple syndromes in batch
  /// @param syndromes Multiple syndrome measurements to decode
  /// @return The decoded error corrections
  using decoder::decode_batch; // keep the batch_opt_results overload visible
  std::vector<decoder_result>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes) override;

  /// @brief Decode multiple syndromes in batch, forwarding `tok` to the inner
  /// decoders.
  /// @param syndromes Multiple syndrome measurements to decode
  /// @param tok The cancellation token to use
  /// @return std::nullopt if an inner decoder honored a stop (the partial
  /// block is dropped), an empty vector until the final window is complete,
  /// otherwise one entry per shot.
  std::optional<std::vector<decoder_result>>
  decode_batch(const std::vector<std::vector<float_t>> &syndromes,
               cancellation_token tok) override;

  /// @brief Reset the decoder, also dropping any partially streamed block.
  void reset_decoder() override;

  /// @brief Get the number of syndromes per round
  /// @return The number of syndromes measured in each round
  std::size_t get_num_syndromes_per_round() const;

  /// @brief Get the boundary-layer width, i.e. the number of detectors in the
  /// first/last detector layers. Equals get_num_syndromes_per_round() for a
  /// uniform layout.
  std::size_t get_num_boundary_syndromes() const;

  /// @brief The global row index at which detector layer @p r begins in the
  /// [B | S | ... | S | B] layout (round_start(num_detector_layers) ==
  /// syndrome_size).
  std::size_t get_layer_offset(std::size_t r) const;

  /// @brief The number of detector layers in the layout.
  std::size_t get_num_detector_layers() const;

  /// @brief Destructor
  virtual ~sliding_window();

  // Plugin registration macros
  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      sliding_window, static std::unique_ptr<decoder> create(
                          cudaq::qec::decoder_init inputs,
                          std::optional<decoder_output_request> output,
                          const cudaqx::heterogeneous_map &params) {
        const auto request =
            output.value_or(decoder_output_request{decode_result_type::errors});
        return std::make_unique<sliding_window>(std::move(inputs), request,
                                                params);
      })
};

} // namespace cudaq::qec
