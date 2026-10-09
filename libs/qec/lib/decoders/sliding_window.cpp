/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "sliding_window.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include <cassert>
#include <fmt/core.h>
#include <vector>

namespace cudaq::qec {

namespace {

decoder_init canonicalize_sliding_window_inputs(decoder_init inputs) {
  // Canonical CSC is the steady-state contract for decode_window's column
  // slices and validate_inputs's per-column reads. canonicalize_H() is
  // basis-preserving and retains the authoritative source, so no source kind
  // needs special-casing here.
  return inputs.canonicalize_H();
}

} // namespace

void sliding_window::validate_inputs() {
  uint32_t num_rows = H.num_rows();
  if (num_boundary_syndromes > num_syndromes_per_round)
    throw std::invalid_argument(
        "sliding_window constructor: num_boundary_syndromes must be <= "
        "num_syndromes_per_round");
  // Memory circuits always emit at least a lock-in and a read-out round.
  if (num_rows < 2 * num_syndromes_per_round)
    throw std::invalid_argument(
        "sliding_window constructor: num_rows must be >= "
        "2 * num_syndromes_per_round; memory circuits always produce "
        "at least a lock-in and a read-out round");
  // The detector rows must form a [B | K*S | B] layout.
  if (num_rows < 2 * num_boundary_syndromes ||
      (num_rows - 2 * num_boundary_syndromes) % num_syndromes_per_round != 0)
    throw std::invalid_argument(
        "sliding_window constructor: number of PCM rows is inconsistent with "
        "the given num_syndromes_per_round and num_boundary_syndromes");
  if (window_size < 1 || window_size > num_detector_layers) {
    throw std::invalid_argument(
        fmt::format("sliding_window constructor: window_size ({}) must "
                    "be between 1 and the number of detector layers ({})",
                    window_size, num_detector_layers));
  }
  if (step_size < 1 || step_size > window_size) {
    throw std::invalid_argument(
        fmt::format("sliding_window constructor: step_size ({}) must "
                    "be between 1 and window_size ({})",
                    step_size, window_size));
  }
  if ((num_detector_layers - window_size) % step_size != 0) {
    throw std::invalid_argument(
        fmt::format("sliding_window constructor: detector layers - "
                    "window_size ({}) must be divisible by step_size ({})",
                    num_detector_layers - window_size, step_size));
  }
  if (num_syndromes_per_round == 0) {
    throw std::invalid_argument("sliding_window constructor: "
                                "num_syndromes_per_round must be non-zero");
  }
  if (inner_decoder_name.empty()) {
    throw std::invalid_argument(
        "sliding_window constructor: inner_decoder_name must be non-empty");
  }
  if (inner_decoder_params.empty()) {
    CUDA_QEC_WARN("sliding_window constructor: inner_decoder_params is empty. "
                  "Is that intentional?");
  }
  if (error_rate_vec.empty()) {
    throw std::invalid_argument(
        "sliding_window constructor: error_rate_vec must be non-empty");
  }

  // Enforce topological column order. Ctor-time materialization only.
  if (!cudaq::qec::pcm_is_sorted(this->H.to_nested_csc(),
                                 this->num_syndromes_per_round,
                                 this->num_boundary_syndromes)) {
    throw std::invalid_argument("sliding_window constructor: PCM must be "
                                "sorted. See cudaq::qec::simplify_pcm.");
  }
}

/// Helper function to initialize the window.
/// @param batch_size The number of independent syndromes (the batch size) to
/// initialize the window for. This will be 1 for non-batched mode.
void sliding_window::initialize_window(std::size_t batch_size) {
  // Initialize the syndrome mods and rw_results.
  auto t0 = std::chrono::high_resolution_clock::now();
  window_proc_times_arr.fill(0.0);
  syndrome_mods.resize(batch_size);
  for (std::size_t s = 0; s < batch_size; ++s) {
    syndrome_mods[s].clear();
    syndrome_mods[s].resize(this->syndrome_size);
  }
  rw_results.clear();
  rw_results.resize(batch_size);
  for (std::size_t s = 0; s < batch_size; ++s) {
    rw_results[s].converged = true; // Gets set to false if we fail to decode
    rw_results[s].result.resize(this->block_size);
  }
  window_proc_times.resize(num_windows);
  std::fill(window_proc_times.begin(), window_proc_times.end(), 0.0);
  this->batch_size = batch_size;
  window_rounds.clear();
  rounds_since_last_reset = 0;
  num_windows_decoded = 0;
  CUDA_QEC_DBG("Initializing window");
  auto t1 = std::chrono::high_resolution_clock::now();
  window_proc_times_arr[WindowProcTimes::INITIALIZE_WINDOW] =
      std::chrono::duration<double>(t1 - t0).count() * 1000;
}

sliding_window::sliding_window(cudaq::qec::decoder_init inputs,
                               decode_result_type requested_output,
                               const cudaqx::heterogeneous_map &params)
    : sliding_window(std::move(inputs),
                     decoder_output_request{requested_output}, params) {}

sliding_window::sliding_window(cudaq::qec::decoder_init inputs,
                               decoder_output_request request,
                               const cudaqx::heterogeneous_map &params)
    // Canonical CSC is the steady-state contract for decode_window's column
    // slices and for validate_inputs's per-column .front()/.back() reads.
    : decoder(canonicalize_sliding_window_inputs(std::move(inputs)), request),
      H(get_inputs().detector_error_matrix()) {
  if (request.has_auxiliary_outputs())
    throw std::invalid_argument(
        "sliding_window does not produce auxiliary outputs");
  // This decoder composes an error frame from its windows. Producing
  // observables requires an observable mapping to project through; reject at
  // construction rather than on the first decode.
  if (request.primary == decode_result_type::observables &&
      !get_inputs().has_observable_model())
    throw std::invalid_argument(
        "sliding_window was constructed for observable output but its model "
        "supplies no observable mapping");
  // Fetch parameters from the params map.
  window_size = params.get<std::size_t>("window_size", window_size);
  step_size = params.get<std::size_t>("step_size", step_size);
  num_syndromes_per_round = params.get<std::size_t>("num_syndromes_per_round",
                                                    num_syndromes_per_round);
  num_boundary_syndromes =
      params.get<std::size_t>("num_boundary_syndromes", num_boundary_syndromes);
  straddle_start_round =
      params.get<bool>("straddle_start_round", straddle_start_round);
  straddle_end_round =
      params.get<bool>("straddle_end_round", straddle_end_round);
  error_rate_vec = get_inputs().error_rates();
  inner_decoder_name =
      params.get<std::string>("inner_decoder_name", inner_decoder_name);
  inner_decoder_params = params.get<cudaqx::heterogeneous_map>(
      "inner_decoder_params", inner_decoder_params);

  if (num_syndromes_per_round == 0)
    throw std::invalid_argument("sliding_window constructor: "
                                "num_syndromes_per_round must be non-zero");

  // Treat a 0 boundary width as the uniform layout (B == S).
  if (num_boundary_syndromes == 0)
    num_boundary_syndromes = num_syndromes_per_round;

  const std::size_t num_rows = this->H.num_rows();
  // The [B | S...S | B] detector-layer layout drives all round<->row mapping.
  layout = details::round_layout(num_syndromes_per_round,
                                 num_boundary_syndromes, num_rows);
  num_detector_layers = layout.num_rounds; // 2 boundary + K interior
  num_windows = (num_detector_layers - window_size) / step_size + 1;

  validate_inputs();

  // Hand the base its streaming geometry. Everything else the realtime path
  // needs came from the model at base construction; layer widths and offsets
  // are a property of how this decoder consumes rounds, not of the model, and
  // the base cannot ask for them while this constructor is still running.
  std::vector<std::size_t> detector_layer_offsets(num_detector_layers + 1);
  for (std::size_t r = 0; r <= num_detector_layers; ++r)
    detector_layer_offsets[r] = get_layer_offset(r);
  initialize_streaming_layout(num_syndromes_per_round,
                              std::move(detector_layer_offsets));

  // Build the per-window inner decoders from the real (unpadded) sub-PCMs. The
  // boundary-aware round layout is handled by get_pcm_for_rounds.
  // this->H is canonical CSC (ctor init list), so skip the per-call
  // canonicalize in get_pcm_for_rounds.
  for (std::size_t w = 0; w < num_windows; ++w) {
    std::size_t start_round = w * step_size;
    std::size_t end_round = start_round + window_size - 1;
    auto [H_round, first_column, last_column] = cudaq::qec::get_pcm_for_rounds(
        this->H, num_syndromes_per_round, start_round, end_round,
        straddle_start_round, straddle_end_round, /*pcm_is_canonical=*/true,
        num_boundary_syndromes);
    first_columns.push_back(first_column);

    // Slice model rates to the same error-column basis as the window H.
    std::vector<double> error_vec_mod(error_rate_vec.begin() + first_column,
                                      error_rate_vec.begin() + last_column + 1);

    CUDA_QEC_INFO("Creating a decoder for rounds {}-{} (dims {} x {}) "
                  "first_column = {}, last_column = {}",
                  start_round, end_round, H_round.shape()[0],
                  H_round.shape()[1], first_column, last_column);

    if (last_column - first_column + 1 != H_round.shape()[1]) {
      throw std::invalid_argument(
          fmt::format("last_column - first_column + 1 ({}) must be equal to "
                      "the number of columns in H_round ({})",
                      last_column - first_column + 1, H_round.shape()[1]));
    }

    auto inner_O = sparse_binary_matrix::from_csr(
        0, H_round.shape()[1], std::vector<std::uint32_t>{0}, {});
    std::optional<std::vector<std::size_t>> inner_error_ids;
    if (const auto &ids = get_inputs().error_ids())
      inner_error_ids = std::vector<std::size_t>(
          ids->begin() + first_column, ids->begin() + last_column + 1);
    // Slicing detector rows and error columns re-indexes both, so this window
    // gets its own matrices. Nothing is inherited: a raw DEM names the outer
    // detectors and would not describe these.
    decoder_init inner_inputs(sparse_binary_matrix(H_round), std::move(inner_O),
                              std::move(error_vec_mod),
                              /*measurement_to_detectors=*/std::nullopt,
                              std::move(inner_error_ids));
    auto inner_decoder =
        decoder::get(inner_decoder_name, std::move(inner_inputs),
                     decode_result_type::errors, inner_decoder_params);
    inner_decoders.push_back(std::move(inner_decoder));
  }
}

decoder_result sliding_window::decode(const std::vector<float_t> &syndrome) {
  // A default-constructed token never requests a stop, so this always has a
  // value.
  return decode(syndrome, cancellation_token{}).value();
}

std::optional<decoder_result>
sliding_window::decode(const std::vector<float_t> &syndrome,
                       cancellation_token tok) {
  auto results = decode_batch({syndrome}, std::move(tok));
  if (!results)
    return std::nullopt; // abandoned
  if (results->empty())
    return decoder_result(); // not ready until the final window
  return std::move((*results)[0]);
}

std::vector<decoder_result> sliding_window::decode_batch(
    const std::vector<std::vector<float_t>> &syndromes) {
  return decode_batch(syndromes, cancellation_token{}).value();
}

std::optional<std::vector<decoder_result>>
sliding_window::decode_batch(const std::vector<std::vector<float_t>> &syndromes,
                             cancellation_token tok) {
  if (syndromes.empty()) {
    CUDA_QEC_DBG("Returning empty decoder_result (no syndrome)");
    return std::vector<decoder_result>{};
  }
  if (syndromes[0].size() == this->syndrome_size) {
    CUDA_QEC_DBG("Decoding whole block");
    // Decode the whole thing, feeding one detector layer at a time.
    std::optional<std::vector<decoder_result>> results;
    std::vector<std::vector<float_t>> syndromes_round(syndromes.size());
    for (std::size_t r = 0; r < num_detector_layers; ++r) {
      std::size_t round_start = layout.round_start(r);
      std::size_t round_end = round_start + layout.round_width(r);
      for (std::size_t s = 0; s < syndromes.size(); ++s) {
        syndromes_round[s].resize(round_end - round_start);
        std::copy(syndromes[s].begin() + round_start,
                  syndromes[s].begin() + round_end, syndromes_round[s].begin());
      }
      results = decode_batch(syndromes_round, tok);
      if (!results)
        return std::nullopt;
    }
    return results;
  }
  if (rounds_since_last_reset == 0)
    initialize_window(syndromes.size());

  if (syndromes.size() != batch_size)
    throw std::invalid_argument(
        fmt::format("sliding_window: batch size changed mid-stream ({} vs {})",
                    syndromes.size(), batch_size));

  const std::size_t expected = layout.round_width(rounds_since_last_reset);
  for (const auto &r : syndromes)
    // Inputs are either a whole block (handled above) or one detector layer per
    // call. Multi-round chunks are rejected.
    if (r.size() != expected)
      throw std::invalid_argument(fmt::format(
          "sliding_window: round {} has width {} but expected {} for this "
          "round in the boundary layout",
          rounds_since_last_reset, r.size(), expected));

  // Maybe FIXME:
  // Copies this layer into the rolling buffer. Revisit with a pooled buffer if
  // buffer if per-round allocation shows up in profiles.
  window_rounds.push_back(syndromes);
  ++rounds_since_last_reset;

  if (window_rounds.size() < window_size)
    return std::vector<decoder_result>{}; // not ready yet

  // A full window is buffered; decode it.
  CUDA_QEC_DBG("Decoding window {}/{}", num_windows_decoded + 1, num_windows);
  if (!decode_window(tok)) {
    // Abandoned: drop the partial block so the next round starts a new one.
    reset_stream();
    return std::nullopt;
  }
  ++num_windows_decoded;

  if (num_windows_decoded == num_windows) {
    // Final window decoded: hand back the accumulated results and reset.
    auto results = std::move(rw_results);
    reset_stream();
    // The only site that produces composed error frames; the whole-block path
    // returns results already converted by its recursive call, and every other
    // return is the empty streaming sentinel.
    // Composed frames are error frames; whether they are projected is fixed at
    // construction.
    if (get_result_type() == decode_result_type::observables)
      for (auto &r : results) {
        if (r.result.empty())
          continue; // streaming sentinel
        std::vector<float_t> observables(get_num_observables(), 0.0);
        project_errors_to_observables(r.result.data(), observables.data(),
                                      observables.size());
        r.result = std::move(observables);
      }
    return results;
  }

  // Slide the window: drop the oldest step_size rounds.
  window_rounds.erase(window_rounds.begin(), window_rounds.begin() + step_size);
  CUDA_QEC_DBG("Returning empty decoder_result");
  return std::vector<decoder_result>{}; // not ready yet
}

void sliding_window::reset_stream() {
  window_rounds.clear();
  rw_results.clear();
  syndrome_mods.clear();
  rounds_since_last_reset = 0;
  num_windows_decoded = 0;
}

void sliding_window::reset_decoder() {
  decoder::reset_decoder();
  reset_stream();
}

/// This is an internal helper function that decodes a single window. Regular
/// users should use the regular `cudaq::qec::decoder::decode` or
/// `cudaq::qec::decoder::decode_batch` functions instead of trying to access
/// this function.
bool sliding_window::decode_window(cancellation_token tok) {
  auto t0 = std::chrono::high_resolution_clock::now();
  const auto &w = this->num_windows_decoded;
  // Detector range of window w's rounds.
  std::size_t syndrome_start = layout.round_start(w * step_size);
  std::size_t num_window_syndromes =
      layout.round_start(w * step_size + window_size) - syndrome_start;
  auto t3 = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<float_t>> window_syndromes(batch_size);
  for (std::size_t s = 0; s < batch_size; ++s) {
    // Assemble each batch element's window syndrome by concatenating its
    // buffered rounds
    auto &syn = window_syndromes[s];
    syn.reserve(num_window_syndromes);
    for (std::size_t slot = 0; slot < window_size; ++slot)
      syn.insert(syn.end(), window_rounds[slot][s].begin(),
                 window_rounds[slot][s].end());
    if (w > 0) {
      // Apply the accumulated syndrome mods from the previously committed
      // windows.
      for (std::size_t r = 0; r < num_window_syndromes; ++r)
        syn[r] = static_cast<float_t>(static_cast<std::uint8_t>(syn[r]) ^
                                      syndrome_mods[s][syndrome_start + r]);
    }
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  CUDA_QEC_DBG("Window {}: syndrome_start = {}, num_window_syndromes = {}", w,
               syndrome_start, num_window_syndromes);

  auto inner =
      inner_decoders[w]->decode_batch(window_syndromes, std::move(tok));
  if (!inner) {
    CUDA_QEC_DBG("Window {}: inner decoder abandoned the window", w);
    return false;
  }
  auto &inner_results = *inner;
  if (!inner_results[0].converged) {
    CUDA_QEC_DBG("Window {}: inner decoder failed to converge", w);
  }
  auto t5 = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint8_t>> window_results(batch_size);
  for (std::size_t s = 0; s < batch_size; ++s) {
    this->rw_results[s].converged &= inner_results[s].converged;
    cudaq::qec::convert_vec_soft_to_hard(inner_results[s].result,
                                         window_results[s]);
  }
  // Commit to everything up to the first column of the next window.
  auto t6 = std::chrono::high_resolution_clock::now();
  if (w < num_windows - 1) {
    // Prepare for the next window.
    auto next_window_first_column = first_columns[w + 1];
    auto this_window_first_column = first_columns[w];
    auto num_to_commit = next_window_first_column - this_window_first_column;
    CUDA_QEC_DBG("  Committing {} bits from window {}", num_to_commit, w);
    for (std::size_t s = 0; s < batch_size; ++s) {
      for (std::size_t c = 0; c < num_to_commit; ++c) {
        rw_results[s].result[c + this_window_first_column] =
            window_results[s][c];
      }
    }
    // Back out committed errors from the next window's syndrome by flipping
    // the rows where the corresponding H columns have a 1. Read directly off
    // the canonical CSC arrays.
    std::size_t syndrome_start_next_window =
        layout.round_start((w + 1) * step_size);
    std::size_t syndrome_end_next_window =
        layout.round_start((w + 1) * step_size + 1) - 1;
    const auto &h_ptr = this->H.ptr();
    const auto &h_indices = this->H.indices();
    for (std::size_t s = 0; s < batch_size; ++s) {
      for (std::size_t c = 0; c < num_to_commit; ++c) {
        if (rw_results[s].result[c + this_window_first_column]) {
          // Flip next-round syndrome bits where PCM has a 1 in this column.
          const auto pcm_col_ix =
              static_cast<std::size_t>(c + this_window_first_column);
          for (auto p = h_ptr[pcm_col_ix]; p < h_ptr[pcm_col_ix + 1]; ++p) {
            const auto r = h_indices[p];
            if (r >= syndrome_start_next_window &&
                r <= syndrome_end_next_window)
              syndrome_mods[s][r] = syndrome_mods[s][r] ^ true;
          }
        }
      }
    }
  } else {
    // This is the last window. Append ALL of window_result to
    // decoded_result.
    auto this_window_first_column = first_columns[w];
    auto num_to_commit = window_results[0].size();
    CUDA_QEC_DBG("  Committing {} bits from window {}", num_to_commit, w);
    for (std::size_t s = 0; s < batch_size; ++s) {
      for (std::size_t c = 0; c < num_to_commit; ++c) {
        rw_results[s].result[c + this_window_first_column] =
            window_results[s][c];
      }
    }
  }
  auto t7 = std::chrono::high_resolution_clock::now();
  window_proc_times.at(w) +=
      std::chrono::duration<double>(t7 - t0).count() * 1000;
  window_proc_times_arr[WindowProcTimes::INDEX_CALCULATION] =
      std::chrono::duration<double>(t3 - t0).count() * 1000;
  window_proc_times_arr[WindowProcTimes::MODIFY_SYNDROME_SLICE] =
      std::chrono::duration<double>(t4 - t3).count() * 1000;
  window_proc_times_arr[WindowProcTimes::INNER_DECODE] =
      std::chrono::duration<double>(t5 - t4).count() * 1000;
  window_proc_times_arr[WindowProcTimes::CONVERT_TO_HARD] =
      std::chrono::duration<double>(t6 - t5).count() * 1000;
  window_proc_times_arr[WindowProcTimes::COMMIT_TO_RESULT] =
      std::chrono::duration<double>(t7 - t6).count() * 1000;
  CUDA_QEC_INFO("Window {} time: {:.3f} ms (0:{:.3f}ms 1:{:.3f}ms 2:{:.3f}ms "
                "3:{:.3f}ms 4:{:.3f}ms 5:{:.3f}ms 6:{:.3f}ms 7:{:.3f}ms)",
                w, window_proc_times[w], window_proc_times_arr[0],
                window_proc_times_arr[1], window_proc_times_arr[2],
                window_proc_times_arr[3], window_proc_times_arr[4],
                window_proc_times_arr[5], window_proc_times_arr[6],
                window_proc_times_arr[7]);
  return true;
}

sliding_window::~sliding_window() {}

std::size_t sliding_window::get_num_syndromes_per_round() const {
  return num_syndromes_per_round;
}

std::size_t sliding_window::get_num_boundary_syndromes() const {
  return num_boundary_syndromes;
}

std::size_t sliding_window::get_layer_offset(std::size_t r) const {
  return layout.round_start(r);
}

std::size_t sliding_window::get_num_detector_layers() const {
  return num_detector_layers;
}

CUDAQ_EXT_PT_REGISTER_TYPE(sliding_window)

// Parameter schema for the realtime decoding YAML (`decoder_custom_args` for
// `type: sliding_window`). `inner_decoder_params` is a discriminated section
// parsed with the schema registered under the value of `inner_decoder_name`
// (whichever decoder that names must have registered its own schema).
// Unknown-key and required-key checks are applied by the framework from the
// param specs alone; the `validate` hook adds the cross-field constraints
// those specs cannot express.
namespace {
struct sliding_window_schema_registrar {
  sliding_window_schema_registrar() {
    using k = decoding::config::param_kind;
    decoding::config::decoder_schema schema{
        "sliding_window",
        {
            {"window_size", k::uint64},
            {"step_size", k::uint64},
            {"num_syndromes_per_round", k::uint64},
            {"num_boundary_syndromes", k::uint64},
            {"straddle_start_round", k::boolean},
            {"straddle_end_round", k::boolean},
            {"inner_decoder_name", k::string, /*required=*/true},
            {"inner_decoder_params", k::discriminated, false, "",
             "inner_decoder_name", /*materialize_empty=*/false},
        }};
    schema.validate = [](const cudaqx::heterogeneous_map &args) {
      if (args.contains("window_size") && args.contains("step_size")) {
        auto window_size = args.get<std::size_t>("window_size");
        auto step_size = args.get<std::size_t>("step_size");
        if (step_size < 1 || step_size > window_size)
          throw std::runtime_error(fmt::format(
              "sliding_window parameters: step_size ({}) must be between 1 "
              "and window_size ({})",
              step_size, window_size));
      }
      if (args.contains("num_boundary_syndromes") &&
          args.contains("num_syndromes_per_round")) {
        auto num_boundary_syndromes =
            args.get<std::size_t>("num_boundary_syndromes");
        auto num_syndromes_per_round =
            args.get<std::size_t>("num_syndromes_per_round");
        if (num_boundary_syndromes > num_syndromes_per_round)
          throw std::runtime_error(fmt::format(
              "sliding_window parameters: num_boundary_syndromes ({}) must be "
              "<= num_syndromes_per_round ({})",
              num_boundary_syndromes, num_syndromes_per_round));
      }
    };
    decoding::config::register_decoder_schema(std::move(schema));
  }
};
sliding_window_schema_registrar register_sliding_window_schema;
} // namespace

} // namespace cudaq::qec
