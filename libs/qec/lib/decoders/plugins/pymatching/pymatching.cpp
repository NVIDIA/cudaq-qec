/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <map>
#include <stdexcept>
#include <vector>

// Enable this to debug decode times.
#define PERFORM_TIMING 0

namespace cudaq::qec {

namespace {
/// @brief Claims exclusive use of a decoder for the duration of a decode.
///
/// The matching state and scratch buffers belong to the decoder, not the call,
/// so overlapping decodes corrupt each other. Rejecting the second is the best
/// available outcome. Costs one uncontended atomic exchange per decode.
class decode_exclusive_guard {
public:
  explicit decode_exclusive_guard(std::atomic<bool> &active) : active_(active) {
    if (active_.exchange(true))
      throw std::runtime_error(
          "pymatching: a decode is already running on this decoder. Its "
          "matching state and scratch buffers are per decoder, so decodes "
          "cannot overlap; serialize the calls or give each concurrent caller "
          "its own decoder.");
  }

  // A throwing constructor means no destructor, so a rejected call cannot
  // release the owner's claim.
  ~decode_exclusive_guard() { active_.store(false); }

  decode_exclusive_guard(const decode_exclusive_guard &) = delete;
  decode_exclusive_guard &operator=(const decode_exclusive_guard &) = delete;

private:
  std::atomic<bool> &active_;
};
} // namespace

/// @brief This is a wrapper around the PyMatching library that implements the
/// MWPM decoder.
class pymatching : public decoder {
private:
  pm::UserGraph user_graph;
  pm::Mwpm *mwpm = nullptr;

  // Input parameters
  std::vector<double> error_rate_vec;
  // Error-output instances default to DISALLOW. Observable-output instances
  // default to INDEPENDENT to match upstream PyMatching's
  // from_detector_error_model behavior.
  pm::MERGE_STRATEGY merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;
  bool merge_strategy_explicit = false;

  // Map of edge pairs to column indices. This does not seem particularly
  // efficient.
  std::map<std::pair<int64_t, int64_t>, size_t> edge2col_idx;
  std::map<std::pair<int64_t, int64_t>, double> edge2weight;

  bool decode_to_observables = false;
  // Scratch buffers reused across decode() calls so a steady-state decode
  // allocates nothing. decode() already mutates the shared Mwpm state, so it
  // was never safe to call concurrently on one instance; reusing these adds no
  // new constraint. Each is reset explicitly at its use site: PyMatching XORs
  // into the observable array and appends to the edge list, so neither may
  // carry over the previous call's contents.
  std::vector<uint64_t> detection_events;
  std::vector<uint8_t> obs_scratch;
  std::vector<int64_t> edges_scratch;

  // Set for the duration of decode(). See decode_exclusive_guard.
  std::atomic<bool> decode_active{false};

  // Helper function to make a canonical edge from two nodes.
  std::pair<int64_t, int64_t> make_canonical_edge(int64_t node1,
                                                  int64_t node2) {
    return std::make_pair(std::min(node1, node2), std::max(node1, node2));
  }

  void record_error_column(const std::pair<int64_t, int64_t> &edge,
                           std::size_t column, double weight) {
    auto [column_it, inserted] = edge2col_idx.try_emplace(edge, column);
    if (inserted) {
      edge2weight.emplace(edge, weight);
      return;
    }

    const bool replace = merge_strategy_enum == pm::MERGE_STRATEGY::REPLACE;
    const bool smaller =
        merge_strategy_enum == pm::MERGE_STRATEGY::SMALLEST_WEIGHT &&
        weight < edge2weight.at(edge);
    if (replace || smaller) {
      column_it->second = column;
      edge2weight.at(edge) = weight;
    }
  }

#if PERFORM_TIMING
  // 0: reduce syndrome to detection events, 1: matching, 2: total.
  static constexpr size_t NUM_TIMING_STEPS = 3;
  std::array<double, NUM_TIMING_STEPS> decode_times;
#endif

public:
  pymatching(cudaq::qec::decoder_init inputs,
             decode_result_type requested_output,
             const cudaqx::heterogeneous_map &params)
      : decoder(std::move(inputs), requested_output) {
    if (requested_output ==
        decode_result_type::observables_and_residual_detectors)
      throw std::invalid_argument(
          "pymatching does not produce residual-detector auxiliary output");
    const auto &H = get_inputs().detector_error_matrix();
    error_rate_vec = get_inputs().error_rates();
    decode_to_observables = requested_output == decode_result_type::observables;

    if (!error_rate_vec.empty()) {
      if (error_rate_vec.size() != block_size) {
        throw std::runtime_error("error_rate_vec must be of size block_size");
      }
      // Validate that the values in the error_rate_vec are between 0 and 0.5.
      // Values > 0.5 would have negative LLR, which is not supported by
      // PyMatching.
      for (auto error_rate : error_rate_vec) {
        if (error_rate <= 0.0 || error_rate > 0.5) {
          throw std::runtime_error(
              "error_rate_vec value is out of range (0, 0.5]");
        }
      }
    }

    if (params.contains("merge_strategy")) {
      std::string merge_strategy = params.get<std::string>("merge_strategy");
      if (merge_strategy == "disallow") {
        merge_strategy_enum = pm::MERGE_STRATEGY::DISALLOW;
      } else if (merge_strategy == "independent") {
        merge_strategy_enum = pm::MERGE_STRATEGY::INDEPENDENT;
      } else if (merge_strategy == "smallest_weight") {
        merge_strategy_enum = pm::MERGE_STRATEGY::SMALLEST_WEIGHT;
      } else if (merge_strategy == "keep_original") {
        merge_strategy_enum = pm::MERGE_STRATEGY::KEEP_ORIGINAL;
      } else if (merge_strategy == "replace") {
        merge_strategy_enum = pm::MERGE_STRATEGY::REPLACE;
      } else {
        throw std::runtime_error(
            "merge_strategy must be one of: disallow, independent, "
            "smallest_weight, keep_original, replace");
      }
      merge_strategy_explicit = true;
    }

    std::vector<std::vector<size_t>> errs2observables(block_size);
    if (decode_to_observables) {
      const auto &O = get_inputs().observable_flips_matrix();
      if (O.num_cols() != block_size)
        throw std::runtime_error(
            "Observable matrix column count must equal block_size");
      const auto O_sparse = O.to_nested_csr();
      for (std::size_t observable = 0; observable < O_sparse.size();
           ++observable)
        for (auto error : O_sparse[observable])
          errs2observables[error].push_back(observable);
      if (!merge_strategy_explicit)
        merge_strategy_enum = pm::MERGE_STRATEGY::INDEPENDENT;
    }

    user_graph =
        decode_to_observables
            ? pm::UserGraph(H.num_rows(), get_inputs().num_observables())
            : pm::UserGraph(H.num_rows());

    H.validate_sorted_unique_indices("pymatching");

    // PyMatching dispatches on per-column nnz in {1, 2}. We deliberately do
    // NOT canonicalize H as a whole here: that would couple this wrapper to the
    // column-level semantics of a shared utility and risk silently reordering
    // or dropping the caller's columns. Instead we iterate the caller's columns
    // in their original order. This guarantees by construction that result
    // index `col` always maps back to the caller's column `col`.
    std::vector<std::vector<std::uint32_t>> H_e2d = H.to_nested_csc();
    for (std::size_t col = 0; col < block_size; col++) {
      double weight = 1.0;
      if (col < error_rate_vec.size()) {
        weight = -std::log(error_rate_vec[col] / (1.0 - error_rate_vec[col]));
      }
      const auto &col_rows = H_e2d[col];
      if (col_rows.size() == 2) {
        if (!decode_to_observables)
          record_error_column(make_canonical_edge(col_rows[0], col_rows[1]),
                              col, weight);
        user_graph.add_or_merge_edge(col_rows[0], col_rows[1],
                                     errs2observables.at(col), weight, 0.0,
                                     merge_strategy_enum);
      } else if (col_rows.size() == 1) {
        if (!decode_to_observables)
          record_error_column(make_canonical_edge(col_rows[0], -1), col,
                              weight);
        user_graph.add_or_merge_boundary_edge(col_rows[0],
                                              errs2observables.at(col), weight,
                                              0.0, merge_strategy_enum);
      } else {
        throw std::runtime_error("Invalid column in H: " + std::to_string(col) +
                                 " has " + std::to_string(col_rows.size()) +
                                 " ones. Must have 1 or 2 ones.");
      }
    }
    this->mwpm = decode_to_observables
                     ? &user_graph.get_mwpm()
                     : &user_graph.get_mwpm_with_search_graph();
    detection_events.reserve(syndrome_size);
    edges_scratch.reserve(block_size * 2);
    obs_scratch.resize(get_inputs().num_observables());
#if PERFORM_TIMING
    std::fill(decode_times.begin(), decode_times.end(), 0.0);
#endif
  }

  /// @brief Decode the syndrome using the MWPM decoder.
  /// @param syndrome The syndrome to decode.
  /// @return The decoder result.
  /// @throws std::runtime_error if no matching solution is found or a decode
  /// is already running on this decoder, or std::out_of_range if an edge is
  /// not found in the edge2col_idx map.
  decoder_result decode(const std::vector<float_t> &syndrome) override {
    const decode_exclusive_guard guard(decode_active);
    const auto result_size =
        decode_to_observables ? get_inputs().num_observables() : block_size;
    decoder_result result{false, std::vector<float_t>(result_size, float_t{0})};
    auto *output = result.result.data();
#if PERFORM_TIMING
    auto t0 = std::chrono::high_resolution_clock::now();
#endif

    detection_events.clear();
    detection_events.reserve(syndrome.size());
    for (size_t i = 0; i < syndrome.size(); i++)
      if (cudaq::qec::convert_soft_to_hard(syndrome[i]))
        detection_events.push_back(i);
#if PERFORM_TIMING
    auto t1 = std::chrono::high_resolution_clock::now();
#endif
    if (decode_to_observables) {
      if (mwpm->flooder.graph.num_observables < 64) {
        auto res = pm::decode_detection_events_for_up_to_64_observables(
            *mwpm, detection_events, /*edge_correlations=*/false);
        for (size_t i = 0; i < mwpm->flooder.graph.num_observables; i++) {
          output[i] =
              static_cast<float_t>(res.obs_mask & (1ULL << i) ? 1.0 : 0.0);
        }
      } else {
        pm::total_weight_int weight = 0;
        // PyMatching XORs its prediction in, so start from a cleared frame.
        obs_scratch.assign(mwpm->flooder.graph.num_observables, 0);
        pm::decode_detection_events(*mwpm, detection_events, obs_scratch.data(),
                                    weight, /*edge_correlations=*/false);
        for (size_t i = 0; i < mwpm->flooder.graph.num_observables; i++) {
          output[i] = static_cast<float_t>(obs_scratch[i]);
        }
      }
    } else {
      // PyMatching appends to the edge list without clearing it first.
      edges_scratch.clear();
      pm::decode_detection_events_to_edges(*mwpm, detection_events,
                                           edges_scratch);
      // Loop over the edge pairs to reconstruct errors.
      const auto &edges = edges_scratch;
      assert(edges.size() % 2 == 0);
      for (size_t i = 0; i < edges.size(); i += 2) {
        auto edge = make_canonical_edge(edges.at(i), edges.at(i + 1));
        auto col_idx = edge2col_idx.at(edge);
        output[col_idx] = 1.0;
      }
    }
#if PERFORM_TIMING
    auto t2 = std::chrono::high_resolution_clock::now();
    decode_times[0] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() /
        1e6;
    decode_times[1] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() /
        1e6;
    decode_times[2] +=
        std::chrono::duration_cast<std::chrono::microseconds>(t2 - t0).count() /
        1e6;
#endif
    result.converged = true;
    return result;
  }

  virtual ~pymatching() {
#if PERFORM_TIMING
    for (int i = 0; i < NUM_TIMING_STEPS; i++) {
      std::cout << "Decode time[" << i << "]: " << decode_times[i] << " seconds"
                << std::endl;
    }
#endif
  }

  CUDAQ_EXTENSION_CUSTOM_CREATOR_FUNCTION(
      pymatching, static std::unique_ptr<decoder> create(
                      cudaq::qec::decoder_init inputs,
                      std::optional<decode_result_type> output,
                      const cudaqx::heterogeneous_map &params) {
        return std::make_unique<pymatching>(
            std::move(inputs), output.value_or(decode_result_type::errors),
            params);
      })
};

CUDAQ_EXT_PT_REGISTER_TYPE(pymatching)

// Parameter schema for the realtime decoding YAML (`decoder_custom_args` for
// `type: pymatching`, and the trt_decoder `global_decoder_params` section when
// `global_decoder: pymatching`). Registered here so the schema ships with the
// decoder itself.
namespace {
struct pymatching_schema_registrar {
  pymatching_schema_registrar() {
    using k = cudaq::qec::decoding::config::param_kind;
    cudaq::qec::decoding::config::register_decoder_schema(
        {"pymatching",
         {
             {"merge_strategy", k::string},
         }});
  }
};
pymatching_schema_registrar register_pymatching_schema;
} // namespace

} // namespace cudaq::qec
