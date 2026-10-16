/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/qec/detector_error_model.h"
#include "sparse_dem_from_stim_text.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/pcm_utils.h"
#include "cudaq/qec/sparse_binary_matrix.h"

#include "stim.h"

#include <exception>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::qec {

namespace {

/// What the Stim parse yields before any matrix layout is chosen: per-error
/// detector and observable hit lists, already GF(2)-reduced and sorted.
struct parsed_stim_dem {
  std::size_t num_detectors = 0;
  std::size_t num_observables = 0;
  std::vector<std::vector<std::size_t>> detector_hits;
  std::vector<std::vector<std::size_t>> observable_hits;
  std::vector<double> error_rates;
};

parsed_stim_dem parse_stim_dem(const std::string &dem_text,
                               bool use_decomp_suggestions) {
  auto dem = [&dem_text]() {
    try {
      return stim::DetectorErrorModel(dem_text);
    } catch (const std::exception &e) {
      throw std::runtime_error(std::string("Stim DEM parse failed: ") +
                               e.what());
    }
  }();
  const std::size_t num_detectors =
      static_cast<std::size_t>(dem.count_detectors());
  const std::size_t num_observables =
      static_cast<std::size_t>(dem.count_observables());

  std::vector<std::vector<std::size_t>> detector_hits;
  std::vector<std::vector<std::size_t>> observable_hits;
  std::vector<double> error_rates;
  std::size_t instruction_index = 0;

  dem.iter_flatten_error_instructions([&](const stim::DemInstruction &inst) {
    if (inst.arg_data.empty())
      throw std::runtime_error(
          "Stim DEM error instruction missing probability argument (index " +
          std::to_string(instruction_index) + ")");
    const double prob = inst.arg_data[0];
    if (!(prob >= 0.0 && prob <= 1.0))
      throw std::runtime_error("Stim DEM error probability " +
                               std::to_string(prob) +
                               " out of range [0, 1] at instruction index " +
                               std::to_string(instruction_index));

    std::set<std::size_t> dets_parity;
    std::set<std::size_t> obs_parity;

    auto toggle = [](std::set<std::size_t> &s, std::size_t v) {
      if (!s.erase(v)) {
        s.insert(v);
      }
    };

    auto push_target = [&](const stim::DemTarget &target) {
      if (target.is_relative_detector_id()) {
        toggle(dets_parity, static_cast<std::size_t>(target.val()));
      } else if (target.is_observable_id()) {
        toggle(obs_parity, static_cast<std::size_t>(target.val()));
      } else {
        throw std::runtime_error(
            "Stim DEM error instruction (index " +
            std::to_string(instruction_index) +
            ") contains an unsupported target kind; only D* (detector) and "
            "L* (observable) targets are supported by the fallback parser");
      }
    };

    auto flush = [&]() {
      if (!dets_parity.empty() || !obs_parity.empty()) {
        detector_hits.push_back({dets_parity.begin(), dets_parity.end()});
        observable_hits.push_back({obs_parity.begin(), obs_parity.end()});
        error_rates.push_back(prob);
        dets_parity.clear();
        obs_parity.clear();
      }
    };

    for (const auto &target : inst.target_data) {
      if (target.is_separator()) {
        if (use_decomp_suggestions) {
          flush();
        }
        continue;
      }
      push_target(target);
    }
    flush();
    ++instruction_index;
  });

  if (detector_hits.empty())
    throw std::runtime_error(
        "Stim DEM contains no error mechanisms after flattening");

  parsed_stim_dem parsed;
  parsed.num_detectors = num_detectors;
  parsed.num_observables = num_observables;
  parsed.detector_hits = std::move(detector_hits);
  parsed.observable_hits = std::move(observable_hits);
  parsed.error_rates = std::move(error_rates);
  return parsed;
}

/// Shared by both projections so an out-of-range id is reported identically
/// whichever one the caller asked for.
void validate_hit_ids(const parsed_stim_dem &parsed) {
  for (const auto &hits : parsed.detector_hits)
    for (auto det : hits)
      if (det >= parsed.num_detectors)
        throw std::runtime_error(
            "Stim DEM detector id out of range while extracting H");
  for (const auto &hits : parsed.observable_hits)
    for (auto ob : hits)
      if (ob >= parsed.num_observables)
        throw std::runtime_error(
            "Stim DEM observable id out of range while extracting O");
}

} // namespace

detector_error_model dem_from_stim_text(const std::string &dem_text,
                                        bool use_decomp_suggestions) {
  auto parsed = parse_stim_dem(dem_text, use_decomp_suggestions);
  validate_hit_ids(parsed);
  const std::size_t num_cols = parsed.detector_hits.size();

  detector_error_model result;
  result.detector_error_matrix =
      cudaqx::tensor<uint8_t>({parsed.num_detectors, num_cols});
  result.observables_flips_matrix =
      cudaqx::tensor<uint8_t>({parsed.num_observables, num_cols});
  result.error_rates = std::move(parsed.error_rates);

  for (std::size_t err = 0; err < num_cols; ++err) {
    for (auto det : parsed.detector_hits[err])
      result.detector_error_matrix.at({det, err}) ^= 1;
    for (auto ob : parsed.observable_hits[err])
      result.observables_flips_matrix.at({ob, err}) ^= 1;
  }

  return result;
}

namespace details {

sparse_dem sparse_dem_from_stim_text(const std::string &dem_text) {
  auto parsed = parse_stim_dem(dem_text, /*use_decomp_suggestions=*/false);
  validate_hit_ids(parsed);

  using index_type = sparse_binary_matrix::index_type;
  const std::size_t num_cols = parsed.detector_hits.size();
  const auto index_limit =
      static_cast<std::size_t>(std::numeric_limits<index_type>::max());
  if (num_cols > index_limit || parsed.num_detectors > index_limit ||
      parsed.num_observables > index_limit)
    throw std::runtime_error(
        "Stim DEM dimensions exceed the sparse matrix index type");

  // Total nonzeros are counted in size_t and checked before anything is sized
  // or cast. Accumulating a prefix sum directly in index_type would wrap on an
  // oversized model, under-allocate the index array, and then write past its
  // end -- memory corruption instead of a clean rejection.
  auto total_nnz = [](const std::vector<std::vector<std::size_t>> &groups) {
    std::size_t total = 0;
    for (const auto &group : groups)
      total += group.size();
    return total;
  };
  const std::size_t h_nnz = total_nnz(parsed.detector_hits);
  const std::size_t o_nnz = total_nnz(parsed.observable_hits);
  if (h_nnz > index_limit || o_nnz > index_limit)
    throw std::runtime_error(
        "Stim DEM nonzero count exceeds the sparse matrix index type");

  // Fill the compressed arrays directly. Going through nested per-group vectors
  // would allocate one small buffer per error mechanism and per observable, and
  // then copy them all again while flattening.

  // H: a hit list is already the compressed column for its error mechanism.
  std::vector<index_type> col_ptrs(num_cols + 1, 0);
  std::size_t h_running = 0;
  for (std::size_t err = 0; err < num_cols; ++err) {
    h_running += parsed.detector_hits[err].size();
    col_ptrs[err + 1] = static_cast<index_type>(h_running);
  }
  std::vector<index_type> row_indices(h_nnz);
  std::size_t written = 0;
  for (const auto &hits : parsed.detector_hits)
    for (auto det : hits)
      row_indices[written++] = static_cast<index_type>(det);

  // O is stored by observable, so count each row first, then scatter. Walking
  // error mechanisms in order leaves every row's indices ascending.
  std::vector<std::size_t> observable_counts(parsed.num_observables, 0);
  for (const auto &hits : parsed.observable_hits)
    for (auto ob : hits)
      ++observable_counts[ob];
  std::vector<index_type> row_ptrs(parsed.num_observables + 1, 0);
  std::size_t o_running = 0;
  for (std::size_t ob = 0; ob < parsed.num_observables; ++ob) {
    o_running += observable_counts[ob];
    row_ptrs[ob + 1] = static_cast<index_type>(o_running);
  }
  std::vector<index_type> col_indices(o_nnz);
  std::vector<index_type> cursor(row_ptrs.begin(), row_ptrs.end() - 1);
  for (std::size_t err = 0; err < num_cols; ++err)
    for (auto ob : parsed.observable_hits[err])
      col_indices[cursor[ob]++] = static_cast<index_type>(err);

  sparse_dem projection;
  projection.detector_error_matrix = sparse_binary_matrix::from_csc(
      static_cast<index_type>(parsed.num_detectors),
      static_cast<index_type>(num_cols), std::move(col_ptrs),
      std::move(row_indices));
  projection.observables_flips_matrix = sparse_binary_matrix::from_csr(
      static_cast<index_type>(parsed.num_observables),
      static_cast<index_type>(num_cols), std::move(row_ptrs),
      std::move(col_indices));
  projection.error_rates = std::move(parsed.error_rates);
  return projection;
}

} // namespace details

std::size_t detector_error_model::num_detectors() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

std::size_t detector_error_model::num_error_mechanisms() const {
  auto shape = detector_error_matrix.shape();
  if (shape.size() == 2)
    return shape[1];
  return 0;
}

std::size_t detector_error_model::num_observables() const {
  auto shape = observables_flips_matrix.shape();
  if (shape.size() == 2)
    return shape[0];
  return 0;
}

void detector_error_model::canonicalize_for_rounds(
    uint32_t num_syndromes_per_round, bool remove_zero_syndrome_errors) {
  auto row_indices = dense_to_sparse(detector_error_matrix);
  auto column_order =
      get_sorted_pcm_column_indices(row_indices, num_syndromes_per_round);
  canonicalize_for_rounds_impl(row_indices, column_order,
                               remove_zero_syndrome_errors);
}

void detector_error_model::canonicalize_for_rounds_with_boundary(
    uint32_t num_syndromes_per_round, uint32_t num_boundary_syndromes,
    bool remove_zero_syndrome_errors) {
  // A boundary wider than the interior would misassign rounds silently.
  if (num_boundary_syndromes > num_syndromes_per_round)
    throw std::invalid_argument(
        "canonicalize_for_rounds_with_boundary: num_boundary_syndromes (" +
        std::to_string(num_boundary_syndromes) +
        ") must be <= num_syndromes_per_round (" +
        std::to_string(num_syndromes_per_round) + ")");
  auto row_indices = dense_to_sparse(detector_error_matrix);
  auto column_order = get_sorted_pcm_column_indices(
      row_indices, num_syndromes_per_round, num_boundary_syndromes);
  canonicalize_for_rounds_impl(row_indices, column_order,
                               remove_zero_syndrome_errors);
}

void detector_error_model::canonicalize_for_rounds_impl(
    const std::vector<std::vector<std::uint32_t>> &row_indices,
    const std::vector<std::uint32_t> &column_order,
    bool remove_zero_syndrome_errors) {
  const std::size_t num_obs = this->num_observables();
  const auto num_cols = column_order.size();
  const bool has_error_ids =
      error_ids.has_value() && error_ids->size() == error_rates.size();

  if (row_indices.size() > error_rates.size()) {
    throw std::runtime_error(
        "canonicalize_for_rounds: row_indices size (" +
        std::to_string(row_indices.size()) +
        ") is greater than the number of error rates (" +
        std::to_string(error_rates.size()) +
        "). This likely means either 'error_rates' was populated incorrectly "
        "or the detector_error_matrix  was computed incorrectly.");
  }

  // March through the columns in topological order and merge columns that share
  // the SAME full signature: identical detector rows AND identical observable
  // rows. Columns that differ in either are distinct error mechanisms and are
  // kept separate (merging on detectors alone would relabel observable-flip
  // probability mass). The merge key is therefore (detector rows, observable
  // rows); because the sort above only orders by detector rows, columns with
  // the same detectors but different observables can be interleaved, so we
  // group by key explicitly rather than relying on adjacency.
  using signature_t =
      std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>>;
  std::map<signature_t, std::size_t> sig_to_out;
  std::vector<std::uint32_t> final_column_order;
  // For each retained output column, accumulate probability mass grouped by the
  // exclusive-set it belongs to. Within one exclusive set (same error id) the
  // alternatives are mutually exclusive, so their rates add. Across exclusive
  // sets the mechanisms are independent, so they are combined with the XOR rule
  // P(A xor B) = P(A) + P(B) - 2 P(A) P(B). When error ids are absent every
  // column is treated as its own independent mechanism (keyed by its original
  // column index), reproducing the all-XOR behavior.
  std::vector<std::map<std::size_t, double>> out_exclusive;

  // Track the first observable signature seen for each detector signature so we
  // can flag columns that share a syndrome but flip a different observable.
  // These are kept as distinct mechanisms (above), but they are worth
  // surfacing: they often indicate an ambiguous/degenerate decoding situation.
  // Cap the per-invocation warnings since short-distance codes can have many
  // such mechanisms, and emit a single summary for the remainder.
  constexpr std::size_t max_same_syndrome_diff_obs_warnings = 10;
  std::size_t num_same_syndrome_diff_obs = 0;
  std::map<std::vector<std::uint32_t>,
           std::pair<std::vector<std::uint32_t>, std::uint32_t>>
      first_obs_for_detector;

  for (std::size_t c = 0; c < num_cols; c++) {
    const auto column_index = column_order[c];
    const auto &curr_row_indices = row_indices[column_index];
    const double rate = error_rates[column_index];

    // Build the observable-flip signature for this column.
    std::vector<std::uint32_t> obs_indices;
    for (std::size_t r = 0; r < num_obs; r++)
      if (this->observables_flips_matrix.at({r, column_index}))
        obs_indices.push_back(static_cast<std::uint32_t>(r));

    // Skip columns that carry no information: zero probability, or no detector
    // signature AND no observable flip. A column with no detectors but a
    // nonzero observable flip is a genuine (undetectable) logical error and is
    // retained by default so the model's observable-flip mass is preserved.
    // Such a column has no syndrome for a round-based decoder to act on, so
    // callers that only consume the detector matrix for decoding can drop all
    // zero-syndrome columns via remove_zero_syndrome_errors.
    const bool zero_syndrome = curr_row_indices.empty();
    if (rate == 0.0 || (zero_syndrome && obs_indices.empty()) ||
        (remove_zero_syndrome_errors && zero_syndrome))
      continue;

    signature_t sig{curr_row_indices, obs_indices};
    auto [it, inserted] = sig_to_out.try_emplace(sig, out_exclusive.size());
    if (inserted) {
      out_exclusive.emplace_back();
      final_column_order.push_back(column_index);

      // A new full signature. If this detector syndrome was already seen with a
      // different observable signature, this is a "same syndrome, different
      // observable" mechanism; flag it (capped).
      auto [dit, first_seen] = first_obs_for_detector.try_emplace(
          curr_row_indices, obs_indices, column_index);
      if (!first_seen && dit->second.first != obs_indices) {
        if (num_same_syndrome_diff_obs < max_same_syndrome_diff_obs_warnings)
          CUDA_QEC_WARN(
              "detector_error_model::canonicalize_for_rounds: identical "
              "syndromes exist in detector_error_matrix but have different "
              "observables in observables_flips_matrix; keeping column {} as a "
              "distinct error mechanism (previous column {})",
              column_index, dit->second.second);
        num_same_syndrome_diff_obs++;
      }
    }
    const std::size_t exclusive_key =
        has_error_ids ? error_ids->at(column_index) : column_index;
    out_exclusive[it->second][exclusive_key] += rate;
  }

  // Emit a single summary if we suppressed any per-column warnings above.
  if (num_same_syndrome_diff_obs > max_same_syndrome_diff_obs_warnings)
    CUDA_QEC_WARN(
        "detector_error_model::canonicalize_for_rounds: found {} columns with "
        "identical syndromes but different observables; suppressed {} "
        "additional warnings (only the first {} were shown).",
        num_same_syndrome_diff_obs,
        num_same_syndrome_diff_obs - max_same_syndrome_diff_obs_warnings,
        max_same_syndrome_diff_obs_warnings);

  std::vector<double> new_weights;
  std::vector<std::size_t> new_error_ids;
  new_weights.reserve(out_exclusive.size());
  for (std::size_t i = 0; i < out_exclusive.size(); i++) {
    double weight = 0.0;
    for (const auto &[id, p] : out_exclusive[i])
      weight = weight + p - 2.0 * weight * p;
    new_weights.push_back(weight);
    // Assign each output column a fresh unique id. Canonicalization does not
    // preserve any cross-column exclusivity structure: if a source mechanism's
    // mutually-exclusive outcomes land in different signature columns, that
    // relationship is no longer recoverable from the ids, so we do not pretend
    // it is. Unique ids simply mark every canonicalized column as an
    // independent mechanism, which is the relation the merged rates were
    // composed under.
    if (has_error_ids)
      new_error_ids.push_back(i);
  }

  std::swap(this->error_rates, new_weights);
  if (has_error_ids)
    std::swap(*this->error_ids, new_error_ids);

  // These two data structures should have the same number of columns.
  // (number of canonicalized error mechanisms)
  // Create the reordered, reduced Detector Error Matrix.
  this->detector_error_matrix = cudaq::qec::reorder_pcm_columns(
      this->detector_error_matrix, final_column_order);

  // Create the reordered, reduced Observables Flips Matrix.
  this->observables_flips_matrix = cudaq::qec::reorder_pcm_columns(
      this->observables_flips_matrix, final_column_order);
}

} // namespace cudaq::qec
