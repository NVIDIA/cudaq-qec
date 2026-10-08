/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Memory-circuit-specific DEM function: dem_chunks_to_d_sparse.
//
// Encodes the XOR-of-consecutive-rounds detector convention used by CSS
// memory experiments. Not valid for circuits whose detectors are not pairs
// of adjacent syndrome rounds.

#include "cudaq/qec/dem_chunks_memory.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// D_sparse[det_id] = measurement bit positions that XOR-combine to form that
// detector under the memory-experiment convention:
//   det k       (r=0): {k}               — vs. zero initial state
//   det r*d+k (r>0): {(r-1)*d+k, r*d+k} — consecutive syndrome XOR
std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const std::vector<extended_dem> &dem_chunks,
                       seam_id from_seam, seam_id to_seam) {
  if (dem_chunks.empty())
    throw std::invalid_argument(
        "dem_chunks_to_d_sparse: dem_chunks must be non-empty");

  uint32_t d = 0;
  if (dem_chunks[0].has_seam(to_seam))
    d = dem_chunks[0].get_seam(to_seam).num_rows();
  if (d == 0 && dem_chunks[0].has_seam(from_seam))
    d = dem_chunks[0].get_seam(from_seam).num_rows();
  if (d == 0)
    throw std::invalid_argument(
        "dem_chunks_to_d_sparse: chunk 0 has no seam rows on either side, "
        "so its rounds cannot be counted");

  // dem_chunks_to_rounds validates seam widths and contractibility.
  const std::size_t T = dem_chunks_to_rounds(dem_chunks, from_seam, to_seam);
  const std::size_t n_det = T * static_cast<std::size_t>(d);

  if (T > 0) {
    const auto max_bit = static_cast<std::size_t>(T - 1) * d + (d - 1);
    if (max_bit > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error(
          "dem_chunks_to_d_sparse: measurement bit index exceeds uint32_t max");
  }

  std::vector<std::vector<uint32_t>> d_sparse(n_det);
  for (std::size_t r = 0; r < T; ++r) {
    for (uint32_t k = 0; k < d; ++k) {
      const std::size_t det = r * d + k;
      if (r == 0) {
        d_sparse[det].assign(1, k);
      } else {
        d_sparse[det] = {static_cast<uint32_t>((r - 1) * d + k),
                         static_cast<uint32_t>(r * d + k)};
      }
    }
  } // end - for(r)
  return d_sparse;
} // end - dem_chunks_to_d_sparse()

namespace {

// Expanded per distinct phase rather than per round: the only thing the
// expansion is consulted for below is a phase's detector count, which is its
// to_seam rows plus its interior rows. Neither depends on where the phase
// sits, so expanding every round would build a chain proportional to
// num_rounds to read a handful of per-phase constants -- the cost the chunk
// form exists to avoid.
std::map<phase_id, uint64_t> detectors_per_phase(const dem_chunks_spec &spec) {
  std::map<phase_id, uint64_t> detectors;
  const std::vector<seam_id> ids{spec.seam.to_seam, spec.seam.from_seam};
  for (const auto &entry : spec.phases) {
    const auto chunk = dem_chunk_from_spec(
        entry.spec, ids, "dem_chunks." + std::to_string(entry.id.value));
    detectors[entry.id] = (chunk.has_seam(spec.seam.to_seam)
                               ? chunk.get_seam(spec.seam.to_seam).num_rows()
                               : 0) +
                          uint64_t{chunk.num_interior_rows()};
  }
  return detectors;
} // end - detectors_per_phase()

} // namespace

std::vector<std::vector<uint32_t>>
dem_chunks_to_d_sparse(const dem_chunks_spec &spec) {
  const seam_id to_seam = spec.seam.to_seam;
  const auto sequence = spec.phase_sequence();
  if (!spec.has_D_sparse()) {
    // The memory convention needs the chain itself, for the per-round width
    // and the contractibility checks dem_chunks_to_rounds() makes.
    return dem_chunks_to_d_sparse(dem_chunks_from_spec(spec),
                                  spec.seam.from_seam, to_seam);
  }
  spec.validate();
  const auto detectors_of = detectors_per_phase(spec);

  std::vector<std::vector<uint32_t>> d_sparse;
  uint64_t previous_start = 0, previous_count = 0, start = 0;
  for (std::size_t i = 0; i < sequence.size(); ++i) {
    const auto &phase = spec.get_phase(sequence[i]);
    const uint64_t count = spec.measurements_of(phase);
    const std::string context = "dem_chunks_to_d_sparse: chunk " +
                                std::to_string(i) + " (phase '" +
                                sequence[i].name() + "')";
    const uint64_t detectors = detectors_of.at(sequence[i]);
    const uint64_t first_row = d_sparse.size();
    std::vector<uint32_t> row;
    for (const std::int64_t value : phase.D_sparse) {
      if (value == -1) {
        d_sparse.push_back(std::move(row));
        row.clear();
        continue;
      }
      const auto index = static_cast<uint64_t>(value);
      uint64_t measurement = previous_start + index;
      if (index >= previous_count) {
        if (index - previous_count >= count)
          throw std::invalid_argument(
              context + ": D_sparse index " + std::to_string(index) +
              " exceeds the " + std::to_string(previous_count) + " + " +
              std::to_string(count) + " measurements it may name");
        measurement = start + (index - previous_count);
      }
      if (measurement > std::numeric_limits<uint32_t>::max())
        throw std::overflow_error(context +
                                  ": measurement index exceeds uint32_t max");
      row.push_back(static_cast<uint32_t>(measurement));
    }
    if (d_sparse.size() - first_row != detectors)
      throw std::invalid_argument(context + ": D_sparse has " +
                                  std::to_string(d_sparse.size() - first_row) +
                                  " rows but the chunk contributes " +
                                  std::to_string(detectors) + " detectors");
    previous_start = start;
    previous_count = count;
    start += count;
  }
  return d_sparse;
} // end - dem_chunks_to_d_sparse(spec)

// Summed over the expanded sequence rather than over the distinct phases: a
// repeating phase contributes its count once per round it is visited.
std::optional<std::uint64_t>
dem_chunks_measurement_count(const dem_chunks_spec &spec) {
  std::uint64_t total = 0;
  if (!spec.has_D_sparse())
    return std::nullopt;
  for (const auto id : spec.phase_sequence()) {
    const auto &phase = spec.get_phase(id);
    if (!phase.num_measurements && !spec.measurements_per_round)
      return std::nullopt;
    total += spec.measurements_of(phase);
  }
  return total;
} // end - dem_chunks_measurement_count()

} // namespace cudaq::qec
