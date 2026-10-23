/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// experiment_template: splits a dem_chunks_spec into prefix, repeating and
// suffix phases once, prebuilds each phase's chunk, and answers per-round
// questions in closed form, so no query expands the whole shot.

#include "decoding_task_experiment_template.h"
#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>

namespace cudaq::qec {

// Whether some fault of chunk c touches its from_seam band, the next round,
// whether or not it touches its own round too.
static bool reaches_next_round(const extended_dem &c, seam_id from) {
  if (!c.has_seam(from))
    return false;
  const auto band = c.get_seam(from);
  const auto h = c.H.to_csc();
  return std::any_of(h.indices().begin(), h.indices().end(), [&](auto row) {
    return row >= band.row_begin && row < band.row_end;
  });
}

// Takes the phase layout from the spec, then builds each phase's chunk twice:
// inside the shot, and ending it without from_seam. A spec without a
// repeating phase is all prefix.
experiment_template::experiment_template(dem_chunks_spec spec)
    : spec_(std::move(spec)) {
  const auto to = spec_.seam.to_seam;
  const auto from = spec_.seam.from_seam;
  std::optional<std::pair<phase_id, std::size_t>> uniform;
  std::set<phase_id> chain;
  spec_.validate();
  repeats_ = spec_.has_repeating_phase();
  if (repeats_) {
    auto layout = spec_.layout();
    prefix_ = std::move(layout.prefix);
    bulk_ = layout.repeating;
    suffix_ = std::move(layout.suffix);
    last_ = suffix_.empty() ? bulk_ : suffix_.back();
    min_rounds_ = std::max<uint64_t>(2, spec_.minimum_rounds());
  } else {
    prefix_ = spec_.phase_sequence();
    last_ = prefix_.back();
    min_rounds_ = prefix_.size();
  }
  if (repeats_ && spec_.num_rounds && *spec_.num_rounds < min_rounds_)
    throw std::invalid_argument("experiment_template: num_rounds " +
                                std::to_string(*spec_.num_rounds) +
                                " is below the " + std::to_string(min_rounds_) +
                                " rounds the phase chain needs");
  num_observables_ = spec_.num_observables();
  chain.insert(prefix_.begin(), prefix_.end());
  chain.insert(suffix_.begin(), suffix_.end());
  if (repeats_)
    chain.insert(bulk_);

  // A phase no connection reaches is in no shot, so it is not built.
  for (const auto &entry : spec_.phases) {
    if (!chain.count(entry.id))
      continue;
    const auto context = "dem_chunks." + entry.id.name();
    phase_chunks p{dem_chunk_from_spec(entry.spec, {to, from}, context),
                   dem_chunk_from_spec(entry.spec, {to}, context)};
    // Zone edges must fall on chunk edges, so a chunk is one round: its
    // detectors are its to_seam band, with no interior rows of later rounds.
    if (p.inner.num_interior_rows() != 0)
      throw std::invalid_argument("experiment_template: phase '" +
                                  entry.id.name() +
                                  "' spans more than one round");
    p.rows = spec_.detectors_of(entry.spec, context);
    p.faults = p.inner.num_faults();
    // The round's width in the stream: its count with D_sparse rows, else
    // one measurement per detector, the memory convention, which ignores
    // any count.
    if (!spec_.has_D_sparse())
      p.measurements = p.rows;
    else if (entry.spec.num_measurements || spec_.measurements_per_round)
      p.measurements = spec_.measurements_of(entry.spec);
    if (reaches_next_round(p.inner, from))
      max_span_ = 1;
    if (!spec_.has_D_sparse())
      check_uniform_round(entry.id, p, uniform);
    phases_.emplace(entry.id, std::move(p));
  } // end - for(entry)

  // A round arriving before FINAL is read at the bulk width, so only the
  // shot's last phase may differ from it.
  for (std::size_t i = 0; repeats_ && i + 1 < suffix_.size(); ++i)
    if (phases_.at(suffix_[i]).measurements != phases_.at(bulk_).measurements)
      throw std::invalid_argument(
          "experiment_template: suffix phase '" + suffix_[i].name() +
          "' precedes FINAL, so it needs the bulk measurement width");
} // end - experiment_template::experiment_template()

// Detector k of a round without D_sparse rows differences measurement k of
// that round and of the one before, the memory convention of
// dem_chunks_to_d_sparse(), which needs every seam band as wide as the first
// phase's round. A wider round would read its own measurements as the
// previous round's.
void experiment_template::check_uniform_round(
    phase_id id, const phase_chunks &p,
    std::optional<std::pair<phase_id, std::size_t>> &uniform) const {
  const auto from = spec_.seam.from_seam;
  const std::size_t next =
      p.inner.has_seam(from) ? p.inner.get_seam(from).num_rows() : 0;
  const auto fail = [&](const std::string &what) {
    throw std::invalid_argument(
        "experiment_template: without D_sparse rows every round needs the "
        "same detectors, so a round's measurements line up with the last "
        "round's; phase '" +
        id.name() + "' has " + std::to_string(p.rows) +
        " detectors a round, but " + what);
  };
  if (!uniform)
    uniform = std::pair{id, p.rows};
  if (p.rows != uniform->second)
    fail("phase '" + uniform->first.name() + "' has " +
         std::to_string(uniform->second));
  if (next != 0 && next != p.rows)
    fail("its outgoing seam reaches " + std::to_string(next));
} // end - experiment_template::check_uniform_round()

// Rejects shot lengths the spec does not admit, and rounds past last. Checks
// R first, so last = R - 1 cannot wrap unchecked.
void experiment_template::check_round(std::size_t r, std::size_t last,
                                      std::size_t R) const {
  if (R < min_rounds_)
    throw std::invalid_argument("experiment_template: a " + std::to_string(R) +
                                "-round shot is shorter than the minimum of " +
                                std::to_string(min_rounds_));
  if (!repeats_ && R != min_rounds_)
    throw std::invalid_argument(
        "experiment_template: a spec without a repeating phase admits only " +
        std::to_string(min_rounds_) + "-round shots, not " + std::to_string(R));
  if (r > last)
    throw std::invalid_argument(
        "experiment_template: round " + std::to_string(r) + " is past " +
        std::to_string(last) + " in a " + std::to_string(R) + "-round shot");
} // end - experiment_template::check_round()

// Prefix rounds count from the start, suffix rounds from the end.
phase_id experiment_template::phase(std::size_t r, std::size_t R) const {
  const auto tail = R - suffix_.size();
  check_round(r, R - 1, R);
  if (r < prefix_.size())
    return prefix_[r];
  return r < tail ? bulk_ : suffix_[r - tail];
}

// The shot's last chunk has no next round, so it carries no from_seam.
const extended_dem &experiment_template::chunk(std::size_t r,
                                               std::size_t R) const {
  const auto &p = phases_.at(phase(r, R));
  return r + 1 < R ? p.inner : p.last;
}

// O(prefix + suffix): the bulk rounds contribute one multiplied term.
template <typename Size>
std::size_t experiment_template::sum_before(std::size_t r, std::size_t R,
                                            Size size) const {
  const auto tail = R - suffix_.size();
  std::size_t total = 0;
  check_round(r, R, R);
  for (std::size_t i = 0; i < std::min(r, prefix_.size()); ++i)
    total += size(phases_.at(prefix_[i]));
  if (repeats_ && r > prefix_.size())
    total += (std::min(r, tail) - prefix_.size()) * size(phases_.at(bulk_));
  for (std::size_t i = tail; i < r; ++i)
    total += size(phases_.at(suffix_[i - tail]));
  return total;
} // end - experiment_template::sum_before()

// Detector rows before round r.
std::size_t experiment_template::row_offset(std::size_t r,
                                            std::size_t R) const {
  return sum_before(r, R, [](const phase_chunks &p) { return p.rows; });
}

// Fault columns before round r.
std::size_t experiment_template::column_offset(std::size_t r,
                                               std::size_t R) const {
  return sum_before(r, R, [](const phase_chunks &p) { return p.faults; });
}

// Measurements before round r; every phase it sums needs a count.
std::size_t experiment_template::measurement_offset(std::size_t r,
                                                    std::size_t R) const {
  return sum_before(r, R, [](const phase_chunks &p) {
    if (!p.measurements)
      throw std::invalid_argument(
          "experiment_template: a phase has no measurement count "
          "(num_measurements / measurements_per_round)");
    return *p.measurements;
  });
}

} // namespace cudaq::qec
