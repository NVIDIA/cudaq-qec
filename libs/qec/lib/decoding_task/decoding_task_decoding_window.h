/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Internal header: decoding windows built from dem_chunks. A window is the
// problem statement of one decode task, as in the EDEM window specification:
// its zones, ports, imports, sends and measurement maps. A window request
// (window_request.h) names a layer's windows relative to an anchor;
// window_from_dem_chunks() builds window k of a shot. Never included from a
// public header.

#pragma once

#include "decoding_task_experiment_template.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include "cudaq/qec/window_request.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

// The request types are experimental in the public API; the window code uses
// them throughout.
using experimental::input_boundary;
using experimental::parallel_resolution;
using experimental::port_disposition;
using experimental::round_range;
using experimental::send_mode;
using experimental::window_request;
using experimental::window_round_limit;

/// Half-open model columns [begin, end).
struct column_range {
  std::size_t begin = 0;
  std::size_t end = 0;
  bool operator==(const column_range &) const = default;
};

/// Half-open window indices [begin, end).
struct index_range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
};

/// A built port. A window edge at the shot's first or last round is a
/// physical boundary, not a port.
struct window_port {
  bool exists = false;
  port_disposition disposition = port_disposition::open;
  input_boundary boundary = input_boundary::columns; ///< An open input's
  double prior = 0.0;  ///< An open input's boundary-column prior
  round_range imports; ///< A closed output's imported rounds
  bool operator==(const window_port &) const = default;
};

/// Why a commit column's parallel column, one with the same rows in the
/// window, lets the decoder's choice between them change the commitment or
/// the logical.
enum class parallel_conflict {
  other_zone,   ///< The parallel column is in another zone
  boundary,     ///< It is a virtual-outcome or truncated crossing column
  outside_rows, ///< Same zone, but rows outside the window differ
  observable,   ///< Same zone and full rows, different observable effect
};

/// A commit column and a parallel column the decoder could pick instead.
struct parallel_columns {
  std::size_t commit_column = 0;
  std::size_t other_column = 0;
  parallel_conflict conflict = parallel_conflict::other_zone;
  bool operator==(const parallel_columns &) const = default;
};

/// A built send: its rounds, clipped to the window, and their own rows.
struct window_send {
  std::string id;
  send_mode mode = send_mode::delta;
  std::vector<round_range> rounds;
  std::vector<std::size_t> rows;
  bool operator==(const window_send &) const = default;
};

/// An affine map over GF(2): output = matrix * input + offset. The inputs
/// and outputs are partitioned into ports, listed by size.
struct affine_map {
  sparse_binary_matrix matrix; ///< One row per output, one column per input
  std::vector<std::uint8_t> offset;
  std::vector<std::size_t> input_sizes;
  std::vector<std::size_t> output_sizes;
};

/// A window's model in the forms a decode kernel uses. Columns are fault
/// columns, then boundary or crossing columns, then logical-error columns;
/// rows are the window's own, then its imported ones. The logical-error
/// columns touch no detector, so H stops before them.
struct window_model {
  sparse_binary_matrix h;     ///< Rows by the decoder's columns, canonical CSC
  sparse_binary_matrix o;     ///< Observables by every column, canonical CSC
  std::vector<double> priors; ///< One per column
};

/// A window decode problem; mirrors the EDEM window spec. Built from
/// dem_chunks in this release.
struct decoding_window {
  std::shared_ptr<const experiment_template> context;
  std::int64_t k = 0;
  std::size_t shot_rounds = 0;
  round_range rounds; ///< The window's occurrences
  round_range commit; ///< Commit zone
  std::vector<round_range> buffers;
  window_port input;
  window_port output;
  std::vector<window_send> sends;
  /// Own rows, then imported rows. Fault columns in occurrence order, then
  /// for an open input one virtual-outcome column per input seam detector,
  /// or the truncated crossing columns, then one logical-error column per
  /// observable.
  window_model model;
  std::size_t num_own_rows = 0;
  std::size_t num_fault_columns = 0;
  std::size_t num_virtual_outcome_columns = 0;
  std::size_t num_crossing_columns = 0;
  std::size_t num_logical_error_columns = 0;
  column_range commit_columns;
  /// Each commit column's first conflicting parallel column, found while
  /// the full rows are known; empty when the column options resolve them.
  std::vector<parallel_columns> parallel_conflicts;
  /// For each fault column, the unmerged fault columns it stands for,
  /// counted from the window's first. The one its corrections are reported
  /// against comes first: the most likely, and under most_likely the most
  /// likely of the effect whose columns together are most likely.
  std::vector<std::vector<std::size_t>> fault_column_sources;
  /// (v_in, m) -> (own rows, then the virtual outcomes on the commit zone's
  /// outgoing wire and on the output port's wire, where those are ports).
  affine_map m2d;
  std::size_t num_virtual_outcomes = 0; ///< Width of v_in
  std::size_t num_measurements = 0;     ///< Measurements of the rounds
  std::optional<affine_map> m2o;        ///< m -> raw observables, on request

  /// Columns the inner decoder sees: every column but the logical-error ones,
  /// which touch no detector.
  std::size_t num_decoder_columns() const {
    return num_fault_columns + num_virtual_outcome_columns +
           num_crossing_columns;
  }
};

/// The instances k of @p request whose commit zone, clipped to the shot,
/// is non-empty in a @p num_rounds round shot.
/// @throws std::invalid_argument if the request has no commit zone or a
/// period below 1.
index_range window_instances(const window_request &request,
                             std::size_t num_rounds);

/// The rounds, zones, ports and sends of window @p k of @p request in a
/// @p num_rounds round shot: window_from_dem_chunks() without the model and
/// the maps, so it stitches no chunks.
/// @throws std::invalid_argument as window_from_dem_chunks(), except for
/// m2o.
decoding_window place_window(std::shared_ptr<const experiment_template> context,
                             const window_request &request, std::int64_t k,
                             std::size_t num_rounds);

/// Build window @p k of @p request in a @p num_rounds round shot.
/// @param with_m2o Also build m2o, which needs the window to hold the
/// shot's readout round.
/// @throws std::invalid_argument as check_window_request() if the request is
/// invalid, or naming the operation if window k does not exist or m2o cannot
/// be built.
decoding_window
window_from_dem_chunks(std::shared_ptr<const experiment_template> context,
                       const window_request &request, std::int64_t k,
                       std::size_t num_rounds, bool with_m2o = false);

/// The canonical workload after decoding.
struct window_outputs {
  std::vector<std::uint8_t> committed;          ///< e' = C AND e
  std::vector<std::vector<std::uint8_t>> sends; ///< One per window send
  std::vector<std::uint8_t> frame;              ///< O * e'
};

/// Which windows of a layer share a model. position is k while the shot's
/// prefix reaches the rounds the window's model reads; terminal is R minus
/// the anchor while the shot's end does. A key with neither is the layer's
/// bulk. Windows with equal keys are the same problem up to a shift of
/// rounds, so a layer has finitely many variants for every shot length.
struct variant_key {
  std::optional<std::int64_t> position;
  std::optional<std::int64_t> terminal;
  /// True for the layer's bulk.
  bool bulk() const { return !position && !terminal; }
  auto operator<=>(const variant_key &) const = default;
};

/// The variant key of window @p k of @p request in a @p num_rounds shot.
variant_key window_variant(const experiment_template &context,
                           const window_request &request, std::int64_t k,
                           std::size_t num_rounds);

/// A shot length past which a layer gains no variant: every key appears in
/// some shot of [context.min_rounds(), variant_probe_rounds()).
std::size_t variant_probe_rounds(const experiment_template &context,
                                 const window_request &request);

/// Check that two windows are the same problem up to a shift of rounds.
/// @throws std::invalid_argument naming the first field they differ in.
void check_same_window_up_to_shift(const decoding_window &a,
                                   const decoding_window &b);

/// Check that the layer's bulk windows k_bulk and k_bulk + 1 are the same
/// problem up to a shift, which lets one bulk model serve every bulk window.
/// @throws std::invalid_argument as check_same_window_up_to_shift().
void check_translation_invariance(
    std::shared_ptr<const experiment_template> context,
    const window_request &request);

/// A window's model as dense matrices, for tests and diagnostics: H spans
/// every column, with no entries in the logical-error columns. At large
/// distances this takes gigabytes.
detector_error_model dense_model(const window_model &model);

/// Commit @p estimate and compute the window's sends and frame update.
/// @param detectors The window's detector values, one per model row.
/// @param estimate The decoder's estimate, one per model column.
/// @throws std::invalid_argument if a vector has the wrong length.
window_outputs commit_window(const decoding_window &window,
                             const std::vector<std::uint8_t> &detectors,
                             const std::vector<std::uint8_t> &estimate);

} // namespace cudaq::qec
