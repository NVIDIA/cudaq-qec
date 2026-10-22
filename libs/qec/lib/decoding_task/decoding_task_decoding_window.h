/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Internal header: decoding windows built from dem_chunks. A window is the
// problem statement of one decode task, as in the EDEM window specification:
// its zones, ports, imports, sends and measurement maps. Window requests name
// a layer's windows relative to an anchor; window_from_dem_chunks() builds
// window k of a shot. Never included from a public header.

#pragma once

#include "decoding_task_experiment_template.h"
#include "cudaq/qec/detector_error_model.h"
#include "cudaq/qec/sparse_binary_matrix.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec {

/// Half-open rounds [begin, end): relative to the anchor in a request,
/// absolute in a built window.
struct round_range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
  /// True when the range holds no round.
  bool empty() const { return end <= begin; }
  bool operator==(const round_range &) const = default;
};

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

/// Whether a port is open (boundary columns or omitted rows) or closed
/// (received deltas or imports).
enum class port_disposition { open, closed };

/// What a send delivers: the committed faults' flips, or the detectors left
/// for the recipient to explain.
enum class send_mode { delta, residual };

/// The windows of one layer: window k is anchored at offset + k * period,
/// and every round range is relative to that anchor.
struct window_request {
  /// The port where the window's first round begins.
  struct input_port {
    std::optional<port_disposition> disposition; ///< Required
    std::optional<double> prior; ///< Boundary-column prior; open only
  };
  /// The port past the window's last round.
  struct output_port {
    std::optional<port_disposition> disposition; ///< Required
    /// Imported rounds, starting at the wire; closed only. Defaults to the
    /// template's max_span() rounds past the wire.
    std::optional<round_range> imports;
  };
  /// A named send. The ids `forward` and `backward` are the spec's defaults
  /// and need no mode or rounds.
  struct send_request {
    std::string id;
    std::optional<send_mode> mode;
    std::vector<round_range> rounds;
  };

  std::int64_t period = 0;
  std::int64_t offset = 0;
  std::optional<round_range> commit;
  std::vector<round_range> buffers;
  input_port input;
  output_port output;
  std::vector<send_request> sends;
};

/// A built port. A window edge at the shot's first or last round is a
/// physical boundary, not a port.
struct window_port {
  bool exists = false;
  port_disposition disposition = port_disposition::open;
  double prior = 0.0;  ///< An open input's boundary-column prior
  round_range imports; ///< A closed output's imported rounds
};

/// A built send: its rounds, clipped to the window, and their own rows.
struct window_send {
  std::string id;
  send_mode mode = send_mode::delta;
  std::vector<round_range> rounds;
  std::vector<std::size_t> rows;
};

/// An affine map over GF(2): output = matrix * input + offset. The inputs
/// and outputs are partitioned into ports, listed by size.
struct affine_map {
  sparse_binary_matrix matrix; ///< One row per output, one column per input
  std::vector<std::uint8_t> offset;
  std::vector<std::size_t> input_sizes;
  std::vector<std::size_t> output_sizes;
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
  /// one virtual-outcome column per input seam detector, then one
  /// logical-error column per observable.
  detector_error_model model;
  std::size_t num_own_rows = 0;
  std::size_t num_fault_columns = 0;
  std::size_t num_virtual_outcome_columns = 0;
  std::size_t num_logical_error_columns = 0;
  column_range commit_columns;
  /// (v_in, m) -> (own rows, then the virtual outcomes on the commit zone's
  /// outgoing wire and on the output port's wire, where those are ports).
  affine_map m2d;
  std::size_t num_virtual_outcomes = 0; ///< Width of v_in
  std::size_t num_measurements = 0;     ///< Measurements of the rounds
  std::optional<affine_map> m2o;        ///< m -> raw observables, on request

  /// Columns the inner decoder sees: every column but the logical-error ones,
  /// which touch no detector.
  std::size_t num_decoder_columns() const {
    return num_fault_columns + num_virtual_outcome_columns;
  }
};

/// The instances k of @p request whose commit zone, clipped to the shot,
/// is non-empty in a @p num_rounds round shot.
/// @throws std::invalid_argument if the request has no commit zone or a
/// period below 1.
index_range window_instances(const window_request &request,
                             std::size_t num_rounds);

/// Build window @p k of @p request in a @p num_rounds round shot.
/// @param with_m2o Also build m2o, which needs the window to hold the
/// shot's readout round.
/// @throws std::invalid_argument naming the operation if the request is
/// invalid, window k does not exist, or m2o cannot be built.
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

/// Commit @p estimate and compute the window's sends and frame update.
/// @param detectors The window's detector values, one per model row.
/// @param estimate The decoder's estimate, one per model column.
/// @throws std::invalid_argument if a vector has the wrong length.
window_outputs commit_window(const decoding_window &window,
                             const std::vector<std::uint8_t> &detectors,
                             const std::vector<std::uint8_t> &estimate);

} // namespace cudaq::qec
