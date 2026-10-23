/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Test-only serial reference for parallel-window decoding. It closes the
// whole shot once, slices every window from it by the window rules (zones,
// ports, imports and sends), independently of window_from_dem_chunks() and
// schedule_windows(), and runs the canonical workload one window at a time,
// producers first. Also holds the window layouts the window tests share.

#pragma once

#include "cuda-qx/core/heterogeneous_map.h"
#include "task_graph_test_fixtures.h"
#include "../lib/decoding_task/decoding_task_commitment.h"
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/detector_error_model.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cudaq::qec {

/// Window layers and the send edges between them.
struct window_graph {
  std::vector<window_layer> layers;
  std::vector<task_graph_edge> edges;
};

/// The parallel-window graph: A[k-1].forward to B[k].receive_0 and
/// A[k].backward to B[k].receive_1.
inline window_graph parallel_graph(const layered_requests &requests) {
  window_graph g;
  g.layers = {{"A", "a", requests.a}, {"B", "b", requests.b}};
  g.edges = {{"A", "forward", "B", "receive_0", -1},
             {"A", "backward", "B", "receive_1", 0}};
  return g;
}

/// A sliding window whose forward delta feeds its next instance.
inline window_graph sliding_graph(std::int64_t step, std::int64_t size) {
  window_graph g;
  g.layers = {{"W", "w", sliding(step, size)}};
  g.edges = {{"W", "forward", "W", "receive_0", -1}};
  return g;
}

/// The interior-commit layout for the default sizes, with M = 1: A commits
/// c_a - 2 rounds between b + 1 buffer rounds, and B's gap grows to c_b + 2.
inline layered_requests interior_commit(std::int64_t c_a, std::int64_t b,
                                        std::int64_t c_b) {
  return layered(c_a - 2, b + 1, c_b + 2);
}

/// A's input truncates the previous round's crossing faults.
inline layered_requests with_truncate(layered_requests requests) {
  requests.a.input = {experimental::port_disposition::open, std::nullopt,
                      experimental::input_boundary::truncate};
  return requests;
}

namespace pw_reference {

/// The closed model of a whole R-round shot, by column.
struct shot {
  detector_error_model dem;
  std::vector<std::vector<std::uint32_t>> h; ///< Rows of each column
  std::vector<std::vector<std::uint32_t>> o; ///< Observables of each column
  std::vector<std::size_t> row;    ///< First row of each round, then the total
  std::vector<std::size_t> column; ///< Each round's first column, then total
  std::int64_t max_span = 0;       ///< Rounds a fault reaches past its own
};

/// Expands @p spec to @p rounds rounds and closes it with dem_close_all(),
/// without the experiment template.
shot whole_shot(dem_chunks_spec spec, std::size_t rounds);

/// Detection events of a set of faults, over the shot's rows.
std::vector<std::uint8_t> syndrome_of(const shot &s,
                                      const std::vector<std::uint8_t> &e);

/// Observables a set of faults flips.
std::vector<std::uint8_t> observables_of(const shot &s,
                                         const std::vector<std::uint8_t> &e);

/// A decoder over the columns @p h, each a list of rows, that returns fault
/// columns.
std::unique_ptr<decoder>
make_decoder(const std::vector<std::vector<std::uint32_t>> &h,
             std::size_t num_rows, std::vector<double> priors,
             const std::string &name, const cudaqx::heterogeneous_map &params);

/// A send of a window: its clipped rounds and their shot rows.
struct reference_send {
  std::string id;
  experimental::send_mode mode = experimental::send_mode::delta;
  std::vector<experimental::round_range> rounds;
  std::vector<std::size_t> rows;
};

/// One window instance, sliced from the whole shot.
struct reference_window {
  std::string node;
  std::int64_t k = 0;
  experimental::round_range rounds;
  experimental::round_range commit;
  std::vector<experimental::round_range> buffers;
  bool input_exists = false;
  bool output_exists = false;
  experimental::round_range imports;
  std::vector<window_dependency> inputs; ///< One per incoming edge
  std::vector<reference_send> sends;
  std::vector<std::size_t> rows; ///< Shot rows: own, then imported
  std::size_t num_own_rows = 0;
  /// Shot columns the decoder sees: the window's own faults, then, with a
  /// truncated input, the previous round's faults that touch its rows.
  std::vector<std::size_t> columns;
  std::size_t num_faults = 0;  ///< The window's own faults, the first columns
  std::size_t num_virtual = 0; ///< Boundary columns after the shot columns
  std::vector<std::uint8_t> committed; ///< Per own fault: in the commit zone
};

/// One window's decode.
struct window_decode {
  std::vector<std::uint8_t> detectors;          ///< d, over the window's rows
  std::vector<std::uint8_t> estimate;           ///< e, over decoder columns
  std::vector<std::size_t> committed;           ///< Shot columns of e'
  std::vector<std::vector<std::uint8_t>> sends; ///< One per send, by row
  std::vector<std::uint8_t> frame;              ///< l = O e'
  bool converged = true;
  bool satisfied = true; ///< H e = d
};

/// One shot's decode.
struct result {
  std::vector<std::uint8_t> errors;      ///< Committed faults, by shot column
  std::vector<std::uint8_t> observables; ///< O times errors
  std::vector<std::uint8_t> frame;       ///< XOR of every window's l
  bool converged = true;                 ///< Every window's decoder converged
  bool satisfied = true;                 ///< Every window's H e = d
  std::vector<window_decode> windows;    ///< In windows() order
};

/// Serial parallel-window decoding of one shot length.
class parallel_window_reference {
public:
  /// Slices every window of @p graph in a @p rounds round shot of @p spec
  /// and builds its decoder.
  /// @throws std::invalid_argument if a request sets a column option, which
  /// the reference does not model, or the dependencies form a cycle.
  parallel_window_reference(const dem_chunks_spec &spec,
                            const window_graph &graph, std::size_t rounds,
                            const std::string &decoder_name,
                            const cudaqx::heterogeneous_map &params = {});

  /// Decodes one shot's detection events. Without @p sends, no delta is
  /// applied and imports keep the shot's detection events: the negative
  /// control.
  result decode(const std::vector<std::uint8_t> &detectors, bool sends = true);

  /// The closed whole-shot model the windows are sliced from.
  const shot &model() const { return shot_; }
  /// Every window, by layer in graph order, then by k.
  const std::vector<reference_window> &windows() const { return windows_; }

private:
  shot shot_;
  std::vector<reference_window> windows_;
  /// Each window's decoder columns, as rows local to the window.
  std::vector<std::vector<std::vector<std::uint32_t>>> h_;
  std::vector<std::unique_ptr<decoder>> decoders_;
  /// Window indices with every producer before its consumers.
  std::vector<std::size_t> order_;
};

} // namespace pw_reference
} // namespace cudaq::qec
