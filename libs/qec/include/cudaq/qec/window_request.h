/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Window requests: the decoding windows of one layer, in the terms of the
// EDEM window specification. A request is one entry of an ingest node's
// `windows` map, or a `window` entry of a configuration's data_artifacts.
//
// Experimental: these types prototype the caller request of the cudaq-edem
// composer, which its specification leaves open. They may change, or be
// replaced, when cudaq-edem windows land; the YAML keys are meant to stay.

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::qec::experimental {

/// Half-open rounds [begin, end): relative to the anchor in a request,
/// absolute in a built window.
struct round_range {
  std::int64_t begin = 0;
  std::int64_t end = 0;
  /// True when the range holds no round.
  bool empty() const { return end <= begin; }
  bool operator==(const round_range &) const = default;
};

/// Whether a port is open (boundary columns or omitted rows) or closed
/// (received deltas or imports).
enum class port_disposition { open, closed };

/// What a send delivers: the committed faults' flips, or the detectors left
/// for the recipient to explain.
enum class send_mode { delta, residual };

/// What stands for the outside faults at an open input: the wire's boundary
/// columns, or the previous chunk's crossing faults truncated to the window.
enum class input_boundary { columns, truncate };

/// How a same-zone class of columns with identical rows but different
/// observable effects is handled: rejected by commitment validation, or
/// merged into its most likely member's response.
enum class parallel_resolution { reject, most_likely };

/// The windows of one layer: window k is anchored at offset + k * period,
/// and every round range is relative to that anchor.
struct window_request {
  /// The port where the window's first round begins.
  struct input_port {
    std::optional<port_disposition> disposition; ///< Required
    std::optional<double> prior; ///< Boundary-column prior; open only
    std::optional<input_boundary> boundary; ///< Open only; default columns
    bool operator==(const input_port &) const = default;
  };
  /// The port past the window's last round.
  struct output_port {
    std::optional<port_disposition> disposition; ///< Required
    /// Imported rounds, starting at the wire; closed only. Defaults to as
    /// many rounds past the wire as a fault spans beyond its own round.
    std::optional<round_range> imports;
    bool operator==(const output_port &) const = default;
  };
  /// A named send. The ids `forward` and `backward` are the spec's defaults
  /// and need no mode or rounds.
  struct send_request {
    std::string id;
    std::optional<send_mode> mode;
    std::vector<round_range> rounds;
    bool operator==(const send_request &) const = default;
  };
  /// How fault columns are built. Merged columns combine their priors as
  /// p = p1 + p2 - 2 p1 p2.
  struct column_options {
    /// Merge same-zone columns with identical full rows, including rows
    /// omitted outside the window, and identical observable effect.
    bool merge_identical = false;
    parallel_resolution conflicting_parallel = parallel_resolution::reject;
    bool operator==(const column_options &) const = default;
  };

  std::int64_t period = 0;
  std::int64_t offset = 0;
  std::optional<round_range> commit;
  std::vector<round_range> buffers;
  input_port input;
  output_port output;
  std::vector<send_request> sends;
  column_options columns;
  bool operator==(const window_request &) const = default;
};

/// One layer of an ingest node: its name, which is also the node's output
/// port for the layer, and the window request it publishes views of.
struct ingest_window {
  std::string layer;
  window_request request;
  bool operator==(const ingest_window &) const = default;
};

/// Check @p request against the window construction rules: a period of at
/// least 1, one non-empty commit zone, disjoint contiguous zones, a
/// disposition on each port, a valid prior, imports at the output wire, and
/// well-formed sends.
/// @throws std::invalid_argument naming the first rule the request breaks.
void check_window_request(const window_request &request);

} // namespace cudaq::qec::experimental
