/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// The `ingest` node kind: the only raw-fed node of a windowed graph. It forms
// detectors from raw rounds and publishes a view of each layer's windows, one
// output port per entry of its `windows` map, and with `read_logical` the
// raw logical value on `logical`. The kind is registered so configurations
// that use it parse and validate; the node itself is not available yet.

#include "decoding_task_node_kind_registry.h"
#include <memory>
#include <stdexcept>
#include <string>

namespace cudaq::qec {
namespace {

/// The kind's own parameter; its windows are a node field, not a parameter.
decoding::config::decoder_schema ingest_params() {
  using decoding::config::param_kind;
  decoding::config::decoder_schema schema;
  schema.params.push_back({"read_logical", param_kind::boolean});
  return schema;
}

// It reads its template from the DEM and takes windows; it wraps no decoder.
const bool registered = [] {
  task_graph_node_kind_spec spec;
  spec.name = "ingest";
  spec.requires_decoder = false;
  spec.requires_dem = true;
  spec.takes_windows = true;
  register_node_kind(
      std::move(spec), ingest_params(),
      [](node_build_context &ctx) -> std::unique_ptr<task_graph_node> {
        throw std::invalid_argument(
            "task graph node '" + ctx.id +
            "': the 'ingest' kind is not available yet");
      });
  return true;
}();

} // namespace
} // namespace cudaq::qec
