/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "SessionRegistry.h"
#include "../realtime_decoding.h"
#include "cudaq/qec/logger.h"
#include "cudaq/qec/realtime/decoding_config.h"
#include <filesystem>

#include <fstream>
#include <iterator>
#include <stdexcept>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::multi_decoder_config;

// ---------------------------------------------------------------------------
// SessionRegistry
// ---------------------------------------------------------------------------

void SessionRegistry::load_from_config(const std::string &yaml_path) {
  std::ifstream f(yaml_path);
  if (!f.is_open())
    throw std::runtime_error("Cannot open config file: " + yaml_path);

  std::string yaml_str((std::istreambuf_iterator<char>(f)),
                       std::istreambuf_iterator<char>());
  // A model path in the document is relative to the document, not to wherever
  // the server happened to be started from.
  auto base_dir = std::filesystem::absolute(yaml_path).parent_path();
  load_from_config(multi_decoder_config::from_yaml_str(yaml_str), yaml_path,
                   base_dir);
}

namespace {

// Build the table once for the whole document: rows are server-level, not
// per-decoder.
//
// id 0 is always a pass-through BULK row unless the document names it.
// Declaring a section is additive, not a replacement: id 0 is what a producer
// sends unless it names a row, so a document that adds only a `final` row would
// otherwise turn every ordinary enqueue into BAD_REQUEST.
SyndromeMappingTable make_mapping_table(const multi_decoder_config &config) {
  // from_yaml_str() already validated, but load_from_config() is also reached
  // from configure_decoders() with a programmatically built config, which has
  // never seen the parser.  Without this an unknown shape reads as BULK, a
  // duplicate id resolves arbitrarily, and a populated vps is ignored -- all
  // silently.
  config.validate_syndrome_mapping();

  SyndromeMappingTable table = default_syndrome_mapping_table();
  for (const auto &row : config.syndrome_mapping) {
    syndrome_mapping_row out;
    out.type = row.type == "final" ? round_type::FINAL : round_type::BULK;
    for (const auto &vp : row.vps)
      out.vps.emplace_back(vp.begin(), vp.end());
    table.insert_or_assign(row.id, std::move(out));
  }
  return table;
}

} // namespace

void SessionRegistry::load_from_config(const multi_decoder_config &config,
                                       const std::string &source_name,
                                       const std::filesystem::path &base_dir) {
  const auto mapping_table = make_mapping_table(config);
  for (const auto &dc : config.decoders) {
    if (dc.id < 0)
      throw std::runtime_error("Negative decoder id " + std::to_string(dc.id) +
                               " in " + source_name);
    const uint64_t id = static_cast<uint64_t>(dc.id);
    if (sessions_.count(id))
      throw std::runtime_error("Duplicate decoder id " + std::to_string(dc.id) +
                               " in " + source_name);

    // Record each decoder's dispatch shape.  Mixed shapes are allowed: the
    // decoding_server process binds a consumer (host dispatcher or
    // device-graph scheduler) per decoder; only the single-transceiver
    // DecodingServer paths require uniformity (see required_dispatch()).
    if (sessions_.empty())
      dispatch_ = dc.dispatch;
    else if (dc.dispatch != dispatch_)
      mixed_ = true;
    dispatch_by_id_[id] = dc.dispatch;

    CUDA_QEC_INFO("SessionRegistry: creating decoder id={} type={}", dc.id,
                  dc.describe_decoder_type());

    // host sessions are served inline by the CQR HOST_CALL plugin on the
    // dispatcher thread; the decoding_server process binds device_graph
    // sessions to their ring consumers via dispatch_for(). Only a
    // device_graph entry should ask create() to capture -- a decoder that
    // claims graph dispatch unconditionally (e.g. nv-qldpc-decoder) can still
    // lack the device-only settings capture needs (relay-BP, an O matrix, no
    // OSD) when actually configured for HOST dispatch.
    const bool want_device_graph = dc.dispatch == DecoderDispatch::device_graph;
    if (dc.task_graph.has_value()) {
      auto graph = cudaq::qec::decoding::host::build_task_graph(dc, base_dir);
      // A device_graph ring fires a captured cudaGraphExec_t, which only a
      // single-node graph whose one node supports_graph_dispatch() can
      // supply (see decoding_task_graph::capture_graph()) -- never a
      // combine()-produced graph, and never a graph whose inner decoder is
      // CPU-only.  Checked here, where the config still names the decoder,
      // rather than later as a missing capture inside
      // DecodingSession::create().
      if (want_device_graph && !graph->supports_graph_dispatch())
        throw std::runtime_error(
            "Decoder " + std::to_string(dc.id) + " in " + source_name +
            " combines a task_graph with dispatch: device_graph, but its "
            "node does not support graph dispatch (a multi-node/combine() "
            "graph, or a CPU-only inner decoder); such task graphs are "
            "supported with dispatch: host only");
      // build_task_graph() never calls set_decoder_id() the way
      // create_realtime_decoder() does for a plain decoder. Set it here,
      // before create() captures: the generic get_corrections/reset_decoder
      // DEVICE_CALL handlers key g_decoder_state_table[] off the wire
      // request's decoder_id (this config's id) -- an unset (default 0)
      // inner decoder would register under the wrong slot for any id != 0.
      if (auto *dec = graph->inner_decoder())
        dec->set_decoder_id(static_cast<uint32_t>(id));
      sessions_.emplace(id, DecodingSession::create(std::move(graph),
                                                    want_device_graph,
                                                    mapping_table));
    } else {
      auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
          dc, cudaq::qec::decoding::host::resolve_decoder_init(dc, base_dir));
      if (!want_device_graph && decoder->supports_graph_dispatch())
        CUDA_QEC_INFO("SessionRegistry: decoder id={} type={} supports graph "
                      "dispatch but is configured for host dispatch; serving "
                      "it inline on the dispatcher thread without capturing "
                      "its decode graph",
                      dc.id, dc.describe_decoder_type());
      sessions_.emplace(id, DecodingSession::create(std::move(decoder),
                                                    want_device_graph,
                                                    mapping_table));
    }
  }

  CUDA_QEC_INFO("SessionRegistry: loaded {} decoder session(s)",
                sessions_.size());
}

DecodingSession &SessionRegistry::get(uint64_t decoder_id) {
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

const DecodingSession &SessionRegistry::get(uint64_t decoder_id) const {
  auto it = sessions_.find(decoder_id);
  if (it == sessions_.end())
    throw std::out_of_range("Unknown decoder_id: " +
                            std::to_string(decoder_id));
  return *it->second;
}

} // namespace cudaq::qec::decoding_server
