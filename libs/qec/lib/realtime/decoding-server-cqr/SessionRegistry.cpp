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

void SessionRegistry::load_from_config(const multi_decoder_config &config,
                                       const std::string &source_name,
                                       const std::filesystem::path &base_dir) {
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
                  dc.type);

    auto decoder = cudaq::qec::decoding::host::create_realtime_decoder(
        dc, cudaq::qec::decoding::host::resolve_decoder_init(dc, base_dir));
    // Host sessions are served inline by the CQR HOST_CALL plugin on the
    // dispatcher thread and never fire the decoder's device graph, so only
    // device_graph sessions capture graph resources here; the
    // decoding_server process binds those to their ring consumers via
    // dispatch_for().
    const bool capture_graph = dc.dispatch == DecoderDispatch::device_graph;
    if (!capture_graph && decoder->supports_graph_dispatch())
      CUDA_QEC_INFO("SessionRegistry: decoder id={} type={} supports graph "
                    "dispatch but is configured for host dispatch; serving "
                    "it inline on the dispatcher thread without capturing "
                    "its decode graph",
                    dc.id, dc.type);
    sessions_.emplace(
        id, DecodingSession::create(std::move(decoder), capture_graph));
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
