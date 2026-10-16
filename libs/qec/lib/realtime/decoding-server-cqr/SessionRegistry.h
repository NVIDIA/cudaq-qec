/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <filesystem>

#include "DecodingSession.h"
#include "cudaq/qec/realtime/decoding_config.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace cudaq::qec::decoding_server {

using cudaq::qec::decoding::config::DecoderDispatch;

/// Owns all DecodingSession instances, keyed by uint64_t decoder_id.
///
/// Populated eagerly at startup from the YAML config.  The map is read-only
/// after load_from_config() returns, so no locking is required at runtime.
class SessionRegistry {
public:
  /// Parse \p yaml_path and construct one DecodingSession per decoder entry.
  /// Decoder entries may mix dispatch shapes (host and device_graph).
  /// @throws std::runtime_error on duplicate id, missing required fields, or
  /// decoder init failure.
  void load_from_config(const std::string &yaml_path);

  /// Same, from an already-parsed config (the in-process application path,
  /// where the config was handed to configure_decoders rather than a file).
  /// \p source_name is used in error messages only.
  /// \p base_dir is the directory a relative model path (`stim_dem_path`)
  /// resolves against; the file overload above passes the config's own
  /// directory. Defaults to the working directory for a config with no
  /// originating file.
  void load_from_config(
      const cudaq::qec::decoding::config::multi_decoder_config &config,
      const std::string &source_name,
      const std::filesystem::path &base_dir = std::filesystem::current_path());

  DecodingSession &get(uint64_t decoder_id);
  const DecodingSession &get(uint64_t decoder_id) const;

  /// Hot-path lookup: nullptr when the id is unknown (no exception).
  /// The map is read-only after load_from_config(), so this is safe for
  /// concurrent dispatcher threads.
  DecodingSession *find(uint64_t decoder_id) noexcept {
    auto it = sessions_.find(decoder_id);
    return it == sessions_.end() ? nullptr : it->second.get();
  }

  /// Dispatch shape shared by ALL sessions.  Valid after load_from_config();
  /// throws when the config mixes shapes (mixed configs are composed by the
  /// decoding_server process, which binds a consumer per decoder -- the
  /// single-transceiver DecodingServer paths require a uniform shape).
  DecoderDispatch required_dispatch() const {
    if (mixed_)
      throw std::runtime_error(
          "config mixes host and device_graph dispatch; a single-transceiver "
          "DecodingServer cannot serve it (the decoding_server process "
          "composes per-decoder consumers instead)");
    return dispatch_;
  }

  /// Dispatch shape of one decoder (valid after load_from_config()).
  DecoderDispatch dispatch_for(uint64_t decoder_id) const {
    return dispatch_by_id_.at(decoder_id);
  }

  bool mixed_dispatch() const { return mixed_; }

  const std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> &
  sessions() const {
    return sessions_;
  }

private:
  std::unordered_map<uint64_t, std::unique_ptr<DecodingSession>> sessions_;
  std::unordered_map<uint64_t, DecoderDispatch> dispatch_by_id_;
  DecoderDispatch dispatch_{DecoderDispatch::host};
  bool mixed_ = false;
};

} // namespace cudaq::qec::decoding_server
