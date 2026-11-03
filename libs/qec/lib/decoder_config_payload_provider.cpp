/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Bridges the decoder-config payload publisher to CUDA-Q's ExtraPayloadProvider
// registry. This translation unit is the reason cudaq-qec links cudaq-common;
// keeping it here (rather than in the realtime-decoding library) lets the
// decoders and realtime libraries stay free of any cudaq-common dependency.
//
// The publisher is installed at library-load time, so the remote decoder-config
// upload is enabled exactly when cudaq-qec (and thus the full CUDA-Q runtime)
// is loaded. Minimal deployments that do not load cudaq-qec simply skip the
// upload.

#include "common/ExtraPayloadProvider.h"
#include "cudaq/qec/decoder_config_payload.h"

#include <memory>
#include <string>
#include <utility>

namespace {

/// @brief Provides the decoder-config YAML as extra payload for job requests.
class decoder_provider : public cudaq::ExtraPayloadProvider {
  // Pre-serialized YAML so it can be reused across requests without re-parsing.
  std::string decoder_config_yaml_;

public:
  explicit decoder_provider(std::string yaml)
      : decoder_config_yaml_(std::move(yaml)) {}
  ~decoder_provider() override = default;
  std::string name() const override { return "decoder"; }
  std::string getPayloadType() const override { return "gpu_decoder_config"; }
  std::string getExtraPayload(const cudaq::RuntimeTarget &) override {
    return decoder_config_yaml_;
  }
};

/// @brief Installs the payload publisher into the decoders library at load
/// time.
const struct payload_publisher_installer {
  payload_publisher_installer() {
    cudaq::qec::set_decoder_config_payload_publisher(
        [](const std::string &yaml) {
          cudaq::registerExtraPayloadProvider(
              std::make_unique<decoder_provider>(yaml));
        });
  }
} g_payload_publisher_installer;

} // namespace
