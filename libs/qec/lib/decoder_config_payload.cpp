/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudaq/qec/decoder_config_payload.h"

#include <utility>

namespace cudaq::qec {

namespace {
decoder_config_payload_publisher g_publisher;
} // namespace

void set_decoder_config_payload_publisher(
    decoder_config_payload_publisher publisher) {
  g_publisher = std::move(publisher);
}

void publish_decoder_config_payload(const std::string &yaml) {
  if (g_publisher)
    g_publisher(yaml);
}

} // namespace cudaq::qec
