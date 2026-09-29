/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <functional>
#include <string>

namespace cudaq::qec {

/// @brief Publisher invoked with the serialized decoder-config YAML so a
/// CUDA-Q integration layer can upload it to remote targets.
using decoder_config_payload_publisher =
    std::function<void(const std::string &yaml)>;

/// @brief Install the decoder-config payload publisher.
///
/// This is the rendezvous point that lets the CUDA-Q ExtraPayloadProvider
/// integration live in the cudaq-coupled `cudaq-qec` library (which installs
/// the publisher at load time) while the realtime-decoding library merely
/// invokes it. The hook itself carries no cudaq dependency, so both the
/// realtime library and this (decoders) library stay free of cudaq-common.
__attribute__((visibility("default"))) void
set_decoder_config_payload_publisher(
    decoder_config_payload_publisher publisher);

/// @brief Invoke the installed publisher with the given YAML, if any.
///
/// No-op when no publisher has been installed (e.g. minimal deployments that
/// do not load `cudaq-qec` / the CUDA-Q runtime).
__attribute__((visibility("default"))) void
publish_decoder_config_payload(const std::string &yaml);

} // namespace cudaq::qec
