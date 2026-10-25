// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compile against the installed SDK, not against the plugin's private sources.
// Listing/validating registration must not construct an engine or require a
// GPU.
#include "cudaq/qec/decoder.h"
#include "cudaq/qec/decoder_config_schema.h"
#include "cudaq/qec/decoding_task_graph.h"
#include <algorithm>
#include <iostream>
#include <string>

namespace {
bool check_decoder(const std::string &name, bool native_streaming) {
  const auto names = cudaq::qec::decoder::get_registered();
  if (std::find(names.begin(), names.end(), name) == names.end() ||
      !cudaq::qec::decoding::config::find_decoder_schema(name)) {
    std::cerr << "Missing installed decoder or parameter schema: " << name
              << '\n';
    return false;
  }
  if (native_streaming &&
      (!cudaq::qec::decoder_uses_native_streaming_source(
           name, cudaq::qec::decoder_model_source::dem_chunks) ||
       cudaq::qec::decoder_needs_model_matrices(
           name, cudaq::qec::decoder_model_source::dem_chunks))) {
    std::cerr << "Missing installed native DEM-chunk streaming declaration: "
              << name << '\n';
    return false;
  }
  std::cout << "Registered installed "
            << (native_streaming ? "native streaming decoder: " : "decoder: ")
            << name << '\n';
  return true;
}

int usage() {
  std::cerr << "Usage: plugin-registration-probe [NODE_KIND ...] "
               "[--node-kind NAME] [--decoder NAME] "
               "[--native-streaming-decoder NAME]\n";
  return 2;
}
} // namespace

int main(int argc, char **argv) {
  if (argc < 2)
    return usage();
  for (int argument = 1; argument < argc; ++argument) {
    std::string kind = argv[argument];
    if (kind == "--decoder" || kind == "--native-streaming-decoder" ||
        kind == "--node-kind") {
      const auto option = kind;
      if (++argument == argc || !argv[argument][0] || argv[argument][0] == '-')
        return usage();
      kind = argv[argument];
      if (option != "--node-kind") {
        if (!check_decoder(kind, option == "--native-streaming-decoder"))
          return 1;
        continue;
      }
    } else if (kind.empty() || kind.front() == '-')
      return usage();
    if (!cudaq::qec::find_task_graph_node_kind(kind) ||
        !cudaq::qec::task_graph_node_params_schema(kind)) {
      std::cerr << "Missing installed task-graph kind or parameter schema: "
                << kind << '\n';
      return 1;
    }
    std::cout << "Registered installed task-graph kind: " << kind << '\n';
  }
  return 0;
}
