// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// Compile against the installed SDK, not against the plugin's private sources.
// Listing/validating registration must not construct an engine or require a
// GPU.
#include "cudaq/qec/decoding_task_graph.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: plugin-registration-probe NODE_KIND [NODE_KIND ...]\n";
    return 2;
  }
  for (int argument = 1; argument < argc; ++argument) {
    const std::string kind = argv[argument];
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
