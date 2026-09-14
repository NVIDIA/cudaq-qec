/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdlib>
#include <fstream>
#include <string>

extern "C" int qec_cc_test_plugin_symbol() { return 1; }

namespace {
struct unload_marker {
  ~unload_marker() {
    if (const char *path = std::getenv("QEC_CC_PLUGIN_UNLOAD_MARKER")) {
      std::ofstream out(path);
      out << "closed\n";
    }
  }
};
unload_marker g_unload_marker;
} // namespace
