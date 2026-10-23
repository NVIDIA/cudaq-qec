/* -*- C++ -*-
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
