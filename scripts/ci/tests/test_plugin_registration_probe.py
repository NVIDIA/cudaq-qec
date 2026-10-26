# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile the actual no-GPU probe against small public-interface fixtures."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

PROBE = Path(__file__).resolve().parents[1] / "plugin_registration_probe.cpp"


@unittest.skipUnless(shutil.which("c++"),
                     "Probe contracts require a C++ compiler")
class PluginRegistrationProbeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="qec-probe-contract-")
        cls.addClassCleanup(cls.temporary.cleanup)
        root = Path(cls.temporary.name)
        headers = root / "cudaq/qec"
        headers.mkdir(parents=True)
        (headers / "decoder.h").write_text('''#pragma once
#include <string>
#include <vector>
namespace cudaq::qec {
enum class decoder_model_source { dem_chunks };
class decoder {
public:
  static std::vector<std::string> get_registered() {
    return {"fixture", "fixture_native", "missing_schema"};
  }
};
inline bool decoder_uses_native_streaming_source(
    const std::string &name, decoder_model_source) {
  return name == "fixture_native";
}
inline bool decoder_needs_model_matrices(
    const std::string &name, decoder_model_source) {
  return name != "fixture_native";
}
}
''')
        (headers / "decoder_config_schema.h").write_text('''#pragma once
#include <string>
namespace cudaq::qec::decoding::config {
inline const void *find_decoder_schema(const std::string &name) {
  static const int schema = 0;
  return name == "fixture" || name == "fixture_native" ? &schema : nullptr;
}
}
''')
        (headers / "decoding_task_graph.h").write_text('''#pragma once
#include <string>
namespace cudaq::qec {
inline const void *find_task_graph_node_kind(const std::string &name) {
  static const int kind = 0;
  return name == "fixture_node" ? &kind : nullptr;
}
inline const void *task_graph_node_params_schema(const std::string &name) {
  return find_task_graph_node_kind(name);
}
}
''')
        cls.binary = root / "probe"
        subprocess.run([
            "c++", "-std=c++20", f"-I{root}",
            str(PROBE), "-o",
            str(cls.binary)
        ],
                       check=True,
                       capture_output=True,
                       text=True)

    def probe(self, *arguments):
        return subprocess.run([str(self.binary), *arguments],
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                              check=False,
                              capture_output=True,
                              text=True)

    def test_legacy_positional_and_explicit_node_kind_protocols(self):
        for arguments in (("fixture_node",), ("--node-kind", "fixture_node")):
            with self.subTest(arguments=arguments):
                self.assertEqual(0, self.probe(*arguments).returncode)
        self.assertEqual(1, self.probe("missing_node").returncode)

    def test_ordinary_decoder_requires_both_registry_and_schema(self):
        self.assertEqual(0, self.probe("--decoder", "fixture").returncode)
        for name in ("missing_decoder", "missing_schema"):
            with self.subTest(name=name):
                self.assertEqual(1, self.probe("--decoder", name).returncode)

    def test_native_streaming_declaration_is_checked_without_construction(self):
        self.assertEqual(
            0,
            self.probe("--native-streaming-decoder",
                       "fixture_native").returncode)
        self.assertEqual(
            1,
            self.probe("--native-streaming-decoder", "fixture").returncode)

    def test_checks_can_be_combined(self):
        self.assertEqual(
            0,
            self.probe("fixture_node", "--decoder", "fixture",
                       "--native-streaming-decoder",
                       "fixture_native").returncode)

    def test_invalid_command_lines_are_usage_errors(self):
        for arguments in ((), ("--decoder",), ("--unknown", "fixture"),
                          ("--decoder", "--node-kind")):
            with self.subTest(arguments=arguments):
                self.assertEqual(2, self.probe(*arguments).returncode)


if __name__ == "__main__":
    unittest.main()
