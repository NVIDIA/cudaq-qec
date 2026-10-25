# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the final-image smoke; real replay runs separately."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import installed_plugin_smoke as smoke


class InstalledPluginSmokeTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="qec-installed-smoke-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "install"
        self.module = self.prefix / "lib/task-graph-plugins/fixture.so"
        self.module.parent.mkdir(parents=True)
        self.module.write_bytes(b"ELF fixture")
        self.manifest = self.prefix / "lib/external-plugins/sha/source-manifest.json"
        self.manifest.parent.mkdir(parents=True)
        self.document = {
            "schema_version":
                1,
            "delivery":
                "native",
            "modules": [{
                "id": "fixture",
                "kind": "task_graph",
                "path": "task-graph-plugins/fixture.so"
            }]
        }
        self.write_manifest()

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.document), encoding="utf-8")

    def options(self):
        probe = self.prefix / "bin/plugin-registration-probe"
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_bytes(b"probe fixture")
        return smoke.arguments([
            f"--prefix={self.prefix}", f"--output={self.root / 'result'}",
            "--require-provider=fixture", "--require-node-kind=fixture_node",
            f"--registration-probe={probe}"
        ])

    def test_installed_paths_are_resolved_from_manifest(self):
        result = smoke.installed_modules(self.prefix, [self.manifest],
                                         ["fixture"])
        self.assertEqual(1, len(result))
        self.assertEqual(self.module, result[0]["file"])
        self.assertEqual("fixture", result[0]["id"])

    def test_manifest_paths_cannot_escape_plugin_directory(self):
        for path in ("../fixture.so", "/tmp/fixture.so",
                     "task-graph-plugins/../fixture.so",
                     "task-graph-plugins//fixture.so",
                     "decoder-plugins/fixture.so"):
            with self.subTest(path=path):
                self.document["modules"][0]["path"] = path
                self.write_manifest()
                with self.assertRaises(ValueError):
                    smoke.installed_modules(self.prefix, [self.manifest], [])

    def test_required_provider_cannot_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, "required installed providers"):
            smoke.installed_modules(self.prefix, [], ["fixture"])
        self.module.unlink()
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            smoke.installed_modules(self.prefix, [self.manifest], [])

    def test_symlink_module_cannot_pass_installed_check(self):
        self.module.unlink()
        outside = self.root / "outside.so"
        outside.write_bytes(b"outside")
        self.module.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "not a regular file"):
            smoke.installed_modules(self.prefix, [self.manifest], [])

    def test_task_graph_archive_requires_real_kind_expectation(self):
        args = self.options()
        args.require_node_kind = []
        with self.assertRaisesRegex(ValueError, "explicit expected node kinds"):
            smoke.smoke(args)
        self.assertFalse(args.output.exists())

    def test_final_rpath_and_dependencies_are_checked(self):
        module = smoke.installed_modules(self.prefix, [self.manifest], [])[0]
        for rpath, dependency, expected in (("$ORIGIN/..\n",
                                             "libok.so => /lib/libok.so\n",
                                             True), ("/tmp/build/lib\n", "",
                                                     False), ("$ORIGIN/..:\n",
                                                              "", False),
                                            ("$ORIGIN/..\n",
                                             "libmissing.so => not found\n",
                                             False)):
            commands = [{
                "returncode": 0,
                "stdout": rpath,
                "stderr": ""
            }, {
                "returncode": 0,
                "stdout": dependency,
                "stderr": ""
            }]
            with self.subTest(rpath=rpath,
                              dependency=dependency), mock.patch.object(
                                  smoke, "run_command", side_effect=commands):
                if expected:
                    self.assertEqual("fixture",
                                     smoke.check_module(module)["id"])
                else:
                    with self.assertRaises(ValueError):
                        smoke.check_module(module)

    def test_success_checks_registration_negative_case_and_real_replay(self):
        args = self.options()
        commands = [{"returncode": 0}, {"returncode": 1}]
        replay = {"status": "passed", "result": {"shots": 8, "requests": 24}}
        with mock.patch.object(smoke, "check_module", return_value={"id": "fixture"}), \
             mock.patch.object(smoke, "run_command", side_effect=commands) as run, \
             mock.patch.object(smoke.playback, "replay", return_value=replay) as launch:
            report = smoke.smoke(args)
        self.assertEqual("passed", report["status"])
        self.assertEqual(smoke.MISSING_NODE_KIND,
                         run.call_args_list[1].args[0][-1])
        self.assertEqual(args.prefix / "bin/decoding_server",
                         launch.call_args.args[0].server)
        self.assertEqual("udp", launch.call_args.args[0].transport)
        self.assertEqual(
            "passed",
            json.loads((args.output / "summary.json").read_text())["status"])

    def test_failed_registration_preserves_report_and_prevents_replay(self):
        args = self.options()
        with mock.patch.object(smoke, "check_module", return_value={"id": "fixture"}), \
             mock.patch.object(smoke, "run_command", return_value={"returncode": 1}), \
             mock.patch.object(smoke.playback, "replay") as launch:
            with self.assertRaisesRegex(ValueError,
                                        "registration/schema check"):
                smoke.smoke(args)
            launch.assert_not_called()
        self.assertEqual(
            "failed",
            json.loads((args.output / "summary.json").read_text())["status"])

    def ordinary_options(self):
        self.document["modules"][0].update(kind="decoder",
                                           path="decoder-plugins/fixture.so")
        self.module = self.prefix / "lib/decoder-plugins/fixture.so"
        self.module.parent.mkdir(parents=True)
        self.module.write_bytes(b"ordinary ELF fixture")
        self.write_manifest()
        args = self.options()
        args.require_node_kind = []
        args.require_decoder = ["fixture_decoder"]
        args.require_native_streaming_decoder = ["fixture_decoder"]
        return args

    def test_ordinary_native_decoder_checks_need_no_task_graph_kind(self):
        args = self.ordinary_options()
        commands = [{"returncode": 0}, {"returncode": 1}, {"returncode": 1}]
        replay = {"status": "passed", "result": {"shots": 8, "requests": 24}}
        with mock.patch.object(smoke, "check_module", return_value={"id": "fixture"}), \
             mock.patch.object(smoke, "run_command", side_effect=commands) as run, \
             mock.patch.object(smoke.playback, "replay", return_value=replay):
            report = smoke.smoke(args)
        self.assertEqual("passed", report["status"])
        self.assertEqual([
            args.registration_probe, "--decoder", "fixture_decoder",
            "--native-streaming-decoder", "fixture_decoder"
        ], run.call_args_list[0].args[0])
        self.assertEqual(
            [args.registration_probe, "--decoder", smoke.MISSING_DECODER],
            run.call_args_list[1].args[0])
        self.assertEqual([
            args.registration_probe, "--native-streaming-decoder",
            smoke.MISSING_DECODER
        ], run.call_args_list[2].args[0])
        self.assertNotIn("missing_kind", report)

    def test_ordinary_registration_requirements_need_a_probe(self):
        args = self.ordinary_options()
        args.registration_probe = None
        with self.assertRaisesRegex(ValueError, "installed registration probe"):
            smoke.smoke(args)
        self.assertFalse(args.output.exists())

    def test_native_streaming_negative_probe_must_reject_unknown_decoder(self):
        args = self.ordinary_options()
        commands = [{"returncode": 0}, {"returncode": 1}, {"returncode": 0}]
        with mock.patch.object(smoke, "check_module", return_value={"id": "fixture"}), \
             mock.patch.object(smoke, "run_command", side_effect=commands), \
             mock.patch.object(smoke.playback, "replay") as launch:
            with self.assertRaisesRegex(ValueError,
                                        "missing_native_streaming_decoder"):
                smoke.smoke(args)
            launch.assert_not_called()

    def test_existing_output_is_preserved(self):
        args = self.options()
        args.output.mkdir()
        marker = args.output / "marker"
        marker.write_text("retain", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            smoke.smoke(args)
        self.assertEqual("retain", marker.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
