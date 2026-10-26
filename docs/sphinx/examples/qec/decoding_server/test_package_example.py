# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launcher behavior tests; no registry credentials or GPU are required."""

import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_package_example as example


class PackageExampleTest(unittest.TestCase):

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "config.yaml"
        self.config.write_text("decoders: []\n")
        self.schedule = self.root / "schedule.txt"
        self.schedule.write_text("0 reset\n")
        self.output = self.root / "output.csv"
        self.calls = []
        self.ready = True
        self.status = "0"
        self.completed = "1"
        self.mismatch = "0"
        self.server_errors = 0
        self.client_confirms_routing = True
        self.final_readout_status = None

    def fake_docker(self, *args, check=True):
        self.calls.append(args)
        text = ""
        if args[0] == "logs":
            if self.ready:
                text = "QEC_DECODING_SERVER_READY port=54321 transport=udp ring0=54321\n"
            text += (
                "QEC_DECODING_SERVER_DECODER_STATS id=0 decodes=1 enqueues=1 "
                "corrections=1 resets=1 errors=%d\n"
                "QEC_DECODING_SERVER_DISPATCHED count=3\n" % self.server_errors)
        elif args[0] == "inspect":
            text = "false\n"
        return subprocess.CompletedProcess(args, 0, stdout=text, stderr="")

    def fake_client(self, command, **kwargs):
        self.calls.append(tuple(command))
        if "qec-client" in command:
            self.assertIn("QEC_DECODING_SERVER_PORT=54321", command)
            self.assertIn("QEC_DECODING_SERVER_TRANSPORT=udp", command)
            self.assertIn("CUDAQ_DEVICE_CALL_CHANNEL=host_dispatch", command)
            self.assertEqual(command[command.index("--entrypoint") + 1],
                             "qec-client")
            self.assertEqual(command[0:2], ["docker", "run"])
            if self.client_confirms_routing:
                kwargs["stdout"].write("CQR service dispatch count: 0\n")
            return subprocess.CompletedProcess(command, 0)
        writer = csv.DictWriter(kwargs["stdout"],
                                fieldnames=[
                                    "op", "status", "dispatched",
                                    "read_completed", "correction_bits",
                                    "correction_mismatch"
                                ])
        writer.writeheader()
        if self.final_readout_status is not None:
            writer.writerow(
                dict(op="enqueue_data",
                     status=self.final_readout_status,
                     dispatched="1"))
        writer.writerow(
            dict(op="get_corrections",
                 status=self.status,
                 dispatched="1",
                 read_completed=self.completed,
                 correction_bits="010",
                 correction_mismatch=self.mismatch))
        return subprocess.CompletedProcess(command, 0)

    def run_example(self, *options, client=False):
        argv = [
            "run_package_example.py", "--image", "test-image", "--config",
            str(self.config), "--work-dir",
            str(self.root), "--output",
            str(self.output), *options
        ]
        argv += ["--client", "qec-client"] if client else [
            "--schedule", str(self.schedule)
        ]
        with patch.object(sys, "argv", argv), patch.object(example, "docker", self.fake_docker), \
             patch.object(example.subprocess, "run", self.fake_client), \
             contextlib.redirect_stdout(io.StringIO()):
            example.main()

    def assert_cleanup(self):
        self.assertTrue(any(call[0] == "stop" for call in self.calls))
        self.assertTrue(any(call[0] == "rm" for call in self.calls))
        self.assertTrue(self.output.with_suffix(".csv.server.log").exists())

    def test_success_uses_discovered_endpoint_and_cleans_up(self):
        self.run_example("--expect-no-errors")
        playback = next(call for call in self.calls if call[0] == "docker")
        self.assertIn("--udp-endpoint=0:127.0.0.1:54321", playback)
        self.assert_cleanup()

    def test_not_ready_is_a_failure_and_still_cleans_up(self):
        self.status = "4"
        with self.assertRaisesRegex(RuntimeError, "failed playback"):
            self.run_example()
        self.assert_cleanup()

    def test_enqueue_data_stream_completion_succeeds(self):
        self.final_readout_status = "100"
        self.run_example()
        self.assert_cleanup()

    def test_enqueue_data_rpc_success_is_not_stream_completion(self):
        self.final_readout_status = "0"
        with self.assertRaisesRegex(RuntimeError, "failed playback"):
            self.run_example()
        self.assert_cleanup()

    def test_missing_correction_is_a_failure(self):
        self.completed = "0"
        with self.assertRaisesRegex(RuntimeError, "completed corrections"):
            self.run_example()
        self.assert_cleanup()

    def test_logical_mismatches_are_reported_but_not_transport_failures(self):
        self.mismatch = "1"
        self.run_example()
        self.assert_cleanup()

    def test_connectivity_check_rejects_logical_mismatch(self):
        self.mismatch = "1"
        with self.assertRaisesRegex(RuntimeError, "correction mismatch"):
            self.run_example("--expect-no-errors")
        self.assert_cleanup()

    def test_server_error_fails_even_when_client_succeeds(self):
        self.server_errors = 1
        with self.assertRaisesRegex(RuntimeError, "Server statistics"):
            self.run_example()

    def test_early_server_exit_saves_log_and_cleans_up(self):
        self.ready = False
        with self.assertRaisesRegex(RuntimeError, "before readiness"):
            self.run_example()
        self.assert_cleanup()

    def test_cudaq_client_receives_discovered_endpoint(self):
        self.run_example(client=True)
        self.assert_cleanup()

    def test_cudaq_client_must_confirm_external_routing(self):
        self.client_confirms_routing = False
        with self.assertRaisesRegex(RuntimeError, "external server routing"):
            self.run_example(client=True)
        self.assert_cleanup()


if __name__ == "__main__":
    unittest.main()
