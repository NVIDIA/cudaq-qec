# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Offline orchestration checks; real UDP replay remains an integration test."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_decoding_server as replay


class ReplayDecodingServerTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qec-replay-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def options(self):
        for name in ("server", "playback", "config", "schedule"):
            (self.root / name).write_text("fixture\n", encoding="utf-8")
        for name in ("server", "playback"):
            (self.root / name).chmod(0o700)
        (self.root / "expected").write_text("expected\n1\n", encoding="utf-8")
        return replay.arguments([
            item for name in ("server", "playback", "config", "schedule",
                              "expected", "output")
            for item in (f"--{name}", str(self.root / name))
        ])

    def test_ready_requires_exact_unique_ring_endpoints(self):
        self.assertIsNone(replay.parse_ready("building\n", {0}))
        self.assertEqual(
            {
                2: 1234,
                7: 4321
            },
            replay.parse_ready(
                "QEC_DECODING_SERVER_READY port=1234 ring2=1234 ring7=4321\n",
                {2, 7}))
        for record in ("ring0=0", "ring0=65536", "ring0=1 ring0=2", "ring1=1",
                       "ring0=1 ring1=1"):
            with self.subTest(record=record), self.assertRaises(ValueError):
                replay.parse_ready(f"QEC_DECODING_SERVER_READY port=1 {record}",
                                   {0})

    def test_server_counts_are_not_inferred_from_client_success(self):
        log = ("QEC_DECODING_SERVER_RING decoder=2 dispatched=3\n"
               "QEC_DECODING_SERVER_RING decoder=7 dispatched=5\n"
               "QEC_DECODING_SERVER_DISPATCHED count=8\n")
        self.assertEqual(
            8,
            replay.validate_server_counts(log, {
                2: 3,
                7: 5
            })["requests"])
        for bad in ("", log.replace("count=8", "count=9"), log + log,
                    log.replace("decoder=7", "decoder=8")):
            with self.subTest(log=bad), self.assertRaises(ValueError):
                replay.validate_server_counts(bad, {2: 3, 7: 5})

    def test_owned_flags_cannot_be_overridden(self):
        args = self.options()
        for option in ("--config=/other", "--port=1234", "--transport=inproc",
                       "position"):
            args.server_arg = [option]
            with self.subTest(option=option), self.assertRaises(ValueError):
                replay.replay(args)
        self.assertFalse(args.output.exists())

    def test_empty_oracle_never_starts_server(self):
        args = self.options()
        args.expected.write_text("expected\n", encoding="utf-8")
        with mock.patch.object(replay.subprocess, "Popen") as launch:
            with self.assertRaises(ValueError):
                replay.replay(args)
            launch.assert_not_called()

    def test_existing_output_is_not_overwritten(self):
        args = self.options()
        args.output.mkdir()
        retained = args.output / "retained"
        retained.write_text("user data", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            replay.replay(args)
        self.assertEqual("user data", retained.read_text(encoding="utf-8"))

    def test_completed_run_records_sources_and_uses_server_defaults(self):
        args = self.options()
        source = self.root / "rounds.txt"
        source.write_text("1\n", encoding="utf-8")
        args.source = [replay.source_argument(f"7:{source}")]
        process = mock.Mock()
        process.poll.return_value = None

        def client(command, **_):
            (args.output / "events.csv").write_text("event_index\n0\n",
                                                    encoding="utf-8")
            self.assertIn(f"--source=7:{source}", command)
            self.assertIn("--udp-endpoint=0:127.0.0.1:1234", command)

        def stopped(**_):
            (args.output / "server.log").write_text(
                "QEC_DECODING_SERVER_RING decoder=0 dispatched=3\n"
                "QEC_DECODING_SERVER_DISPATCHED count=3\n",
                encoding="utf-8")
            return 0

        process.wait.side_effect = stopped
        with mock.patch.object(replay.subprocess, "Popen", return_value=process) as launch, \
             mock.patch.object(replay, "wait_ready", return_value={0: 1234}), \
             mock.patch.object(replay.subprocess, "run", side_effect=client), \
             mock.patch.object(replay, "validate_records", return_value={
                 "requests_per_decoder": {"0": 3}, "requests": 3, "shots": 1
             }):
            report = replay.replay(args)
        command = launch.call_args.args[0]
        self.assertFalse(any(arg.startswith("--transport=") for arg in command))
        self.assertIn("--timeout=700", command)
        self.assertEqual("passed", report["status"])
        self.assertEqual(replay.fingerprint(source), report["sources"]["7"])
        process.terminate.assert_called_once()

    def test_invalid_and_duplicate_sources_are_rejected(self):
        for value in ("missing-separator", "-1:file", "0:"):
            with self.subTest(value=value), self.assertRaises(
                    argparse.ArgumentTypeError):
                replay.source_argument(value)
        args = self.options()
        args.source = [(1, args.schedule), (1, args.schedule)]
        with self.assertRaisesRegex(ValueError, "unique"):
            replay.replay(args)
        args.source = [(1, self.root / "absent")]
        with self.assertRaisesRegex(ValueError, "regular file"):
            replay.replay(args)

    def test_startup_failure_retains_evidence_and_stops_owned_child(self):
        args = self.options()
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(replay.subprocess, "Popen", return_value=process), \
             mock.patch.object(replay, "wait_ready", side_effect=TimeoutError("not ready")):
            with self.assertRaises(TimeoutError):
                replay.replay(args)
        process.terminate.assert_called_once()
        report = json.loads((args.output / "summary.json").read_text())
        self.assertEqual("failed", report["status"])
        self.assertEqual("not ready", report["error"])
        self.assertTrue((args.output / "server.log").is_file())

    def test_client_failure_stops_server_and_retains_logs(self):
        args = self.options()
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        with mock.patch.object(replay.subprocess, "Popen", return_value=process), \
             mock.patch.object(replay, "wait_ready", return_value={0: 1234}), \
             mock.patch.object(replay.subprocess, "run", side_effect=subprocess.TimeoutExpired("client", 1)):
            with self.assertRaises(subprocess.TimeoutExpired):
                replay.replay(args)
        process.terminate.assert_called_once()
        self.assertEqual(
            "failed",
            json.loads((args.output / "summary.json").read_text())["status"])
        self.assertTrue((args.output / "client.log").is_file())

    def test_forced_shutdown_is_not_a_pass(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("server", 10), -9]
        with self.assertRaises(TimeoutError):
            replay.stop_server(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()

    def test_readiness_rejects_dead_server(self):
        log = self.root / "server.log"
        log.write_text("", encoding="utf-8")
        process = mock.Mock()
        process.poll.return_value = 1
        with self.assertRaisesRegex(RuntimeError, "before READY"):
            replay.wait_ready(process, log, {0}, 1)

    def test_nonpositive_or_nonfinite_timeout_is_rejected(self):
        for value in ("0", "-1", "inf", "nan"):
            with self.subTest(value=value), self.assertRaises(
                    argparse.ArgumentTypeError):
                replay.positive_seconds(value)


if __name__ == "__main__":
    unittest.main()
