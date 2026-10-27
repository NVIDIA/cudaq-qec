# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the shell loop guards and acceptance logic without network/hardware."""

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("hsb_fpga_decoding_server_test.sh")


def shell_function(name):
    """Read one production function through its unindented closing brace."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines(keepends=True)
    starts = [
        index for index, line in enumerate(lines)
        if re.match(r"^" + re.escape(name) + r"\(\)\s*\{", line)
    ]
    if len(starts) != 1:
        raise AssertionError(f"Expected one shell function named {name}")
    start = starts[0]
    for end in range(start + 1, len(lines)):
        if lines[end].rstrip("\n") == "}":
            return "".join(lines[start:end + 1])
    raise AssertionError(f"Missing closing brace for shell function {name}")


class HsbPlaybackLoopTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="hsb-loop-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.playback = self.root / "playback.log"
        self.server = self.root / "server.log"
        self.emulator = self.root / "emulator.log"
        self.playback.write_text(
            "  get_corrections frames: 2\n"
            "  ILA verification: PASS\n"
            "  RESULT: PASS\n"
            "  Playback frames: 30\n"
            "  Correction responses: 30\n"
            "  Completed measurements: 10\n"
            "  RPC status failures: 0\n",
            encoding="utf-8")
        self.server.write_text(
            "QEC_DECODING_SERVER_DECODER_STATS id=0 decodes=12 enqueues=30 "
            "corrections=12 resets=12 errors=0\n",
            encoding="utf-8")
        self.emulator.write_text(
            "  Windows sent: 36\n"
            "  Responses received: 36\n"
            "  Send errors: 0\n"
            "  Timeouts: 0\n"
            "*** EMULATOR: PASS\n",
            encoding="utf-8")

    def run_functions(self, command, transport="cpu_roce", mocks=""):
        functions = "\n".join(
            shell_function(name)
            for name in ("verify_loop_statistics", "wait_for_loop_exit",
                         "wait_for_pattern", "finish_loop_run"))
        setup = r'''
set -euo pipefail
_log() { echo "$*"; }
_info() { echo "$*"; }
_banner() { echo "$*"; }
_err() { echo "$*" >&2; }
# Never signal or wait on an external process; advance the clock without sleep.
kill() { return 1; }
wait() { return 0; }
sleep() { SECONDS=$((SECONDS + 1)); }
PLAYBACK_LOG=$1
SERVER_LOG=$2
EMULATOR_LOG=$3
TRANSPORT=$4
LOOP_SECONDS=1
MIN_DECODES=10
SERVER_PID=101
LOOP_PROCESS_EXIT_TIMEOUT_SECONDS=1
LOOP_LOG_FLUSH_TIMEOUT_SECONDS=1
'''
        return subprocess.run([
            "bash", "-c", functions + setup + mocks + "\n" + command,
            "loop-test",
            str(self.playback),
            str(self.server),
            str(self.emulator), transport
        ],
                              capture_output=True,
                              text=True,
                              check=False,
                              timeout=5)

    def assert_rejected(self, result, message):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(message, result.stdout + result.stderr)

    def test_cli_rejects_loop_errors_before_network_setup(self):
        sentinel = self.root / "network-called"
        tools = self.root / "tools"
        tools.mkdir()
        for name in ("sudo", "ip", "ibdev2netdev", "ping", "rdma", "ethtool",
                     "mlnx_qos"):
            tool = tools / name
            tool.write_text(
                '#!/bin/sh\nprintf "%s\\n" "$0" >> "$HSB_TEST_SENTINEL"\nexit 97\n',
                encoding="utf-8")
            tool.chmod(0o700)
        environment = dict(os.environ,
                           PATH=str(tools) + os.pathsep +
                           os.environ.get("PATH", ""),
                           HSB_TEST_SENTINEL=str(sentinel))
        cases = ((["--min-decodes",
                   "1"], "--min-decodes requires --loop-seconds"),
                 (["--emulate", "--loop-seconds",
                   "1"], "--loop-seconds requires --min-decodes"),
                 (["--loop-seconds", "1", "--min-decodes",
                   "1"], "supported only with --emulate"),
                 (["--emulate", "--loop-seconds", "0", "--min-decodes",
                   "1"], "--loop-seconds must be a positive integer"),
                 (["--emulate", "--loop-seconds", "bad", "--min-decodes",
                   "1"], "--loop-seconds must be a positive integer"),
                 (["--emulate", "--loop-seconds", "1", "--min-decodes",
                   "0"], "--min-decodes must be a positive integer"), ([
                       "--emulate", "--loop-seconds", "1", "--min-decodes", "1",
                       "--no-verify"
                   ], "requires ILA verification"), ([
                       "--emulate", "--loop-seconds", "1", "--min-decodes", "1",
                       "--timeout", "0"
                   ], "--timeout must be a positive integer"),
                 (["--loop-seconds"], "--loop-seconds requires a value"),
                 (["--min-decodes"], "--min-decodes requires a value"),
                 (["--loop-seconds",
                   "--emulate"], "--loop-seconds requires a value"))
        for arguments, message in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(SCRIPT), "--setup-network", *arguments],
                    env=environment,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5)
                self.assert_rejected(result, message)
                self.assertFalse(sentinel.exists(), "Network setup was reached")

    def test_default_mode_stays_real_fpga_and_valid_loop_can_skip_run(self):
        for arguments, mode in ((["--no-run"], "Mode: Real FPGA"), ([
                "--emulate", "--loop-seconds", "1", "--min-decodes", "10",
                "--no-run"
        ], "Mode: FPGA Emulation")):
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ["bash", str(SCRIPT), *arguments],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=5)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(mode, result.stdout)
                self.assertIn("Skipping test run", result.stdout)

    def test_cpu_acceptance_subtracts_finite_preflight_decodes(self):
        command = 'verify_loop_statistics "$SERVER_LOG" "$EMULATOR_LOG"'
        result = self.run_functions(command)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("10 loop measurements, 0 errors", result.stdout)
        original = self.server.read_text()
        for before, after, message in ((
                "decodes=12", "decodes=11",
                "loop_decodes=9"), ("decodes=12", "decodes=1",
                                    "below the finite ILA correction count"),
                                       ("errors=0", "errors=1", "errors=1")):
            with self.subTest(change=after):
                self.server.write_text(original.replace(before, after))
                self.assert_rejected(self.run_functions(command), message)

    def test_device_acceptance_uses_completed_rpc_measurements_not_host_stats(
            self):
        self.server.write_text("")
        command = 'verify_loop_statistics "$SERVER_LOG" "$EMULATOR_LOG"'
        result = self.run_functions(command, transport="gpu_roce")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("device-graph completion verified", result.stdout)
        self.playback.write_text(self.playback.read_text().replace(
            "Completed measurements: 10", "Completed measurements: 9"))
        self.assert_rejected(self.run_functions(command, transport="gpu_roce"),
                             "measurements=9 required=10")

    def test_missing_preflight_counters_and_rpc_failures_are_rejected(self):
        command = 'verify_loop_statistics "$SERVER_LOG" "$EMULATOR_LOG"'
        original = self.playback.read_text()
        for before, after, message in (("  ILA verification: PASS\n", "",
                                        "Finite ILA verification"),
                                       ("ILA verification: PASS",
                                        "ILA verification: FAIL",
                                        "Finite ILA verification"),
                                       ("  RESULT: PASS\n", "",
                                        "Bounded loop playback"),
                                       ("RESULT: PASS", "RESULT: FAIL",
                                        "Bounded loop playback"),
                                       ("  Playback frames: 30\n", "",
                                        "parse loop-only"),
                                       ("Correction responses: 30",
                                        "Correction responses: 29",
                                        "Loop transport failed"),
                                       ("RPC status failures: 0",
                                        "RPC status failures: 1",
                                        "Loop RPC status failures")):
            with self.subTest(change=before):
                self.playback.write_text(original.replace(before, after))
                self.assert_rejected(self.run_functions(command), message)
        self.playback.write_text(original)
        self.server.write_text("")
        self.assert_rejected(self.run_functions(command),
                             "without decoder statistics")

    def test_emulator_transport_failures_are_rejected(self):
        original = self.emulator.read_text()
        for before, after in (("Responses received: 36",
                               "Responses received: 35"), ("Send errors: 0",
                                                           "Send errors: 1"),
                              ("Timeouts: 0", "Timeouts: 1")):
            with self.subTest(change=after):
                self.emulator.write_text(original.replace(before, after))
                self.assert_rejected(
                    self.run_functions(
                        'verify_loop_statistics "$SERVER_LOG" "$EMULATOR_LOG"'),
                    "Emulator loop transport failed")

    def test_finish_waits_for_flushed_logs_and_runs_acceptance(self):
        result = self.run_functions(
            'finish_loop_run "$SERVER_LOG" "$EMULATOR_LOG" 102')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("RESULT: PASS", result.stdout)

    def test_log_flush_timeouts_are_bounded_failures(self):
        self.server.write_text("")
        command = 'finish_loop_run "$SERVER_LOG" "$EMULATOR_LOG" 102'
        self.assert_rejected(self.run_functions(command),
                             "statistics were not flushed after shutdown")
        self.emulator.write_text(self.emulator.read_text().replace(
            "*** EMULATOR: PASS\n", ""))
        self.assert_rejected(self.run_functions(command, transport="gpu_roce"),
                             "results were not flushed after exit")

    def test_process_wait_timeout_and_nonzero_exit_are_failures(self):
        result = self.run_functions('wait_for_loop_exit 101 fixture',
                                    mocks=r'''
kill() { if [[ $1 == -0 ]]; then return 0; fi; echo "signal $*"; }
''')
        self.assert_rejected(result, "did not exit within 1s")
        self.assertIn("signal -KILL 101", result.stdout)
        result = self.run_functions('wait_for_loop_exit 101 fixture',
                                    mocks='''
wait() { return 7; }
''')
        self.assert_rejected(result, "exited with an error")


if __name__ == "__main__":
    unittest.main()
