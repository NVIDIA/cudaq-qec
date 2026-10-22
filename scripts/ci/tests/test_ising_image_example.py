# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Exercise the shipped shell example with fake processes, no GPU or network."""

import csv
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER = REPOSITORY / "docker/release/decoding-server-examples/run-ising-example.sh"
HEADERS = "decoder_id,op,status,dispatched,request_status,correction_mismatch,read_completed,correction_bits".split(
    ",")


class IsingImageExampleTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prefix = self.root / "install"
        binary = self.prefix / "bin"
        binary.mkdir(parents=True)
        self.assets = self.prefix / "share/cudaq-qec/runtime-assets/fixture"
        examples = self.assets / "examples"
        examples.mkdir(parents=True)
        (self.assets / "manifest.json").write_text("{}")
        for suffix in ("yaml", "schedule"):
            (examples / f"x-global.{suffix}").write_text("fixture")
        (examples / "x-global.csv").write_text("decoder_id,expected\n0,01\n")
        self.events = self.root / "fixture.csv"
        self.rows = [["0", "reset", "0", "1", "0", "0", "0", ""],
                     ["0", "stream", "100", "1", "0 0", "0", "0", ""],
                     ["0", "enqueue_data", "100", "1", "0", "0", "0", ""],
                     ["0", "get_corrections", "0", "1", "0", "0", "1", "01"]]
        server = binary / "decoding_server"
        server.write_text('''#!/usr/bin/env python3
import signal
import sys
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
print("QEC_DECODING_SERVER_READY port=1234 transport=udp ring0=1234", flush=True)
try:
    while True:
        signal.pause()
finally:
    print("QEC_DECODING_SERVER_RING decoder=0 dispatched=5", flush=True)
    print("QEC_DECODING_SERVER_DISPATCHED count=5", flush=True)
''')
        server.chmod(0o755)
        playback = binary / "playback-emulator"
        playback.write_text('''#!/usr/bin/env bash
set -eu
test "$PWD" = "$EXPECTED_ASSET_ROOT"
test -z "${CUDAQ_QEC_ISING_MATCHING:-}"
for arg in "$@"; do
    case "$arg" in --out=*) cp "$EXAMPLE_EVENT_FIXTURE" "${arg#--out=}" ;; esac
done
''')
        playback.chmod(0o755)

    def run_example(self, name="x-global", **overrides):
        with self.events.open("w", newline="") as data:
            writer = csv.writer(data, lineterminator="\n")
            writer.writerow(HEADERS)
            writer.writerows(self.rows)
        environment = dict(os.environ,
                           CUDAQX_INSTALL_DIR=str(self.prefix),
                           QEC_RUNTIME_ASSET_ID="fixture",
                           TMPDIR=str(self.root),
                           EXAMPLE_EVENT_FIXTURE=str(self.events),
                           EXPECTED_ASSET_ROOT=str(self.assets),
                           CUDAQ_QEC_ISING_MATCHING="diagnostic-override")
        environment.update(overrides)
        return subprocess.run(["bash", str(RUNNER), name],
                              env=environment,
                              capture_output=True,
                              text=True,
                              timeout=10,
                              check=False)

    def test_installed_relative_example_and_external_oracle_pass(self):
        result = self.run_example()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("Validated 1 completed shots (5 requests)", result.stdout)

    def test_client_success_cannot_hide_failure_or_wrong_correction(self):
        for row, column, value in ((1, 2, "103"), (1, 4, "0 9"), (3, 6, "0"),
                                   (3, 7, "1"), (3, 5, "1"), (3, 3, "0")):
            with self.subTest(row=row, column=column):
                previous = self.rows[row][column]
                self.rows[row][column] = value
                result = self.run_example()
                self.assertNotEqual(0, result.returncode)
                self.assertIn("Example failed", result.stderr)
                self.rows[row][column] = previous

    def test_missing_or_invalid_asset_and_case_fail_before_server(self):
        for case, overrides in (("unknown", {}), ("x-global", {
                "QEC_RUNTIME_ASSET_ID": ""
        }), ("x-global", {
                "QEC_RUNTIME_ASSET_ID": "../other"
        })):
            with self.subTest(case=case, overrides=overrides):
                self.assertNotEqual(
                    0,
                    self.run_example(case, **overrides).returncode)


if __name__ == "__main__":
    unittest.main()
