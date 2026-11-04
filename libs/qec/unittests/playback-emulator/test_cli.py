#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise playback CLI argument handling and CSV publication."""

import argparse
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

EMULATOR = None
NOISE_KEYS = (
    "before_measure_flip_probability",
    "after_clifford_depolarization",
    "before_round_data_depolarization",
    "after_reset_flip_probability",
)


class PlaybackCliTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="playback-cli-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "config.yaml"
        self.config.write_text("""\
decoders:
  - id: 0
    type: single_error_lut
    block_size: 1
    syndrome_size: 1
    H_sparse: [0, -1]
    O_sparse: [0, -1]
    D_sparse: [0, -1]
""")
        self.schedule = self.root / "schedule.txt"
        self.schedule.write_text("0 reset\n")

    def invoke(self, *args, stdout=subprocess.PIPE):
        return subprocess.run([
            EMULATOR, f"--config={self.config}", f"--schedule={self.schedule}",
            "--backend=null", *args
        ],
                              text=True,
                              stdout=stdout,
                              stderr=subprocess.PIPE,
                              timeout=15)

    def assert_reset_csv(self, text):
        records = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["op"], "reset")
        self.assertEqual(records[0]["status"], "0")
        self.assertEqual(records[0]["dispatched"], "1")

    def test_csv_stdout(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)

    def test_csv_file(self):
        output = self.root / "result.csv"
        result = self.invoke(f"--out={output}")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assert_reset_csv(output.read_text())

    def test_csv_open_failure(self):
        for output in (self.root / "missing" / "result.csv", self.root):
            with self.subTest(output=output):
                result = self.invoke(f"--out={output}")
                self.assertEqual(result.returncode, 1)
                self.assertIn("cannot open CSV file", result.stderr)
                self.assertIn(str(output), result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_csv_file_write_failure(self):
        result = self.invoke("--out=/dev/full")
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot write CSV file: /dev/full", result.stderr)

    @unittest.skipUnless(Path("/dev/full").exists(), "requires /dev/full")
    def test_csv_stdout_write_failure(self):
        with open("/dev/full", "w") as full:
            result = self.invoke(stdout=full)
        self.assertEqual(result.returncode, 1)
        self.assertIn("cannot write CSV to stdout", result.stderr)

    def test_requested_lead_in(self):
        start = time.monotonic()
        result = self.invoke("--lead-in=1s")
        elapsed = time.monotonic() - start
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)
        # Only a lower bound: process startup and a loaded runner may add time.
        self.assertGreaterEqual(elapsed, 1.0)

    def test_zero_lead_in(self):
        result = self.invoke("--lead-in=0ns")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)

    def stim_source(self, **noise):
        return ("--stim-source=0:code=surface_code,task=rotated_memory_z,"
                "distance=3,seed=123," +
                ",".join(f"{key}={value}" for key, value in noise.items()))

    def test_rejects_malformed_and_nonfinite_noise(self):
        for key in NOISE_KEYS:
            for value in ("0.1garbage", "nan", "inf"):
                with self.subTest(key=key, value=value):
                    result = self.invoke(self.stim_source(**{key: value}))
                    self.assertEqual(result.returncode, 1)
                    self.assertIn(f"--stim-source {key}", result.stderr)
                    self.assertIn("finite probability in [0,1]", result.stderr)

    def test_rejects_invalid_noise_numbers(self):
        key = "after_clifford_depolarization"
        for value in ("", "invalid", "1e999", "-0.1", "1.1"):
            with self.subTest(value=value):
                result = self.invoke(self.stim_source(**{key: value}))
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"--stim-source {key}", result.stderr)
                self.assertIn("finite probability in [0,1]", result.stderr)

    def test_valid_noise_probabilities(self):
        self.schedule.write_text("0 stream source=0 rounds=3\n"
                                 "- enqueue_data source=0\n")
        for value in ("0", "1e-3", "+0.1", " 0.1", "0x1p-2"):
            with self.subTest(value=value):
                result = self.invoke(
                    self.stim_source(**{key: value for key in NOISE_KEYS}))
                self.assertEqual(result.returncode, 0, result.stderr)
                records = list(csv.DictReader(io.StringIO(result.stdout)))
                self.assertEqual([r["status"] for r in records], ["100", "100"])
                self.assertEqual([len(r["syndrome_bits"]) for r in records],
                                 [24, 9])

    def test_unit_probability_parses(self):
        result = self.invoke(
            self.stim_source(**{key: "1" for key in NOISE_KEYS}))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_reset_csv(result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--emulator", required=True)
    args, remaining = parser.parse_known_args()
    EMULATOR = str(Path(args.emulator).resolve())
    unittest.main(argv=[sys.argv[0], *remaining])
