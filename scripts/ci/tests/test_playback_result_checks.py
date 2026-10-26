# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Completed-shot playback checks without a GPU, server, or private modules."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "playback_result_checks.py"
SPEC = importlib.util.spec_from_file_location("playback_checks", SCRIPT)
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def event(index, operation, decoder_id=0, ids=None, correction="1"):
    ids = ids if ids is not None else [index + 1]
    is_data = operation in ("stream", "enqueue_data")
    return {
        "event_index": str(index),
        "decoder_id": str(decoder_id),
        "op": operation,
        "call_ns": "10",
        "return_ns": "30",
        "status": str(CHECKS.STREAM_OK if is_data else CHECKS.RPC_OK),
        "rounds_streamed": str(len(ids) if is_data else 0),
        "read_completed": str(int(operation == "get_corrections")),
        "correction_bits": correction if operation == "get_corrections" else "",
        "correction_mismatch": "0",
        "request_ids": " ".join(str(value) for value in ids),
        "dispatched": "1",
        "request_dispatch_ns": " ".join("15" for _ in ids),
        "request_return_ns": " ".join("25" for _ in ids),
        "request_status": " ".join("0" for _ in ids)
    }


def shot_rows(final=True, correction="1"):
    operations = ["reset", "stream"]
    if final:
        operations.append("enqueue_data")
    operations.append("get_corrections")
    return [
        event(index, operation, correction=correction)
        for index, operation in enumerate(operations)
    ]


class ExpectedCorrectionsTest(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "expected.csv"

    def load(self, text):
        self.path.write_text(text, encoding="utf-8")
        return CHECKS.load_expected(self.path)

    def test_default_decoder_and_opaque_fixture_metadata(self):
        self.assertEqual(
            self.load("shot,rounds,expected\n0,73,01\n1,1000,10\n"),
            {0: ["01", "10"]})

    def test_interleaved_decoder_oracles_preserve_per_decoder_order(self):
        self.assertEqual(self.load("decoder_id,expected\n2,1\n0,00\n2,0\n"), {
            2: ["1", "0"],
            0: ["00"]
        })

    def test_rejects_empty_missing_or_duplicate_headers(self):
        for text in ("", "shot\n0\n", "expected\n", "expected,expected\n1,1\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_rejects_invalid_bits_and_identity(self):
        for text in ("expected\n2\n", "expected\n01x\n",
                     "decoder_id,expected\n-1,1\n", "decoder_id,expected\n,1\n",
                     "decoder_id,expected\n0,\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_rejects_truncated_or_extra_csv_values(self):
        for text in ("expected,shot\n1\n", "expected\n1,2\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)


class PlaybackRecordsTest(unittest.TestCase):

    def test_final_shot_returns_json_ready_counters(self):
        self.assertEqual(
            CHECKS.validate_records(shot_rows(), {0: ["1"]}, True), {
                "events": 4,
                "requests": 4,
                "requests_per_decoder": {
                    "0": 4
                },
                "shots": 1,
                "shots_per_decoder": {
                    "0": 1
                },
                "nonzero_corrections": 1,
                "reads": [{
                    "decoder_id": 0,
                    "shot": 0,
                    "correction_bits": "1"
                }]
            })

    def test_ordinary_decoder_does_not_require_final(self):
        summary = CHECKS.validate_records(shot_rows(final=False), {0: ["1"]})
        self.assertEqual(summary["shots"], 1)
        with self.assertRaisesRegex(ValueError, "lifecycle"):
            CHECKS.validate_records(shot_rows(final=False), {0: ["1"]}, True)

    def test_zero_correction_and_multiple_observable_bits_are_not_vacuous(self):
        rows = shot_rows(correction="000")
        summary = CHECKS.validate_records(rows, {0: ["000"]}, True)
        self.assertEqual(summary["shots"], 1)
        self.assertEqual(summary["nonzero_corrections"], 0)

    def test_interleaved_decoders_may_reuse_request_ids(self):
        rows = []
        for index, operation in enumerate(
            ("reset", "stream", "enqueue_data", "get_corrections")):
            for decoder_id in (0, 2):
                rows.append(
                    event(len(rows),
                          operation,
                          decoder_id,
                          ids=[index + 1],
                          correction=str(decoder_id // 2)))
        summary = CHECKS.validate_records(rows, {0: ["0"], 2: ["1"]}, True)
        self.assertEqual(summary["requests_per_decoder"], {"0": 4, "2": 4})
        self.assertEqual(summary["shots_per_decoder"], {"0": 1, "2": 1})

    def test_batched_stream_matches_request_and_round_counts(self):
        rows = shot_rows()
        rows[1] = event(1, "stream", ids=[2, 10, 11])
        summary = CHECKS.validate_records(rows, {0: ["1"]}, True)
        self.assertEqual(summary["requests"], 6)

    def test_multiple_shots_preserve_order_and_counts(self):
        rows = shot_rows(correction="0")
        rows.extend(
            event(index, operation) for index, operation in enumerate(
                ("reset", "stream", "enqueue_data",
                 "get_corrections"), start=4))
        summary = CHECKS.validate_records(rows, {0: ["0", "1"]}, True)
        self.assertEqual(summary["shots"], 2)
        self.assertEqual([read["shot"] for read in summary["reads"]], [0, 1])

    def test_rejects_bad_expected_oracles(self):
        for expected in ({}, {
                0: []
        }, {
                0: [""]
        }, {
                0: ["2"]
        }, {
                -1: ["1"]
        }, {
                "0": ["1"]
        }, {
                True: ["1"]
        }):
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                CHECKS.validate_records(shot_rows(), expected)

    def test_rejects_empty_missing_extra_or_incomplete_shots(self):
        for rows, expected in (([], {
                0: ["1"]
        }), (shot_rows()[:-1], {
                0: ["1"]
        }), (shot_rows(), {
                0: ["1", "1"]
        }), (shot_rows(), {
                0: ["1"],
                2: ["0"]
        })):
            with self.subTest(rows=rows,
                              expected=expected), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, expected, True)

    def test_rejects_extra_completed_shot(self):
        rows = shot_rows()
        rows.extend(
            event(index, operation)
            for index, operation in enumerate(("reset", "get_corrections"),
                                              start=4))
        with self.assertRaisesRegex(ValueError, "extra shot"):
            CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_invalid_event_identity_and_shape(self):
        for key, value in (("event_index", "7"), ("decoder_id", "3"),
                           ("decoder_id", "-1"), ("op", "enqueue")):
            rows = shot_rows()
            rows[0][key] = value
            with self.subTest(key=key,
                              value=value), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, {0: ["1"]})
        rows = shot_rows()
        del rows[0]["request_status"]
        with self.assertRaises(ValueError):
            CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_undispatched_failed_or_not_ready_events(self):
        for key, value in (("dispatched", "0"), ("dispatched", "true"),
                           ("status", "3"), ("status", "4"), ("status", "-1")):
            rows = shot_rows()
            rows[0][key] = value
            with self.subTest(key=key,
                              value=value), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_failed_request_hidden_by_success_event(self):
        rows = shot_rows()
        rows[1]["request_status"] = "3"
        with self.assertRaisesRegex(ValueError, "unsuccessful RPC"):
            CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_missing_or_inconsistent_request_vectors(self):
        for key, value in (("request_ids", ""), ("request_status", ""),
                           ("request_dispatch_ns",
                            "15 16"), ("request_return_ns", "25 26")):
            rows = shot_rows()
            rows[1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_duplicate_request_ids_within_decoder(self):
        for replacement in (event(1, "stream",
                                  ids=[1]), event(1, "stream", ids=[2, 2])):
            rows = shot_rows()
            rows[1] = replacement
            with self.subTest(row=replacement), self.assertRaisesRegex(
                    ValueError, "repeats a request ID"):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_reply_before_dispatch_and_negative_timing(self):
        for key, value in (("call_ns", "31"), ("return_ns", "-1"),
                           ("request_dispatch_ns", "-1"), ("request_return_ns",
                                                           "14")):
            rows = shot_rows()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_round_count_disagreement(self):
        for index, value in ((0, "1"), (1, "0"), (2, "2"), (3, "1")):
            rows = shot_rows()
            rows[index]["rounds_streamed"] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_reset_final_and_read_are_single_request_operations(self):
        for index, operation in ((0, "reset"), (2, "enqueue_data"),
                                 (3, "get_corrections")):
            rows = shot_rows()
            rows[index] = event(index, operation, ids=[10, 11])
            with self.subTest(operation=operation), self.assertRaisesRegex(
                    ValueError, "exactly one request"):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_rejects_wrong_malformed_missing_or_incomplete_corrections(self):
        for key, value in (("correction_bits", "0"), ("correction_bits", ""),
                           ("correction_bits", "2"), ("correction_bits", "11"),
                           ("read_completed", "0"), ("correction_mismatch",
                                                     "1")):
            rows = shot_rows()
            rows[-1][key] = value
            with self.subTest(key=key,
                              value=value), self.assertRaises(ValueError):
                CHECKS.validate_records(rows, {0: ["1"]}, True)

    def test_rejects_illegal_lifecycle_transitions(self):
        for operations in (("stream", "get_corrections"), ("reset", "reset",
                                                           "get_corrections"),
                           ("reset", "enqueue_data", "stream",
                            "get_corrections"),
                           ("reset", "enqueue_data", "enqueue_data",
                            "get_corrections"), ("get_corrections",)):
            rows = [
                event(index, operation)
                for index, operation in enumerate(operations)
            ]
            with self.subTest(operations=operations), self.assertRaisesRegex(
                    ValueError, "lifecycle"):
                CHECKS.validate_records(rows, {0: ["1"]})

    def test_final_requirement_is_a_boolean(self):
        with self.assertRaises(ValueError):
            CHECKS.validate_records(shot_rows(), {0: ["1"]}, "false")


if __name__ == "__main__":
    unittest.main()
