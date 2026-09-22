# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Consistency tests for the `cudaq_qec.playback` bindings.

The emulator's behaviour is covered in C++ (libs/qec/tools/playback-emulator/
tests). What only a Python test can reach is the seam itself: the names the
module exports, the fields each bound struct exposes, the two places the
binding re-derives something C++ already computes (`record.status` and
`run_result.request_ids()`), the backend-selection check that lives in the
binding and nowhere else, and how a C++ exception arrives on this side.

So the shape of most tests here is: take one run, then check that two
independent paths out of the same C++ value agree -- a bound attribute
against the corresponding cell of `write_csv()`, which is written by C++ and
never passes through nanobind.
"""

import socket

import pytest

import cudaq
import cudaq_qec as qec

# The bindings are only built where the emulator itself is: a
# realtime-enabled CUDA-Q install (see libs/qec/python/CMakeLists.txt).
pb = getattr(qec, "playback", None)
pytestmark = pytest.mark.skipif(
    pb is None, reason="cudaq_qec.playback not built in this configuration")


def rows(result):
    """`write_csv()` split into a header list and a list of column lists."""
    lines = result.write_csv().splitlines()
    header = lines[0].split(",")
    return header, [line.split(",") for line in lines[1:]]


def cell(header, row, name):
    return row[header.index(name)]


def bits_of(bits):
    """The rendering write_csv() documents: one '0'/'1' character per bit, in
    log order. Re-derived here so the Python view of an arena can be checked
    against the column C++ wrote from it."""
    return "".join("1" if b else "0" for b in bits)


def a_run():
    """One successful run covering three of the four ops (reset, stream,
    get_corrections); `enqueue_data` has no schedule spelling reachable from
    here. Runs entirely against the null backend, which never fails."""
    return pb.run(
        "0 reset\n"
        "1 stream source=0 rounds=2\n"
        "2 get_corrections return_size=1\n",
        1000,
        {0: {
            "type": "static",
            "rounds": [[1, 0, 1]] * 8
        }},
        null_decoder_ids=[0],
    )


def an_aborted_run():
    """A run whose `get_corrections` never gets an answer -- the UDP
    endpoint is bound-then-closed, so nobody is listening -- producing a
    genuine RPC timeout that hard-aborts the run. The trailing resets are
    spaced widely enough in wall-clock time (20ms apart, well past the 50ms
    UDP timeout) that at least one of them is guaranteed to still be in the
    future when the abort lands; this is a deadline comparison against a
    fixed timeout, not a race the reader thread has to win quickly (compare
    the C++ suite's long-tail-of-cheap-events pattern for that other kind of
    race, used where nothing bounds how long the failure takes to surface)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    schedule = "0 get_corrections return_size=1\n"
    for tick in range(1, 21):
        schedule += f"{tick} reset\n"
    return pb.run(
        schedule,
        20_000_000,  # 20ms/tick: the 20-event tail spans 400ms
        {},
        udp_endpoints={0: f"127.0.0.1:{port}"},
        udp_timeout_ms=50,
    )


# -- the exported surface ----------------------------------------------------


def test_module_exports_exactly_the_documented_surface():
    assert sorted(n for n in dir(pb) if not n.startswith("_")) == [
        "operation",
        "record",
        "run",
        "run_result",
    ]


def test_every_record_and_run_result_field_is_reachable():
    # A field dropped from the binding but still present in C++ would
    # otherwise only show up as an AttributeError in a demo.
    result = a_run()
    for name in (
            "event_index",
            "decoder_id",
            "op",
            "dispatched",
            "deadline_ns",
            "call_ns",
            "return_ns",
            "status",
            "rounds_streamed",
            "read_completed",
            "syndrome_offset",
            "syndrome_count",
            "correction_offset",
            "correction_count",
            "correction_mismatch",
            "request_id_offset",
            "request_id_count",
    ):
        assert hasattr(result.records[0], name), name
    for name in ("records", "syndrome_log", "correction_log", "request_id_log",
                 "warnings", "t0_ns", "tick_ns"):
        assert hasattr(result, name), name
    assert callable(result.request_ids)
    assert callable(result.write_csv)


def test_run_accepts_every_documented_keyword_by_name():
    # The nb::arg names are API: a demo calling run(schedule=..., tick_ns=...)
    # breaks silently if one is renamed on the C++ side.
    result = pb.run(
        schedule="0 reset\n",
        tick_ns=1000,
        sources={},
        decoders=None,
        udp_endpoints=None,
        udp_timeout_ms=200,
        null_decoder_ids=[0],
        lead_in_ns=1_000_000,
    )
    assert len(result.records) == 1


# -- bound values against the CSV C++ writes ---------------------------------


def test_one_csv_row_per_record_with_matching_scalar_fields():
    result = a_run()
    header, data = rows(result)
    assert len(data) == len(result.records)
    for rec, row in zip(result.records, data):
        assert cell(header, row, "event_index") == str(rec.event_index)
        assert cell(header, row, "decoder_id") == str(rec.decoder_id)
        assert cell(header, row, "deadline_ns") == str(rec.deadline_ns)
        assert cell(header, row, "call_ns") == str(rec.call_ns)
        assert cell(header, row, "return_ns") == str(rec.return_ns)
        assert cell(header, row, "rounds_streamed") == str(rec.rounds_streamed)
        assert cell(header, row,
                    "read_completed") == str(int(rec.read_completed))
        assert cell(header, row,
                    "correction_mismatch") == str(int(rec.correction_mismatch))
        assert cell(header, row, "dispatched") == str(int(rec.dispatched))


def test_operation_enum_names_match_the_csv_op_column():
    result = a_run()
    header, data = rows(result)
    for rec, row in zip(result.records, data):
        assert rec.op.name == cell(header, row, "op")
    # `enqueue` lowers to a one-round stream, so three lines cover three of
    # the four values; the fourth must still exist and be distinct.
    assert {r.op for r in result.records} == {
        pb.operation.reset,
        pb.operation.stream,
        pb.operation.get_corrections,
    }
    assert pb.operation.enqueue_data not in {r.op for r in result.records}


def test_status_string_agrees_with_the_numeric_status_column():
    # record.status is a binding-side property that re-derives which enum a
    # numeric status belongs to (the two ranges are disjoint: RpcStatus 0..6,
    # stream_terminate 100..103, and -1 for never-dispatched). Nothing in C++
    # shares that code, so it is checked against the raw number here.
    numeric_to_name = {
        "0": "OK",
        "2": "BAD_REQUEST",
        "3": "INTERNAL_ERROR",
        "4": "NOT_READY",
        "100": "OK",
        "101": "SOURCE_EXHAUSTED",
        "102": "EXHAUSTED_ROUNDS",
        "103": "ERROR",
        "-1": "NOT_DISPATCHED",
    }
    seen = set()
    # a_run() covers OK; an_aborted_run() covers INTERNAL_ERROR and, past
    # the abort, NOT_DISPATCHED -- together, all three shapes of status.
    for result in (a_run(), an_aborted_run()):
        header, data = rows(result)
        for rec, row in zip(result.records, data):
            raw = cell(header, row, "status")
            assert raw in numeric_to_name, f"unmapped status {raw}"
            assert rec.status == numeric_to_name[raw]
            seen.add(rec.status)
    assert {"OK", "INTERNAL_ERROR", "NOT_DISPATCHED"} <= seen


def test_request_ids_helper_agrees_with_the_offsets_and_the_csv_column():
    result = a_run()
    header, data = rows(result)
    total = 0
    for i, (rec, row) in enumerate(zip(result.records, data)):
        ids = result.request_ids(i)
        assert len(ids) == rec.request_id_count
        assert ids == list(
            result.request_id_log[rec.request_id_offset:rec.request_id_offset +
                                  rec.request_id_count])
        assert cell(header, row, "request_ids") == " ".join(str(x) for x in ids)
        total += len(ids)
    # Every id the run issued belongs to exactly one record, and the log is
    # strictly increasing.
    assert total == len(result.request_id_log)
    assert sorted(set(result.request_id_log)) == list(result.request_id_log)


def test_request_ids_rejects_an_event_index_that_does_not_exist():
    result = a_run()
    with pytest.raises(IndexError):
        result.request_ids(len(result.records))


def test_syndrome_log_slice_matches_the_bits_column_written_from_it():
    result = a_run()
    header, data = rows(result)
    for rec, row in zip(result.records, data):
        bits = list(
            result.syndrome_log[rec.syndrome_offset:rec.syndrome_offset +
                                rec.syndrome_count])
        assert cell(header, row, "syndrome_bits") == bits_of(bits)
        corrections = list(
            result.correction_log[rec.correction_offset:rec.correction_offset +
                                  rec.correction_count])
        assert cell(header, row, "correction_bits") == bits_of(corrections)
    # The streamed rounds really did reach the log, so the check above is not
    # vacuously comparing two empty strings.
    assert list(result.syndrome_log) == [1, 0, 1, 1, 0, 1]


def test_an_aborted_run_reports_a_warning_and_stops_dispatching():
    result = an_aborted_run()
    assert len(result.warnings) == 1
    assert "aborting the run" in result.warnings[0]
    dispatched = [r.dispatched for r in result.records]
    assert dispatched[0] is True  # the failing get_corrections itself ran
    assert False in dispatched  # the abort pre-empted at least one reset
    # dispatch never reorders or skips backward: every True precedes every
    # False.
    assert dispatched == sorted(dispatched, reverse=True)


# -- how C++ exceptions arrive here ------------------------------------------


@pytest.mark.parametrize(
    "schedule",
    [
        "0 frobnicate\n",  # unknown operation
        "0 reset session=7\n",  # decoder_id not in the config
        "0 enqueue\n",  # missing a required operand
        "0 enqueue source=0b012\n",  # malformed bit string
        "2 reset\n1 reset\n",  # ticks out of order
    ],
)
def test_a_schedule_error_arrives_as_a_value_error(schedule):
    # parse() throws std::invalid_argument, which nanobind maps to
    # ValueError. Demos catch it as such, so the mapping is part of the API.
    with pytest.raises(ValueError):
        pb.run(schedule, 1000, {}, null_decoder_ids=[0])


def test_a_schedule_error_names_the_offending_line():
    with pytest.raises(ValueError, match="line 2"):
        pb.run("0 reset\n0 frobnicate\n", 1000, {}, null_decoder_ids=[0])


def test_a_missing_syndrome_source_is_a_value_error():
    with pytest.raises(ValueError, match="source_id=9"):
        pb.run("0 enqueue source=9\n", 1000, {}, null_decoder_ids=[0])


@pytest.mark.parametrize(
    "backends",
    [
        {},  # none named
        dict(null_decoder_ids=[0], udp_endpoints={0: "127.0.0.1:1"}),
        dict(null_decoder_ids=[0], decoders=qec.multi_decoder_config()),
    ],
)
def test_exactly_one_backend_must_be_named(backends):
    # This check lives in the binding's own run_schedule() wrapper and has no
    # C++ test, because there is no C++ caller that can get it wrong.
    with pytest.raises(ValueError, match="exactly one"):
        pb.run("0 reset\n", 1000, {}, **backends)


# -- syndrome sources --------------------------------------------------------


def test_a_static_source_spec_replays_identically_across_separate_runs():
    schedule = "0 stream source=0 rounds=3\n"
    spec = {0: {"type": "static", "rounds": [[1, 1, 0]] * 3}}

    # A fresh source is built per run(), so the same spec replays identically.
    first = pb.run(schedule, 1000, spec, null_decoder_ids=[0])
    assert first.records[0].rounds_streamed == 3
    assert list(first.syndrome_log) == [1, 1, 0] * 3

    second = pb.run(schedule, 1000, spec, null_decoder_ids=[0])
    assert second.records[0].rounds_streamed == 3
    assert list(second.syndrome_log) == [1, 1, 0] * 3


def test_a_source_spec_missing_its_type_key_is_a_value_error():
    with pytest.raises(ValueError, match="type"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   "rounds": [[1]]
               }},
               null_decoder_ids=[0])


def test_an_unrecognized_source_type_is_a_value_error():
    with pytest.raises(ValueError, match="unknown source type"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   "type": "not_a_real_type"
               }},
               null_decoder_ids=[0])


def test_a_stim_memory_source_spec_drives_a_run_and_rejects_bad_params():
    params = dict(type="stim_memory",
                  seed=1,
                  code="repetition_code",
                  task="memory",
                  distance=3)
    result = pb.run("0 stream source=0 rounds=4\n",
                    1000, {0: params},
                    null_decoder_ids=[0])
    assert result.records[0].rounds_streamed == 4
    assert len(
        result.syndrome_log) == 4 * 2  # 2 ancilla bits/round at distance 3

    # The C++ constructor throws std::runtime_error, which nanobind maps to
    # RuntimeError rather than ValueError. Stim inlines the round body
    # instead of emitting a REPEAT block for 1-2 rounds, so there is no
    # round for stim_memory_source to derive.
    with pytest.raises(RuntimeError):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   **params, "rounds": 1
               }},
               null_decoder_ids=[0])

    # An unrecognized code family throws std::invalid_argument, which
    # nanobind maps to ValueError.
    with pytest.raises(ValueError):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   **params, "code": "not_a_real_code"
               }},
               null_decoder_ids=[0])


def test_a_cudaq_memory_source_spec_drives_a_run_and_rejects_bad_params():
    code = qec.get_code("repetition", distance=3)
    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
    params = dict(type="cudaq_memory",
                  code=code,
                  state_prep="prep0",
                  max_rounds=3,
                  seed=1,
                  noise=noise)

    result = pb.run("0 stream source=0 rounds=3\n",
                    1000, {0: params},
                    null_decoder_ids=[0])
    assert result.records[0].rounds_streamed == 3
    assert len(
        result.syndrome_log) == 3 * 2  # 2 ancilla bits/round at distance 3

    # "noise" is optional -- a fresh, empty NoiseModel is used if omitted.
    no_noise = {k: v for k, v in params.items() if k != "noise"}
    result2 = pb.run("0 stream source=0 rounds=3\n",
                     1000, {0: no_noise},
                     null_decoder_ids=[0])
    assert result2.records[0].rounds_streamed == 3

    # An unrecognized state_prep name is a ValueError, not a crash.
    with pytest.raises(ValueError, match="state_prep"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   **params, "state_prep": "not_a_real_op"
               }},
               null_decoder_ids=[0])
