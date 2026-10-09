# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Consistency tests for the `cudaq_qec.playback` bindings.

The emulator's behaviour is covered in C++ (libs/qec/unittests/
playback-emulator). What only a Python test can reach is the seam itself: the
names the module exports, the keys each record dict carries, the binding's
slicing of the run's logs into per-record values, the backend-selection check
that lives in the binding and nowhere else, and how a C++ exception arrives on
this side.

So the shape of most tests here is: take one run, then check that two
independent paths out of the same C++ value agree -- a record dict entry
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


def a_run():
    """One successful run covering all four ops. Runs entirely against the
    null backend, which never fails."""
    return pb.run(
        "0 reset\n"
        "1 stream source=0 rounds=2\n"
        "2 enqueue_data source=0\n"
        "3 get_corrections return_size=1\n",
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
        "run",
        "run_result",
    ]


def test_every_record_key_and_run_result_field_is_reachable():
    result = a_run()
    header, _ = rows(result)
    for rec in result.records:
        assert set(rec) == set(header)
    for name in ("records", "warnings", "t0_ns", "tick_ns"):
        assert hasattr(result, name), name
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
        cpu_roce_endpoints=None,
        cpu_roce_options=None,
        cpu_roce_timeout_ms=200,
        null_decoder_ids=[0],
        lead_in_ns=1_000_000,
    )
    assert len(result.records) == 1


def test_run_defaults_match_the_cli():
    # tick_ns and sources default like the CLI's --tick=1us and no --source.
    result = pb.run("1 reset\n", null_decoder_ids=[0])
    assert result.tick_ns == 1000
    assert result.records[0]["deadline_ns"] == 1000


@pytest.mark.parametrize("kwargs", [
    dict(tick_ns=0, null_decoder_ids=[0]),
    dict(udp_endpoints={0: "127.0.0.1:1"}, udp_timeout_ms=0),
    dict(cpu_roce_endpoints={0: "127.0.0.1:1"},
         cpu_roce_options={
             "device": "none",
             "local_ip": "127.0.0.1"
         },
         cpu_roce_timeout_ms=0),
    dict(cpu_roce_endpoints={0: "127.0.0.1:1"},
         cpu_roce_options={
             "device": "none",
             "local_ip": "127.0.0.1",
             "connect_timeout_ms": 0
         }),
])
def test_a_zero_tick_or_timeout_is_a_value_error(kwargs):
    # Checked in the C++ library, so the CLI rejects the same values.
    with pytest.raises(ValueError, match="must be positive"):
        pb.run("0 reset\n", **kwargs)


# -- bound values against the CSV C++ writes ---------------------------------


def test_one_csv_row_per_record_with_matching_scalar_fields():
    result = a_run()
    header, data = rows(result)
    assert len(data) == len(result.records)
    for rec, row in zip(result.records, data):
        for name in ("event_index", "decoder_id", "op", "deadline_ns",
                     "call_ns", "return_ns", "status", "rounds_streamed",
                     "syndrome_bits", "correction_bits"):
            assert cell(header, row, name) == str(rec[name]), name
        for name in ("read_completed", "correction_mismatch", "dispatched"):
            assert cell(header, row, name) == str(int(rec[name])), name
    assert {r["op"] for r in result.records} == {
        "reset",
        "stream",
        "enqueue_data",
        "get_corrections",
    }


def test_status_matches_the_csv_for_ok_error_and_never_dispatched():
    # a_run() covers OK; an_aborted_run() covers INTERNAL_ERROR (3) and, past
    # the abort, never-dispatched (-1).
    seen = set()
    for result in (a_run(), an_aborted_run()):
        header, data = rows(result)
        for rec, row in zip(result.records, data):
            assert cell(header, row, "status") == str(rec["status"])
            seen.add(rec["status"])
    assert {0, 3, -1} <= seen


def test_per_request_lists_match_the_csv_columns():
    result = a_run()
    header, data = rows(result)
    all_ids = []
    for rec, row in zip(result.records, data):
        for name in ("request_ids", "request_dispatch_ns", "request_return_ns",
                     "request_status"):
            assert cell(header, row,
                        name) == " ".join(str(x) for x in rec[name]), name
        assert len(rec["request_status"]) == len(rec["request_ids"])
        assert set(rec["request_status"]) <= {0}
        all_ids += rec["request_ids"]
    # Every id the run issued belongs to exactly one record, in issue order.
    assert all_ids and sorted(set(all_ids)) == all_ids


def test_bits_reach_the_records():
    # The bits columns are compared above; this checks they are not vacuously
    # empty.
    result = a_run()
    assert "".join(r["syndrome_bits"] for r in result.records) == "101" * 3


def test_an_aborted_run_reports_a_warning_and_stops_dispatching():
    result = an_aborted_run()
    assert len(result.warnings) == 1
    assert "aborting the run" in result.warnings[0]
    dispatched = [r["dispatched"] for r in result.records]
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


def test_at_least_one_backend_must_be_named():
    # This check lives in the binding's own run_schedule() wrapper and has no
    # C++ test, because there is no C++ caller that can get it wrong.
    with pytest.raises(ValueError, match="at least one"):
        pb.run("0 reset\n", 1000, {})


def test_a_decoder_id_named_by_two_backends_is_a_value_error():
    with pytest.raises(ValueError, match="decoder_id 0 is named by more"):
        pb.run("0 reset\n",
               1000, {},
               null_decoder_ids=[0],
               udp_endpoints={0: "127.0.0.1:1"})


def test_backends_can_be_mixed_per_decoder():
    # Decoder 0 is null (always OK); decoder 1 is UDP to a port nobody listens
    # on (always times out). Each record must carry its own backend's verdict.
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    result = pb.run("0 reset\n"
                    "0 reset session=1\n",
                    1000, {},
                    null_decoder_ids=[0],
                    udp_endpoints={1: f"127.0.0.1:{port}"},
                    udp_timeout_ms=50)
    by_decoder = {r["decoder_id"]: r["status"] for r in result.records}
    assert by_decoder == {0: 0, 1: 3}  # OK, INTERNAL_ERROR


def test_cpu_roce_options_are_parsed_before_any_transport_is_touched():
    # The key checks live in the binding's parser, ahead of the factory; a full
    # option set reaches the factory and fails there (no device / no transport).
    eps = {0: "127.0.0.1:1"}
    with pytest.raises(ValueError, match="cpu_roce_options"):
        pb.run("0 reset\n", 1000, {}, cpu_roce_endpoints=eps)
    opts = {"device": "none", "local_ip": "127.0.0.1"}
    with pytest.raises(ValueError, match="unknown cpu_roce_options key"):
        pb.run("0 reset\n",
               1000, {},
               cpu_roce_endpoints=eps,
               cpu_roce_options=dict(opts, pages=8))
    opts.update(slots=8, slot_size=256, connect_timeout_ms=100)
    with pytest.raises(RuntimeError):
        pb.run("0 reset\n",
               1000, {},
               cpu_roce_endpoints=eps,
               cpu_roce_options=opts,
               cpu_roce_timeout_ms=100)


# -- syndrome sources --------------------------------------------------------


def test_a_static_source_spec_replays_identically_across_separate_runs():
    schedule = "0 stream source=0 rounds=3\n"
    spec = {0: {"type": "static", "rounds": [[1, 1, 0]] * 3}}

    # A fresh source is built per run(), so the same spec replays identically.
    first = pb.run(schedule, 1000, spec, null_decoder_ids=[0])
    assert first.records[0]["rounds_streamed"] == 3
    assert first.records[0]["syndrome_bits"] == "110" * 3

    second = pb.run(schedule, 1000, spec, null_decoder_ids=[0])
    assert second.records[0]["rounds_streamed"] == 3
    assert second.records[0]["syndrome_bits"] == "110" * 3


def test_a_source_spec_missing_its_type_key_is_a_value_error():
    with pytest.raises(ValueError, match="type"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   "rounds": [[1]]
               }},
               null_decoder_ids=[0])


@pytest.mark.parametrize("spec", [
    {
        "type": "static",
        "rounds": [[1]],
        "seed": 1
    },
    {
        "type": "stim_memory",
        "code": "repetition_code",
        "task": "memory",
        "distance": 3,
        "distnace": 3
    },
])
def test_an_unknown_source_key_is_a_value_error(spec):
    with pytest.raises(ValueError, match="unknown .* source key"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: spec},
               null_decoder_ids=[0])


def test_an_unrecognized_source_type_is_a_value_error():
    with pytest.raises(ValueError, match="unknown source type"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   "type": "not_a_real_type"
               }},
               null_decoder_ids=[0])


def test_a_stim_memory_source_spec_drives_a_run_and_rejects_bad_params():
    # "seed" defaults to 1, like the CLI's --stim-source.
    params = dict(type="stim_memory",
                  code="repetition_code",
                  task="memory",
                  distance=3)
    result = pb.run("0 stream source=0 rounds=4\n",
                    1000, {0: params},
                    null_decoder_ids=[0])
    assert result.records[0]["rounds_streamed"] == 4
    assert len(result.records[0]
               ["syndrome_bits"]) == 4 * 2  # 2 ancilla bits/round at distance 3

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
    assert result.records[0]["rounds_streamed"] == 3
    assert len(result.records[0]
               ["syndrome_bits"]) == 3 * 2  # 2 ancilla bits/round at distance 3

    # "noise" is optional -- a fresh, empty NoiseModel is used if omitted.
    no_noise = {k: v for k, v in params.items() if k != "noise"}
    result2 = pb.run("0 stream source=0 rounds=3\n",
                     1000, {0: no_noise},
                     null_decoder_ids=[0])
    assert result2.records[0]["rounds_streamed"] == 3

    # An unrecognized state_prep name is a ValueError, not a crash.
    with pytest.raises(ValueError, match="state_prep"):
        pb.run("0 stream source=0 rounds=1\n",
               1000, {0: {
                   **params, "state_prep": "not_a_real_op"
               }},
               null_decoder_ids=[0])
