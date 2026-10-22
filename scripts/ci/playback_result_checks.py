# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Strict correctness checks for completed-shot playback smoke tests.

This profile checks ordinary decoder shots and optionally FINAL-tagged shots.
It intentionally rejects cancellation, partial reads, and failed requests even
when those operations are useful in a different playback experiment. Expected
corrections are supplied independently; a successful client exit or an unset
``correction_mismatch`` flag is not an oracle.
"""

import csv
from pathlib import Path
import re

RPC_OK = 0
STREAM_OK = 100
SINGLE_REQUEST_OPERATIONS = {"reset", "enqueue_data", "get_corrections"}
EVENT_STATUS = {
    "reset": RPC_OK,
    "stream": STREAM_OK,
    "enqueue_data": STREAM_OK,
    "get_corrections": RPC_OK
}
REQUIRED_EVENT_FIELDS = {
    "event_index", "decoder_id", "op", "call_ns", "return_ns", "status",
    "rounds_streamed", "read_completed", "correction_bits",
    "correction_mismatch", "request_ids", "dispatched", "request_dispatch_ns",
    "request_return_ns", "request_status"
}


def _integer(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise ValueError(f"{name} must be a nonnegative decimal integer")
    return int(value)


def _bits(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[01]+", value) is None:
        raise ValueError(f"{name} must be a nonempty binary string")
    return value


def _integers(value, name):
    if not isinstance(value, str):
        raise ValueError(f"{name} must contain decimal integers")
    return [_integer(token, name) for token in value.split()]


def _validated_expected(expected):
    if not isinstance(expected, dict) or not expected:
        raise ValueError(
            "At least one nonempty expected-correction list is required")
    result = {}
    for decoder_id, corrections in expected.items():
        if type(decoder_id) is not int or decoder_id < 0:
            raise ValueError(
                "Expected decoder identities must be nonnegative integers")
        if not isinstance(corrections, list) or not corrections:
            raise ValueError(f"Decoder {decoder_id} needs expected corrections")
        result[decoder_id] = [
            _bits(bits, f"Decoder {decoder_id} expected correction")
            for bits in corrections
        ]
    return result


def load_expected(path):
    """Read ordered corrections per decoder from an external CSV oracle.

    ``expected`` is required; ``decoder_id`` defaults to zero when its column
    is absent. Other fixture metadata is not interpreted. Order within each
    decoder, rather than ordering between independent decoders, identifies
    shots.
    """
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        names = reader.fieldnames
        if (not names or "expected" not in names or
                len(names) != len(set(names))):
            raise ValueError(
                "Oracle CSV needs unique columns including expected")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed oracle CSV row {line}")
            decoder_id = _integer(row.get("decoder_id", "0"),
                                  f"Oracle row {line} decoder_id")
            correction = _bits(row["expected"], f"Oracle row {line} expected")
            result.setdefault(decoder_id, []).append(correction)
    return _validated_expected(result)


def validate_records(rows, expected, require_final=False):
    """Validate completed shots and every RPC, returning JSON-ready counters.

    ``rows`` contains playback CSV dictionaries; ``expected`` maps integer
    decoder IDs to ordered nonempty correction strings. Request IDs are local
    to each decoder session and may overlap between decoders.
    """
    expected = _validated_expected(expected)
    if type(require_final) is not bool:
        raise ValueError("require_final must be boolean")
    states = {decoder_id: "reset" for decoder_id in expected}
    counts = {decoder_id: 0 for decoder_id in expected}
    requests = {decoder_id: 0 for decoder_id in expected}
    request_ids = {decoder_id: set() for decoder_id in expected}
    reads = []
    event_count = 0
    nonzero_corrections = 0
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not REQUIRED_EVENT_FIELDS <= row.keys():
            raise ValueError(f"Event {index} has missing playback CSV fields")
        if _integer(row["event_index"], f"Event {index} index") != index:
            raise ValueError(
                "Playback event indices must be contiguous from zero")
        decoder_id = _integer(row["decoder_id"], f"Event {index} decoder_id")
        if decoder_id not in expected:
            raise ValueError(f"No correction oracle for decoder {decoder_id}")
        operation = row["op"]
        if operation not in EVENT_STATUS:
            raise ValueError(
                f"Event {index} has unsupported operation {operation!r}")
        if row["dispatched"] != "1":
            raise ValueError(f"Event {index} was not dispatched")
        if _integer(row["status"],
                    f"Event {index} status") != EVENT_STATUS[operation]:
            raise ValueError(f"Event {index} has unsuccessful playback status")
        call = _integer(row["call_ns"], f"Event {index} call_ns")
        returned = _integer(row["return_ns"], f"Event {index} return_ns")
        if returned < call:
            raise ValueError(f"Event {index} returned before dispatch")
        ids = _integers(row["request_ids"], f"Event {index} request_ids")
        dispatched = _integers(row["request_dispatch_ns"],
                               f"Event {index} request_dispatch_ns")
        completed = _integers(row["request_return_ns"],
                              f"Event {index} request_return_ns")
        statuses = _integers(row["request_status"],
                             f"Event {index} request_status")
        if not ids or not (len(ids) == len(dispatched) == len(completed) ==
                           len(statuses)):
            raise ValueError(f"Event {index} has inconsistent request vectors")
        if len(ids) != len(
                set(ids)) or request_ids[decoder_id].intersection(ids):
            raise ValueError(
                f"Event {index} repeats a request ID in decoder {decoder_id}")
        if any(status != RPC_OK for status in statuses):
            raise ValueError(f"Event {index} contains an unsuccessful RPC")
        if any(end < start for start, end in zip(dispatched, completed)):
            raise ValueError(
                f"Event {index} contains a reply preceding its request")
        if operation in SINGLE_REQUEST_OPERATIONS and len(ids) != 1:
            raise ValueError(f"Event {index} must contain exactly one request")
        rounds = _integer(row["rounds_streamed"],
                          f"Event {index} rounds_streamed")
        if operation in ("stream", "enqueue_data"):
            if rounds != len(ids):
                raise ValueError(
                    f"Event {index} round count differs from request count")
        elif rounds != 0:
            raise ValueError(
                f"Event {index} unexpectedly reports streamed rounds")
        state = states[decoder_id]
        if operation == "reset" and state == "reset":
            states[decoder_id] = "bulk"
        elif operation == "stream" and state == "bulk":
            pass
        elif operation == "enqueue_data" and state == "bulk":
            states[decoder_id] = "final"
        elif (operation == "get_corrections" and
              (state == "final" or (state == "bulk" and not require_final))):
            shot = counts[decoder_id]
            if shot >= len(expected[decoder_id]):
                raise ValueError(f"Decoder {decoder_id} returned an extra shot")
            bits = _bits(row["correction_bits"],
                         f"Event {index} correction_bits")
            if (row["read_completed"] != "1" or
                    row["correction_mismatch"] != "0" or
                    bits != expected[decoder_id][shot]):
                raise ValueError(
                    f"Correction mismatch or incomplete read for decoder {decoder_id}, shot {shot}"
                )
            counts[decoder_id] += 1
            nonzero_corrections += "1" in bits
            reads.append({
                "decoder_id": decoder_id,
                "shot": shot,
                "correction_bits": bits
            })
            states[decoder_id] = "reset"
        else:
            raise ValueError(
                f"Invalid shot lifecycle at event {index}: {operation} after {state}"
            )
        request_ids[decoder_id].update(ids)
        requests[decoder_id] += len(ids)
        event_count += 1
    if not event_count:
        raise ValueError("Playback returned no events")
    if any(state != "reset" for state in states.values()):
        raise ValueError("Playback ended with an incomplete shot")
    if any(counts[key] != len(values) for key, values in expected.items()):
        raise ValueError(
            "Playback shot count does not match the correction oracle")
    return {
        "events": event_count,
        "requests": sum(requests.values()),
        "requests_per_decoder": {
            str(key): value for key, value in requests.items()
        },
        "shots": len(reads),
        "shots_per_decoder": {
            str(key): value for key, value in counts.items()
        },
        "nonzero_corrections": nonzero_corrections,
        "reads": reads
    }
