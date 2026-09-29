# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import cudaq_qec as qec


def _reference_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "test_data" / "chromobius"


def _read_dem() -> str:
    return (_reference_dir() /
            "basic_reference.dem").read_text(encoding="utf-8")


def _parse_bits(bits: str) -> list[float]:
    values = []
    for bit in bits.strip():
        if bit not in "01":
            raise ValueError(f"Invalid reference bit: {bit!r}")
        values.append(float(bit == "1"))
    return values


def _read_shots() -> list[tuple[str, list[float], list[float]]]:
    shots = []
    path = _reference_dir() / "basic_reference.tsv"
    for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        columns = line.split()
        if len(columns) != 3:
            raise ValueError(
                f"Malformed reference row {path}:{line_no}: {line}")
        name, syndrome_bits, expected_bits = columns
        shots.append(
            (name, _parse_bits(syndrome_bits), _parse_bits(expected_bits)))
    return shots


def _as_list(values):
    return values.tolist() if hasattr(values, "tolist") else list(values)


def _make_decoder():
    dem = _read_dem()
    shots = _read_shots()
    return qec.get_decoder("chromobius", dem), shots


def test_chromobius_matches_upstream_predict_reference():
    decoder, shots = _make_decoder()
    for name, syndrome, expected in shots:
        result = decoder.decode(syndrome)
        assert result.converged, name
        actual = _as_list(result.result)
        assert actual == expected, (name, actual, expected)


def test_chromobius_batch_matches_upstream_predict_reference():
    decoder, shots = _make_decoder()
    batch_results = decoder.decode_batch([syndrome for _, syndrome, _ in shots])
    for batch_result, (name, _, expected) in zip(batch_results, shots):
        assert batch_result.converged, name
        actual = _as_list(batch_result.result)
        assert actual == expected, (name, actual, expected)
