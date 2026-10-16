# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
import pathlib
import pytest

# Force stim as the default simulator for emulation
# Note: this must be done before importing `cudaq`
os.environ["CUDAQ_DEFAULT_SIMULATOR"] = "stim"

import cudaq
import cudaq_qec as qec

from surface_code_1 import run

CASES = [
    pytest.param(
        {
            "distance": 3,
            "p_cnot": 0.01,
            "num_rounds": 3,
            "num_shots": 200,
            "target": "stim",
            "decoder_type": "multi_error_lut",
            "number_of_non_zero_values_threshold": 30,
            "number_of_corrections_decoder_threshold": 4
        },
        id="d3-local"),
    # This must be disabled for now because the multi_error_lut decoder is not
    # powerful enough to pass this test. The nv-qldpc-decoder can pass this test,
    # but that is not available on the GitHub repo.
    # pytest.param(
    #     {
    #         "distance": 5,
    #         "p_cnot": 0.01,
    #         "num_rounds": 20,
    #         "num_shots": 1000,
    #         "target": "stim",
    #         "decoder_type": "multi_error_lut",
    #         "number_of_non_zero_values_threshold": 40,
    #         "number_of_corrections_decoder_threshold": 40
    #     },
    #     id="d5-local",
    # ),
    pytest.param(
        {
            "distance": 3,
            "p_cnot": 0.01,
            "num_rounds": 3,
            "num_shots": 100,
            "decoder_type": "multi_error_lut",
            "target": "quantinuum",
            "emulate": True,
            "machine_name": "Helios-1Dummy",
            "number_of_non_zero_values_threshold": 0,
            "number_of_corrections_decoder_threshold": 0
        },
        id="d3-quantinuum-emulate-in-process"),
    pytest.param(
        {
            "distance": 5,
            "p_cnot": 0.01,
            "num_rounds": 5,
            "num_shots": 100,
            "decoder_type": "multi_error_lut",
            "target": "quantinuum",
            "emulate": True,
            "machine_name": "Helios-1Dummy",
            "number_of_non_zero_values_threshold": 0,
            "number_of_corrections_decoder_threshold": 0
        },
        id="d5-quantinuum-emulate-in-process",
    ),
]


# Fixtures
@pytest.fixture(scope="module", params=CASES)
def case(request):
    return request.param


@pytest.fixture(autouse=True)
def reset_cudaq_target():
    cudaq.set_target("stim")


@pytest.fixture(scope="module")
def dem_file(case, tmp_path_factory):
    d = tmp_path_factory.mktemp(f"dem_d{case['distance']}")
    dem_path = d / f"dem_d{case['distance']}.yaml"
    run([
        "--distance",
        str(case["distance"]),
        "--num_rounds",
        str(case["num_rounds"]),
        "--p_cnot",
        str(case.get("p_cnot", 0.001)),
        "--decoder_type",
        case["decoder_type"],
        "--save_dem",
        str(dem_path),
    ])
    assert dem_path.exists() and dem_path.stat().st_size > 0
    return dem_path


def test_run_from_dem(case, dem_file):
    argv = [
        "--distance",
        str(case["distance"]),
        "--num_rounds",
        str(case["num_rounds"]),
        "--p_cnot",
        str(case.get("p_cnot", 0.001)),
        "--num_shots",
        str(case["num_shots"]),
        "--decoder_type",
        case["decoder_type"],
        "--load_dem",
        str(dem_file),
    ]
    if "target" in case:
        argv += ["--target", case["target"]]
    if case.get("emulate", False):
        argv += ["--emulate"]
    if "machine_name" in case:
        argv += ["--machine_name", case["machine_name"]]

    result = run(argv)
    qec.finalize_decoders()
    assert result["num_non_zero"] <= case["number_of_non_zero_values_threshold"]
    assert result["num_corrections"] >= case[
        "number_of_corrections_decoder_threshold"]


def test_build_dem_with_zero_p_cnot_raises(case, tmp_path_factory):
    d = tmp_path_factory.mktemp(f"zero_p_d{case['distance']}")
    dem_path = d / "bad.yaml"
    with pytest.raises(RuntimeError,
                       match="Cannot build a DEM with p_cnot = 0.0"):
        run([
            "--distance",
            str(case["distance"]),
            "--num_rounds",
            str(case["num_rounds"]),
            "--p_cnot",
            "0.0",
            "--save_dem",
            str(dem_path),
        ])


def test_quantinuum_requires_machine_name(case, dem_file):
    with pytest.raises(
            RuntimeError,
            match="machine_name must be set when target is quantinuum"):
        run([
            "--distance",
            str(case["distance"]),
            "--num_rounds",
            str(case["num_rounds"]),
            "--num_shots",
            "1",
            "--load_dem",
            str(dem_file),
            "--target",
            "quantinuum",
            # no --machine_name → should fail
        ])


def test_quantinuum_requires_project_id_remote(case, dem_file):
    with pytest.raises(RuntimeError):
        run([
            "--distance",
            str(case["distance"]),
            "--num_rounds",
            str(case["num_rounds"]),
            "--num_shots",
            "1",
            "--load_dem",
            str(dem_file),
            "--target",
            "quantinuum",
            "--machine_name",
            "Helios-1SC",
            # no --project_id → should fail
        ])
