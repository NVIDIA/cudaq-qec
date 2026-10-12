# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""End-to-end nv-qldpc calibration + noise learning on a stim-generated surface code (CI-sized, GPU).

Self-contained (stim only): a distance-3 rotated memory circuit with stim's four noise strengths gives the structure;
a "device" built with different strengths gives the detector data. Only ``H`` reaches ``calibrate`` / ``learn_noise``;
the device's rates are used to score the result. The parametric mode uses ``design_map_by_gate`` (one parameter per
distinct noise instruction), so the hidden parameters are the four stim strengths.
"""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
nv_qldpc = pytest.importorskip("cudaq_qec.plugins.nv_qldpc")
try:
    nv_qldpc.inspect(b"")  # loads the plugin library; empty bytes are rejected
except ValueError:
    pass  # library loaded
except (FileNotFoundError, OSError) as e:
    pytest.skip(f"nv-qldpc-decoder plugin library not available: {e}",
                allow_module_level=True)
from cudaq_qec.plugins.nv_qldpc import p25, tying  # noqa: E402

# Four distinct nominal strengths (so each noise instruction maps to its own hidden value) and the device's.
NOMINAL = dict(after_clifford_depolarization=2e-3,
               before_round_data_depolarization=1e-3,
               before_measure_flip_probability=3e-3,
               after_reset_flip_probability=2.5e-3)
HIDDEN = dict(after_clifford_depolarization=3.2e-3,
              before_round_data_depolarization=0.6e-3,
              before_measure_flip_probability=1.8e-3,
              after_reset_flip_probability=3.9e-3)


def memory_circuit(noise):
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  distance=3,
                                  rounds=3,
                                  **noise)


@pytest.fixture(scope="module")
def setup():
    structure = memory_circuit(NOMINAL)
    dem = tying.circuit_dem(structure)
    H, _, _ = tying.dem_arrays(dem)
    device = memory_circuit(HIDDEN)
    _, _, truth_rates = tying.dem_arrays(tying.circuit_dem(device))
    _, _, nominal_rates = tying.dem_arrays(dem)
    dets = device.compile_detector_sampler(seed=5).sample(262144).astype(
        np.uint8)
    try:
        calibration, report = nv_qldpc.calibrate(
            H,
            seed=1,
            steps=3000,
            shots=65536,
            device_memory_budget_bytes=2 << 30,
            return_report=True)
    except ValueError as e:  # no usable GPU (CPU-only wheel test job)
        if "CUDA" in str(e):
            pytest.skip(f"GPU not available: {e}")
        raise
    return dict(structure=structure,
                dem=dem,
                H=H,
                dets=dets,
                truth=truth_rates,
                nominal=nominal_rates,
                calibration=calibration,
                cal_report=report)


def median_log_err(rates, truth):
    return float(np.median(np.abs(np.log(rates / truth))))


def test_calibration_report(setup):
    rep = setup["cal_report"]
    assert rep["converged"]
    assert not rep["prob_regression_flag"]
    assert rep["bce_model"] < rep["bce_baseline"]
    ins = nv_qldpc.inspect(setup["calibration"])
    assert ins["n"] == setup["H"].shape[1]
    assert 1e-4 <= ins["training_rate_min"] < ins["training_rate_max"] <= 1e-2


def test_per_mechanism_beats_nominal(setup):
    rates, report = nv_qldpc.learn_noise(setup["H"], setup["dets"],
                                         setup["calibration"])
    assert rates.shape == (setup["H"].shape[1],)
    assert np.all(rates > 0)
    e_learned = median_log_err(rates, setup["truth"])
    e_nominal = median_log_err(setup["nominal"], setup["truth"])
    assert e_learned < 0.5 * e_nominal, (e_learned, e_nominal)
    assert report["passes_run"] >= 1
    assert "start_rate" in report and report["start_rate"] > 0


def test_tied_beats_nominal(setup):
    classes = tying.round_classes(setup["structure"])
    rates, report = nv_qldpc.learn_noise(setup["H"],
                                         setup["dets"],
                                         setup["calibration"],
                                         mode="tied",
                                         class_ids=classes.class_ids)
    assert report["n_classes"] == classes.n_classes
    e_learned = median_log_err(rates, setup["truth"])
    e_nominal = median_log_err(setup["nominal"], setup["truth"])
    assert e_learned < 0.5 * e_nominal, (e_learned, e_nominal)


def test_parametric_by_gate_recovers_hidden_strengths(setup):
    design = p25.design_map_by_gate(setup["structure"], setup["dem"])
    rates, report = nv_qldpc.learn_noise(setup["H"],
                                         setup["dets"],
                                         setup["calibration"],
                                         mode="parametric",
                                         design=design,
                                         passes=12,
                                         tol=0.02)
    q = np.asarray(report["q_hat"])
    assert report["n_params"] == design["n_params"] == len(q)
    # design_map_by_gate names parameters "<GATE> (<args>)" with the structure circuit's own probability as the args;
    # the hidden strength is the device circuit's value for the same instruction.
    nominal_to_hidden = {NOMINAL[k]: HIDDEN[k] for k in NOMINAL}
    checked = 0
    for name, value, ident in zip(design["names"], q, report["q_identifiable"]):
        if not ident:
            continue
        nominal_p = float(name.split("(")[1].rstrip(")").split(",")[0])
        truth = nominal_to_hidden[round(nominal_p, 12)]
        assert abs(np.log(value / truth)) < 0.25, (name, value, truth)
        checked += 1
    assert checked >= 2
    e_learned = median_log_err(rates, setup["truth"])
    e_nominal = median_log_err(setup["nominal"], setup["truth"])
    assert e_learned < 0.5 * e_nominal, (e_learned, e_nominal)
