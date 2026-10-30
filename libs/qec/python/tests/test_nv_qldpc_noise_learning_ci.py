# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under    #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""End-to-end nv-qldpc TCBP model setup + noise learning on a stim-generated surface code (CI-sized, GPU).

Self-contained (stim only): a distance-3 rotated memory circuit with stim's four noise strengths gives the structure;
a "device" built with different strengths gives the detector data. Only ``H`` reaches ``init_tcbp_model`` / ``learn_noise``;
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


def memory_circuit(noise, rounds=3):
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  distance=3,
                                  rounds=rounds,
                                  **noise)


# Time-translation classes of the distance-3 memory circuit by number of
# rounds, as `tying.round_classes` (boundary_layers=2) finds them. With 3 or 4
# rounds every round is a boundary round and nothing is tied (one class per
# column); from 6 rounds on the bulk repeats and the count stays at 412.
TIED_ROUNDS = 6
TIED_CLASSES = 412


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
    tcbp_model, report = nv_qldpc.init_tcbp_model(
        H,
        seed=1,
        steps=3000,
        shots=65536,
        device_memory_budget_bytes=2 << 30,
        return_report=True)
    return dict(structure=structure,
                dem=dem,
                H=H,
                dets=dets,
                truth=truth_rates,
                nominal=nominal_rates,
                tcbp_model=tcbp_model,
                model_report=report)


@pytest.fixture(scope="module")
def setup_tied():
    """Longer memory (TIED_ROUNDS rounds): the fixture for the tied mode,
    where the 3-round circuit would tie nothing."""
    structure = memory_circuit(NOMINAL, TIED_ROUNDS)
    dem = tying.circuit_dem(structure)
    H, _, nominal_rates = tying.dem_arrays(dem)
    device = memory_circuit(HIDDEN, TIED_ROUNDS)
    _, _, truth_rates = tying.dem_arrays(tying.circuit_dem(device))
    dets = device.compile_detector_sampler(seed=5).sample(131072).astype(
        np.uint8)
    tcbp_model = nv_qldpc.init_tcbp_model(H,
                                          seed=1,
                                          steps=3000,
                                          shots=65536,
                                          device_memory_budget_bytes=2 << 30)
    return dict(structure=structure,
                H=H,
                dets=dets,
                truth=truth_rates,
                nominal=nominal_rates,
                tcbp_model=tcbp_model)


def median_log_err(rates, truth):
    return float(np.median(np.abs(np.log(rates / truth))))


def test_tcbp_model_report(setup):
    rep = setup["model_report"]
    assert rep["converged"]
    assert not rep["prob_regression_flag"]
    assert rep["bce_model"] < rep["bce_baseline"]
    ins = nv_qldpc.inspect(setup["tcbp_model"])
    assert ins["n"] == setup["H"].shape[1]
    assert 1e-4 <= ins["training_rate_min"] < ins["training_rate_max"] <= 1e-2


def test_per_mechanism_beats_nominal(setup):
    rates, report = nv_qldpc.learn_noise(setup["H"], setup["dets"],
                                         setup["tcbp_model"])
    assert rates.shape == (setup["H"].shape[1],)
    assert np.all(rates > 0)
    e_learned = median_log_err(rates, setup["truth"])
    e_nominal = median_log_err(setup["nominal"], setup["truth"])
    assert e_learned < 0.5 * e_nominal, (e_learned, e_nominal)
    assert report["passes_run"] >= 1
    assert "start_rate" in report and report["start_rate"] > 0


def test_tied_beats_nominal(setup_tied):
    s = setup_tied
    n = s["H"].shape[1]
    classes = tying.round_classes(s["structure"])
    # a known count, not the input echoed back: real tying on this circuit
    assert classes.n_untied == 0
    assert classes.n_classes == TIED_CLASSES < n
    assert len(np.unique(classes.class_ids)) == TIED_CLASSES
    rates, report = nv_qldpc.learn_noise(s["H"],
                                         s["dets"],
                                         s["tcbp_model"],
                                         mode="tied",
                                         class_ids=classes.class_ids)
    assert report["n_classes"] == TIED_CLASSES
    assert rates.shape == (n,)
    # tied mode returns one rate per class: members agree exactly
    for cid in range(TIED_CLASSES):
        member = rates[classes.class_ids == cid]
        assert member.size >= 1 and np.ptp(member) == 0.0, cid
    e_learned = median_log_err(rates, s["truth"])
    e_nominal = median_log_err(s["nominal"], s["truth"])
    assert e_learned < 0.5 * e_nominal, (e_learned, e_nominal)


def test_parametric_by_gate_recovers_hidden_strengths(setup):
    design = p25.design_map_by_gate(setup["structure"], setup["dem"])
    rates, report = nv_qldpc.learn_noise(setup["H"],
                                         setup["dets"],
                                         setup["tcbp_model"],
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


def test_sparse_h_same_rates(setup):
    """A scipy sparse H learns the same rates as the dense one and is not
    densified on the way."""
    sp = pytest.importorskip("scipy.sparse")

    class NoDense(sp.csr_matrix):

        def toarray(self, *a, **k):
            raise AssertionError("sparse H was densified")

        todense = toarray

    dets = setup["dets"][:65536]
    dense, _ = nv_qldpc.learn_noise(setup["H"],
                                    dets,
                                    setup["tcbp_model"],
                                    seed=7)
    sparse, _ = nv_qldpc.learn_noise(NoDense(setup["H"]),
                                     dets,
                                     setup["tcbp_model"],
                                     seed=7)
    np.testing.assert_array_equal(sparse, dense)
