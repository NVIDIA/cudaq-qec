# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named option presets for the nv-qldpc decoder (cudaq_qec.plugins.nv_qldpc.presets).

The dict-level tests run anywhere. Construction and decoding need the
plugin library and a GPU, and a TCBP model for the tcbp* presets; those skip
when the plugin is absent.
"""
import shutil
import subprocess

import numpy as np
import pytest

qec = pytest.importorskip("cudaq_qec")
nv_qldpc = pytest.importorskip("cudaq_qec.plugins.nv_qldpc")
from cudaq_qec.plugins.nv_qldpc import presets  # noqa: E402

TCBP = sorted(presets.REQUIRES_TCBP_MODEL)


def decoder(H, name, **kw):
    return qec.get_decoder("nv-qldpc-decoder", H, **presets.preset(name, **kw))


# ---------------------------------------------------------------------------
# Dict level: no plugin, no GPU
# ---------------------------------------------------------------------------
def test_exports_and_derived_sets():
    assert nv_qldpc.PRESETS is presets.PRESETS
    assert nv_qldpc.preset is presets.preset
    assert not hasattr(presets, "get_decoder")
    assert presets.names() == list(presets.PRESETS)
    # The tcbp* presets read a model; only the -gap preset needs observables.
    assert presets.REQUIRES_TCBP_MODEL == {
        n for n in presets.PRESETS if n.startswith("tcbp")
    }
    assert presets.REQUIRES_OBSERVABLES == {"tcbp-jobs-sc-gap"}
    for p in presets.PRESETS.values():
        assert "error_rate_vec" not in p and "error_rate" not in p


def test_preset_copies_and_overrides():
    a = presets.preset("relay-bp")
    a["srelay_config"]["num_sets"] = 1
    a["gamma_dist"].append(0.0)
    assert presets.PRESETS["relay-bp"]["srelay_config"]["num_sets"] == 300
    assert presets.PRESETS["relay-bp"]["gamma_dist"] == [-0.24, 0.66]
    b = presets.preset("bp60", max_iterations=50, bp_batch_size=64)
    assert b["max_iterations"] == 50 and b["bp_batch_size"] == 64
    assert presets.PRESETS["bp60"]["max_iterations"] == 60
    g = presets.preset("tcbp-jobs-sc-gap", tcbp_model=b"\x00")
    g["opt_results"]["jobs_gap"] = False
    assert presets.PRESETS["tcbp-jobs-sc-gap"]["opt_results"]["jobs_gap"]
    with pytest.raises(KeyError, match="unknown nv-qldpc preset 'nope'"):
        presets.preset("nope")


def test_tcbp_model_override_forms(tmp_path):
    raw = bytes(range(16))
    a = presets.preset("tcbp", tcbp_model=raw)["tcbp_model"]
    assert a.dtype == np.uint8 and a.flags.writeable
    assert a.tolist() == list(raw)
    b = presets.preset("tcbp", tcbp_model=bytearray(raw))["tcbp_model"]
    assert b.tolist() == list(raw)
    ro = np.frombuffer(raw, dtype=np.uint8)
    c = presets.preset("tcbp-jobs", tcbp_model=ro)["tcbp_model"]
    assert c.flags.writeable and c.tolist() == list(raw)
    path = tmp_path / "m.tcbp"
    path.write_bytes(raw)
    assert presets.preset("tcbp", tcbp_model=path)["tcbp_model"] == str(path)
    assert presets.preset("tcbp",
                          tcbp_model=str(path))["tcbp_model"] == str(path)
    for name in TCBP:
        with pytest.raises(ValueError, match="tcbp_model"):
            presets.preset(name)
        # An explicit None is "not given", not a model to convert.
        with pytest.raises(ValueError, match="tcbp_model"):
            presets.preset(name, tcbp_model=None)
    assert "tcbp_model" not in presets.preset("bp60", tcbp_model=None)
    for name in presets.names():
        if name not in TCBP:
            assert "tcbp_model" not in presets.preset(name)


# ---------------------------------------------------------------------------
# Construction and decoding: plugin + GPU
# ---------------------------------------------------------------------------
def _plugin_available():
    try:
        nv_qldpc.inspect(b"")
    except ValueError:
        return True
    except (FileNotFoundError, OSError):
        return False
    return True


def _gpu_available():
    # Matches test_nv_qldpc_tcbp.py: the plugin and init_tcbp_model are
    # GPU-only, and the CPU wheel-test job ships the plugin without a GPU.
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi"], capture_output=True).returncode == 0


needs_plugin = pytest.mark.skipif(
    not (_plugin_available() and _gpu_available()),
    reason="nv-qldpc-decoder plugin library and a GPU are required")


@pytest.fixture(scope="module")
def code():
    stim = pytest.importorskip("stim")
    tying = nv_qldpc.tying
    circuit = stim.Circuit.generated("surface_code:rotated_memory_z",
                                     distance=3,
                                     rounds=3,
                                     after_clifford_depolarization=0.003,
                                     before_round_data_depolarization=0.003,
                                     before_measure_flip_probability=0.003,
                                     after_reset_flip_probability=0.003)
    dem = circuit.detector_error_model(flatten_loops=True)
    H, O, rates = tying.dem_arrays(dem)
    rng = np.random.default_rng(7)
    E = (rng.random((256, H.shape[1])) < rates[None, :]).astype(np.uint8)
    S = ((E @ H.T) % 2).astype(np.uint8)
    return dict(H=H, O=O, rates=rates, E=E, S=S)


@pytest.fixture(scope="module")
def tcbp_model(code):
    if not _plugin_available():
        pytest.skip("nv-qldpc-decoder plugin library not available")
    return nv_qldpc.init_tcbp_model(code["H"],
                                    seed=1,
                                    steps=300,
                                    shots=8192,
                                    device_memory_budget_bytes=1 << 30)


def _model_form(tcbp_model, form, tmp_path):
    if form == "bytes":
        return tcbp_model
    if form == "array":
        return np.frombuffer(tcbp_model, dtype=np.uint8)  # read-only view
    p = tmp_path / "model.tcbp"
    p.write_bytes(tcbp_model)
    return p


def _check_corrections(H, S, res, all_shots):
    """Shape checks, then syndrome consistency of the hard correction: for
    every shot when a post-processor (OSD, JOBS) is on, else for the shots
    BP converged on. `converged` reports BP convergence, not OSD success."""
    result = np.asarray(res.result)
    converged = np.asarray(res.converged, dtype=bool)
    assert result.shape == (S.shape[0], H.shape[1])
    assert converged.shape == (S.shape[0],)
    e = (result > 0.5).astype(np.uint8)
    ok = ((e @ H.T) % 2 == S).all(axis=1)
    if all_shots:
        assert ok.all()
    else:
        assert ok[converged].all()


@needs_plugin
@pytest.mark.parametrize(
    "name",
    [n for n in presets.names() if n.startswith(("bp60", "bp10-minllr"))])
def test_bp_presets_construct_and_decode(code, name):
    dec = decoder(code["H"],
                  name,
                  error_rate_vec=code["rates"],
                  bp_batch_size=64)
    _check_corrections(code["H"],
                       code["S"],
                       dec.decode_batch(code["S"]),
                       all_shots=name != "bp60")


@needs_plugin
def test_relay_bp_preset_constructs_and_decodes(code):
    dec = decoder(code["H"],
                  "relay-bp",
                  error_rate_vec=code["rates"],
                  bp_batch_size=64)
    _check_corrections(code["H"],
                       code["S"],
                       dec.decode_batch(code["S"]),
                       all_shots=False)


@needs_plugin
@pytest.mark.parametrize("form", ["bytes", "array", "path"])
def test_tcbp_preset_is_a_soft_predecoder(code, tcbp_model, tmp_path, form):
    dec = decoder(code["H"],
                  "tcbp",
                  tcbp_model=_model_form(tcbp_model, form, tmp_path),
                  error_rate_vec=code["rates"],
                  bp_batch_size=64)
    res = dec.decode_batch(code["S"])
    result = np.asarray(res.result, dtype=np.float64)
    assert result.shape == (code["S"].shape[0], code["H"].shape[1])
    assert np.all(result >= 0.0) and np.all(result <= 1.0)
    # Probabilities, not hard decisions: some values are strictly inside (0, 1).
    assert np.any((result > 1e-6) & (result < 1 - 1e-6))


@needs_plugin
@pytest.mark.parametrize("name",
                         ["tcbp-osd-cs1000", "tcbp-jobs", "tcbp-jobs-sc"])
@pytest.mark.parametrize("form", ["bytes", "array", "path"])
def test_tcbp_osd_presets_construct_and_decode(code, tcbp_model, tmp_path, name,
                                               form):
    dec = decoder(code["H"],
                  name,
                  tcbp_model=_model_form(tcbp_model, form, tmp_path),
                  error_rate_vec=code["rates"],
                  bp_batch_size=64)
    _check_corrections(code["H"],
                       code["S"],
                       dec.decode_batch(code["S"]),
                       all_shots=True)


@needs_plugin
def test_tcbp_jobs_sc_gap_returns_gaps(code, tcbp_model):
    dec = decoder(code["H"],
                  "tcbp-jobs-sc-gap",
                  O=code["O"],
                  tcbp_model=tcbp_model,
                  error_rate_vec=code["rates"],
                  bp_batch_size=64)
    res = dec.decode_batch(code["S"])
    _check_corrections(code["H"], code["S"], res, all_shots=True)
    gaps = np.array([o["jobs_gap"] for o in res.opt_results], dtype=float)
    classes = np.array([o["jobs_gap_class"] for o in res.opt_results])
    assert gaps.shape == classes.shape == (code["S"].shape[0],)
    assert np.all(gaps >= 0.0)  # +inf where the search saw no competitor
    assert np.all((classes == -1) == np.isinf(gaps))
    # sc: every shot is searched, so a competitor is seen for most shots.
    assert np.isfinite(gaps).any()


@needs_plugin
def test_overrides_reach_the_decoder(code):
    # The documented tuning override: window OSD to 4 x K* columns.
    dec = decoder(code["H"],
                  "bp10-minllr-osd-cs1000",
                  error_rate_vec=code["rates"],
                  bp_batch_size=64,
                  osd_sparsify=True,
                  osd_sparsify_pad=4.0)
    _check_corrections(code["H"],
                       code["S"],
                       dec.decode_batch(code["S"]),
                       all_shots=True)
    # An override the plugin rejects: osd_method=4 needs use_sparsity=True.
    with pytest.raises(Exception, match="(?i)spars"):
        decoder(code["H"],
                "bp10-minllr-osd-cs1000",
                error_rate_vec=code["rates"],
                osd_method=4,
                use_sparsity=False)
