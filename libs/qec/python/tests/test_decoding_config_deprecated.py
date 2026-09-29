# SPDX-FileCopyrightText: Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the deprecated typed decoder-config classes.

The typed config classes (nv_qldpc_decoder_config and friends) are deprecated
dict-backed shims over the schema-driven interface (see cudaq_qec/_compat.py).
This module proves they still behave like the originals: the bulk of it is the
pre-schema typed-config test suite restored verbatim, plus shim-specific tests
(deprecation warnings, dict equivalence). The only edits to the restored tests
are marked inline: reading decoder_config.decoder_custom_args now returns a
plain dict, never a typed object. Delete this file together with _compat.py
when the deprecation period ends.
"""

import math

import numpy as np
import pytest

import cudaq_qec as qec

# The deprecated classes warn on every construction; that is asserted once in
# test_deprecated_typed_configs_warn_on_construction and silenced everywhere
# else so the restored tests run unmodified.
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")

trt_schema_missing = qec.decoder_param_schema("trt_decoder") is None
chromobius_schema_missing = qec.decoder_param_schema("chromobius") is None
pymatching_schema_missing = qec.decoder_param_schema("pymatching") is None
nv_qldpc_schema_missing = qec.decoder_param_schema("nv-qldpc-decoder") is None


def is_nv_qldpc_decoder_available():
    """
    Helper function to check if the NV-QLDPC decoder is available.
    """
    try:
        H_list = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1],
                  [0, 0, 1, 0, 1, 1, 1]]
        H_np = np.array(H_list, dtype=np.uint8)
        qec.get_decoder("nv-qldpc-decoder", H_np)
        return True
    except Exception:
        return False


# Shim-specific tests (not part of the restored pre-schema suite)


def make_nv_qldpc_decoder_config(id=0):
    dc = qec.decoder_config()
    dc.id = id
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    return dc


def test_deprecated_typed_configs_warn_on_construction():
    for cls in (qec.nv_qldpc_decoder_config, qec.multi_error_lut_config,
                qec.trt_decoder_config, qec.pymatching_config,
                qec.chromobius_config, qec.qecrt.config.srelay_bp_config,
                qec.qecrt.config.single_error_lut_config,
                qec.qecrt.config.sliding_window_config):
        with pytest.warns(DeprecationWarning):
            cls()


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_deprecated_config_matches_dict_built_yaml():
    cfg = qec.nv_qldpc_decoder_config()
    cfg.use_sparsity = True
    cfg.error_rate = 0.01
    cfg.max_iterations = 50
    cfg.bp_seed = -1
    cfg.srelay_config = qec.qecrt.config.srelay_bp_config()
    cfg.srelay_config.pre_iter = 5
    cfg.srelay_config.stopping_criterion = "NConv"

    old_style = make_nv_qldpc_decoder_config()
    old_style.set_decoder_custom_args(cfg)

    new_style = make_nv_qldpc_decoder_config()
    new_style.decoder_custom_args = {
        "use_sparsity": True,
        "error_rate": 0.01,
        "max_iterations": 50,
        "bp_seed": -1,
        "srelay_config": {
            "pre_iter": 5,
            "stopping_criterion": "NConv",
        },
    }

    assert old_style.to_yaml_str() == new_style.to_yaml_str()
    assert old_style == new_style


def test_deprecated_config_assignable_to_property():
    cfg = qec.pymatching_config()
    cfg.error_rate_vec = [0.1, 0.2, 0.3]
    cfg.merge_strategy = "smallest_weight"

    dc = qec.decoder_config()
    # The shim carries its own schema name, so conversion works even before
    # dc.type is assigned.
    dc.decoder_custom_args = cfg
    dc.type = "pymatching"
    args = dc.decoder_custom_args
    # Model rates are top-level decoder_config state, so the shim promotes
    # them out of the per-decoder parameters instead of dropping them.
    assert "error_rate_vec" not in args
    assert list(dc.error_rate_vec) == [0.1, 0.2, 0.3]
    assert args["merge_strategy"] == "smallest_weight"
    dc.validate_custom_args()


def test_deprecated_config_unset_fields_read_as_none_and_are_omitted():
    cfg = qec.nv_qldpc_decoder_config()
    assert cfg.max_iterations is None
    cfg.max_iterations = 50
    assert cfg.max_iterations == 50
    cfg.max_iterations = None  # clears, like the old std::optional fields
    assert cfg.to_heterogeneous_map() == {}


def test_deprecated_config_rejects_unknown_attributes():
    cfg = qec.pymatching_config()
    with pytest.raises(AttributeError):
        cfg.merge_stratgey = "smallest_weight"
    with pytest.raises(AttributeError):
        _ = cfg.merge_stratgey


def test_deprecated_sliding_window_inner_params_collapse():
    cfg = qec.qecrt.config.sliding_window_config()
    cfg.window_size = 3
    cfg.step_size = 1
    cfg.error_rate_vec = [0.1, 0.2, 0.3]
    cfg.inner_decoder_name = "multi_error_lut"
    cfg.multi_error_lut_params = qec.multi_error_lut_config()
    cfg.multi_error_lut_params.lut_error_depth = 2

    assert cfg.to_heterogeneous_map() == {
        "window_size": 3,
        "step_size": 1,
        "error_rate_vec": [0.1, 0.2, 0.3],
        "inner_decoder_name": "multi_error_lut",
        "inner_decoder_params": {
            "lut_error_depth": 2
        },
    }

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "sliding_window"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.set_decoder_custom_args(cfg)
    dc.validate_custom_args()

    round_tripped = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    args = round_tripped.decoder_custom_args
    assert args["inner_decoder_params"]["lut_error_depth"] == 2


def test_deprecated_config_from_heterogeneous_map_round_trip():
    source = {
        "window_size": 3,
        "error_rate_vec": [0.1, 0.2, 0.3],
        "inner_decoder_name": "multi_error_lut",
        "inner_decoder_params": {
            "lut_error_depth": 2
        },
    }
    cfg = qec.qecrt.config.sliding_window_config.from_heterogeneous_map(source)
    assert cfg.window_size == 3
    assert cfg.inner_decoder_name == "multi_error_lut"
    assert cfg.multi_error_lut_params.lut_error_depth == 2
    assert cfg.to_heterogeneous_map() == source


def test_deprecated_config_schema_validation_still_applies():
    cfg = qec.qecrt.config.sliding_window_config()
    cfg.window_size = 2
    cfg.step_size = 5  # step_size > window_size: schema validate hook rejects
    cfg.error_rate_vec = [0.1]
    cfg.inner_decoder_name = "single_error_lut"

    dc = qec.decoder_config()
    dc.type = "sliding_window"
    dc.set_decoder_custom_args(cfg)
    with pytest.raises(RuntimeError, match="step_size"):
        dc.validate_custom_args()


# ---------------------------------------------------------------------------
# The pre-schema typed-config test suite, restored verbatim (except where
# marked) from the version this branch removed.
# ---------------------------------------------------------------------------

# nv_qldpc_decoder_config tests

FIELDS = {
    "use_sparsity": (bool, True, False),
    "error_rate": (float, 1e-3, 5e-2),
    "error_rate_vec": (list, [0.01, 0.02, 0.03], [0.2, 0.1]),
    "max_iterations": (int, 25, 50),
    "n_threads": (int, 4, 8),
    "use_osd": (bool, False, True),
    "osd_method": (int, 1, 2),
    "osd_order": (int, 7, 3),
    "bp_batch_size": (int, 64, 128),
    "osd_batch_size": (int, 16, 32),
    "iter_per_check": (int, 2, 3),
    "clip_value": (float, 10.0, 7.5),
    "bp_method": (int, 0, 1),
    "scale_factor": (float, 0.5, 1.25),
    "proc_float": (str, "fp32", "fp64"),
}


def test_nv_qldpc_decoder_config_defaults_are_none():
    nv = qec.nv_qldpc_decoder_config()
    for name in FIELDS:
        assert getattr(nv, name) is None, f"Expected {name} to default to None"


@pytest.mark.parametrize("name, meta", list(FIELDS.items()))
def test_nv_qldpc_decoder_config_set_and_get_each_optional(name, meta):
    nv = qec.nv_qldpc_decoder_config()

    py_type, sample_val, alt_val = meta

    # Initially None
    assert getattr(nv, name) is None

    # Set to a valid value and get back
    setattr(nv, name, sample_val)
    got = getattr(nv, name)
    if py_type is float:
        assert isinstance(got, float)
        assert math.isclose(got, float(sample_val), rel_tol=1e-12, abs_tol=0.0)
    elif py_type is list:
        assert isinstance(got, list)
        assert all(isinstance(x, float)
                   for x in got), f"{name} must be a list of float"
        assert got == sample_val
    else:
        assert isinstance(got, py_type)
        assert got == sample_val

    # Change to an alternate valid value
    setattr(nv, name, alt_val)
    got2 = getattr(nv, name)
    if py_type is float:
        assert math.isclose(got2, float(alt_val), rel_tol=1e-12, abs_tol=0.0)
    else:
        assert got2 == alt_val

    # Set value to None
    setattr(nv, name, None)
    assert getattr(nv, name) is None


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_nv_qldpc_decoder_config_setting_wrong_types_raises_typeerror():
    nv = qec.nv_qldpc_decoder_config()

    with pytest.raises(TypeError):
        nv.max_iterations = "ten"

    with pytest.raises(TypeError):
        nv.use_sparsity = "True"

    with pytest.raises(TypeError):
        nv.error_rate = "0.1"

    with pytest.raises(TypeError):
        nv.error_rate_vec = [0.1, "nope", 0.3]

    with pytest.raises(TypeError):
        nv.error_rate_vec = 3.14


def test_nv_qldpc_decoder_config_error_rate_vec_accepts_python_list_of_float():
    nv = qec.nv_qldpc_decoder_config()

    vals = [0.0, 0.125, 0.25]
    nv.error_rate_vec = vals
    got = nv.error_rate_vec
    assert isinstance(got, list)
    assert all(isinstance(x, float) for x in got)
    assert got == vals


def test_nv_qldpc_decoder_config_toggle_multiple_fields_and_clear():
    nv = qec.nv_qldpc_decoder_config()

    nv.use_sparsity = True
    nv.error_rate = 0.0123
    nv.error_rate_vec = [0.1, 0.2, 0.3]
    nv.max_iterations = 100
    nv.n_threads = 8
    nv.use_osd = True
    nv.osd_method = 2
    nv.osd_order = 4
    nv.bp_batch_size = 32
    nv.osd_batch_size = 16
    nv.iter_per_check = 3
    nv.clip_value = 7.5
    nv.bp_method = 1
    nv.scale_factor = 0.8
    nv.proc_float = "fp64"

    assert nv is not None
    assert nv.use_sparsity is True
    assert math.isclose(nv.error_rate, 0.0123)
    assert nv.error_rate_vec == [0.1, 0.2, 0.3]
    assert nv.max_iterations == 100
    assert nv.n_threads == 8

    nv.use_sparsity = None
    nv.error_rate = None
    nv.error_rate_vec = None
    nv.max_iterations = None
    nv.n_threads = None

    assert nv.use_sparsity is None
    assert nv.error_rate is None
    assert nv.error_rate_vec is None
    assert nv.max_iterations is None
    assert nv.n_threads is None


# multi_error_lut_config tests

FIELDS_MULTI_ERROR_LUT = {
    "lut_error_depth": (int, 1, 3),
}

# pymatching_config tests

FIELDS_PYMATCHING = {
    "error_rate_vec": (list, [0.1, 0.2, 0.3], [0.2, 0.1, 0.2]),
    "merge_strategy": (str, "smallest_weight", "disallow"),
}

FIELDS_CHROMOBIUS = {
    "drop_mobius_errors_involving_remnant_errors": (bool, True, False),
    "ignore_decomposition_failures": (bool, True, False),
    "include_coords_in_mobius_dem": (bool, True, False),
    "return_weight": (bool, True, False),
    "write_mobius_match_to_stderr": (bool, True, False),
}

# trt_decoder_config tests

FIELDS_TRT_DECODER = {
    "onnx_load_path": (str, "/path/to/model.onnx", "/other/path/model.onnx"),
    "engine_load_path": (str, "/path/to/engine.trt", "/other/engine.trt"),
    "engine_save_path": (str, "/path/to/save.trt", "/other/save.trt"),
    "precision": (str, "fp16", "fp32"),
    "memory_workspace": (int, 1073741824, 2147483648),  # 1GB, 2GB
}


def test_multi_error_lut_config_defaults_are_none():
    m = qec.multi_error_lut_config()
    for name in FIELDS_MULTI_ERROR_LUT:
        assert getattr(m, name) is None, f"Expected {name} to default to None"


def test_pymatching_config_defaults_are_none():
    pm = qec.pymatching_config()
    for name in FIELDS_PYMATCHING:
        assert getattr(pm, name) is None, f"Expected {name} to default to None"


def test_chromobius_config_defaults_are_none():
    chromobius = qec.chromobius_config()
    for name in FIELDS_CHROMOBIUS:
        assert getattr(chromobius, name) is None


@pytest.mark.parametrize("name, meta", list(FIELDS_PYMATCHING.items()))
def test_pymatching_config_set_and_get_each_optional(name, meta):
    pm = qec.pymatching_config()

    py_type, sample_val, alt_val = meta

    assert getattr(pm, name) is None

    setattr(pm, name, sample_val)
    got = getattr(pm, name)
    assert isinstance(got, py_type)
    assert got == sample_val

    setattr(pm, name, alt_val)
    got2 = getattr(pm, name)
    assert got2 == alt_val

    setattr(pm, name, None)
    assert getattr(pm, name) is None


@pytest.mark.parametrize("name, meta", list(FIELDS_CHROMOBIUS.items()))
def test_chromobius_config_set_and_get_each_optional(name, meta):
    chromobius = qec.chromobius_config()

    py_type, sample_val, alt_val = meta

    assert getattr(chromobius, name) is None

    setattr(chromobius, name, sample_val)
    got = getattr(chromobius, name)
    assert isinstance(got, py_type)
    assert got == sample_val

    setattr(chromobius, name, alt_val)
    got2 = getattr(chromobius, name)
    assert got2 == alt_val

    setattr(chromobius, name, None)
    assert getattr(chromobius, name) is None


def test_configure_valid_multi_error_lut_decoders():
    nv = qec.multi_error_lut_config()
    nv.lut_error_depth = 2

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    # The decoding server constructs for observable output, so a server config
    # must supply an observable mapping.
    dc.O_sparse = [0, -1]
    dc.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        dc.syndrome_size, 2, include_first_round=False)
    dc.set_decoder_custom_args(nv)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    ret = qec.configure_decoders(mdc)
    qec.finalize_decoders()
    assert isinstance(ret, int)
    assert ret == 0


def test_trt_decoder_config_defaults_are_none():
    trt = qec.trt_decoder_config()
    for name in FIELDS_TRT_DECODER:
        assert getattr(trt, name) is None, f"Expected {name} to default to None"


@pytest.mark.parametrize("name, meta", list(FIELDS_TRT_DECODER.items()))
def test_trt_decoder_config_set_and_get_each_optional(name, meta):
    trt = qec.trt_decoder_config()

    py_type, sample_val, alt_val = meta

    # Initially None
    assert getattr(trt, name) is None

    # Set to a valid value and get back
    setattr(trt, name, sample_val)
    got = getattr(trt, name)
    assert isinstance(got, py_type)
    assert got == sample_val

    # Change to an alternate valid value
    setattr(trt, name, alt_val)
    got2 = getattr(trt, name)
    assert got2 == alt_val

    # Set value to None
    setattr(trt, name, None)
    assert getattr(trt, name) is None


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_config_yaml_roundtrip():
    trt = qec.trt_decoder_config()
    trt.engine_load_path = "/path/to/engine.trt"
    trt.engine_output_format = "observables"
    trt.precision = "fp16"
    trt.memory_workspace = 1073741824  # 1GB

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.set_decoder_custom_args(trt)

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    # Basic scalar fields
    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "trt_decoder"
    assert dc2.block_size == 10
    assert dc2.syndrome_size == 3

    # Intentional API change vs. the original test: decoder_custom_args now
    # reads back as a plain dict, never a typed config object.
    trt2 = dc2.decoder_custom_args
    assert trt2 is not None
    assert trt2["engine_load_path"] == "/path/to/engine.trt"
    assert trt2["precision"] == "fp16"
    assert trt2["memory_workspace"] == 1073741824


def test_trt_decoder_config_chromobius_global_params_roundtrip():
    trt = qec.trt_decoder_config()
    chromobius = qec.chromobius_config()
    chromobius.return_weight = True

    trt.global_decoder = "chromobius"
    trt.global_decoder_params = chromobius

    got = trt.global_decoder_params
    assert isinstance(got, qec.chromobius_config)
    assert got.return_weight is True

    as_map = trt.to_heterogeneous_map()
    assert as_map["global_decoder"] == "chromobius"
    assert as_map["global_decoder_params"]["return_weight"] is True

    trt2 = qec.trt_decoder_config.from_heterogeneous_map(as_map)
    got2 = trt2.global_decoder_params
    assert isinstance(got2, qec.chromobius_config)
    assert got2.return_weight is True

    trt2.global_decoder_params = None
    assert trt2.global_decoder_params is None


def test_trt_decoder_config_defaults_omitted_global_params():
    for global_decoder, config_type in (
        ("pymatching", qec.pymatching_config),
        ("chromobius", qec.chromobius_config),
    ):
        trt = qec.trt_decoder_config.from_heterogeneous_map(
            {"global_decoder": global_decoder})

        got = trt.global_decoder_params
        assert isinstance(got, config_type)

        as_map = trt.to_heterogeneous_map()
        assert as_map["global_decoder"] == global_decoder
        assert as_map["global_decoder_params"] == {}

        trt = qec.trt_decoder_config()
        trt.global_decoder = global_decoder
        as_map = trt.to_heterogeneous_map()
        assert as_map["global_decoder"] == global_decoder
        assert as_map["global_decoder_params"] == {}


def test_trt_decoder_config_preserves_unknown_omitted_global_params():
    trt = qec.trt_decoder_config.from_heterogeneous_map(
        {"global_decoder": "my_plugin"})

    assert trt.global_decoder_params is None

    as_map = trt.to_heterogeneous_map()
    assert as_map["global_decoder"] == "my_plugin"
    assert "global_decoder_params" not in as_map

    trt = qec.trt_decoder_config()
    trt.global_decoder = "my_plugin"
    as_map = trt.to_heterogeneous_map()
    assert as_map["global_decoder"] == "my_plugin"
    assert "global_decoder_params" not in as_map


def test_trt_decoder_config_rejects_unknown_global_params():
    with pytest.raises(RuntimeError):
        qec.trt_decoder_config.from_heterogeneous_map({
            "global_decoder": "my_plugin",
            "global_decoder_params": {},
        })


def test_pymatching_config_yaml_roundtrip():
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.1, 0.2, 0.3]
    pm.merge_strategy = "smallest_weight"

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.set_decoder_custom_args(pm)

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and "pymatching" in yaml_text

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2 is not None
    assert dc2.type == "pymatching"

    # Intentional API change vs. the original test: decoder_custom_args now
    # reads back as a plain dict, never a typed config object.
    pm2 = dc2.decoder_custom_args
    assert pm2 is not None
    assert "error_rate_vec" not in pm2
    assert list(dc2.error_rate_vec) == [0.1, 0.2, 0.3]
    assert pm2["merge_strategy"] == "smallest_weight"


@pytest.mark.skipif(
    trt_schema_missing or chromobius_schema_missing,
    reason="trt_decoder/chromobius plugins (and their schemas) not available")
def test_trt_decoder_chromobius_global_config_yaml_roundtrip():
    chromobius = qec.chromobius_config()
    chromobius.ignore_decomposition_failures = True
    chromobius.return_weight = False

    trt = qec.trt_decoder_config()
    trt.engine_output_format = "residual_detectors"
    trt.global_decoder = "chromobius"
    trt.global_decoder_params = chromobius

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.set_decoder_custom_args(trt)

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and "chromobius" in yaml_text

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2 is not None
    assert dc2.type == "trt_decoder"

    # Intentional API change vs. the original test: decoder_custom_args now
    # reads back as a plain dict, never a typed config object.
    trt2 = dc2.decoder_custom_args
    assert trt2 is not None
    assert trt2["global_decoder"] == "chromobius"

    chromobius2 = trt2["global_decoder_params"]
    assert chromobius2 is not None
    assert chromobius2["ignore_decomposition_failures"] is True
    assert chromobius2["return_weight"] is False


@pytest.mark.skipif(
    trt_schema_missing or pymatching_schema_missing,
    reason="trt_decoder/pymatching plugins (and their schemas) not available")
def test_trt_decoder_nested_pymatching_rates_promote_and_roundtrip():
    # A nested composition carries model rates on the child, which belong to
    # the top-level model rather than the child's parameters.
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.1, 0.2, 0.3]
    pm.merge_strategy = "independent"

    trt = qec.trt_decoder_config()
    trt.engine_output_format = "residual_detectors"
    trt.global_decoder = "pymatching"
    trt.global_decoder_params = pm

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.set_decoder_custom_args(trt)

    assert list(dc.error_rate_vec) == [0.1, 0.2, 0.3]
    nested = dc.decoder_custom_args["global_decoder_params"]
    assert "error_rate_vec" not in nested
    assert nested["merge_strategy"] == "independent"
    dc.validate_custom_args()

    dc2 = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    assert list(dc2.error_rate_vec) == [0.1, 0.2, 0.3]
    nested2 = dc2.decoder_custom_args["global_decoder_params"]
    assert "error_rate_vec" not in nested2
    assert nested2["merge_strategy"] == "independent"


@pytest.mark.skipif(
    trt_schema_missing or pymatching_schema_missing,
    reason="trt_decoder/pymatching plugins (and their schemas) not available")
def test_trt_decoder_nested_pymatching_conflicting_rates_rejected():
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.9, 0.9, 0.9]

    trt = qec.trt_decoder_config()
    trt.engine_output_format = "residual_detectors"
    trt.global_decoder = "pymatching"
    trt.global_decoder_params = pm

    dc = qec.decoder_config()
    dc.type = "trt_decoder"
    dc.error_rate_vec = [0.1, 0.2, 0.3]
    with pytest.raises(RuntimeError, match="error_rate_vec is supplied both"):
        dc.set_decoder_custom_args(trt)


# decoder_config tests


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_decoder_config_yaml_roundtrip_and_custom_args():
    # Build NV config and embed into DecoderConfig via helper
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.max_iterations = 50
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.set_decoder_custom_args(nv)

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    # Basic scalar fields
    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "nv-qldpc-decoder"
    assert dc2.block_size == 10
    assert dc2.syndrome_size == 3

    # Intentional API change vs. the original test: decoder_custom_args now
    # reads back as a plain dict, never a typed config object.
    nv2 = dc2.decoder_custom_args
    assert nv2 is not None
    assert nv2["use_sparsity"] is True
    assert math.isclose(nv2["error_rate"], 0.01)
    assert nv2["max_iterations"] == 50


# multi_decoder_config tests


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_multi_decoder_config_yaml_roundtrip():
    # Build NV config and embed into DecoderConfig via helper
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    nv.max_iterations = 50

    d1 = qec.decoder_config()
    d1.id = 0
    d1.type = "nv-qldpc-decoder"
    d1.block_size = 10
    d1.syndrome_size = 3
    d1.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d1.set_decoder_custom_args(nv)

    lut_config = qec.multi_error_lut_config()
    lut_config.lut_error_depth = 3

    d2 = qec.decoder_config()
    d2.id = 1
    d2.type = "multi_error_lut"
    d2.block_size = 10
    d2.syndrome_size = 3
    d2.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d2.set_decoder_custom_args(lut_config)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [d1, d2]

    yaml_text = mdc.to_yaml_str()
    assert isinstance(yaml_text, str) and "0" in yaml_text and "1" in yaml_text

    mdc2 = qec.multi_decoder_config.from_yaml_str(yaml_text)
    assert mdc2 is not None
    assert len(mdc2.decoders) == 2
    ids = sorted({md.id for md in mdc2.decoders})
    assert ids == [0, 1]


def test_configure_decoders_from_str_smoke():
    multi_decoder_config = qec.multi_decoder_config()
    yaml_str = multi_decoder_config.to_yaml_str()
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()

    if nv_qldpc_schema_missing:
        return
    nv = qec.nv_qldpc_decoder_config()
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]

    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "nv-qldpc-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    decoder_config.set_decoder_custom_args(nv)
    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    yaml_str = multi_decoder_config.to_yaml_str()
    # Do not instantiate the decoder if it is not available.
    if not is_nv_qldpc_decoder_available():
        return
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()


def test_configure_valid_decoders():
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    nv.max_iterations = 50

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    # The decoding server constructs for observable output, so a server config
    # must supply an observable mapping.
    dc.O_sparse = [0, -1]
    dc.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        dc.syndrome_size, 2, include_first_round=False)
    lut_config = qec.multi_error_lut_config()
    lut_config.lut_error_depth = 2
    dc.set_decoder_custom_args(lut_config)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    ret = qec.configure_decoders(mdc)
    qec.finalize_decoders()
    assert isinstance(ret, int)
    assert ret == 0


def make_pymatching_multi_decoder_config(pm, h_sparse=None):
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = h_sparse if h_sparse is not None else [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.set_decoder_custom_args(pm)

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc


def configure_pymatching_status(pm, h_sparse=None):
    try:
        return qec.configure_decoders(
            make_pymatching_multi_decoder_config(pm, h_sparse))
    finally:
        qec.finalize_decoders()


def test_configure_valid_pymatching_decoder():
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.1, 0.1, 0.1]
    pm.merge_strategy = "smallest_weight"

    ret = configure_pymatching_status(pm)
    assert isinstance(ret, int)
    assert ret == 0


@pytest.mark.parametrize(
    "error_rate_vec",
    ([0.0, 0.1, 0.1], [0.1, 0.6, 0.1]),
)
def test_configure_invalid_pymatching_error_rate_vec(error_rate_vec):
    pm = qec.pymatching_config()
    pm.error_rate_vec = error_rate_vec
    pm.merge_strategy = "smallest_weight"

    ret = configure_pymatching_status(pm)
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_rejects_pymatching_error_rate_vec_size_mismatch():
    # Promoted rates are validated with the top-level field, which rejects a
    # size mismatch by raising rather than through the status code.
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.1, 0.1]
    pm.merge_strategy = "smallest_weight"

    with pytest.raises(RuntimeError, match="error_rate_vec size"):
        configure_pymatching_status(pm)


def test_configure_invalid_pymatching_merge_strategy():
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.1, 0.1, 0.1]
    pm.merge_strategy = "not-a-strategy"

    ret = configure_pymatching_status(pm)
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_pymatching_non_graphlike_h_sparse():
    pm = qec.pymatching_config()
    pm.error_rate_vec = [0.1, 0.1, 0.1]
    pm.merge_strategy = "smallest_weight"

    ret = configure_pymatching_status(pm, h_sparse=[0, -1, 0, -1, 0, -1])
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_decoders():
    nv = qec.nv_qldpc_decoder_config()
    nv.use_sparsity = True
    nv.error_rate = 0.01
    nv.error_rate_vec = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    nv.max_iterations = 50

    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "invalid-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    # A resolvable model, so the failure under test is the unregistered
    # decoder type at construction rather than an unresolvable configuration.
    decoder_config.O_sparse = [0, -1]
    decoder_config.D_sparse = [0, -1, 1, -1, 2, -1]
    decoder_config.set_decoder_custom_args(nv)

    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    ret = qec.configure_decoders(multi_decoder_config)
    assert isinstance(ret, int)
    assert ret != 0


def _clear_kind_cache(cls):
    if "_kind_cache" in cls.__dict__:
        delattr(cls, "_kind_cache")


def test_typed_config_assignment_when_schema_lookup_fails(monkeypatch):
    # A missing or failing parameter schema is optional: assignment still
    # stores the value because no kind check can be applied.
    cls = qec.nv_qldpc_decoder_config
    _clear_kind_cache(cls)

    def raise_schema(name):
        raise RuntimeError("schema lookup failed")

    monkeypatch.setattr(qec, "decoder_param_schema", raise_schema)
    try:
        cfg = cls()
        cfg.max_iterations = 50
        assert cfg.max_iterations == 50
    finally:
        _clear_kind_cache(cls)


def test_typed_config_equality_and_repr():
    # Equality is type-plus-set-fields; repr lists those fields in _fields order.
    a = qec.pymatching_config()
    a.error_rate_vec = [0.1, 0.2]
    a.merge_strategy = "smallest_weight"
    b = qec.pymatching_config()
    b.error_rate_vec = [0.1, 0.2]
    b.merge_strategy = "smallest_weight"
    c = qec.pymatching_config()
    c.merge_strategy = "disallow"
    assert a == b
    assert a != c
    assert repr(a) == ("pymatching_config(error_rate_vec=[0.1, 0.2], "
                       "merge_strategy='smallest_weight')")
    assert repr(c) == "pymatching_config(merge_strategy='disallow')"


# A 1-D list: accepted length, but not a list of rows.
_not_a_matrix = [0.1, 0.2]


@pytest.mark.parametrize(
    "field, kind, bad, expected",
    [
        ("use_sparsity", "bool", 1, "a bool"),
        ("osd_method", "int32", 1.5, "an int"),
        ("osd_method", "int32", True, "an int"),
        ("max_iterations", "uint64", -1, "a non-negative int"),
        ("max_iterations", "uint64", True, "a non-negative int"),
        ("error_rate", "float64", "0.1", "a float"),
        ("proc_float", "string", 32, "a str"),
        ("error_rate_vec", "float64_vec", 0.1, "a list of floats"),
        ("error_rate_vec", "float64_vec", [0.1, "x"], "a list of floats"),
        ("explicit_gammas", "float64_matrix", 0.1, "a list of lists of floats"),
        ("explicit_gammas", "float64_matrix", _not_a_matrix,
         "a list of lists of floats"),
        ("srelay_config", "subschema", 5, "a config object or dict"),
        ("composition", "discriminated", "bp", "a config object or dict"),
    ],
)
def test_nv_qldpc_decoder_config_kind_errors(monkeypatch, field, kind, bad,
                                             expected):
    # Drive every schema kind through the public property setters on a real
    # nv_qldpc_decoder_config field, using a synthetic schema for this image.
    cls = qec.nv_qldpc_decoder_config
    _clear_kind_cache(cls)

    def synthetic_schema(name):
        return [{"key": field, "kind": kind}]

    monkeypatch.setattr(qec, "decoder_param_schema", synthetic_schema)
    try:
        cfg = cls()
        with pytest.raises(
                TypeError,
                match=(f"nv_qldpc_decoder_config.{field} expects {expected}, "
                       f"got {type(bad).__name__}")):
            setattr(cfg, field, bad)
    finally:
        _clear_kind_cache(cls)


def test_nv_qldpc_decoder_config_nested_srelay_from_map():
    # from_heterogeneous_map rebuilds nested srelay_bp_config and round-trips.
    source = {
        "use_sparsity": True,
        "max_iterations": 50,
        "srelay_config": {
            "pre_iter": 5,
            "num_sets": 3,
        },
    }
    cfg = qec.nv_qldpc_decoder_config.from_heterogeneous_map(source)
    assert isinstance(cfg.srelay_config, qec.qecrt.config.srelay_bp_config)
    assert cfg.srelay_config.pre_iter == 5
    assert cfg.srelay_config.num_sets == 3
    assert cfg.use_sparsity is True
    assert cfg.max_iterations == 50
    assert cfg.to_heterogeneous_map() == source


if __name__ == "__main__":
    pytest.main()
