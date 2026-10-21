# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import json
import math

import numpy as np
import pytest

import cudaq_qec as qec

# Decoder custom args are plain dicts. Their keys are governed by the
# parameter schema each decoder registers (see qec.decoder_param_schema).

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


# Schema introspection tests


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_nv_qldpc_decoder_param_schema_introspection():
    schema = qec.decoder_param_schema("nv-qldpc-decoder")
    assert schema is not None
    by_key = {entry["key"]: entry for entry in schema}
    assert by_key["max_iterations"]["kind"] == "int32"
    assert "error_rate_vec" not in by_key
    assert by_key["srelay_config"]["kind"] == "subschema"
    assert by_key["srelay_config"]["subschema"] == "srelay_bp"


def test_decoder_param_schema_introspection():
    sw_schema = qec.decoder_param_schema("sliding_window")
    assert sw_schema is not None
    by_key = {entry["key"]: entry for entry in sw_schema}
    assert "error_rate_vec" not in by_key
    assert by_key["inner_decoder_params"]["kind"] == "discriminated"
    assert by_key["inner_decoder_params"]["discriminator"] == \
        "inner_decoder_name"

    assert qec.decoder_param_schema("no-such-decoder") is None

    names = qec.registered_decoder_schemas()
    assert "pymatching" in names
    assert "multi_error_lut" in names


# decoder_config custom args tests


def test_decoder_custom_args_is_a_dict():
    dc = qec.decoder_config()
    assert dc.decoder_custom_args == {}

    dc.decoder_custom_args = {"lut_error_depth": 2}
    assert dc.decoder_custom_args == {"lut_error_depth": 2}

    dc.set_decoder_custom_args({"lut_error_depth": 3})
    assert dc.decoder_custom_args == {"lut_error_depth": 3}


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_decoder_config_yaml_roundtrip_and_custom_args():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.error_rate_vec = \
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    dc.decoder_custom_args = {
        "use_sparsity": True,
        "error_rate": 0.01,
        "max_iterations": 50,
        "srelay_config": {
            "pre_iter": 5,
            "stopping_criterion": "NConv",
        },
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    # Basic scalar fields
    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "nv-qldpc-decoder"
    assert dc2.block_size == 10
    assert dc2.syndrome_size == 3

    args = dc2.decoder_custom_args
    assert args["use_sparsity"] is True
    assert math.isclose(args["error_rate"], 0.01)
    assert args["max_iterations"] == 50
    assert args["srelay_config"]["pre_iter"] == 5
    assert args["srelay_config"]["stopping_criterion"] == "NConv"


def test_pymatching_config_yaml_roundtrip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.error_rate_vec = [0.1, 0.2, 0.3]
    dc.decoder_custom_args = {
        "merge_strategy": "smallest_weight",
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and "pymatching" in yaml_text
    assert "\nerror_rate_vec:" not in yaml_text
    assert "decoder_custom_args:\n  error_rate_vec:" in yaml_text

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2 is not None
    assert dc2.type == "pymatching"

    args = dc2.decoder_custom_args
    assert list(dc2.error_rate_vec) == [0.1, 0.2, 0.3]
    assert args["merge_strategy"] == "smallest_weight"


def test_legacy_custom_arg_error_rates_are_accepted_but_not_plugin_params():
    dc = qec.decoder_config()
    dc.type = "pymatching"
    dc.decoder_custom_args = {
        "error_rate_vec": [0.1, 0.2, 0.3],
        "merge_strategy": "smallest_weight",
    }

    dc.validate_custom_args()
    yaml_text = dc.to_yaml_str()
    assert "decoder_custom_args:\n  error_rate_vec:" in yaml_text


def test_unknown_custom_arg_key_is_rejected():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"merge_strategy": "smallest_weight"}
    yaml_text = qec_yaml_for(dc)

    misspelled = yaml_text.replace("merge_strategy", "merge_stratgey")
    with pytest.raises(RuntimeError):
        qec.multi_decoder_config.from_yaml_str(misspelled)


def test_validate_custom_args_checks_dict_built_configs():
    # Dicts assigned to decoder_custom_args never pass through the YAML
    # parser; validate_custom_args applies the same schema checks explicitly.
    dc = qec.decoder_config()
    dc.type = "pymatching"
    dc.decoder_custom_args = {"merge_strategy": "smallest_weight"}
    dc.validate_custom_args()

    dc.decoder_custom_args = {"merge_stratgey": "smallest_weight"}
    with pytest.raises(RuntimeError, match="merge_stratgey"):
        dc.validate_custom_args()

    # Non-empty args for a type with no registered schema are rejected.
    dc.type = "decoder_without_registered_schema"
    dc.decoder_custom_args = {"anything": 1}
    with pytest.raises(RuntimeError, match="no registered parameter schema"):
        dc.validate_custom_args()

    # multi_decoder_config validates every decoder.
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    with pytest.raises(RuntimeError):
        mdc.validate_custom_args()


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_custom_args_dict_values_convert_to_schema_types():
    # The setter converts dict values to the canonical types the registered
    # schema declares: Python ints are accepted for f64 params and negative
    # ints for int32 params (the generic conversion stores every int as
    # size_t, which would reject both).
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "nv-qldpc-decoder"
    dc.block_size = 1
    dc.syndrome_size = 1
    dc.H_sparse = [0, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1]

    dc.decoder_custom_args = {"clip_value": 2}  # int for an f64 param
    assert "clip_value" in dc.to_yaml_str()

    dc.decoder_custom_args = {"bp_seed": -1}  # negative int32
    assert dc.decoder_custom_args == {"bp_seed": -1}
    round_tripped = qec.multi_decoder_config.from_yaml_str(qec_yaml_for(dc))
    assert round_tripped.decoders[0].decoder_custom_args == {"bp_seed": -1}

    # Mismatched types raise a clear error naming the parameter.
    with pytest.raises(RuntimeError, match="bp_seed"):
        dc.decoder_custom_args = {"bp_seed": "oops"}


def test_decoder_config_from_yaml_str_rejects_malformed_yaml():
    with pytest.raises(RuntimeError, match="Invalid decoder configuration"):
        qec.decoder_config.from_yaml_str("id: [oops")


def test_validate_custom_args_runs_schema_validate_hook():
    # sliding_window registers a validate hook for cross-field constraints
    # (step_size must be between 1 and window_size).
    dc = qec.decoder_config()
    dc.type = "sliding_window"
    dc.error_rate_vec = [0.01, 0.01]
    dc.decoder_custom_args = {
        "window_size": 4,
        "step_size": 2,
        "inner_decoder_name": "single_error_lut",
    }
    dc.validate_custom_args()

    dc.decoder_custom_args = {
        "window_size": 2,
        "step_size": 4,
        "inner_decoder_name": "single_error_lut",
    }
    with pytest.raises(RuntimeError, match="step_size"):
        dc.validate_custom_args()

    # Missing required key (inner_decoder_name).
    dc.decoder_custom_args = {"window_size": 2}
    with pytest.raises(RuntimeError, match="inner_decoder_name"):
        dc.validate_custom_args()


def test_decoder_config_json_schema_validates_yaml_documents():
    # The exported JSON Schema lets standard third-party tooling validate
    # user-provided configuration YAML without loading this library.
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    import json

    schema = json.loads(qec.decoder_config_json_schema())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    # Every registered decoder schema appears in the export.
    assert set(qec.registered_decoder_schemas()) == set(
        schema["$defs"]["decoder_params"])

    # A real emitted configuration validates.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.error_rate_vec = [0.1, 0.1, 0.1]
    dc.decoder_custom_args = {
        "merge_strategy": "smallest_weight",
    }
    document = yaml.safe_load(qec_yaml_for(dc))
    validator.validate(document)

    # Both locations are accepted independently, but never together.
    top_level = yaml.safe_load(qec_yaml_for(dc))
    rates = top_level["decoders"][0]["decoder_custom_args"].pop(
        "error_rate_vec")
    top_level["decoders"][0]["error_rate_vec"] = rates
    validator.validate(top_level)

    duplicate = yaml.safe_load(qec_yaml_for(dc))
    duplicate["decoders"][0]["error_rate_vec"] = [0.1, 0.1, 0.1]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(duplicate)

    # Unknown custom-arg keys fail validation.
    bad = yaml.safe_load(qec_yaml_for(dc))
    bad["decoders"][0]["decoder_custom_args"]["merge_stratgey"] = \
        bad["decoders"][0]["decoder_custom_args"].pop("merge_strategy")
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad)

    # Missing required custom-arg keys fail validation (sliding_window
    # requires inner_decoder_name).
    missing = yaml.safe_load(qec_yaml_for(dc))
    missing["decoders"][0]["type"] = "sliding_window"
    missing["decoders"][0]["decoder_custom_args"] = {"window_size": 2}
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(missing)

    # Custom args for a type with no registered schema fail validation.
    unregistered = yaml.safe_load(qec_yaml_for(dc))
    unregistered["decoders"][0]["type"] = "decoder_without_registered_schema"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(unregistered)

    # Missing envelope fields fail validation.
    no_type = yaml.safe_load(qec_yaml_for(dc))
    del no_type["decoders"][0]["type"]
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(no_type)


def qec_yaml_for(dc):
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc.to_yaml_str()


def test_decoder_config_json_schema_covers_dispatch_and_transport():
    # The exported JSON schema is generated separately from the YAML parser, so
    # validate a real document that populates `dispatch:` and the server-level
    # `transport:` section. The other schema test only exercises documents that
    # omit these keys (they default, and to_yaml_str drops defaults), so it
    # cannot catch the schema drifting behind the parser.
    jsonschema = pytest.importorskip("jsonschema")
    yaml = pytest.importorskip("yaml")
    import json

    schema = json.loads(qec.decoder_config_json_schema())
    jsonschema.Draft202012Validator.check_schema(schema)
    validator = jsonschema.Draft202012Validator(schema)

    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.dispatch = qec.decoder_dispatch.device_graph
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.error_rate_vec = [0.1, 0.1, 0.1]
    dc.decoder_custom_args = {
        "merge_strategy": "smallest_weight",
    }

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    mdc.transport.provider = "udp"
    mdc.transport.args = ["--num-slots=8"]
    mdc.transport.device_graph.provider = "gpu_roce"
    mdc.transport.device_graph.args = ["--pinned-rings"]

    document = yaml.safe_load(mdc.to_yaml_str())

    # Sanity: the emitted YAML really carries the keys we mean to validate.
    # to_yaml_str drops defaults, so if the bindings failed to set them this
    # test would be vacuous rather than failing.
    assert document["decoders"][0]["dispatch"] == "device_graph"
    assert document["transport"]["provider"] == "udp"
    assert document["transport"]["device_graph"]["args"] == ["--pinned-rings"]

    # The populated document validates against the exported schema.
    validator.validate(document)

    # An unknown dispatch value is rejected (the schema's dispatch enum is
    # host|device_graph).
    bad_dispatch = yaml.safe_load(mdc.to_yaml_str())
    bad_dispatch["decoders"][0]["dispatch"] = "gpu_roce"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad_dispatch)

    # A per-decoder `transport:` key is rejected: additionalProperties is false
    # on the decoder envelope, so a config carrying this (non-existent) key
    # fails validation rather than being silently ignored.
    old_key = yaml.safe_load(mdc.to_yaml_str())
    old_key["decoders"][0]["transport"] = "gpu_roce"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(old_key)

    # An unknown key in the transport section is rejected (additionalProperties
    # false on transport_config).
    bad_transport = yaml.safe_load(mdc.to_yaml_str())
    bad_transport["transport"]["bogus"] = 1
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(bad_transport)


# trt_decoder tests (need the trt_decoder plugin for its parameter schema)

trt_schema_missing = qec.decoder_param_schema("trt_decoder") is None


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_config_yaml_roundtrip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 10
    dc.syndrome_size = 3
    dc.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    dc.decoder_custom_args = {
        "engine_load_path": "/path/to/engine.trt",
        "engine_output_format": "errors",
        "precision": "fp16",
        "memory_workspace": 1073741824,  # 1GB
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and len(yaml_text) > 0

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)

    assert dc2 is not None
    assert dc2.id == 0
    assert dc2.type == "trt_decoder"

    args = dc2.decoder_custom_args
    assert args["engine_load_path"] == "/path/to/engine.trt"
    assert args["precision"] == "fp16"
    assert args["memory_workspace"] == 1073741824


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_chromobius_global_config_yaml_roundtrip():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {
        "onnx_load_path": "/tmp/predecoder.onnx",
        "engine_output_format": "observables_and_residual_detectors",
        "global_decoder": "chromobius",
        "global_decoder_params": {
            "ignore_decomposition_failures": True,
            "return_weight": False,
        },
    }

    yaml_text = dc.to_yaml_str()
    assert isinstance(yaml_text, str) and "chromobius" in yaml_text

    dc2 = qec.decoder_config.from_yaml_str(yaml_text)
    assert dc2 is not None
    assert dc2.type == "trt_decoder"

    args = dc2.decoder_custom_args
    assert args["global_decoder"] == "chromobius"
    assert args["global_decoder_params"]["ignore_decomposition_failures"] \
        is True
    assert args["global_decoder_params"]["return_weight"] is False


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_default_global_params_materialized():
    # A named global decoder with a registered schema gets an empty params
    # section materialized on parse.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {
        "engine_output_format": "residual_detectors",
        "global_decoder": "pymatching",
    }

    mdc2 = qec.multi_decoder_config.from_yaml_str(qec_yaml_for(dc))
    args = mdc2.decoders[0].decoder_custom_args
    assert args["global_decoder_params"] == {}

    # ... and already on FIRST emission (matching the old typed path), so
    # emitted YAML is stable across round trips.
    first = qec_yaml_for(dc)
    assert "global_decoder_params" in first
    assert mdc2.to_yaml_str() == first


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_validate_custom_args_checks_value_kinds():
    # A dict assigned before `type` is set takes the generic conversion
    # (ints stored as size_t), which an f64 param cannot read back at
    # emission. validate_custom_args must name the offending key instead of
    # letting emission fail with a low-context error later.
    dc = qec.decoder_config()
    dc.decoder_custom_args = {"clip_value": 2}  # type not set yet
    dc.type = "nv-qldpc-decoder"
    with pytest.raises(RuntimeError, match="clip_value"):
        dc.validate_custom_args()

    # Assigned after `type`, the same dict converts to the schema's declared
    # types and validates.
    dc.decoder_custom_args = {"clip_value": 2}
    dc.validate_custom_args()


def test_non_schema_keys_dropped_from_emission_and_decoder_params():
    # A key outside the registered schema cannot round-trip through YAML, so
    # it is warned-and-dropped from the emitted YAML (and from the map local
    # decoders receive) rather than taking effect locally but silently
    # vanishing when the config is serialized for a remote target.
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "multi_error_lut"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.decoder_custom_args = {"lut_error_depth": 2, "not_a_real_param": 42}

    yaml_text = qec_yaml_for(dc)
    assert "lut_error_depth" in yaml_text
    assert "not_a_real_param" not in yaml_text

    # The stored args are untouched; validate_custom_args still rejects them
    # for callers who want a hard error instead of the warn-and-drop.
    assert dc.decoder_custom_args["not_a_real_param"] == 42
    with pytest.raises(RuntimeError, match="not_a_real_param"):
        dc.validate_custom_args()


@pytest.mark.skipif(
    trt_schema_missing,
    reason="trt_decoder plugin (and its parameter schema) not available")
def test_trt_decoder_rejects_unknown_global_params():
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "trt_decoder"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    yaml_text = """
decoders:
  - id: 0
    type: trt_decoder
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args:
      global_decoder: my_plugin
      global_decoder_params: {}
"""
    with pytest.raises(RuntimeError):
        qec.multi_decoder_config.from_yaml_str(yaml_text)


# multi_decoder_config tests


@pytest.mark.skipif(
    nv_qldpc_schema_missing,
    reason="nv-qldpc-decoder plugin (and its parameter schema) not available")
def test_multi_decoder_config_yaml_roundtrip():
    d1 = qec.decoder_config()
    d1.id = 0
    d1.type = "nv-qldpc-decoder"
    d1.block_size = 10
    d1.syndrome_size = 3
    d1.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d1.error_rate_vec = \
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    d1.decoder_custom_args = {
        "use_sparsity": True,
        "error_rate": 0.01,
        "max_iterations": 50,
    }

    d2 = qec.decoder_config()
    d2.id = 1
    d2.type = "multi_error_lut"
    d2.block_size = 10
    d2.syndrome_size = 3
    d2.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    d2.decoder_custom_args = {"lut_error_depth": 3}

    mdc = qec.multi_decoder_config()
    mdc.decoders = [d1, d2]

    yaml_text = mdc.to_yaml_str()
    assert isinstance(yaml_text, str) and "0" in yaml_text and "1" in yaml_text

    mdc2 = qec.multi_decoder_config.from_yaml_str(yaml_text)
    assert mdc2 is not None
    assert len(mdc2.decoders) == 2
    ids = sorted({md.id for md in mdc2.decoders})
    assert ids == [0, 1]
    assert mdc2 == mdc


# configure_decoders tests


def test_configure_valid_multi_error_lut_decoders():
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
    dc.decoder_custom_args = {"lut_error_depth": 2}

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    ret = qec.configure_decoders(mdc)
    qec.finalize_decoders()
    assert isinstance(ret, int)
    assert ret == 0


def test_configure_decoders_from_str_smoke():
    multi_decoder_config = qec.multi_decoder_config()
    yaml_str = multi_decoder_config.to_yaml_str()
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()

    if nv_qldpc_schema_missing:
        return
    decoder_config = qec.decoder_config()
    decoder_config.id = 0
    decoder_config.type = "nv-qldpc-decoder"
    decoder_config.block_size = 10
    decoder_config.syndrome_size = 3
    decoder_config.H_sparse = [1, 2, 3, -1, 6, 7, 8, -1, -1]
    decoder_config.O_sparse = [0, -1]
    decoder_config.D_sparse = qec.generate_timelike_sparse_detector_matrix(
        decoder_config.syndrome_size, 2, include_first_round=False)
    decoder_config.error_rate_vec = \
        [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1]
    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    yaml_str = multi_decoder_config.to_yaml_str()
    # Do not instantiate the decoder if it is not available.
    if not is_nv_qldpc_decoder_available():
        return
    status = qec.configure_decoders_from_str(yaml_str)
    assert isinstance(status, int)
    qec.finalize_decoders()


def test_configure_decoders_from_file_smoke(tmp_path):
    path = tmp_path / "decoders.yaml"
    path.write_text(qec.multi_decoder_config().to_yaml_str(), encoding="utf-8")

    status = qec.configure_decoders_from_file(str(path))
    assert isinstance(status, int)
    qec.finalize_decoders()


def make_pymatching_multi_decoder_config(pm_args, h_sparse=None):
    dc = qec.decoder_config()
    dc.id = 0
    dc.type = "pymatching"
    dc.block_size = 3
    dc.syndrome_size = 3
    dc.H_sparse = h_sparse if h_sparse is not None else [0, -1, 1, -1, 2, -1]
    dc.O_sparse = [0, -1, 1, -1, 2, -1]
    dc.D_sparse = [0, -1, 1, -1, 2, -1]
    pm_args = dict(pm_args)
    dc.error_rate_vec = pm_args.pop("error_rate_vec", [])
    dc.decoder_custom_args = pm_args

    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    return mdc


def configure_pymatching_status(pm_args, h_sparse=None):
    try:
        return qec.configure_decoders(
            make_pymatching_multi_decoder_config(pm_args, h_sparse))
    finally:
        qec.finalize_decoders()


def test_configure_valid_pymatching_decoder():
    ret = configure_pymatching_status({
        "error_rate_vec": [0.1, 0.1, 0.1],
        "merge_strategy": "smallest_weight",
    })
    assert isinstance(ret, int)
    assert ret == 0


def test_configure_rejects_pymatching_error_rate_vec_size_mismatch():
    with pytest.raises(RuntimeError, match="error_rate_vec size"):
        configure_pymatching_status({
            "error_rate_vec": [0.1, 0.1],
            "merge_strategy": "smallest_weight",
        })


@pytest.mark.parametrize("error_rate_vec", ([0.0, 0.1, 0.1], [0.1, 0.6, 0.1]))
def test_configure_invalid_pymatching_error_rate_vec(error_rate_vec):
    ret = configure_pymatching_status({
        "error_rate_vec": error_rate_vec,
        "merge_strategy": "smallest_weight",
    })
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_pymatching_merge_strategy():
    ret = configure_pymatching_status({
        "error_rate_vec": [0.1, 0.1, 0.1],
        "merge_strategy": "not-a-strategy",
    })
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_pymatching_non_graphlike_h_sparse():
    ret = configure_pymatching_status(
        {
            "error_rate_vec": [0.1, 0.1, 0.1],
            "merge_strategy": "smallest_weight",
        },
        h_sparse=[0, -1, 0, -1, 0, -1])
    assert isinstance(ret, int)
    assert ret != 0


def test_configure_invalid_decoders():
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
    decoder_config.decoder_custom_args = {"max_iterations": 50}

    multi_decoder_config = qec.multi_decoder_config()
    multi_decoder_config.decoders = [decoder_config]
    ret = qec.configure_decoders(multi_decoder_config)
    assert isinstance(ret, int)
    assert ret != 0


if __name__ == "__main__":
    pytest.main()

# --- exported JSON Schema: the two model sources ----------------------------
#
# The schema must describe the language the runtime actually accepts. It keys
# the DEM source on a NON-EMPTY stim_dem_path, matching resolve_decoder_init.


def _decoder_doc(**overrides):
    doc = {"id": 0, "type": "pymatching", "D_sparse": [0, -1, 1, -1]}
    doc.update(overrides)
    return {"decoders": [doc]}


_MATRIX_KEYS = {
    "block_size": 4,
    "syndrome_size": 2,
    "H_sparse": [0, -1, 1, -1],
    "O_sparse": [0, -1],
}


def _schema_accepts(doc):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(qec.qecrt.config.decoder_config_json_schema())
    try:
        jsonschema.validate(doc, schema)
        return True
    except jsonschema.ValidationError:
        return False


def test_json_schema_accepts_matrix_source():
    assert _schema_accepts(_decoder_doc(**_MATRIX_KEYS))


def test_json_schema_accepts_dem_source():
    assert _schema_accepts(_decoder_doc(stim_dem_path="model.dem"))


def test_json_schema_rejects_both_sources():
    assert not _schema_accepts(
        _decoder_doc(stim_dem_path="model.dem", **_MATRIX_KEYS))


def test_json_schema_rejects_neither_source():
    assert not _schema_accepts(_decoder_doc())


def test_json_schema_treats_empty_dem_path_as_matrix_source():
    # An explicitly empty path is not a DEM source at runtime, so the schema
    # must accept it alongside matrices and reject it on its own.
    assert _schema_accepts(_decoder_doc(stim_dem_path="", **_MATRIX_KEYS))
    assert not _schema_accepts(_decoder_doc(stim_dem_path=""))


# --- task graphs ------------------------------------------------------------


def _task_graph_doc(node, edges=None, **overrides):
    graph = {"nodes": [node]}
    if edges is not None:
        graph["edges"] = edges
    doc = {"id": 0, "task_graph": graph, "D_sparse": [0, -1, 1, -1]}
    doc.update(_MATRIX_KEYS)
    doc.update(overrides)
    return {"decoders": [doc]}


_WINDOW_NODE = {
    "id": "w",
    "type": "single_window",
    "inner": {
        "type": "pymatching"
    },
}


def test_registered_task_graph_node_kinds():
    kinds = qec.registered_task_graph_node_kinds()
    assert "sliding_window" in kinds
    assert "single_window" in kinds
    assert "xor" in kinds

    # A kind's node parameters are an ordinary parameter schema, so the same
    # introspection call describes them.
    schema = qec.decoder_param_schema("task_graph_node.sliding_window")
    assert schema is not None
    by_key = {entry["key"]: entry for entry in schema}
    assert by_key["window_size"]["kind"] == "uint64"
    assert by_key["end_guard"]["kind"] == "uint64"


def test_task_graph_config_roundtrips_through_yaml():
    node = qec.task_graph_node()
    node.id = "w"
    node.type = "sliding_window"
    node.inner_type = "pymatching"
    node.node_params = {"window_size": 3, "step_size": 2, "end_guard": 1}
    node.decoder_custom_args = {"merge_strategy": "smallest_weight"}

    graph = qec.task_graph_spec()
    graph.nodes = [node]

    dc = qec.decoder_config()
    dc.id = 0
    dc.block_size = 3
    dc.syndrome_size = 2
    dc.H_sparse = [0, 1, -1, 1, 2, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1]
    dc.task_graph = graph

    assert dc.task_graph.nodes[0].resolved_kind() == "sliding_window"
    assert dc.task_graph.nodes[0].resolved_decoder_type() == "pymatching"
    assert dc.task_graph.nodes[0].node_params["window_size"] == 3

    parsed = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    assert parsed.task_graph is not None
    assert len(parsed.task_graph.nodes) == 1
    node2 = parsed.task_graph.nodes[0]
    assert node2.id == "w"
    assert node2.type == "sliding_window"
    assert node2.inner_type == "pymatching"
    assert node2.node_params["step_size"] == 2
    assert node2.node_params["end_guard"] == 1
    assert node2.decoder_custom_args["merge_strategy"] == "smallest_weight"


def test_task_graph_ensemble_roundtrips_through_yaml():
    a = qec.task_graph_node()
    a.id = "a"
    a.type = "single_window"
    a.inner_type = "pymatching"
    into_a = qec.task_graph_edge()
    into_a.from_node = "input"
    into_a.to_node = "a"
    out_of_a = qec.task_graph_edge()
    out_of_a.from_node = "a"
    out_of_a.to_node = "output"
    member = qec.task_graph_ensemble_member()
    member.nodes = [a]
    member.edges = [into_a, out_of_a]

    ens = qec.task_graph_ensemble()
    ens.join_strategy = "deadline"
    ens.join_params = {"deadline_us": 200}
    ens.members = [member]
    node = qec.task_graph_node()
    node.id = "ens"
    node.ensemble = ens

    graph = qec.task_graph_spec()
    graph.nodes = [node]
    dc = qec.decoder_config()
    dc.id = 0
    dc.block_size = 3
    dc.syndrome_size = 2
    dc.H_sparse = [0, 1, -1, 1, 2, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1]
    dc.task_graph = graph

    parsed = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    node2 = parsed.task_graph.nodes[0]
    assert node2.type == ""
    assert node2.ensemble is not None
    assert node2.ensemble.join_strategy == "deadline"
    assert node2.ensemble.join_params["deadline_us"] == 200
    assert node2.ensemble.members[0].nodes[0].id == "a"
    assert node2.ensemble.members[0].edges[1].to_node == "output"
    assert parsed.task_graph == dc.task_graph


def test_task_graph_edges_carry_ports():
    # The configuration holds the same edge type from_topology() takes, so ports
    # come pre-defaulted rather than empty.
    edge = qec.task_graph_edge()
    edge.from_node = "pre"
    edge.from_output = "residual"
    edge.to_node = "global"

    graph = qec.task_graph_spec()
    graph.nodes = [qec.task_graph_node()]
    graph.edges = [edge]
    assert graph.edges[0].from_output == "residual"
    assert graph.edges[0].to_input == "input"


# A DEM whose detector coordinates carry the colour a colour-code decoder
# needs. Those annotations have no home in detector_error_model, so a decoder
# that reads them can only be built from the text -- which is what `stim_dem`
# is for. Two detectors, one observable.
_COLOUR_DEM = """error(0.1) D0 L0
error(0.1) D0 D1 L0
error(0.1) D1 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
"""


def _stim_dem_node():
    node = qec.task_graph_node()
    node.id = "w"
    node.type = "single_window"
    node.inner_type = "chromobius"
    node.stim_dem = _COLOUR_DEM
    return node


def test_task_graph_node_stim_dem_roundtrips_through_yaml():
    graph = qec.task_graph_spec()
    graph.nodes = [_stim_dem_node()]

    dc = qec.decoder_config()
    dc.id = 0
    dc.block_size = 2
    dc.syndrome_size = 2
    dc.H_sparse = [0, -1, 1, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1]
    dc.task_graph = graph

    parsed = qec.decoder_config.from_yaml_str(dc.to_yaml_str())
    assert parsed.task_graph.nodes[0].stim_dem == _COLOUR_DEM
    # Inline text, not a path: the DEM body has to survive the round trip.
    assert "detector(1, 0, 0, 1) D1" in dc.to_yaml_str()


@pytest.mark.skipif(qec.decoder_param_schema("chromobius") is None,
                    reason="chromobius plugin not available")
def test_task_graph_node_builds_a_colour_decoder_from_stim_dem():
    # The point of the field: a decoder that can only be built from DEM text
    # is unreachable from YAML without it. Before this, the config layer built
    # every inner decoder from the entry-level H and this returned nonzero
    # with "requires a Stim detector error model string".
    node = _stim_dem_node()
    node.node_params = {"num_syndromes_per_round": 2}
    graph = qec.task_graph_spec()
    graph.nodes = [node]

    dc = qec.decoder_config()
    dc.id = 0
    dc.block_size = 2
    dc.syndrome_size = 2
    dc.H_sparse = [0, -1, 1, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1]
    dc.task_graph = graph
    mdc = qec.multi_decoder_config()
    mdc.decoders = [dc]
    try:
        assert qec.configure_decoders(mdc) == 0
    finally:
        qec.finalize_decoders()

    # Not vacuous: the same node without the text falls back to the H path,
    # which this decoder rejects.
    node.stim_dem = ""
    graph.nodes = [node]
    dc.task_graph = graph
    mdc.decoders = [dc]
    try:
        assert qec.configure_decoders(mdc) != 0
    finally:
        qec.finalize_decoders()


def test_predecoder_kind_is_registered_with_its_own_schema():
    assert "predecoder" in qec.registered_task_graph_node_kinds()
    schema = qec.decoder_param_schema("task_graph_node.predecoder")
    assert schema is not None
    by_key = {e["key"]: e for e in schema}
    assert by_key["window_size"]["kind"] == "uint64"
    # The non-vacuous half: a pre-decoder covers its whole DEM and empties on
    # every fire, so it declares no step_size and no guards. This fails if the
    # kind was registered by copy-pasting the sliding-window schema.
    assert "step_size" not in by_key
    assert "start_guard" not in by_key
    assert "end_guard" not in by_key


def test_predecoder_topology_roundtrips_through_yaml():
    pre = qec.task_graph_node()
    pre.id = "pre"
    pre.type = "PreDecoder"  # alias in, canonical name out
    pre.inner_type = "single_error_lut"
    pre.node_params = {"residual_from": "parity_check"}
    glob = qec.task_graph_node()
    glob.id = "global"
    glob.type = "single_window"
    glob.inner_type = "multi_error_lut"
    comb = qec.task_graph_node()
    comb.id = "combine"
    comb.type = "xor"

    graph = qec.task_graph_spec()
    graph.nodes = [pre, glob, comb]
    graph.edges = [
        _edge("measurements", "pre"),
        _edge("pre", "global"),
        _edge("pre", "combine", to_input="a"),
        _edge("global", "combine", to_input="b"),
        _edge("combine", "output"),
    ]
    graph.edges[1].from_output = "residual"

    assert graph.nodes[0].resolved_kind() == "predecoder"

    dc = qec.decoder_config()
    dc.id = 0
    dc.block_size = 3
    dc.syndrome_size = 2
    dc.H_sparse = [0, 1, -1, 1, 2, -1]
    dc.O_sparse = [0, -1]
    dc.D_sparse = [0, -1, 1, -1]
    dc.task_graph = graph

    yaml_text = dc.to_yaml_str()
    # Emission writes the canonical spelling, and the residual port survives
    # because it is not the default one.
    assert "predecoder" in yaml_text and "PreDecoder" not in yaml_text
    assert "residual" in yaml_text

    parsed = qec.decoder_config.from_yaml_str(yaml_text)
    assert parsed.task_graph.nodes[0].type == "predecoder"
    assert parsed.task_graph.edges[1].from_output == "residual"


_REP5_CHUNKS = """    dem_chunks:
      seam: {from: next_round, to: prev_round}
      connections:
        - {from: init, to: bulk}
        - {from: bulk, to: bulk}
        - {from: bulk, to: final}
      phases:
        - name: init
          spec: {num_faults: 9, H_sparse: [0,1,5,-1,1,2,6,-1,2,3,7,-1,3,4,8,-1], O_sparse: [0,-1], error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]}
        - name: bulk
          spec: {num_faults: 9, H_sparse: [0,1,5,-1,1,2,6,-1,2,3,7,-1,3,4,8,-1], O_sparse: [0,-1], error_rates: [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]}
        - name: final
          spec: {num_faults: 5, H_sparse: [0,1,-1,1,2,-1,2,3,-1,3,4,-1], O_sparse: [0,-1], error_rates: [0.02,0.02,0.02,0.02,0.02]}
"""


def _predecoder_yaml(with_global):
    """pre --residual--> global --> xor, or `pre` alone. The pre-decoder needs
    the chunk form: its residual is `s XOR H*e`, and H is not recoverable from
    flat D and O."""
    if with_global:
        nodes = """        - id: pre
          type: predecoder
          window_size: 5
          residual_from: parity_check
          inner: {type: single_error_lut}
        - id: global
          type: single_window
          window_size: 5
          inner: {type: multi_error_lut, lut_error_depth: 2}
        - id: combine
          type: xor
      edges:
        - {from: measurements, to: pre}
        - {from: pre.residual, to: global}
        - {from: pre.pre_L, to: combine.a}
        - {from: global, to: combine.b}
        - {from: combine, to: output}
"""
    else:
        nodes = """        - id: pre
          type: predecoder
          window_size: 5
          residual_from: parity_check
          inner: {type: single_error_lut}
"""
    return ("decoders:\n  - id: 0\n    task_graph:\n      nodes:\n" + nodes +
            _REP5_CHUNKS)


# 5 rounds (init, bulk, bulk, bulk, final) x 4 detectors/round. Not decorative:
# at any smaller count enqueue buffers without ever firing and every shot below
# reads back 0, so the x(q[0]) case pins this from beneath while the binding's
# own range check pins it from above.
_REP5_MSYN = 20


def test_predecoder_topology_configures_and_serves_a_shot():
    # The config-layer half of the residual story: the kind is reachable from
    # YAML, the residual edge wires up, and the graph decodes real syndromes.
    # Which node commits which fault column is pinned numerically in C++
    # (ResidualTopologySplitsWorkBetweenPreAndGlobal); the question here is
    # whether the YAML surface yields a graph that decodes at all.
    import cudaq

    @cudaq.kernel
    def shot_kernel(decoder_id: int, num_syndromes: int, num_obs: int,
                    flip: int) -> int:
        q = cudaq.qvector(num_syndromes)
        qec.reset_decoder(decoder_id)
        if flip >= 0:
            x(q[flip])
        syndromes = mz(q)
        qec.enqueue_syndromes(decoder_id, syndromes, 0)
        correction = qec.get_corrections(decoder_id, num_obs, False)
        result = 0
        if correction[0]:
            result = 1
        return result

    def shots(yaml_text, flips):
        cudaq.reset_target()
        assert qec.configure_decoders_from_str(yaml_text) == 0
        try:
            return [
                cudaq.run(shot_kernel, 0, _REP5_MSYN, 1, f, shots_count=1)[0]
                for f in flips
            ]
        finally:
            qec.finalize_decoders()

    # Round 0 detectors are absolute and later rounds are differences, so
    # x(q[j]) flips detectors {j, j+4} for j < 16 and {j} for j >= 16.  q0 gives
    # {0,4}, a lone fault column single_error_lut resolves to observable 1; q3
    # gives {3,7}, which it resolves to observable 0; q17 gives {17}, which
    # matches no column at all, so `pre` abstains and only `global` -- reading
    # the residual -- can answer it.  -1 flips nothing.
    assert shots(_predecoder_yaml(True), [-1, 0, 3, 17]) == [0, 1, 0, 1]

    # The same topology without `global`.  q0 still decodes, and that is what
    # makes the q17 zero mean something: this graph does fire, it simply has
    # nothing able to explain detector 17.  So the 1 above arrived through the
    # residual edge rather than from `pre`.
    assert shots(_predecoder_yaml(False), [0, 17]) == [1, 0]

    # Not vacuous, both verified by mutation: rewiring `pre.residual` to
    # `measurements`
    # drops the q0 case to 0, and dropping `global` drops the q17 case to 0.
    # The all-zero shot this replaced passed with enqueue_syndromes deleted.
    # Giving `pre` a depth-2 inner also breaks it, by letting the pre-only leg
    # answer q17 on its own and collapsing the attribution above.
    #
    # Not claimed: the streaming fold XORs every node's corrections regardless
    # of the output wiring (decoding_task_graph.cpp get_obs_corrections), so
    # deleting `pre -> combine.a` changes no shot here. `combine -> output` is
    # covered, but by configure_decoders rejecting the graph, not by a shot.


def test_predecoder_builds_from_the_flat_form():
    # The flat form already requires H_sparse to build its inner decoder, so a
    # pre-decoder needs no new YAML key -- the entry-level H simply reaches the
    # node now. No dem_chunks here, which is the point.
    yaml_text = """
decoders:
  - id: 0
    task_graph:
      nodes:
        - id: pre
          type: predecoder
          num_syndromes_per_round: 4
          residual_from: parity_check
          inner: {type: single_error_lut}
    block_size: 5
    syndrome_size: 4
    H_sparse: [0,1,-1,1,2,-1,2,3,-1,3,4,-1]
    O_sparse: [0,-1]
    D_sparse: [0,-1,1,-1,2,-1,3,-1]
"""
    assert qec.configure_decoders_from_str(yaml_text) == 0
    qec.finalize_decoders()

    # Not vacuous: the flat form names its five matrices together, so dropping
    # one is rejected. This is not passing because flat configs are waved
    # through -- and it is the parser that refuses, before the graph is built.
    with pytest.raises(RuntimeError, match="D_sparse"):
        qec.configure_decoders_from_str(
            yaml_text.replace("    D_sparse: [0,-1,1,-1,2,-1,3,-1]\n", ""))


def test_predecoder_residual_edge_needs_a_kind_that_produces_one():
    # A sliding_window declares no `residual` output, so the same edge is
    # rejected. Paired with the positive test above, this cannot both pass if
    # compile_wires stopped checking ports.
    # Drop the predecoder's own keys along with its kind, so the rejection can
    # only be about the `residual` port and not an unknown key or port name.
    yaml_text = (_predecoder_yaml(True).replace(
        "type: predecoder", "type: sliding_window").replace(
            "          residual_from: parity_check\n",
            "").replace("pre.pre_L", "pre"))
    assert qec.configure_decoders_from_str(yaml_text) != 0
    qec.finalize_decoders()


def test_json_schema_accepts_a_predecoder_and_rejects_a_guard_on_one():
    node = {
        "id": "pre",
        "type": "predecoder",
        "residual_from": "parity_check",
        "inner": {
            "type": "single_error_lut"
        }
    }
    assert _schema_accepts(_task_graph_doc(node))
    # residual_from has no default, so offline validation catches a predecoder
    # that never declared where its residual comes from.
    # An unknown *value* is not the schema's job -- it types the key as a
    # string, and the node kind's validate hook rejects the spelling.
    bare = {k: v for k, v in node.items() if k != "residual_from"}
    assert not _schema_accepts(_task_graph_doc(bare))
    # start_guard belongs to sliding_window; a predecoder declares no guards.
    # Both directions, so this cannot pass by the schema rejecting everything.
    assert not _schema_accepts(_task_graph_doc(dict(node, start_guard=1)))
    assert _schema_accepts(
        _task_graph_doc(
            dict(bare, type="sliding_window", window_size=2, start_guard=1)))


def test_json_schema_accepts_a_node_with_stim_dem():
    # The node schema is additionalProperties: false, so an unlisted key fails
    # offline validation even though it parses. Both directions asserted, so
    # this cannot pass by the schema accepting everything.
    node = {
        "id": "w",
        "type": "single_window",
        "inner": {
            "type": "chromobius"
        },
        "stim_dem": _COLOUR_DEM
    }
    assert _schema_accepts(_task_graph_doc(node))
    assert not _schema_accepts(
        _task_graph_doc(dict(node, stim_dem_typo=_COLOUR_DEM)))


def test_json_schema_accepts_a_node_with_its_own_matrices():
    # additionalProperties: false, so the block is listed at node level too.
    # Both directions asserted.
    node = {
        "id": "a",
        "type": "single_window",
        "inner": {
            "type": "single_error_lut"
        },
        "block_size": 2,
        "syndrome_size": 2,
        "H_sparse": [0, -1, 1, -1],
        "O_sparse": [0, -1],
        "D_sparse": [0, -1, 1, -1],
    }
    assert _schema_accepts(_task_graph_doc(node))
    # The schema types the block; pairing rules are the parser's.
    assert not _schema_accepts(_task_graph_doc(dict(node, D_sprase=[0, -1])))
    assert not _schema_accepts(_task_graph_doc(dict(node, block_size="two")))


# ---------------------------------------------------------------------------
# Multi-node execution from Python
#
# from_topology() has no Python binding, so a `nodes`/`edges` task graph is
# only reachable from Python through this config layer + configure_decoders().
# The roundtrip tests above never build or decode with a multi-node graph.
# ---------------------------------------------------------------------------


def _node(id_, type_, inner_type=None):
    node = qec.task_graph_node()
    node.id = id_
    node.type = type_
    if inner_type is not None:
        node.inner_type = inner_type
    return node


def _edge(from_node, to_node, to_input="input"):
    edge = qec.task_graph_edge()
    edge.from_node = from_node
    edge.to_node = to_node
    edge.to_input = to_input
    return edge


def _xor_combine_config():
    # Same topology as SessionRegistryLoad.NodesFormMultiNodeGraphServesAShot
    # and DecodingTaskGraph.XorCombineRunDisagreesOnUniqueTwoFaultSyndrome:
    # two single_window branches over one syndrome, wired into an xor node.
    # x and z run different-depth LUT decoders over the same H so a unique
    # two-fault syndrome exists that only z can resolve.
    #
    # The kernel path below is streaming (enqueue_syndromes / get_corrections).
    # xor_node is one-shot-only, so that path XOR-folds the two window nodes'
    # corrections rather than calling xor_node::execute_indexed -- the C++
    # run() test covers the node itself. Construction still validates the
    # combine ports (see the bad-edge test).
    #
    # H is a 4-round bit-flip chain over 5 faults: detector_i = fault_i XOR
    # fault_(i+1). Its null space is the all-ones vector, so two distinct
    # weight-<=2 patterns collide only if they're complements, which needs 4
    # faults (2+2) not 5 (2+3) -- "1001" is produced only by {0, 4}.
    z_node = _node("z", "single_window", "multi_error_lut")
    z_node.decoder_custom_args = {"lut_error_depth": 2}

    graph = qec.task_graph_spec()
    graph.nodes = [
        _node("x", "single_window", "single_error_lut"),
        z_node,
        _node("combine", "xor"),
    ]
    graph.edges = [
        _edge("measurements", "x"),
        _edge("measurements", "z"),
        _edge("x", "combine", to_input="a"),
        _edge("z", "combine", to_input="b"),
        _edge("combine", "output"),
    ]

    config = qec.decoder_config()
    config.id = 0
    config.block_size = 5
    config.syndrome_size = 4
    config.H_sparse = [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1]
    config.O_sparse = [0, -1]
    config.D_sparse = [0, -1, 1, -1, 2, -1, 3, -1]
    config.task_graph = graph
    return config


def test_multi_node_task_graph_with_a_bad_edge_port_is_rejected():
    # Negative counterpart to the shot test below: an edge pointed at a port
    # its destination doesn't have must be rejected, not silently ignored.
    config = _xor_combine_config()
    config.task_graph.edges[2].to_input = "bogus_port"
    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [config]
    try:
        assert qec.configure_decoders(multi_config) != 0
    finally:
        qec.finalize_decoders()


def test_multi_node_task_graph_serves_a_shot_via_configure_decoders():
    import cudaq

    config = _xor_combine_config()
    multi_config = qec.multi_decoder_config()
    multi_config.decoders = [config]
    cudaq.reset_target()
    assert qec.configure_decoders(multi_config) == 0

    # Same in-kernel shape as demo_circuit_qpu (surface_code_1.py):
    # get_corrections() isn't readable from Python outside a kernel, so the
    # kernel returns it directly. reset_decoder() first keeps each shot
    # self-contained -- corrections XOR-accumulate across fires otherwise.
    @cudaq.kernel
    def shot_kernel(decoder_id: int, set_first: bool, set_last: bool,
                    num_obs: int) -> int:
        q = cudaq.qvector(4)
        qec.reset_decoder(decoder_id)
        if set_first:
            x(q[0])
        if set_last:
            x(q[3])
        syndromes = mz(q)
        qec.enqueue_syndromes(decoder_id, syndromes, 0)
        correction = qec.get_corrections(decoder_id, num_obs, False)
        result = 0
        if correction[0]:
            result = 1
        return result

    try:
        # Baseline: zero syndrome, both branches agree at zero.
        assert cudaq.run(shot_kernel, 0, False, False, 1, shots_count=1) == [0]

        # "1000" is fault 0 alone: both decoders agree (observable0 = 1), so
        # the streaming XOR-fold cancels to zero -- rules out forwarding only
        # one branch or a fold hardwired to zero.
        assert cudaq.run(shot_kernel, 0, True, False, 1, shots_count=1) == [0]

        # "1001" is uniquely faults {0, 4} (see _xor_combine_config): x
        # (depth 1) can't resolve it and stays at zero; z (depth 2) resolves
        # it and reports observable0 = 1. The fold of those two streaming
        # corrections is 1.
        assert cudaq.run(shot_kernel, 0, True, True, 1, shots_count=1) == [1]
    finally:
        qec.finalize_decoders()


def test_json_schema_accepts_a_task_graph_without_a_top_level_type():
    assert _schema_accepts(_task_graph_doc(_WINDOW_NODE))


def test_json_schema_rejects_a_task_graph_beside_a_top_level_type():
    assert not _schema_accepts(_task_graph_doc(_WINDOW_NODE, type="pymatching"))


def test_json_schema_rejects_a_kinds_parameter_on_another_kind():
    # step_size belongs to sliding_window; single_window does not slide.
    node = dict(_WINDOW_NODE, step_size=1)
    assert not _schema_accepts(_task_graph_doc(node))
    assert _schema_accepts(
        _task_graph_doc(dict(node, type="sliding_window", window_size=2)))


def test_json_schema_rejects_an_unknown_node_key():
    assert not _schema_accepts(
        _task_graph_doc(dict(_WINDOW_NODE, straddle_end_round=True)))


_GUARDED_WINDOW_YAML = ("""decoders:
  - id: 0
    task_graph:
      nodes:
        - id: w
          type: sliding_window
          window_size: 5
          step_size: 3
          start_guard: 1
          end_guard: 1
          inner: {type: single_error_lut}
""" + _REP5_CHUNKS + """syndrome_mapping:
  - {id: 1, type: final}
""")


def test_a_kernel_can_close_a_guarded_window_by_naming_a_final_row():
    # A guarded window withholds its trailing rounds for a successor that never
    # arrives at the end of a stream; naming a `type: final` row from a kernel
    # commits them.  The only test that covers that from a kernel: the C++ ones
    # drive the graph directly, and the other kernel tests use a plain decoder,
    # which ignores the shape.
    #
    # Break-it: drop the flush_sweep() call from the tagged enqueue_syndrome, or
    # give the window end_guard 0.  Either collapses the two runs.
    import cudaq

    @cudaq.kernel
    def shot_kernel(decoder_id: int, num_syndromes: int, num_obs: int,
                    flip: int, mapping_id: int) -> int:
        q = cudaq.qvector(num_syndromes)
        qec.reset_decoder(decoder_id)
        x(q[flip])
        syndromes = mz(q)
        qec.enqueue_syndromes_mapped(decoder_id, syndromes, 0, mapping_id)
        correction = qec.get_corrections(decoder_id, num_obs, False)
        result = 0
        if correction[0]:
            result = 1
        return result

    # The first measurement of the last round, which lands inside the window's
    # end_guard -- the one region a BULK round withholds.
    per_round = _REP5_MSYN // 5
    flip = _REP5_MSYN - per_round

    cudaq.reset_target()
    assert qec.configure_decoders_from_str(_GUARDED_WINDOW_YAML) == 0
    try:
        # Row 0 is BULK: the window fires and withholds its guarded tail.
        bulk = cudaq.run(shot_kernel, 0, _REP5_MSYN, 1, flip, 0,
                         shots_count=1)[0]
        # Row 1 is FINAL: the same shot, plus the tail.
        final = cudaq.run(shot_kernel, 0, _REP5_MSYN, 1, flip, 1,
                          shots_count=1)[0]
    finally:
        qec.finalize_decoders()

    # The fault sits in the guarded tail, so only the FINAL run accounts for it.
    # A difference rather than two literals: what matters is that naming the row
    # changed the answer.
    assert bulk != final, (
        "naming a `final` row committed nothing the BULK round did not; "
        f"both runs returned {bulk}")


def test_enqueue_syndromes_mapped_names_a_syndrome_mapping_row():
    # A Python device kernel can name a syndrome_mapping row.  It needs its own
    # kernel name because the device-kernel registry holds one argument count
    # per name -- qec.enqueue_syndromes stays 3-argument.
    import cudaq

    yaml_text = """decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
syndrome_mapping:
  - {id: 1, type: final}
"""
    cudaq.reset_target()
    assert qec.configure_decoders(
        qec.multi_decoder_config.from_yaml_str(yaml_text)) == 0

    @cudaq.kernel
    def shot_kernel(decoder_id: int, flip: int, mapping_id: int) -> int:
        q = cudaq.qvector(3)
        qec.reset_decoder(decoder_id)
        if flip >= 0:
            x(q[flip])
        syndromes = mz(q)
        qec.enqueue_syndromes_mapped(decoder_id, syndromes, 0, mapping_id)
        correction = qec.get_corrections(decoder_id, 3, False)
        result = 0
        if correction[0]:
            result = 1
        return result

    try:
        # The mapped kernel reaches the decoder: a fault on qubit 0 decodes to
        # observable 0 through it, exactly as the 3-argument form would.  Zero
        # syndrome stays zero, so the assertion is not satisfied by a decoder
        # that always fires.
        assert cudaq.run(shot_kernel, 0, -1, 0, shots_count=1) == [0]
        assert cudaq.run(shot_kernel, 0, 0, 0, shots_count=1) == [1]
        # The declared row works too.  A plain decoder ignores the round shape,
        # so this pins the id reaching the host, not what the shape does there.
        assert cudaq.run(shot_kernel, 0, 0, 1, shots_count=1) == [1]
    finally:
        qec.finalize_decoders()


def test_enqueue_syndromes_mapped_rejects_an_undeclared_row():
    # An id no row declares is a configuration error, the same answer the
    # decoding server gives.  Break-it: have shape_for_mapping_id fall back to
    # BULK for an unknown id.
    import cudaq

    yaml_text = """decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
"""
    cudaq.reset_target()
    assert qec.configure_decoders(
        qec.multi_decoder_config.from_yaml_str(yaml_text)) == 0

    @cudaq.kernel
    def shot_kernel(decoder_id: int, mapping_id: int) -> int:
        q = cudaq.qvector(3)
        qec.reset_decoder(decoder_id)
        syndromes = mz(q)
        qec.enqueue_syndromes_mapped(decoder_id, syndromes, 0, mapping_id)
        return 0

    try:
        # id 0 is the seeded pass-through row and is always serviceable.
        assert cudaq.run(shot_kernel, 0, 0, shots_count=1) == [0]
        with pytest.raises(RuntimeError):
            cudaq.run(shot_kernel, 0, 7, shots_count=1)
    finally:
        qec.finalize_decoders()


def test_syndrome_mapping_round_trips_through_yaml_and_the_object_api():
    yaml_text = """
decoders:
  - id: 0
    type: single_error_lut
    block_size: 3
    syndrome_size: 3
    H_sparse: [0,-1,1,-1,2,-1]
    O_sparse: [0,-1,1,-1,2,-1]
    D_sparse: [0,-1,1,-1,2,-1]
syndrome_mapping:
  - {id: 0, type: bulk, vps: [[]]}
  - {id: 1, type: final, vps: [[]]}
"""
    parsed = qec.multi_decoder_config.from_yaml_str(yaml_text)
    assert [(r.id, r.type) for r in parsed.syndrome_mapping] == [(0, "bulk"),
                                                                 (1, "final")]
    # The section is optional, which is what keeps every pre-existing
    # configuration parsing unchanged.
    without = yaml_text[:yaml_text.index("syndrome_mapping:")]
    assert qec.multi_decoder_config.from_yaml_str(
        without).syndrome_mapping == []


def test_syndrome_mapping_row_validates():
    row = qec.syndrome_mapping_config()
    row.id, row.type, row.vps = 1, "final", [[]]
    row.validate()

    row.type = "middle"
    with pytest.raises(RuntimeError, match="bulk"):
        row.validate()

    # Per-port scatter is parsed but not honored yet, so a populated index list
    # is refused rather than silently ignored.
    row.type, row.vps = "bulk", [[0, 1]]
    with pytest.raises(RuntimeError, match="does not scatter"):
        row.validate()


def test_json_schema_accepts_syndrome_mapping():
    doc = _task_graph_doc(_WINDOW_NODE)
    doc["syndrome_mapping"] = [{"id": 0, "type": "bulk", "vps": [[]]}]
    assert _schema_accepts(doc)
    # Not vacuous: the row schema constrains `type` and rejects unknown keys.
    doc["syndrome_mapping"] = [{"id": 0, "type": "middle", "vps": [[]]}]
    assert not _schema_accepts(doc)
    doc["syndrome_mapping"] = [{"id": 0, "vps": [[]], "bogus": 1}]
    assert not _schema_accepts(doc)


def test_object_api_validates_syndrome_mapping():
    # from_yaml_str validates on parse, but a programmatically built config
    # never sees the parser -- and an unknown shape would silently read as
    # BULK, a duplicate id would resolve arbitrarily, and a populated vps
    # would be ignored.
    def row(i, t, vps):
        r = qec.syndrome_mapping_config()
        r.id, r.type, r.vps = i, t, vps
        return r

    dec = qec.decoder_config()
    dec.id, dec.type = 0, "single_error_lut"
    dec.block_size = dec.syndrome_size = 3
    dec.H_sparse = dec.O_sparse = dec.D_sparse = [0, -1, 1, -1, 2, -1]

    def configure(rows):
        m = qec.multi_decoder_config()
        m.decoders, m.syndrome_mapping = [dec], rows
        try:
            return qec.configure_decoders(m)
        finally:
            qec.finalize_decoders()

    assert configure([row(1, "final", [[]])]) == 0
    assert configure([]) == 0
    # Break-it: drop validate_syndrome_mapping() from configure_decoders.
    assert configure([row(1, "middle", [[]])]) != 0
    assert configure([row(1, "final", [[]]), row(1, "bulk", [[]])]) != 0
    assert configure([row(1, "final", [[0, 1]])]) != 0
