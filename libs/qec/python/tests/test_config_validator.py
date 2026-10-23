# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for libs/qec/tools/config-validator/validate_decoder_config.py.

The validator is a standalone script (pyyaml + jsonschema, no cudaq_qec
import on the validating machine). These tests load it as a module and run
it against the schema exported by this installation.
"""

import importlib.util
import json
import pathlib

import pytest

import cudaq_qec as qec

_QEC_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TOOL = _QEC_ROOT / "tools" / "config-validator" / "validate_decoder_config.py"
_SERVER_CONFIGS = _QEC_ROOT / "tools" / "decoding-server"

_GOOD_YAML = """\
decoders:
  - id: 0
    type: pymatching
    block_size: 3
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args:
      merge_strategy: smallest_weight
      error_rate_vec: [0.1, 0.1, 0.1]
"""


@pytest.fixture(scope="module")
def tool():
    pytest.importorskip("jsonschema")
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("validate_decoder_config",
                                                  _TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def schema_file(tool, tmp_path_factory):
    path = tmp_path_factory.mktemp("schema") / "decoder_config.schema.json"
    path.write_text(qec.decoder_config_json_schema())
    return path


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def _run(tool, capsys, *argv):
    status = tool.main(list(argv))
    return status, capsys.readouterr().out


def test_yaml12_floats_without_a_dot_in_the_mantissa(tool):
    # PyYAML's YAML 1.1 resolver would read 6e-05 and 1e3 as strings; the
    # server's LLVM YAML parser and this loader read them as floats.
    doc = tool.load_yaml("v: [6e-05, 6.0e-05, 1e3, .5, 0.1, 7, -2]")
    assert doc["v"] == [6e-05, 6e-05, 1000.0, 0.5, 0.1, 7, -2]
    assert all(isinstance(x, float) for x in doc["v"][:5])
    assert all(isinstance(x, int) for x in doc["v"][5:])


def test_valid_config_passes(tool, schema_file, tmp_path, capsys):
    good = _write(tmp_path, "good.yml", _GOOD_YAML)
    status, out = _run(tool, capsys, "--schema", str(schema_file), good)
    assert status == 0
    assert out == f"{good}: OK\n"


def test_scientific_notation_error_rates_pass(tool, schema_file, tmp_path,
                                              capsys):
    text = _GOOD_YAML.replace("[0.1, 0.1, 0.1]", "[6e-05, 6e-05, 1e-3]")
    path = _write(tmp_path, "sci.yml", text)
    status, out = _run(tool, capsys, "--schema", str(schema_file), path)
    assert status == 0, out


def test_typo_is_reported_with_its_location(tool, schema_file, tmp_path,
                                            capsys):
    text = _GOOD_YAML.replace("merge_strategy", "merge_stratgey")
    path = _write(tmp_path, "typo.yml", text)
    status, out = _run(tool, capsys, "--schema", str(schema_file), path)
    assert status == 1
    assert out.startswith(f"{path}: FAIL\n")
    assert "decoders/0/decoder_custom_args: " in out
    assert "merge_stratgey" in out


def test_wrong_type_is_reported(tool, schema_file, tmp_path, capsys):
    text = _GOOD_YAML.replace("syndrome_size: 3", "syndrome_size: three")
    path = _write(tmp_path, "type.yml", text)
    status, out = _run(tool, capsys, "--schema", str(schema_file), path)
    assert status == 1
    assert "decoders/0/syndrome_size: " in out


def test_unknown_decoder_type_with_args_is_explained(tool, schema_file,
                                                     tmp_path, capsys):
    # A type the schema does not know may carry no args; jsonschema would
    # only say the args mapping "has too many properties". Name the cause.
    text = _GOOD_YAML.replace("type: pymatching", "type: pymatchnig")
    path = _write(tmp_path, "unknown.yml", text)
    status, out = _run(tool, capsys, "--schema", str(schema_file), path)
    assert status == 1
    assert "decoders/0: decoder type 'pymatchnig' has no parameter schema" in out
    assert "too many properties" not in out


def test_malformed_yaml_fails_without_a_traceback(tool, schema_file, tmp_path,
                                                  capsys):
    path = _write(tmp_path, "broken.yml", "decoders: [\n  - id: 0\n")
    status, out = _run(tool, capsys, "--schema", str(schema_file), path)
    assert status == 1
    assert "not valid YAML" in out


def test_task_graph_is_rejected_unless_allowed(tool, schema_file, capsys):
    path = str(_SERVER_CONFIGS / "decoding_server_config_task_graph.yaml")
    status, out = _run(tool, capsys, "--schema", str(schema_file), path)
    assert status == 1
    assert "decoders/0: task_graph configurations are not supported" in out

    status, out = _run(tool, capsys, "--schema", str(schema_file),
                       "--allow-task-graph", path)
    assert status == 0, out


def test_repo_example_configs_validate(tool, schema_file, capsys):
    paths = sorted(
        str(p)
        for p in _SERVER_CONFIGS.glob("*.yaml")
        if "task_graph" not in p.name)
    assert paths
    status, out = _run(tool, capsys, "--schema", str(schema_file), *paths)
    assert status == 0, out


def test_schema_comes_from_the_installed_package_when_not_given(
        tool, tmp_path, capsys):
    good = _write(tmp_path, "good.yml", _GOOD_YAML)
    status, out = _run(tool, capsys, good)
    assert status == 0, out


def test_export_schema_strips_task_graph(tool, tmp_path, capsys):
    out_path = tmp_path / "exported.json"
    status, _ = _run(tool, capsys, "--export-schema", str(out_path))
    assert status == 0
    exported = json.loads(out_path.read_text())
    assert "task_graph" not in exported["$defs"]["decoder_config"]["properties"]
    full = json.loads(qec.decoder_config_json_schema())
    assert "task_graph" in full["$defs"]["decoder_config"]["properties"]
    assert exported["$defs"]["decoder_params"] == full["$defs"][
        "decoder_params"]

    out_path = tmp_path / "with_task_graph.json"
    status, _ = _run(tool, capsys, "--allow-task-graph", "--export-schema",
                     str(out_path))
    assert status == 0
    assert json.loads(out_path.read_text()) == full


def test_mixed_results_exit_nonzero_and_report_each_file(
        tool, schema_file, tmp_path, capsys):
    good = _write(tmp_path, "good.yml", _GOOD_YAML)
    bad = _write(tmp_path, "bad.yml",
                 _GOOD_YAML.replace("D_sparse", "D_sparse_matrix"))
    status, out = _run(tool, capsys, "--schema", str(schema_file), good, bad)
    assert status == 1
    assert f"{good}: OK\n" in out
    assert f"{bad}: FAIL\n" in out
    assert "D_sparse_matrix" in out
