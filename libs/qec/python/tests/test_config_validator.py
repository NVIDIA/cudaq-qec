# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for libs/qec/tools/config-validator/validate_decoder_config.py.

The validator is a standalone script (pyyaml + jsonschema, no cudaq_qec
import on the validating machine). These tests load it as a module, run it
against the schema exported by this installation, and -- the part that
matters -- compare its verdicts with the server's own parser
(``multi_decoder_config.from_yaml_str``) on a corpus of documents.
"""

import importlib.util
import json
import pathlib
import re

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


# --- command line behaviour -------------------------------------------------


def test_valid_config_passes(tool, schema_file, tmp_path, capsys):
    good = _write(tmp_path, "good.yml", _GOOD_YAML)
    status, out = _run(tool, capsys, "--schema", str(schema_file), good)
    assert status == 0
    assert out == f"{good}: OK\n"


def test_scientific_notation_error_rates_pass(tool, schema_file, tmp_path,
                                              capsys):
    # PyYAML's YAML 1.1 resolver reads 6e-05 as a string; the server's LLVM
    # YAML parser reads it as a float, and so must the validator.
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
    # A type the schema does not know may carry no args section at all;
    # jsonschema would only say the args mapping "has too many properties".
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


def _installed_lib_dir():
    """Directory with libcudaq-qec-realtime-decoding.so for this cudaq_qec."""
    here = pathlib.Path(qec.__file__).resolve().parent
    for candidate in (here / "lib", here.parent.parent / "lib",
                      here.parent.parent.parent / "lib"):
        if (candidate / "libcudaq-qec-realtime-decoding.so").exists():
            return candidate
    return None


def test_lib_dir_export_matches_the_python_export(tool, tmp_path, capsys):
    # --lib-dir reads the schema through the libraries' C entry point, with
    # no Python package involved; it must be the schema of this installation.
    lib_dir = _installed_lib_dir()
    if lib_dir is None:
        pytest.skip("library directory of this installation not found")
    try:
        exported = tool.schema_from_library_dir(str(lib_dir))
    except RuntimeError as exc:
        if "predates" in str(exc):
            pytest.skip(str(exc))
        raise
    assert exported == json.loads(qec.decoder_config_json_schema())

    out_path = tmp_path / "from-lib.json"
    status, _ = _run(tool, capsys, "--lib-dir", str(lib_dir), "--export-schema",
                     str(out_path))
    assert status == 0
    assert "task_graph" not in json.loads(
        out_path.read_text())["$defs"]["decoder_config"]["properties"]


def test_lib_dir_errors_are_explained(tool, tmp_path, capsys):
    status = tool.main([
        "--lib-dir",
        str(tmp_path), "--export-schema",
        str(tmp_path / "x.json")
    ])
    assert status == 2
    assert "not found" in capsys.readouterr().err


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


# --- scalar rules: the validator reads scalars like llvm::yaml --------------


def test_scalars_are_loaded_as_text(tool):
    doc = tool.load_yaml("a: 6e-05\nb: yes\nc: ~\nd: '3'\ne: 010\n")
    assert doc == {"a": "6e-05", "b": "yes", "c": "~", "d": "3", "e": "010"}


@pytest.mark.parametrize("text, value", [
    ("3", 3),
    ("'3'", "3"),
    ("010", 8),
    ("0o10", 8),
    ("0x1f", 31),
    ("0X1f", 31),
    ("0b11", 3),
    ("-1", -1),
    ("00", 0),
    ("18446744073709551615", 2**64 - 1),
])
def test_integer_rule_accepts(tool, text, value):
    scalar = tool.load_yaml(f"v: {text}")["v"]
    assert tool.LlvmScalars.integer(scalar) == (value if isinstance(value, int)
                                                else int(value))


@pytest.mark.parametrize("text", [
    "+3", "0O10", "0x", "", "1.", "3.0", "1e3", "1_000", "1:30", " 3 ",
    "18446744073709551616", "three", "yes"
])
def test_integer_rule_rejects(tool, text):
    assert tool.LlvmScalars.integer(text) is None


@pytest.mark.parametrize("text, value", [
    ("6e-05", 6e-05),
    ("1e3", 1000.0),
    (".5", 0.5),
    ("1.", 1.0),
    ("1.e3", 1000.0),
    ("+.5e-3", 0.0005),
    ("0x10", 16.0),
    ("0x1p-3", 0.125),
    ("010", 10.0),
    ("1", 1.0),
    ("-0.0", -0.0),
    ("", 0.0),
])
def test_number_rule_accepts(tool, text, value):
    assert tool.LlvmScalars.number(text) == value


def test_number_rule_accepts_inf_and_nan(tool):
    import math
    assert tool.LlvmScalars.number("inf") == math.inf
    assert tool.LlvmScalars.number("-Inf") == -math.inf
    assert math.isnan(tool.LlvmScalars.number("nan"))
    assert tool.LlvmScalars.number("1e400") == math.inf


@pytest.mark.parametrize("text",
                         ["1_000.5", ".inf", ".nan", "yes", "1e", ".e3", "1,5"])
def test_number_rule_rejects(tool, text):
    assert tool.LlvmScalars.number(text) is None


def test_boolean_rule_is_llvm_parsebool(tool):
    yes = [
        "y", "Y", "yes", "Yes", "YES", "true", "True", "TRUE", "on", "On", "ON"
    ]
    no = [
        "n", "N", "no", "No", "NO", "false", "False", "FALSE", "off", "Off",
        "OFF"
    ]
    assert all(tool.LlvmScalars.boolean(t) is True for t in yes)
    assert all(tool.LlvmScalars.boolean(t) is False for t in no)
    for t in ["1", "0", "t", "f", "T", "yEs", "", "~", "null"]:
        assert tool.LlvmScalars.boolean(t) is None


def test_duplicate_keys_and_aliases_are_rejected(tool):
    import yaml
    with pytest.raises(yaml.YAMLError, match="duplicated mapping key"):
        tool.load_yaml("a: 1\nb: 2\na: 3\n")
    with pytest.raises(yaml.YAMLError, match="aliases are not supported"):
        tool.load_yaml("a: &x [1]\nb: *x\n")


# --- differential test: validator verdict == server parser verdict ----------
#
# Every document below goes through the validator and through
# multi_decoder_config.from_yaml_str, the parser the decoding server uses.
# Both must accept or both must reject. from_yaml_str also runs the
# per-decoder validate hooks; cases that only a hook rejects would show up
# here as a disagreement, so the corpus is built from structural and scalar
# variations, which is what the validator claims to cover.

_BASE = """decoders:
  - id: {id}
    type: {type}
    block_size: {block_size}
    syndrome_size: 3
    H_sparse: [0, -1, 1, -1, 2, -1]
    O_sparse: [0, -1, 1, -1, 2, -1]
    D_sparse: [0, -1, 1, -1, 2, -1]
    decoder_custom_args:
{args}"""

_PM_ARGS = ("      merge_strategy: {ms}\n"
            "      error_rate_vec: [{erv}, 0.1, 0.1]\n")
_CH_ARGS = "      return_weight: {b}\n"


def _doc(id="0",
         type="pymatching",
         block_size="3",
         ms="smallest_weight",
         erv="0.1",
         args=None):
    if args is None:
        args = _PM_ARGS.format(ms=ms, erv=erv)
    return _BASE.format(id=id, type=type, block_size=block_size, args=args)


def _corpus_cases():
    cases = {
        "baseline":
            _doc(),
        # floats (error_rate_vec)
        "float 6e-05":
            _doc(erv="6e-05"),
        "float 1e3":
            _doc(erv="1e3"),
        "float .5":
            _doc(erv=".5"),
        "float 1.":
            _doc(erv="1."),
        "float +.5e-3":
            _doc(erv="+.5e-3"),
        "float quoted":
            _doc(erv="'0.1'"),
        "float int":
            _doc(erv="1"),
        "float 0x10":
            _doc(erv="0x10"),
        "float hex p":
            _doc(erv="0x1p-3"),
        "float inf":
            _doc(erv="inf"),
        "float nan":
            _doc(erv="nan"),
        "float 1_000.5":
            _doc(erv="1_000.5"),
        "float .inf":
            _doc(erv=".inf"),
        "float .nan":
            _doc(erv=".nan"),
        "float yes":
            _doc(erv="yes"),
        "float 1e":
            _doc(erv="1e"),
        "float .e3":
            _doc(erv=".e3"),
        # unsigned integers (block_size)
        "int 010":
            _doc(block_size="010"),
        "int 0o10":
            _doc(block_size="0o10"),
        "int 0O10":
            _doc(block_size="0O10"),
        "int 0x10":
            _doc(block_size="0x10"),
        "int 0b11":
            _doc(block_size="0b11"),
        "int quoted":
            _doc(block_size="'3'"),
        "int +3":
            _doc(block_size="+3"),
        "int -1":
            _doc(block_size="-1"),
        "int -0":
            _doc(block_size="-0"),
        "int 1_000":
            _doc(block_size="1_000"),
        "int 1e3":
            _doc(block_size="1e3"),
        "int 3.0":
            _doc(block_size="3.0"),
        "int 1.":
            _doc(block_size="1."),
        "int 1:30":
            _doc(block_size="1:30"),
        "int uint64 max":
            _doc(block_size="18446744073709551615"),
        "int uint64 max + 1":
            _doc(block_size="18446744073709551616"),
        "int spaces":
            _doc(block_size="' 3 '"),
        "int empty":
            _doc(block_size="''"),
        # signed integers (id)
        "id -1":
            _doc(id="-1"),
        "id +1":
            _doc(id="+1"),
        "id quoted":
            _doc(id="'7'"),
        # strings (merge_strategy, type)
        "str on":
            _doc(ms="on"),
        "str ~":
            _doc(ms="~"),
        "str null":
            _doc(ms="null"),
        "str 123":
            _doc(ms="123"),
        "str 1e3":
            _doc(ms="1e3"),
        "str empty":
            _doc(ms="''"),
        "type yes":
            _doc(type="yes"),
        # booleans (chromobius return_weight)
        **{
            f"bool {b}": _doc(type="chromobius", args=_CH_ARGS.format(b=b)) for b in [
                "true", "True", "TRUE", "yes", "Yes", "YES", "y", "Y", "on", "On", "ON", "false", "no", "n", "off", "Off", "'yes'", "1", "0", "t", "T", "yEs", "~", "null", "''"
            ]
        },
        # structure
        "dup key":
            _doc().replace("    syndrome_size: 3\n",
                           "    syndrome_size: 3\n    block_size: 4\n"),
        "dup key in args":
            _doc(args=_PM_ARGS.format(ms="smallest_weight", erv="0.1") +
                 "      merge_strategy: smallest_weight\n"),
        "alias":
            _doc().replace("H_sparse: [0, -1, 1, -1, 2, -1]",
                           "H_sparse: &m [0, -1, 1, -1, 2, -1]").replace(
                               "O_sparse: [0, -1, 1, -1, 2, -1]",
                               "O_sparse: *m"),
        "merge key":
            "base: &b\n  merge_strategy: smallest_weight\n" +
            _doc(args="      <<: *b\n      error_rate_vec: [0.1, 0.1, 0.1]\n"),
        "tab indentation":
            _doc().replace("    type:", "\ttype:"),
        "args {}":
            _doc(args="      {}\n"),
        "args ~":
            _doc(args="").replace("decoder_custom_args:\n",
                                  "decoder_custom_args: ~\n"),
        "args empty value":
            _doc(args=""),
        "unknown top key":
            _doc().replace("decoders:", "extra: 1\ndecoders:"),
        "document marker":
            _doc() + "---\n",
        "trailing comma":
            _doc().replace("[0.1, 0.1, 0.1]", "[0.1, 0.1, 0.1,]"),
        "double comma":
            _doc().replace("[0.1, 0.1, 0.1]", "[0.1,, 0.1, 0.1]"),
        "sparse float":
            _doc().replace("H_sparse: [0, -1, 1, -1, 2, -1]",
                           "H_sparse: [0, -1, 1.0, -1, 2, -1]"),
        "sparse -2":
            _doc().replace("H_sparse: [0, -1, 1, -1, 2, -1]",
                           "H_sparse: [0, -2, 1, -1, 2, -1]"),
        "missing D_sparse":
            _doc().replace("    D_sparse: [0, -1, 1, -1, 2, -1]\n", ""),
        "missing type":
            _doc().replace("    type: pymatching\n", ""),
        "typo in args":
            _doc().replace("merge_strategy", "merge_stratgey"),
        "typo in envelope":
            _doc().replace("D_sparse", "D_sparse_matrix"),
        "unknown type with args":
            _doc(type="pymatchnig"),
        "unknown type, args {}":
            _doc(type="pymatchnig", args="      {}\n"),
        "window_size string":
            _doc(type="sliding_window",
                 args="      inner_decoder_name: pymatching\n"
                 "      window_size: two\n"),
        "sliding_window missing inner":
            _doc(type="sliding_window", args="      window_size: 2\n"),
        "decoders not a list":
            "decoders: 3\n",
        "root is a list":
            "- decoders: []\n",
        "empty document":
            "",
    }
    cases.update(_more_cases())
    cases.update(_round6_cases())
    return cases


_CHUNK_PHASES = (
    "        - name: init\n"
    "          spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.02, 0.02]}\n"
    "        - name: bulk\n"
    "          spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.02, 0.02]}\n"
    "        - name: final\n"
    "          spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], error_rates: [0.02, 0.02]}\n"
)
_CHUNKS = ("decoders:\n"
           "  - id: 0\n"
           "    type: pymatching\n"
           "    dem_chunks:\n"
           "      seam: {from: next_round, to: prev_round}\n"
           "      connections:\n"
           "        - {from: init, to: bulk}\n"
           "        - {from: bulk, to: bulk}\n"
           "        - {from: bulk, to: final}\n"
           "      num_rounds: 3\n"
           "      phases:\n" + _CHUNK_PHASES + "    decoder_custom_args:\n"
           "      merge_strategy: smallest_weight\n")
_SPEC = ("spec: {num_faults: 2, H_sparse: [0, 1, -1], O_sparse: [0, -1], "
         "error_rates: [0.02, 0.02]}\n        - name: bulk")
_CONNECTIONS = (
    "      connections:\n        - {from: init, to: bulk}\n"
    "        - {from: bulk, to: bulk}\n        - {from: bulk, to: final}\n")


def _chunks(old, new):
    assert old in _CHUNKS, old
    return _CHUNKS.replace(old, new)


def _sm(rows):
    return _doc() + "syndrome_mapping:\n" + rows


def _ms(value):
    return _doc().replace("merge_strategy: smallest_weight",
                          f"merge_strategy: {value}")


def _round6_cases():
    cases = {
        # anchors without aliases are fine on both sides
        "bare anchor on list":
            _h("&m [0, -1, 1, -1, 2, -1]"),
        "bare anchor on scalar":
            _doc(block_size="&b 3"),
        "bare anchor on mapping":
            _doc().replace("decoder_custom_args:\n",
                           "decoder_custom_args: &a\n"),
        # 64-bit signed id, 32-bit cuda_device_id
        "id int64 max":
            _doc(id="9223372036854775807"),
        "id int64 max + 1":
            _doc(id="9223372036854775808"),
        "id int64 min":
            _doc(id="-9223372036854775808"),
        "id int64 min - 1":
            _doc(id="-9223372036854775809"),
        "id uint64 max":
            _doc(id="18446744073709551615"),
        "cuda_device_id int32 max":
            _env("    cuda_device_id: 2147483647\n"),
        "cuda_device_id int32 max + 1":
            _env("    cuda_device_id: 2147483648\n"),
        "cuda_device_id int32 min":
            _env("    cuda_device_id: -2147483648\n"),
        # numerals
        "int 08":
            _doc(block_size="08"),
        "int 09":
            _doc(block_size="09"),
        "int 0_8":
            _doc(block_size="0_8"),
        "int 0x-10":
            _doc(block_size="0x-10"),
        "int +0x10":
            _doc(block_size="+0x10"),
        "int 0b":
            _doc(block_size="0b"),
        "int 0b102":
            _doc(block_size="0b102"),
        "id -0x10":
            _doc(id="-0x10"),
        "float 1e05":
            _doc(erv="1e05"),
        "float 1E3":
            _doc(erv="1E3"),
        "float 1e+3":
            _doc(erv="1e+3"),
        "float -.5":
            _doc(erv="-.5"),
        "float +inf":
            _doc(erv="+inf"),
        "float INF":
            _doc(erv="INF"),
        "float Infinity":
            _doc(erv="Infinity"),
        "float NAN":
            _doc(erv="NAN"),
        "float nAn":
            _doc(erv="nAn"),
        "float 0x1p":
            _doc(erv="0x1p"),
        "float 0x.8":
            _doc(erv="0x.8"),
        "float 1.0e":
            _doc(erv="1.0e"),
        "float 1..0":
            _doc(erv="1..0"),
        # strings: chomping, escapes, plain-scalar starts
        "block literal keep":
            _ms("|\n        smallest_weight"),
        "block literal strip":
            _ms("|-\n        smallest_weight"),
        "block literal keep all":
            _ms("|+\n        smallest_weight\n\n"),
        "block folded two lines":
            _ms(">\n        smallest\n        weight"),
        "block literal with indent indicator":
            _ms("|2\n        smallest_weight"),
        "dq escape x":
            _ms('"smallest\\x5fweight"'),
        "dq escape u":
            _ms('"smallest\\u005fweight"'),
        "dq escape N":
            _ms('"a\\Nb"'),
        "dq escape underscore":
            _ms('"a\\_b"'),
        "dq invalid escape":
            _ms('"a\\qb"'),
        "dq line continuation":
            _ms('"smallest_\\\n        weight"'),
        "dq multi-line":
            _ms('"smallest\n        weight"'),
        "plain multi-line":
            _ms("smallest\n        weight"),
        "sq multi-line":
            _ms("'smallest\n        weight'"),
        "plain with colon space":
            _ms("a: b"),
        "plain with space hash":
            _ms("a #b"),
        "plain starting with dash":
            _ms("-x"),
        "plain starting with question":
            _ms("?x"),
        "plain starting with star":
            _ms("*x"),
        "plain starting with ampersand":
            _ms("&x y"),
        "plain starting with percent":
            _ms("%x"),
        "plain starting with at":
            _ms("@x"),
        "plain starting with backtick":
            _ms("`x"),
        "sq trailing spaces":
            _ms("'x  '"),
        "NUL in string":
            _ms("smallest\x00weight"),
        "latin-1 char in string":
            _ms("smallest_w\u00e9ight"),
        # keys
        "quoted key":
            _doc().replace("type: pymatching", "'type': pymatching"),
        "dup key plain and quoted":
            _doc().replace("    syndrome_size: 3\n",
                           "    syndrome_size: 3\n    'block_size': 4\n"),
        "key with escape":
            _doc().replace("type: pymatching", '"ty\\x70e": pymatching'),
        "key capitalised":
            _doc().replace("type: pymatching", "Type: pymatching"),
        "key with space":
            _doc().replace("block_size: 3", "block size: 3"),
        "empty key":
            _env("    '': 1\n"),
        "tilde key":
            _env("    ~: 1\n"),
        # whitespace inside flow collections, block scalar with tab
        "tab inside flow sequence":
            _doc().replace("[0.1, 0.1, 0.1]", "[0.1,\t0.1, 0.1]"),
        "tab inside flow mapping":
            _doc(
                args=
                "      {merge_strategy: smallest_weight,\terror_rate_vec: [0.1, 0.1, 0.1]}\n"
            ),
        "tab indenting block scalar":
            _ms("|\n\tsmallest_weight"),
        "CRLF inside block scalar":
            _ms("|-\r\n        smallest_weight"),
        # directives and documents
        "TAG directive":
            "%TAG ! tag:example.com,2026:\n---\n" + _doc(),
        "unknown directive":
            "%FOO bar\n---\n" + _doc(),
        "YAML 1.3 directive":
            "%YAML 1.3\n---\n" + _doc(),
        "YAML directive twice":
            "%YAML 1.2\n%YAML 1.2\n---\n" + _doc(),
        "comment-only document":
            "# nothing\n",
        "only document marker":
            "---\n",
        "marker then end marker":
            "---\n...\n",
        "end marker then garbage":
            _doc() + "...\ngarbage here\n",
        "explicit document tag":
            "--- !!map\n" + _doc(),
        "marker on the content line":
            "--- " + _doc(),
        # dem_chunks form
        "chunks baseline":
            _CHUNKS,
        "chunks num_rounds 1":
            _chunks("num_rounds: 3", "num_rounds: 1"),
        "chunks num_rounds absent":
            _chunks("      num_rounds: 3\n", ""),
        "chunks num_rounds 0":
            _chunks("num_rounds: 3", "num_rounds: 0"),
        "chunks rate 1.5":
            _chunks("error_rates: [0.02, 0.02]", "error_rates: [1.5, 0.02]"),
        "chunks rate -0.1":
            _chunks("error_rates: [0.02, 0.02]", "error_rates: [-0.1, 0.02]"),
        "chunks rate 1e-3":
            _chunks("error_rates: [0.02, 0.02]", "error_rates: [1e-3, 0.02]"),
        "chunks with block_size":
            _chunks("    dem_chunks:", "    block_size: 3\n    dem_chunks:"),
        "chunks with syndrome_size":
            _chunks("    dem_chunks:", "    syndrome_size: 3\n    dem_chunks:"),
        "chunks with O_sparse":
            _chunks("    dem_chunks:",
                    "    O_sparse: [0, -1]\n    dem_chunks:"),
        "chunks with D_sparse":
            _chunks("    dem_chunks:",
                    "    D_sparse: [0, -1]\n    dem_chunks:"),
        "chunks with top error_rate_vec":
            _chunks("    dem_chunks:",
                    "    error_rate_vec: [0.1]\n    dem_chunks:"),
        "chunks with args error_rate_vec":
            _chunks(
                "      merge_strategy: smallest_weight\n",
                "      merge_strategy: smallest_weight\n      error_rate_vec: [0.1]\n"
            ),
        "chunks with stim_dem_path":
            _chunks("    dem_chunks:",
                    "    stim_dem_path: x.dem\n    dem_chunks:"),
        "chunks with flat H (flat wins)":
            _chunks(
                "    dem_chunks:",
                "    block_size: 2\n    syndrome_size: 1\n    H_sparse: [0, 1, -1]\n    O_sparse: [0, -1]\n    D_sparse: [0, -1]\n    dem_chunks:"
            ),
        "chunks phases empty":
            _chunks("      phases:\n" + _CHUNK_PHASES, "      phases: []\n"),
        "chunks phases missing":
            _chunks("      phases:\n" + _CHUNK_PHASES, ""),
        "chunks seam missing":
            _chunks("      seam: {from: next_round, to: prev_round}\n", ""),
        "chunks connections missing":
            _chunks(_CONNECTIONS, ""),
        "chunks connections empty":
            _chunks(_CONNECTIONS, "      connections: []\n"),
        "chunks unknown key":
            _chunks("      num_rounds: 3\n",
                    "      num_rounds: 3\n      bogus: 1\n"),
        "chunks phase unknown key":
            _chunks("        - name: init\n",
                    "        - name: init\n          extra: 1\n"),
        "chunks spec unknown key":
            _chunks(
                _SPEC,
                _SPEC.replace("error_rates: [0.02, 0.02]}",
                              "error_rates: [0.02, 0.02], bogus: 1}")),
        "chunks spec missing num_faults":
            _chunks(_SPEC, _SPEC.replace("num_faults: 2, ", "")),
        "chunks spec missing error_rates":
            _chunks(_SPEC, _SPEC.replace(", error_rates: [0.02, 0.02]", "")),
        "chunks num_faults 0":
            _chunks("num_faults: 2", "num_faults: 0"),
        "chunks num_faults 3 with 2 rates":
            _chunks(_SPEC, _SPEC.replace("num_faults: 2", "num_faults: 3")),
        "chunks 3 rates with 2 faults":
            _chunks(
                _SPEC,
                _SPEC.replace("error_rates: [0.02, 0.02]",
                              "error_rates: [0.02, 0.02, 0.02]")),
        "chunks connection to unknown phase":
            _chunks("- {from: bulk, to: final}", "- {from: bulk, to: fnal}"),
        "chunks seam bad key":
            _chunks("seam: {from: next_round, to: prev_round}",
                    "seam: {frm: next_round, to: prev_round}"),
        "chunks seam unknown value":
            _chunks("seam: {from: next_round, to: prev_round}",
                    "seam: {from: nxt, to: prev_round}"),
        "chunks phase without name":
            _chunks("        - name: init\n          spec:", "        - spec:"),
        "chunks phase as scalar":
            _chunks("      phases:\n" + _CHUNK_PHASES, "      phases: [a]\n"),
        "chunks phases_supply_D_sparse without D":
            _chunks(
                "      num_rounds: 3\n",
                "      num_rounds: 3\n      phases_supply_D_sparse: true\n"),
        "chunks measurements_per_round":
            _chunks("      num_rounds: 3\n",
                    "      num_rounds: 3\n      measurements_per_round: 2\n"),
        # syndrome_mapping
        "sm one final row":
            _sm("  - id: 1\n    type: final\n    vps: [[]]\n"),
        "sm duplicate id":
            _sm("  - id: 1\n    type: final\n    vps: [[]]\n  - id: 1\n    type: bulk\n    vps: [[]]\n"
               ),
        "sm type middle":
            _sm("  - id: 1\n    type: middle\n    vps: [[]]\n"),
        "sm id 16":
            _sm("  - id: 16\n    type: final\n    vps: [[]]\n"),
        "sm id 15":
            _sm("  - id: 15\n    type: final\n    vps: [[]]\n"),
        "sm id huge":
            _sm("  - id: 99999999\n    type: final\n    vps: [[]]\n"),
        "sm missing id":
            _sm("  - type: final\n    vps: [[]]\n"),
        "sm missing type":
            _sm("  - id: 1\n    vps: [[]]\n"),
        "sm vps not nested":
            _sm("  - id: 1\n    type: final\n    vps: [1, 2]\n"),
        "sm vps missing":
            _sm("  - id: 1\n    type: final\n"),
        "sm unknown key":
            _sm("  - id: 1\n    type: final\n    vps: [[]]\n    extra: 1\n"),
        "sm empty list":
            _sm("  []\n"),
        "sm id -1":
            _sm("  - id: -1\n    type: final\n    vps: [[]]\n"),
        # null-ish plain scalars per field kind
        "O_sparse empty value":
            _o(""),
        "O_sparse tilde":
            _o("~"),
        "O_sparse null":
            _o("null"),
        "O_sparse NULL":
            _o("NULL"),
        "O_sparse quoted empty":
            _o("''"),
        "D_sparse tilde":
            _d("~"),
        "H_sparse tilde":
            _h("~"),
        "error_rate_vec empty value":
            _doc().replace("error_rate_vec: [0.1, 0.1, 0.1]",
                           "error_rate_vec:"),
        "error_rate_vec tilde":
            _doc().replace("error_rate_vec: [0.1, 0.1, 0.1]",
                           "error_rate_vec: ~"),
        "error_rate_vec quoted empty":
            _doc().replace("error_rate_vec: [0.1, 0.1, 0.1]",
                           "error_rate_vec: ''"),
        "block_size empty value":
            _doc(block_size=""),
        "block_size tilde":
            _doc(block_size="~"),
        "id empty value":
            _doc(id=""),
        "id tilde":
            _doc(id="~"),
        "type empty value":
            _doc(type=""),
        "type tilde":
            _doc(type="~"),
        "type quoted empty":
            _doc(type="''"),
        "merge_strategy empty value":
            _ms(""),
        "merge_strategy tilde":
            _ms("~"),
        "merge_strategy null":
            _ms("null"),
        "merge_strategy quoted empty":
            _ms("''"),
        "decoder_custom_args empty value":
            _doc(args=""),
        "decoder_custom_args tilde":
            _doc(args="").replace("decoder_custom_args:\n",
                                  "decoder_custom_args: ~\n"),
        "stim_dem_path empty value":
            _env("    stim_dem_path:\n"),
        "stim_dem_path tilde":
            _env("    stim_dem_path: ~\n"),
        "cuda_device_id empty value":
            _env("    cuda_device_id:\n"),
        "cuda_device_id tilde":
            _env("    cuda_device_id: ~\n"),
        "dispatch empty value":
            _env("    dispatch:\n"),
        "dispatch tilde":
            _env("    dispatch: ~\n"),
        "transport empty value":
            _doc() + "transport:\n",
        "transport tilde":
            _doc() + "transport: ~\n",
        "transport.args empty value":
            _doc() + "transport:\n  args:\n",
        "transport.args tilde":
            _doc() + "transport:\n  args: ~\n",
        "transport.args quoted empty":
            _doc() + "transport:\n  args: ''\n",
        "syndrome_mapping empty value":
            _doc() + "syndrome_mapping:\n",
        "syndrome_mapping tilde":
            _doc() + "syndrome_mapping: ~\n",
        "syndrome_mapping quoted empty":
            _doc() + "syndrome_mapping: ''\n",
        "decoders empty value":
            "decoders:\n",
        "decoders tilde":
            "decoders: ~\n",
        "decoders quoted empty":
            "decoders: ''\n",
        "sequence item tilde":
            _o("[0, ~, -1]"),
        "block sequence empty item":
            _o("\n      - 0\n      -\n      - -1"),
        "rates item tilde":
            _doc(erv="~"),
        "bool tilde":
            _doc(type="chromobius", args=_CH_ARGS.format(b="~")),
        "chunks phase name mapping":
            _chunks("        - name: bulk\n", "        - name: {}\n"),
        "chunks phase name empty":
            _chunks("        - name: bulk\n", "        - name:\n"),
        "chunks phase name list":
            _chunks("        - name: bulk\n", "        - name: [a]\n"),
        "chunks connection from mapping":
            _chunks("- {from: bulk, to: final}", "- {from: {}, to: final}"),
        # transport
        "transport device_graph ok":
            _doc() +
            "transport:\n  provider: udp\n  device_graph:\n    provider: x\n    args: [a]\n",
        "transport device_graph bad key":
            _doc() + "transport:\n  device_graph:\n    providr: x\n",
        "transport args ints":
            _doc() + "transport:\n  args: [1, 2]\n",
        "transport empty mapping":
            _doc() + "transport: {}\n",
        "transport null":
            _doc() + "transport: ~\n",
        "transport scalar":
            _doc() + "transport: udp\n",
    }
    return cases


_LUT = _BASE.replace("type: {type}", "type: multi_error_lut").replace(
    "{args}", "      lut_error_depth: {v}\n")
_DEM = _doc().replace(
    "    H_sparse: [0, -1, 1, -1, 2, -1]\n    O_sparse: [0, -1, 1, -1, 2, -1]\n",
    "    stim_dem_path: /no/such/file.dem\n").replace(
        "      error_rate_vec: [0.1, 0.1, 0.1]\n", "")
_SW_NESTED = _doc(type="sliding_window",
                  args="      inner_decoder_name: pymatching\n"
                  "      window_size: 2\n      step_size: 1\n"
                  "      error_rate_vec: [0.1, 0.1, 0.1]\n{inner}")


def _h(new):
    return _doc().replace("H_sparse: [0, -1, 1, -1, 2, -1]", f"H_sparse: {new}")


def _o(new):
    return _doc().replace("O_sparse: [0, -1, 1, -1, 2, -1]", f"O_sparse: {new}")


def _d(new):
    return _doc().replace("D_sparse: [0, -1, 1, -1, 2, -1]", f"D_sparse: {new}")


def _env(extra):
    return _doc().replace("    block_size: 3\n", f"    block_size: 3\n{extra}")


def _more_cases():
    second = _doc().split("decoders:\n")[1]
    cases = {
        # encodings and layout
        "CRLF line endings":
            _doc().replace("\n", "\r\n"),
        "CR line endings":
            _doc().replace("\n", "\r"),
        "UTF-8 BOM":
            "\ufeff" + _doc(),
        "CRLF and BOM":
            "\ufeff" + _doc().replace("\n", "\r\n"),
        "trailing spaces":
            _doc().replace("type: pymatching", "type: pymatching   "),
        "space before colon":
            _doc().replace("block_size: 3", "block_size : 3"),
        "no space after colon":
            _doc().replace("block_size: 3", "block_size:3"),
        "comment after value":
            _doc().replace("block_size: 3", "block_size: 3  # three"),
        "hash inside scalar":
            _doc(ms="smallest#weight"),
        "tab after colon":
            _doc().replace("block_size: 3", "block_size:\t3"),
        "tab before comment":
            _doc().replace("block_size: 3", "block_size: 3\t# c"),
        "form feed":
            _doc().replace("block_size: 3", "block_size: 3\f"),
        "nbsp indentation":
            _doc().replace("    type:", "\u00a0\u00a0\u00a0\u00a0type:"),
        "unicode in string":
            _doc(ms="smallest_weight_é"),
        "YAML 1.1 directive":
            "%YAML 1.1\n---\n" + _doc(),
        "YAML 1.2 directive":
            "%YAML 1.2\n---\n" + _doc(),
        "leading document marker":
            "---\n" + _doc(),
        "trailing end marker":
            _doc() + "...\n",
        "second document is garbage":
            _doc() + "---\nnot: valid\n",
        "flow mapping args":
            _doc(
                args=
                "      {merge_strategy: smallest_weight, error_rate_vec: [0.1, 0.1, 0.1]}\n"
            ),
        "block sequence H_sparse":
            _h("\n      - 0\n      - -1\n      - 1\n      - -1\n      - 2\n      - -1"
              ),
        "dash at key column":
            _doc().replace("decoders:\n  - id: 0",
                           "decoders:\n- id: 0").replace("\n    ", "\n  "),
        "block literal string":
            _doc().replace("merge_strategy: smallest_weight",
                           "merge_strategy: |\n        smallest_weight"),
        "block folded string":
            _doc().replace("merge_strategy: smallest_weight",
                           "merge_strategy: >-\n        smallest_weight"),
        "double-quoted escape":
            _doc().replace("merge_strategy: smallest_weight",
                           'merge_strategy: "smallest\\x5fweight"'),
        "single-quoted apostrophe":
            _doc(ms="'it''s'"),
        "complex key":
            _env("    ? [a, b]\n    : c\n"),
        "quoted key with trailing space":
            _doc().replace("block_size: 3", "'block_size ': 3"),
        "numeric key":
            _env("    5: x\n"),
        # tags: the server ignores them
        "tag !!int on quoted":
            _doc(block_size="!!int '3'"),
        "tag !!str on number":
            _doc(block_size="!!str 3"),
        "tag !!int on octal":
            _doc(block_size="!!int 010"),
        "tag !!float":
            _doc(erv="!!float 6e-05"),
        "tag !!str in float list":
            _doc(erv="!!str 6e-05"),
        "tag !!bool in string field":
            _doc(ms="!!bool yes"),
        "tag !!timestamp in string":
            _doc(ms="!!timestamp 2026-01-01"),
        "tag !!binary in string":
            _doc(ms="!!binary aGVsbG8="),
        "tag !!null in string":
            _doc(ms="!!null ''"),
        "tag !!seq on list":
            _h("!!seq [0, -1, 1, -1, 2, -1]"),
        "tag !!map on args":
            _doc().replace("decoder_custom_args:\n",
                           "decoder_custom_args: !!map\n"),
        "local tag on scalar":
            _doc(block_size="!foo 3"),
        "local tag on mapping":
            _doc().replace("decoder_custom_args:\n",
                           "decoder_custom_args: !foo\n"),
        "tag !!set in string field":
            _doc(ms="!!set {a, b}"),
        "tag !!omap on list":
            _h("!!omap [a: 1]"),
        # flat form: required fields, empty and zero values
        "block_size absent":
            _doc().replace("    block_size: 3\n", ""),
        "block_size 0":
            _doc(block_size="0"),
        "block_size 00":
            _doc(block_size="00"),
        "syndrome_size absent":
            _doc().replace("    syndrome_size: 3\n", ""),
        "syndrome_size 0":
            _doc().replace("syndrome_size: 3", "syndrome_size: 0"),
        "H_sparse absent":
            _doc().replace("    H_sparse: [0, -1, 1, -1, 2, -1]\n", ""),
        "H_sparse []":
            _h("[]"),
        "H_sparse empty value":
            _h(""),
        "H_sparse scalar":
            _h("5"),
        "H_sparse nested":
            _h("[[0, -1], [1, -1]]"),
        "H_sparse as string":
            _h("'0 -1 1 -1 2 -1'"),
        "O_sparse absent":
            _doc().replace("    O_sparse: [0, -1, 1, -1, 2, -1]\n", ""),
        "O_sparse []":
            _o("[]"),
        "D_sparse []":
            _d("[]"),
        "error_rate_vec []":
            _doc(erv="").replace("[, 0.1, 0.1]", "[]"),
        "error_rate_vec absent":
            _doc().replace("      error_rate_vec: [0.1, 0.1, 0.1]\n", ""),
        "error_rate_vec top-level only":
            _doc().replace(
                "      error_rate_vec: [0.1, 0.1, 0.1]\n", ""
            ).replace(
                "    decoder_custom_args:",
                "    error_rate_vec: [0.1, 0.1, 0.1]\n    decoder_custom_args:"
            ),
        "error_rate_vec both places":
            _doc().replace(
                "    decoder_custom_args:",
                "    error_rate_vec: [0.1, 0.1, 0.1]\n    decoder_custom_args:"
            ),
        "H and error_rate_vec both []":
            _h("[]").replace("[0.1, 0.1, 0.1]", "[]"),
        # flat form: shape rules
        "H rows 2 of 3":
            _h("[0, -1, 1, -1]"),
        "H rows 4 of 3":
            _h("[0, -1, 1, -1, 2, -1, 0, -1]"),
        "H column 3 of block 3":
            _h("[0, -1, 1, -1, 3, -1]"),
        "H without trailing -1":
            _h("[0, -1, 1, -1, 2]"),
        "H empty row":
            _h("[-1, 1, -1, 2, -1]"),
        "H duplicate column":
            _h("[0, 0, -1, 1, -1, 2, -1]"),
        "H entry 1e0":
            _h("[0, -1, 1e0, -1, 2, -1]"),
        "H entry 0x1":
            _h("[0, -1, 0x1, -1, 2, -1]"),
        "H entries quoted":
            _h("['0', '-1', '1', '-1', '2', '-1']"),
        "O rows 2":
            _o("[0, -1, 1, -1]"),
        "O rows 5":
            _o("[0, -1, 1, -1, 2, -1, 0, -1, 1, -1]"),
        "O column 3 of block 3":
            _o("[0, -1, 1, -1, 3, -1]"),
        "D rows 2":
            _d("[0, -1, 1, -1]"),
        "D rows 5":
            _d("[0, -1, 1, -1, 2, -1, 0, -1, 1, -1]"),
        "D column 7":
            _d("[0, -1, 1, -1, 7, -1]"),
        "D first row empty":
            _d("[-1, 1, -1, 2, -1]"),
        "D middle row empty":
            _d("[0, -1, -1, 2, -1]"),
        "error_rate_vec length 2":
            _doc().replace("[0.1, 0.1, 0.1]", "[0.1, 0.1]"),
        "error_rate_vec length 4":
            _doc().replace("[0.1, 0.1, 0.1]", "[0.1, 0.1, 0.1, 0.1]"),
        "error rate 1.5":
            _doc(erv="1.5"),
        "error rate -0.1":
            _doc(erv="-0.1"),
        "syndrome_size 2 with 3 rows":
            _doc().replace("syndrome_size: 3", "syndrome_size: 2"),
        "block_size 2 with column 2":
            _doc(block_size="2"),
        "block_size 10":
            _doc(block_size="10"),
        "block_size 1 with column 1":
            _doc(block_size="1"),
        # stim_dem_path form
        "dem path form":
            _DEM,
        "dem path without sizes":
            _DEM.replace("    block_size: 3\n    syndrome_size: 3\n", ""),
        "dem path with error_rate_vec":
            _DEM.replace(
                "      merge_strategy: smallest_weight\n",
                "      merge_strategy: smallest_weight\n      error_rate_vec: [0.1]\n"
            ),
        "dem path with H_sparse":
            _DEM.replace(
                "    stim_dem_path: /no/such/file.dem\n",
                "    stim_dem_path: /no/such/file.dem\n    H_sparse: [0, -1, 1, -1, 2, -1]\n"
            ),
        "dem path without D_sparse":
            _DEM.replace("    D_sparse: [0, -1, 1, -1, 2, -1]\n", ""),
        "dem path and dem_chunks":
            _DEM.replace(
                "    stim_dem_path: /no/such/file.dem\n",
                "    stim_dem_path: /no/such/file.dem\n    dem_chunks: {seam: {}, connections: [], phases: []}\n"
            ),
        "stim_dem_path tilde":
            _env("    stim_dem_path: ~\n"),
        "stim_dem_path empty":
            _env("    stim_dem_path: ''\n"),
        # envelope fields
        "dispatch bogus":
            _env("    dispatch: gpu\n"),
        "dispatch quoted":
            _env("    dispatch: 'host'\n"),
        "cuda_device_id -1":
            _env("    cuda_device_id: -1\n"),
        "cuda_device_id 0":
            _env("    cuda_device_id: 0\n"),
        "id float":
            _doc(id="0.0"),
        "id huge":
            _doc(id="99999999999"),
        "id absent":
            _doc().replace("  - id: 0\n", "  -\n"),
        "type bool-looking":
            _doc(type="true"),
        # several decoders, transport
        "duplicate decoder id":
            _doc() + second,
        "two decoders":
            _doc() + second.replace("id: 0", "id: 1"),
        "three decoders":
            _doc() + second + second.replace("id: 0", "id: 1"),
        "empty decoders":
            "decoders: []\n",
        "decoders missing":
            "transport:\n  provider: udp\n",
        "transport ok":
            _doc() + "transport:\n  provider: udp\n  args: [a, 1, 'x']\n",
        "transport bad key":
            _doc() + "transport:\n  provder: udp\n",
        "transport args scalar":
            _doc() + "transport:\n  args: a\n",
        # nested parameter sections, int32 parameters
        "sw nested ok":
            _SW_NESTED.format(
                inner=
                "      inner_decoder_params:\n        merge_strategy: smallest_weight\n"
            ),
        "sw nested typo":
            _SW_NESTED.format(
                inner=
                "      inner_decoder_params:\n        merge_stratgey: smallest_weight\n"
            ),
        "sw nested number as string":
            _SW_NESTED.format(
                inner="      inner_decoder_params:\n        merge_strategy: 5\n"
            ),
        "sw unknown inner name":
            _SW_NESTED.format(inner=""
                             ).replace("inner_decoder_name: pymatching",
                                       "inner_decoder_name: pymatchnig"),
        "sw window_size 2^31":
            _SW_NESTED.format(inner="").replace("window_size: 2",
                                                "window_size: 2147483648"),
        "sw window_size 2^63":
            _SW_NESTED.format(inner=""
                             ).replace("window_size: 2",
                                       "window_size: 9223372036854775808"),
        "sw window_size -1":
            _SW_NESTED.format(inner="").replace("window_size: 2",
                                                "window_size: -1"),
        "sw window_size 0":
            _SW_NESTED.format(inner="").replace("window_size: 2",
                                                "window_size: 0"),
        "sw step_size above window":
            _SW_NESTED.format(inner="").replace("step_size: 1", "step_size: 5"),
        "lut_error_depth 1":
            _LUT.replace("{v}", "1").replace("{id}",
                                             "0").replace("{block_size}", "3"),
        "lut_error_depth int32 max":
            _LUT.replace("{v}", "2147483647").replace("{id}", "0").replace(
                "{block_size}", "3"),
        "lut_error_depth int32 max + 1":
            _LUT.replace("{v}", "2147483648").replace("{id}", "0").replace(
                "{block_size}", "3"),
        "lut_error_depth -1":
            _LUT.replace("{v}", "-1").replace("{id}",
                                              "0").replace("{block_size}", "3"),
        "lut_error_depth quoted":
            _LUT.replace("{v}",
                         "'2'").replace("{id}",
                                        "0").replace("{block_size}", "3"),
    }
    return cases


def _corpus():
    cases = _corpus_cases()
    return [
        pytest.param(label,
                     text,
                     marks=pytest.mark.xfail(reason=_SYNTAX_DIFFERENCES[label],
                                             strict=True))
        if label in _SYNTAX_DIFFERENCES else (label, text)
        for label, text in sorted(cases.items())
    ]


# Documents on which the validator and the server are known to disagree.
# Each is recorded as an expected failure so the gap stays visible.
_SYNTAX_DIFFERENCES = {
    "YAML directive twice":
        "LLVM YAML accepts a repeated %YAML directive; PyYAML rejects it",
    "tab inside flow sequence":
        "LLVM YAML accepts a tab as whitespace inside a flow sequence; "
        "PyYAML rejects it",
    "tab inside flow mapping":
        "LLVM YAML accepts a tab as whitespace inside a flow mapping; "
        "PyYAML rejects it",
    "tab after colon":
        "LLVM YAML accepts a tab as separator whitespace after ':'; PyYAML "
        "rejects it",
    "tab before comment":
        "LLVM YAML accepts a tab before a '#' comment; PyYAML rejects it",
    "double comma":
        "LLVM YAML tolerates an empty flow-sequence entry (`[a,, b]`); PyYAML "
        "rejects it",
    "sw window_size 0":
        "sliding_window validate hook (step_size within 1..window_size), "
        "server-only",
    "sw step_size above window":
        "sliding_window validate hook (step_size within 1..window_size), "
        "server-only",
    "block literal with indent indicator":
        "LLVM YAML does not honour an explicit indentation indicator (`|2`) "
        "and misreads the block; PyYAML parses it",
    "chunks phases_supply_D_sparse without D":
        "dem_chunks_spec consistency rule between phases_supply_D_sparse and "
        "the phases' D_sparse, server-only",
}


def _server_accepts(text):
    try:
        qec.multi_decoder_config.from_yaml_str(text)
        return True
    except Exception:  # The binding raises RuntimeError / ValueError.
        return False


@pytest.fixture(scope="module")
def live_validator(tool):
    schema = json.loads(qec.decoder_config_json_schema())
    schema = tool.forbid_task_graph(schema)
    return tool.make_validator(schema), tool.known_decoder_types(schema)


@pytest.mark.parametrize("label, text", _corpus())
def test_validator_agrees_with_server_parser(tool, live_validator, label, text):
    validator, known = live_validator
    problems = tool.validate_text(text, validator, False, known)
    assert (not problems) == _server_accepts(text), (
        f"{label}: validator says {'OK' if not problems else problems}, "
        f"server says {'OK' if _server_accepts(text) else 'reject'}")


def test_corpus_has_both_outcomes(live_validator, tool):
    verdicts = {_server_accepts(text) for text in _corpus_cases().values()}
    assert verdicts == {True, False}


# --- task-graph documents with --allow-task-graph ----------------------------

_TG = (_SERVER_CONFIGS / "decoding_server_config_task_graph.yaml").read_text()


def _tg_cases():
    chunks_tg = _CHUNKS.replace(
        "    type: pymatching\n",
        "    task_graph: {type: single_window, inner_type: pymatching}\n")
    return sorted({
        "tg repo example":
            _TG,
        "tg with type as well":
            _TG.replace("    task_graph:",
                        "    type: pymatching\n    task_graph:"),
        "tg unknown node key":
            _TG.replace("      inner_type: pymatching",
                        "      inner_type: pymatching\n      bogus: 1"),
        "tg without any DEM":
            _TG.replace("    H_sparse: [0, -1, 1, -1, 2, -1]\n", ""),
        "tg over dem_chunks":
            chunks_tg,
        "tg over dem_chunks with type":
            chunks_tg.replace("    task_graph:",
                              "    type: pymatching\n    task_graph:"),
    }.items())


@pytest.fixture(scope="module")
def live_validator_with_task_graphs(tool):
    schema = json.loads(qec.decoder_config_json_schema())
    return tool.make_validator(schema), tool.known_decoder_types(schema)


@pytest.mark.parametrize("label, text", _tg_cases())
def test_task_graph_documents_agree_with_server_parser(
        tool, live_validator_with_task_graphs, label, text):
    validator, known = live_validator_with_task_graphs
    problems = tool.validate_text(text, validator, True, known)
    assert (not problems) == _server_accepts(text), (
        f"{label}: validator says {'OK' if not problems else problems}, "
        f"server says {'OK' if _server_accepts(text) else 'reject'}")


# --- the reported location names the right decoder ---------------------------


def _server_error(text):
    try:
        qec.multi_decoder_config.from_yaml_str(text)
        return ""
    except Exception as exc:  # RuntimeError / ValueError from the binding
        return str(exc)


_LOCATION_CASES = {
    "block_size 0 in second decoder":
        _doc() + _doc().split("decoders:\n")[1].replace(
            "id: 0", "id: 1").replace("block_size: 3", "block_size: 0"),
    "empty H in second decoder":
        _doc() +
        _doc().split("decoders:\n")[1].replace("id: 0", "id: 1").replace(
            "H_sparse: [0, -1, 1, -1, 2, -1]", "H_sparse: []"),
    "missing D in third decoder":
        _doc() + _doc().split("decoders:\n")[1].replace("id: 0", "id: 1") +
        _doc().split("decoders:\n")[1].replace("id: 0", "id: 2").replace(
            "    D_sparse: [0, -1, 1, -1, 2, -1]\n", ""),
    "negative cuda_device_id in second decoder":
        _doc() +
        _doc().split("decoders:\n")[1].replace("id: 0", "id: 1").replace(
            "    block_size: 3\n",
            "    block_size: 3\n    cuda_device_id: -1\n"),
    "empty D row in second decoder":
        _doc() +
        _doc().split("decoders:\n")[1].replace("id: 0", "id: 1").replace(
            "D_sparse: [0, -1, 1, -1, 2, -1]", "D_sparse: [0, -1, -1, 2, -1]"),
}


@pytest.mark.parametrize("label, text", sorted(_LOCATION_CASES.items()))
def test_problem_location_points_at_the_failing_decoder(tool, live_validator,
                                                        label, text):
    validator, known = live_validator
    problems = tool.validate_text(text, validator, False, known)
    assert problems, label
    where = problems[0][0]
    match = re.search(r"decoder (\d+)", _server_error(text))
    assert match, f"{label}: server did not name a decoder"
    assert where.startswith(f"decoders/{match.group(1)}"), (label, where)


# --- mutation fuzz: random edits of valid documents, both sides must agree ----
#
# The hand-written corpus covers the differences we know about. The fuzzer
# applies random line-level edits to valid documents (key typos, dropped and
# duplicated lines, scalar substitutions from a pool of awkward spellings,
# indentation shifts, extra keys) and requires the two verdicts to agree on
# every result. It is deterministic (fixed seed); CONFIG_VALIDATOR_FUZZ_N
# scales the number of documents for a longer local run.

_FUZZ_SCALARS = [
    "0",
    "1",
    "-1",
    "3",
    "010",
    "0x10",
    "0o7",
    "1_000",
    "1e3",
    "6e-05",
    ".5",
    "1.",
    "inf",
    "nan",
    "''",
    "'3'",
    "\"true\"",
    "yes",
    "no",
    "on",
    "off",
    "y",
    "n",
    "True",
    "~",
    "null",
    "[]",
    "{}",
    "[1, 2]",
    "{a: 1}",
    "x",
    "smallest_weight",
    "pymatching",
    "18446744073709551616",
    "2147483648",
    "-0",
    "+1",
    "1e",
    "0b1",
    "1:30",
    "a: b",
    "# c",
    "'it''s'",
    "\"a\\x41\"",
]
_FUZZ_SEEDS = [
    _GOOD_YAML,
    _CHUNKS,
    _doc(type="chromobius", args=_CH_ARGS.format(b="true")),
    _SW_NESTED.format(
        inner=
        "      inner_decoder_params:\n        merge_strategy: smallest_weight\n"
    ),
    _LUT.replace("{v}", "2").replace("{id}", "0").replace("{block_size}", "3"),
    _DEM,
    _doc() + _doc().split("decoders:\n")[1].replace("id: 0", "id: 1") +
    "transport:\n  provider: udp\n  args: [a]\nsyndrome_mapping:\n  - id: 1\n    type: final\n    vps: [[]]\n",
]


def _mutate(text, rng):
    lines = text.split("\n")
    body = [i for i, line in enumerate(lines) if line.strip()]
    i = rng.choice(body)
    line = lines[i]
    op = rng.randrange(8)
    if op == 0 and ":" in line:  # typo in the key
        key, rest = line.split(":", 1)
        stripped = key.strip()
        if len(stripped) > 3:
            k = rng.randrange(len(stripped) - 1)
            typo = stripped[:k] + stripped[k + 1] + stripped[k] + stripped[k +
                                                                           2:]
            lines[i] = key.replace(stripped, typo) + ":" + rest
    elif op == 1:  # drop the line
        del lines[i]
    elif op == 2:  # duplicate the line
        lines.insert(i, line)
    elif op == 3 and ":" in line and not line.rstrip().endswith(
            ":"):  # new value
        key = line.split(":", 1)[0]
        lines[i] = key + ": " + rng.choice(_FUZZ_SCALARS)
    elif op == 4:  # shift indentation
        delta = rng.choice([-2, 2, 1])
        indent = len(line) - len(line.lstrip(" "))
        lines[i] = " " * max(0, indent + delta) + line.lstrip(" ")
    elif op == 5 and ":" in line:  # add an unknown key at the same depth
        indent = len(line) - len(line.lstrip(" "))
        lines.insert(i + 1,
                     " " * indent + "zz_extra: " + rng.choice(_FUZZ_SCALARS))
    elif op == 6 and ":" in line and not line.rstrip().endswith(
            ":"):  # quote value
        key, value = line.split(":", 1)
        lines[i] = key + ": '" + value.strip().replace("'", "''") + "'"
    elif op == 7:  # replace one scalar inside a flow sequence
        if "[" in line and "]" in line:
            head, rest = line.split("[", 1)
            items, tail = rest.rsplit("]", 1)
            parts = [p.strip() for p in items.split(",") if p.strip()]
            if parts:
                parts[rng.randrange(len(parts))] = rng.choice(_FUZZ_SCALARS)
                lines[i] = head + "[" + ", ".join(parts) + "]" + tail
    return "\n".join(lines)


def test_fuzz_mutations_agree_with_server_parser(tool, live_validator):
    import os
    import random
    validator, known = live_validator
    count = int(os.environ.get("CONFIG_VALIDATOR_FUZZ_N", "400"))
    rng = random.Random(20261023)
    disagreements = []
    for n in range(count):
        text = rng.choice(_FUZZ_SEEDS)
        for _ in range(rng.randrange(1, 4)):
            text = _mutate(text, rng)
        if "\t" in text:
            continue  # Known: PyYAML is stricter about tabs.
        ours = not tool.validate_text(text, validator, False, known)
        theirs = _server_accepts(text)
        if ours != theirs:
            disagreements.append(
                f"--- case {n}: validator {'OK' if ours else 'reject'}, server "
                f"{'OK' if theirs else 'reject'}\n{text}\n"
                f"server: {_server_error(text).splitlines()[:1]}")
    assert not disagreements, "\n".join(disagreements[:5]) + (
        f"\n... {len(disagreements)} disagreement(s) in {count} documents")
