#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate CUDA-Q QEC decoder configuration YAML files offline.

Checks ``multi_decoder_config`` YAML documents -- the files the decoding
server loads -- against the JSON Schema that CUDA-Q QEC exports
(``cudaq_qec.decoder_config_json_schema()``). No GPU is needed and no decoder
is loaded. The only dependencies are ``pyyaml`` and ``jsonschema``;
``cudaq_qec`` itself is needed only to produce the schema file::

    pip install pyyaml jsonschema
    python3 validate_decoder_config.py --schema decoder_config.schema.json a.yml b.yml

On a machine where ``cudaq_qec`` is installed the ``--schema`` argument may
be omitted (the schema is generated from the installed package), and
``--export-schema OUT.json`` writes that schema out for machines that do not
have the package.

Exit status is 0 when every file validates, 1 when at least one does not,
and 2 for a usage or schema error.

What is checked: key names and their placement, value types, which model
form a decoder entry uses (flat ``H_sparse``/``O_sparse``/``D_sparse``
matrices, ``dem_chunks``, or ``stim_dem_path``), and the parameter names of
each decoder type. What cannot be checked here: cross-field rules (matrix
row counts against ``syndrome_size``, seam widths between DEM chunks) and
the per-decoder ``validate`` hooks. Those run only when the server loads the
configuration, so a file that passes here may still be rejected at load.

Task-graph configurations (a decoder entry with a ``task_graph`` block) are
reported as invalid unless ``--allow-task-graph`` is given.

Numbers are read with YAML 1.2 rules, which is what the server's LLVM YAML
parser applies: ``6e-05`` and ``1e3`` are floats. PyYAML's default YAML 1.1
resolver reads those spellings as strings, which is why a stock
``yaml.safe_load`` must not be used for these files.
"""

import argparse
import copy
import json
import re
import sys

try:
    import jsonschema
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by hand
    sys.stderr.write(f"{exc.name} is not installed; run: "
                     "pip install pyyaml jsonschema\n")
    sys.exit(2)

# YAML 1.2 core schema float: an optional sign, digits with an optional
# fraction (or a bare fraction), and an optional exponent. The mantissa does
# not need a dot, unlike YAML 1.1. Integers are still resolved first by
# PyYAML's own int resolver because it was registered earlier.
_YAML12_FLOAT = re.compile(
    r"""^[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)
                               (?:[eE][-+]?[0-9]+)?$""", re.X)


class Yaml12Loader(yaml.SafeLoader):
    """``yaml.SafeLoader`` with YAML 1.2 float resolution."""


Yaml12Loader.add_implicit_resolver("tag:yaml.org,2002:float", _YAML12_FLOAT,
                                   list("-+.0123456789"))


def load_yaml(text):
    """Parse ``text`` as a YAML document with YAML 1.2 number rules."""
    return yaml.load(text, Loader=Yaml12Loader)


def schema_from_installed_package():
    """Return the schema exported by the installed ``cudaq_qec`` package."""
    import cudaq_qec  # Only needed on the machine that produces the schema.
    return json.loads(cudaq_qec.decoder_config_json_schema())


def forbid_task_graph(schema):
    """Return a copy of ``schema`` in which ``task_graph`` is not a known key.

    ``decoder_config`` sets ``additionalProperties: false``, so dropping the
    property makes any ``task_graph`` block an unexpected key.
    """
    schema = copy.deepcopy(schema)
    schema.get("$defs", {}).get("decoder_config",
                                {}).get("properties",
                                        {}).pop("task_graph", None)
    return schema


def task_graph_locations(doc):
    """Yield the document paths of decoder entries that use a task graph."""
    decoders = doc.get("decoders") if isinstance(doc, dict) else None
    if not isinstance(decoders, list):
        return
    for index, decoder in enumerate(decoders):
        if isinstance(decoder, dict) and "task_graph" in decoder:
            yield f"decoders/{index}"


def known_decoder_types(schema):
    """Decoder ``type`` names whose parameters the schema describes."""
    return set(schema.get("$defs", {}).get("decoder_params", {}))


def unknown_type_locations(doc, known_types):
    """Yield ``(path, type)`` for decoders whose type the schema does not know.

    The schema lets such a decoder carry no ``decoder_custom_args`` at all,
    and jsonschema then reports that by dumping the whole args mapping. Only
    decoders that do carry args are reported here; the plain schema error
    covers the rest.
    """
    decoders = doc.get("decoders") if isinstance(doc, dict) else None
    if not isinstance(decoders, list):
        return
    for index, decoder in enumerate(decoders):
        if not isinstance(decoder, dict):
            continue
        type_name = decoder.get("type")
        if (isinstance(type_name, str) and type_name not in known_types and
                decoder.get("decoder_custom_args")):
            yield f"decoders/{index}", type_name


def validate_document(doc, validator, allow_task_graph=False, known_types=None):
    """Return ``[(where, message), ...]`` for ``doc``; empty when it is valid."""
    problems = []
    if not allow_task_graph:
        for where in task_graph_locations(doc):
            problems.append((where, "task_graph configurations are not "
                             "supported by this validator (pass "
                             "--allow-task-graph to check them against the "
                             "schema)"))
        if problems:
            return problems

    skip_prefixes = []
    if known_types is not None:
        for where, type_name in unknown_type_locations(doc, known_types):
            problems.append((where, f"decoder type '{type_name}' has no "
                             "parameter schema in this schema file, so its "
                             "decoder_custom_args cannot be checked (the type "
                             "name is misspelled, or its plugin was not loaded "
                             "when the schema was exported)"))
            skip_prefixes.append(f"{where}/decoder_custom_args")

    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
    # A missing required key also trips the "which model form is this" oneOf,
    # whose message just dumps the document; keep the specific errors only.
    specific = [e for e in errors if e.validator not in ("oneOf", "anyOf")]
    for err in specific or errors:
        best = jsonschema.exceptions.best_match([err])
        where = "/".join(str(p) for p in best.absolute_path) or "<root>"
        if any(where == p or where.startswith(p + "/") for p in skip_prefixes):
            continue
        problems.append((where, best.message))
    return problems


def validate_file(path, validator, allow_task_graph=False, known_types=None):
    """Validate one file; returns the problem list (empty when valid)."""
    try:
        with open(path, encoding="utf-8") as handle:
            doc = load_yaml(handle.read())
    except OSError as exc:
        return [("<file>", str(exc))]
    except yaml.YAMLError as exc:
        detail = " ".join(line.strip() for line in str(exc).splitlines())
        return [("<yaml>", f"not valid YAML: {detail}")]
    if not isinstance(doc, dict):
        return [("<root>", "the document must be a mapping with a "
                 "'decoders' list")]
    return validate_document(doc, validator, allow_task_graph, known_types)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate decoding-server YAML configurations against "
        "the CUDA-Q QEC decoder configuration JSON Schema.")
    parser.add_argument("configs",
                        nargs="*",
                        metavar="CONFIG.yml",
                        help="configuration files to validate")
    parser.add_argument(
        "--schema",
        metavar="SCHEMA.json",
        help="JSON Schema file; when omitted the schema is generated from "
        "the installed cudaq_qec package")
    parser.add_argument(
        "--allow-task-graph",
        action="store_true",
        help="accept task_graph configurations (rejected by default)")
    parser.add_argument(
        "--export-schema",
        metavar="OUT.json",
        help="write the schema in use (task_graph removed unless "
        "--allow-task-graph) to OUT.json and exit")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        if args.schema:
            with open(args.schema, encoding="utf-8") as handle:
                schema = json.load(handle)
        else:
            schema = schema_from_installed_package()
    except ImportError:
        sys.stderr.write("cudaq_qec is not installed; pass --schema with a "
                         "schema exported by `--export-schema` on a machine "
                         "that has it\n")
        return 2
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"cannot read schema: {exc}\n")
        return 2

    if not args.allow_task_graph:
        schema = forbid_task_graph(schema)
    elif "task_graph" not in schema.get("$defs",
                                        {}).get("decoder_config",
                                                {}).get("properties", {}):
        sys.stderr.write("note: this schema has no task_graph definition "
                         "(exported without --allow-task-graph?); task_graph "
                         "entries will be reported as unexpected keys\n")

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        sys.stderr.write(f"not a valid JSON Schema: {exc.message}\n")
        return 2
    validator = jsonschema.Draft202012Validator(schema)

    if args.export_schema:
        with open(args.export_schema, "w", encoding="utf-8") as handle:
            json.dump(schema, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"schema written to {args.export_schema}")
        if not args.configs:
            return 0

    if not args.configs:
        build_parser().error("no configuration files given")

    known_types = known_decoder_types(schema)
    status = 0
    for path in args.configs:
        problems = validate_file(path, validator, args.allow_task_graph,
                                 known_types)
        if not problems:
            print(f"{path}: OK")
            continue
        status = 1
        print(f"{path}: FAIL")
        for where, message in problems:
            print(f"  {where}: {message}")
    return status


if __name__ == "__main__":
    sys.exit(main())
