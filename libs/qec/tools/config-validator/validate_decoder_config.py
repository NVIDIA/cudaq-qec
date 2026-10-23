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
matrices, ``dem_chunks``, or ``stim_dem_path``), the parameter names of each
decoder type, and the cross-field rules the server applies while parsing
(required fields per form, ``H_sparse``/``D_sparse`` row counts against
``syndrome_size``, matrix entries below ``block_size``, fields a form
derives and therefore forbids). What cannot be checked here: the per-decoder
``validate`` hooks, DEM-chunk seam consistency, and anything that needs the
constructed decoder. Those run only when the server loads the configuration,
so a file that passes here may still be rejected at load.

Task-graph configurations (a decoder entry with a ``task_graph`` block) are
reported as invalid unless ``--allow-task-graph`` is given.

Scalars are read the way the server's LLVM YAML parser reads them, which is
not how PyYAML reads them by default. The server takes every scalar as raw
text and converts it according to the type of the field it lands in, so
quoting does not matter (``'3'`` is a valid integer, ``yes`` in a string
field is the string ``yes``), integers accept ``0x``/``0b``/``0o`` prefixes
and a leading ``0`` for octal but no sign ``+`` and no ``_``, floats accept
anything ``strtod`` accepts (``6e-05``, ``1.``, ``inf``, hex floats) but not
``_`` or YAML's ``.inf``, and booleans are exactly LLVM's set
(``y yes true on`` / ``n no false off`` in their usual capitalisations, not
``1``/``0``). Tags (``!!int``, ``!foo``) are ignored. Duplicate mapping keys
and YAML anchors/aliases are rejected; only the first document of a stream
is read.
This script loads every scalar as text and applies those rules while
checking the schema, so its verdicts match the server's for scalars.
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

# --- YAML loading: raw scalars, no duplicates, no aliases -------------------


class RawLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` that mirrors the server's structural rules.

    Every scalar stays a string whatever its spelling or tag (the server
    types scalars by the field they land in, see ``LlvmScalars``, and ignores
    tags), duplicate mapping keys are an error, and anchors/aliases are an
    error.
    """

    def compose_node(self, parent, index):
        if self.check_event(yaml.events.AliasEvent):
            event = self.peek_event()
            raise yaml.constructor.ConstructorError(
                None, None, "YAML aliases are not supported by the decoder "
                "configuration parser", event.start_mark)
        return super().compose_node(parent, index)

    def construct_object(self, node, deep=False):
        # Build by node kind only; resolvers and tags (``!!int``, ``!foo``)
        # do not take part, exactly like llvm::yaml.
        if isinstance(node, yaml.ScalarNode):
            return node.value
        if isinstance(node, yaml.SequenceNode):
            return [self.construct_object(child, True) for child in node.value]
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, True)
            if not isinstance(key, str):
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    "mapping keys must be scalars", key_node.start_mark)
            if key in mapping:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping", node.start_mark,
                    f"duplicated mapping key '{key}'", key_node.start_mark)
            mapping[key] = self.construct_object(value_node, True)
        return mapping


def load_yaml(text):
    """Parse ``text`` with ``RawLoader``: scalars are strings.

    Like ``llvm::yaml::Input``, only the first document of a stream is read;
    anything after a ``---`` separator is ignored.
    """
    return next(yaml.load_all(text, Loader=RawLoader), None)


# --- scalar conversion rules of the server (LLVM YAML ScalarTraits) --------


class LlvmScalars:
    """Convert raw scalar text like ``llvm::yaml`` does, or return ``None``."""

    TRUE = {
        "y", "Y", "yes", "Yes", "YES", "true", "True", "TRUE", "on", "On", "ON"
    }
    FALSE = {
        "n", "N", "no", "No", "NO", "false", "False", "FALSE", "off", "Off",
        "OFF"
    }

    # What strtod consumes in full: decimal, hex float, inf/infinity, nan.
    _FLOAT = re.compile(
        r"""^[+-]?(?:
              (?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?
            | 0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)(?:[pP][+-]?[0-9]+)?
            | inf(?:inity)?
            | nan
            )$""", re.X | re.I)

    @classmethod
    def boolean(cls, text):
        if isinstance(text, bool):
            return text
        if not isinstance(text, str):
            return None
        if text in cls.TRUE:
            return True
        if text in cls.FALSE:
            return False
        return None

    @classmethod
    def integer(cls, text):
        """``StringRef::getAsInteger(0)``: auto radix, ``-`` but no ``+``."""
        if isinstance(text, bool):
            return None
        if isinstance(text, int):
            return text
        if not isinstance(text, str):
            return None
        negative = text.startswith("-")
        digits = text[1:] if negative else text
        radix = 10
        if digits[:2] in ("0x", "0X"):
            digits, radix = digits[2:], 16
        elif digits[:2] in ("0b", "0B"):
            digits, radix = digits[2:], 2
        elif digits.startswith("0o"):
            digits, radix = digits[2:], 8
        elif len(digits) > 1 and digits[0] == "0" and digits[1].isdigit():
            digits, radix = digits[1:], 8
        if not digits or not re.fullmatch(r"[0-9a-fA-F]+", digits):
            return None
        try:
            value = int(digits, radix)
        except ValueError:
            return None
        value = -value if negative else value
        # Widest C++ targets are 64-bit; anything beyond overflows there.
        if value > 2**64 - 1 or value < -2**63:
            return None
        return value

    @classmethod
    def number(cls, text):
        """``StringRef::getAsDouble``: strtod that must consume everything."""
        if isinstance(text, bool):
            return None
        if isinstance(text, (int, float)):
            return text
        if not isinstance(text, str):
            return None
        if text == "":
            return 0.0
        if not cls._FLOAT.match(text):
            return None
        try:
            return float.fromhex(text) if "x" in text.lower() else float(text)
        except ValueError:
            return None


def _type_checker():
    base = jsonschema.Draft202012Validator.TYPE_CHECKER
    return base.redefine_many({
        "integer": lambda _, v: LlvmScalars.integer(v) is not None,
        "number": lambda _, v: LlvmScalars.number(v) is not None,
        "boolean": lambda _, v: LlvmScalars.boolean(v) is not None,
    })


def _bound(keyword, compare, describe):

    def check(validator, limit, instance, schema):
        if not isinstance(instance,
                          (str, int, float)) or isinstance(instance, bool):
            return
        if keyword.endswith("imum") and isinstance(
                instance, str) and instance.startswith("-") and limit >= 0:
            # The field is unsigned on the server; its parser rejects the
            # sign outright, even for -0.
            yield jsonschema.ValidationError(
                f"{instance!r} is negative but the field is unsigned")
            return
        value = LlvmScalars.number(instance)
        if value is None:
            return  # The type check reports it.
        if not compare(value, limit):
            yield jsonschema.ValidationError(
                f"{instance!r} is {describe} {limit}")

    return check


def _validator_class():
    return jsonschema.validators.extend(
        jsonschema.Draft202012Validator,
        validators={
            "minimum":
                _bound("minimum", lambda v, m: v >= m,
                       "less than the minimum of"),
            "maximum":
                _bound("maximum", lambda v, m: v <= m,
                       "greater than the maximum of"),
            "exclusiveMinimum":
                _bound("exclusiveMinimum", lambda v, m: v > m,
                       "less than or equal to the minimum of"),
            "exclusiveMaximum":
                _bound("exclusiveMaximum", lambda v, m: v < m,
                       "greater than or equal to the maximum of"),
        },
        type_checker=_type_checker())


def make_validator(schema):
    """A jsonschema validator for ``schema`` with the server's scalar rules."""
    cls = _validator_class()
    cls.check_schema(schema)
    return cls(schema)


# --- schema handling ----------------------------------------------------------


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


def known_decoder_types(schema):
    """Decoder ``type`` names whose parameters the schema describes."""
    return set(schema.get("$defs", {}).get("decoder_params", {}))


# --- document checks ----------------------------------------------------------


def task_graph_locations(doc):
    """Yield the document paths of decoder entries that use a task graph."""
    decoders = doc.get("decoders") if isinstance(doc, dict) else None
    if not isinstance(decoders, list):
        return
    for index, decoder in enumerate(decoders):
        if isinstance(decoder, dict) and "task_graph" in decoder:
            yield f"decoders/{index}"


def unknown_type_locations(doc, known_types):
    """Yield ``(path, type)`` for decoders whose type the schema does not know.

    The server rejects a ``decoder_custom_args`` section, even an empty one,
    for a type with no registered parameter schema; jsonschema would report
    that by dumping the whole args mapping. Decoders without the section are
    left to the schema.
    """
    decoders = doc.get("decoders") if isinstance(doc, dict) else None
    if not isinstance(decoders, list):
        return
    for index, decoder in enumerate(decoders):
        if not isinstance(decoder, dict):
            continue
        type_name = decoder.get("type")
        if (isinstance(type_name, str) and type_name not in known_types and
                "decoder_custom_args" in decoder):
            yield f"decoders/{index}", type_name


# --- the server's parse-time cross-field checks -----------------------------
#
# llvm::yaml's MappingTraits<decoder_config> runs these right after reading a
# decoder entry, so they are part of parsing and the offline validator applies
# them too. They are a transcription of that code and use its wording. Only
# the per-decoder validate hooks and the checks that need the constructed
# decoder (D row count against a Stim model, DEM chunk seams) stay server-only.


def _ints(value):
    if not isinstance(value, list):
        return []
    out = [LlvmScalars.integer(v) for v in value]
    return [] if any(v is None for v in out) else out


def server_checks_for_decoder(decoder, where):
    """Cross-field rules the server enforces while parsing one decoder entry."""
    who = f"decoder {decoder.get('id', '?')}"
    problems = []

    def fail(message):
        problems.append((where, message))
        return problems

    H = _ints(decoder.get("H_sparse"))
    O = _ints(decoder.get("O_sparse"))
    D = _ints(decoder.get("D_sparse"))
    block_size = LlvmScalars.integer(decoder.get("block_size", "0")) or 0
    syndrome_size = LlvmScalars.integer(decoder.get("syndrome_size", "0")) or 0
    args = decoder.get("decoder_custom_args")
    rates = decoder.get("error_rate_vec") or (
        args.get("error_rate_vec") if isinstance(args, dict) else None) or []
    has_chunks = "dem_chunks" in decoder
    from_dem = bool(decoder.get("stim_dem_path"))

    if "task_graph" in decoder:
        if not has_chunks and not H:
            return fail(f"{who}: task_graph requires either dem_chunks or "
                        "H_sparse to supply the DEM")
        if not H:
            return problems  # Chunk form carries its own DEM.

    if has_chunks and from_dem:
        return fail(f"dem_chunks and stim_dem_path must not both be set for "
                    f"{who}; each describes the whole DEM on its own.")

    def reject_derived(key, present, source):
        if present:
            fail(f"{key} must not be set for {who} because it is derived from "
                 f"{source}. Remove it, or describe the whole experiment with "
                 f"H_sparse instead of {source}.")

    if has_chunks and not H:
        chunks = decoder.get("dem_chunks")
        if isinstance(chunks, dict) and "num_rounds" in chunks:
            num_rounds = LlvmScalars.integer(chunks["num_rounds"])
            if num_rounds is not None and num_rounds < 2:
                fail(f"dem_chunks.num_rounds must be at least 2 for {who}; "
                     f"got {num_rounds}.")
        reject_derived("block_size", block_size != 0, "dem_chunks")
        reject_derived("syndrome_size", syndrome_size != 0, "dem_chunks")
        reject_derived("O_sparse", bool(O), "dem_chunks")
        reject_derived("D_sparse", bool(D), "dem_chunks")
        reject_derived("error_rate_vec", bool(rates), "dem_chunks")
        if problems:
            return problems
    elif from_dem:
        reject_derived("H_sparse", bool(H), "stim_dem_path")
        reject_derived("O_sparse", bool(O), "stim_dem_path")
        reject_derived("error_rate_vec", bool(rates), "stim_dem_path")
        if problems:
            return problems
        if not D:
            return fail(f"D_sparse is required for {who} alongside "
                        "stim_dem_path, which describes the DEM but not the "
                        "detectors this decoder is sent.")
    else:
        missing = [
            k for k in ("block_size", "syndrome_size", "H_sparse", "O_sparse",
                        "D_sparse") if k not in decoder
        ]
        if missing:
            return fail(f"{who} is missing required field(s): "
                        f"{', '.join(missing)}. A flat DEM names block_size, "
                        "syndrome_size, H_sparse, O_sparse and D_sparse "
                        "together; use dem_chunks to describe the same DEM one "
                        "round at a time, or stim_dem_path to name the model "
                        "text itself, instead.")
        if not H and syndrome_size > 0:
            return fail(f"H_sparse is required for {who} unless dem_chunks is "
                        "present, which describes the same DEM one round at a "
                        "time instead.")
        if block_size == 0 and H:
            return fail(f"block_size is required for {who}")
        rows = H.count(-1)
        if rows != syndrome_size:
            return fail("Number of rows in H_sparse vector is not equal to "
                        f"syndrome_size: {rows} != {syndrome_size}")

    for name, matrix in (("H_sparse", H), ("O_sparse", O)):
        for value in matrix:
            if value < -1 or (value >= 0 and value >= block_size):
                return fail(f"Value in {name} vector is out of range: {value}")

    if D:
        rows = D.count(-1)
        if not from_dem and rows != syndrome_size:
            return fail("Number of rows in D_sparse vector is not equal to "
                        f"syndrome_size: {rows} != {syndrome_size}")
        if D[0] == -1 or any(a == -1 and b == -1 for a, b in zip(D, D[1:])):
            return fail(f"D_sparse row is empty for {who}")
    return problems


def server_checks(doc):
    """Apply ``server_checks_for_decoder`` to every decoder of ``doc``."""
    problems = []
    decoders = doc.get("decoders") if isinstance(doc, dict) else None
    if not isinstance(decoders, list):
        return problems
    for index, decoder in enumerate(decoders):
        if isinstance(decoder, dict):
            problems.extend(
                server_checks_for_decoder(decoder, f"decoders/{index}"))
    return problems


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
    if not problems:
        # The server runs its cross-field checks on a document that parsed;
        # do the same, so a type error is reported before a shape error.
        problems.extend(server_checks(doc))
    return problems


def validate_text(text, validator, allow_task_graph=False, known_types=None):
    """Validate one YAML document given as text; returns the problem list."""
    try:
        doc = load_yaml(text)
    except yaml.YAMLError as exc:
        detail = " ".join(line.strip() for line in str(exc).splitlines())
        return [("<yaml>", f"not valid YAML: {detail}")]
    if not isinstance(doc, dict):
        return [("<root>", "the document must be a mapping with a "
                 "'decoders' list")]
    return validate_document(doc, validator, allow_task_graph, known_types)


def validate_file(path, validator, allow_task_graph=False, known_types=None):
    """Validate one file; returns the problem list (empty when valid)."""
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        return [("<file>", str(exc))]
    return validate_text(text, validator, allow_task_graph, known_types)


# --- command line -------------------------------------------------------------


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
        validator = make_validator(schema)
    except jsonschema.SchemaError as exc:
        sys.stderr.write(f"not a valid JSON Schema: {exc.message}\n")
        return 2

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
