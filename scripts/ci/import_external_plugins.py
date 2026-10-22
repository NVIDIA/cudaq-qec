#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Validate a binary plugin delivery and stage its modules without loading them.

The mandatory checksum sidecar is ``ARCHIVE.sha256`` and names only ARCHIVE's
basename. The reviewed producer framework commit is provenance, not a runtime
ABI handshake with the consumer. The recorded task-graph ABI must match this
importer's interface; the runtime loader independently verifies the module's
compiled descriptor before registration. GPU execution still needs testing.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile

SCHEMA_VERSION = 1
# Keep this in step with CUDAQ_QEC_TASK_GRAPH_FRAMEWORK_ABI_VERSION in the
# installed C gateway. The runtime checks the binary rather than trusting JSON.
EXPECTED_TASK_GRAPH_FRAMEWORK_ABI_VERSION = 1
MANIFEST_LIMIT_BYTES = 1024 * 1024
CHECKSUM_LIMIT_BYTES = 1024
COPY_CHUNK_BYTES = 1024 * 1024
ELF_HEADER_BYTES = 20
ELF_SHARED_OBJECT = 3
ELF_ARCHITECTURES = {"x86_64": 62, "aarch64": 183}
PLUGIN_DIRECTORIES = {
    "decoder": "decoder-plugins",
    "task_graph": "task-graph-plugins"
}
MODULE_FIELDS = {
    "schema_version", "id", "kind", "path", "framework_commit", "source_dirty",
    "build_type", "architecture", "compiler", "cxx_standard", "cuda_version",
    "tensorrt_version", "glibcxx_use_cxx11_abi", "sha256", "size_bytes",
    "needed", "rpath", "build_id", "required_symbol_versions",
    "framework_abi_version"
}
COMMIT_PATTERN = r"[0-9a-f]{40}"
VERSION_PATTERN = r"[0-9]+\.[0-9]+(?:\.[0-9]+)?"


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _canonical_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Plugin path must be a nonempty string")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or ".." in path.parts or
            any(character in value for character in ("\\", ":", "\x00")) or
            any(ord(character) < 32 for character in value)):
        raise ValueError(f"Noncanonical plugin path: {value!r}")
    return path


def _validate_module(module, architecture, framework_commit, cuda_version):
    if not isinstance(module, dict) or set(module) != MODULE_FIELDS:
        raise ValueError("Unknown or missing plugin module fields")
    if (type(module["schema_version"]) is not int or
            module["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Unsupported plugin module schema")
    if not _matches(r"[A-Za-z0-9_-]+", module["id"]):
        raise ValueError("Invalid plugin identity")
    kind = module["kind"]
    if not isinstance(kind, str) or kind not in PLUGIN_DIRECTORIES:
        raise ValueError("Unsupported plugin kind")
    framework_abi = module["framework_abi_version"]
    if kind == "task_graph":
        if (type(framework_abi) is not int or
                framework_abi != EXPECTED_TASK_GRAPH_FRAMEWORK_ABI_VERSION):
            raise ValueError("Task-graph framework ABI version mismatch")
    elif framework_abi is not None:
        raise ValueError("Decoder plugins must not declare a task-graph ABI")
    path = _canonical_path(module["path"])
    if (len(path.parts) != 2 or path.parts[0] != PLUGIN_DIRECTORIES[kind] or
            path.suffix != ".so" or
            not re.fullmatch(r"[A-Za-z0-9_.+-]+\.so", path.name)):
        raise ValueError("Plugin path does not match its kind")
    if module["build_type"] != "Release":
        raise ValueError("Only Release plugins may be imported")
    if (not _matches(COMMIT_PATTERN, module["framework_commit"]) or
            module["framework_commit"] != framework_commit):
        raise ValueError("Plugin producer framework commit mismatch")
    if module["architecture"] != architecture:
        raise ValueError("Plugin architecture mismatch")
    if (module["source_dirty"] is not None and
            type(module["source_dirty"]) is not bool):
        raise ValueError("source_dirty must be boolean or null")
    abi = module["glibcxx_use_cxx11_abi"]
    if abi is not None and (type(abi) is not int or abi not in (0, 1)):
        raise ValueError("Invalid libstdc++ ABI metadata")
    compiler = module["compiler"]
    if (not isinstance(compiler, dict) or set(compiler) != {
            "id", "version"
    } or not all(
            isinstance(value, str) and value for value in compiler.values())):
        raise ValueError("Compiler identity and version are required")
    if type(module["cxx_standard"]) is not int or module["cxx_standard"] < 17:
        raise ValueError("Invalid C++ standard metadata")
    for key in ("cuda_version", "tensorrt_version"):
        if module[key] is not None and not isinstance(module[key], str):
            raise ValueError(f"Invalid {key} metadata")
    recorded_cuda = module["cuda_version"]
    if recorded_cuda and (not _matches(VERSION_PATTERN, recorded_cuda) or
                          recorded_cuda.split(".")[:2]
                          != cuda_version.split(".")[:2]):
        raise ValueError("Plugin CUDA toolkit mismatch")
    if (not _matches(r"[0-9a-f]{64}", module["sha256"]) or
            type(module["size_bytes"]) is not int or
            module["size_bytes"] < ELF_HEADER_BYTES):
        raise ValueError("Invalid plugin size or SHA-256")
    for key in ("needed", "rpath", "required_symbol_versions"):
        values = module[key]
        if (not isinstance(values, list) or
                not all(isinstance(value, str) and value for value in values)):
            raise ValueError(f"Invalid ELF metadata: {key}")
    if any("/" in value or "\\" in value for value in module["needed"]):
        raise ValueError("Nonrelocatable dependency in manifest")
    if any(not (value == "$ORIGIN" or value.startswith("$ORIGIN/"))
           for value in module["rpath"]):
        raise ValueError("Nonrelocatable RPATH in manifest")
    if any(
            any(character in value
                for character in (";", ":", "\n", "\r", "\x00"))
            for value in module["rpath"]):
        raise ValueError("Unsafe RPATH entry in manifest")
    if (module["build_id"] is not None and
            not _matches(r"[0-9a-f]+", module["build_id"])):
        raise ValueError("Invalid ELF build identity")


def _read_manifest(data, delivery, architecture, framework_commit, cuda_version,
                   required):
    manifest = json.loads(data, object_pairs_hook=_unique_object)
    if (not isinstance(manifest, dict) or
            set(manifest) != {"schema_version", "delivery", "modules"} or
            type(manifest["schema_version"]) is not int or
            manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Invalid plugin manifest schema")
    if manifest["delivery"] != delivery:
        raise ValueError("Plugin delivery mismatch")
    modules = manifest["modules"]
    if not isinstance(modules, list) or not modules:
        raise ValueError("Plugin manifest must list at least one module")
    identities, destinations = set(), set()
    for module in modules:
        _validate_module(module, architecture, framework_commit, cuda_version)
        if module["id"] in identities or module["path"] in destinations:
            raise ValueError("Duplicate plugin identity or destination")
        identities.add(module["id"])
        destinations.add(module["path"])
    missing = set(required) - identities
    if missing:
        raise ValueError(f"Required plugins are missing: {sorted(missing)}")
    return manifest


def _check_regular_member(member):
    _canonical_path(member.name)
    if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or
            member.linkname or member.sparse is not None or
            member.pax_headers or
            member.offset_data != member.offset + tarfile.BLOCKSIZE):
        raise ValueError(f"Unsafe plugin archive member: {member.name}")


def _check_archive_checksum(source, archive_path):
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    if sidecar.is_symlink() or not sidecar.is_file():
        raise ValueError("A regular archive SHA-256 sidecar is required")
    with sidecar.open("rb") as stream:
        data = stream.read(CHECKSUM_LIMIT_BYTES + 1)
    if len(data) > CHECKSUM_LIMIT_BYTES:
        raise ValueError("Archive checksum sidecar is too large")
    pattern = rb"([0-9a-f]{64})  " + re.escape(os.fsencode(archive_path.name))
    match = re.fullmatch(pattern + rb"\n?", data)
    if match is None:
        raise ValueError("Invalid basename-only archive checksum sidecar")
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
        digest.update(chunk)
    if digest.hexdigest() != match.group(1).decode("ascii"):
        raise ValueError("Archive SHA-256 mismatch")
    source.seek(0)


def _copy_module(source, destination, module, architecture, reuse=False):
    digest = hashlib.sha256()
    total = 0
    header = bytearray()
    with destination.open("rb" if reuse else "xb") as target:
        for chunk in iter(lambda: source.read(COPY_CHUNK_BYTES), b""):
            if len(header) < ELF_HEADER_BYTES:
                header.extend(chunk[:ELF_HEADER_BYTES - len(header)])
            total += len(chunk)
            if total > module["size_bytes"]:
                raise ValueError(f"Plugin size mismatch: {module['path']}")
            digest.update(chunk)
            if reuse:
                if target.read(len(chunk)) != chunk:
                    raise ValueError(
                        f"Existing plugin differs: {module['path']}")
            else:
                target.write(chunk)
        if reuse and target.read(1):
            raise ValueError(f"Existing plugin is too large: {module['path']}")
    if total != module["size_bytes"] or digest.hexdigest() != module["sha256"]:
        raise ValueError(f"Plugin fingerprint mismatch: {module['path']}")
    if (len(header) != ELF_HEADER_BYTES or header[:4] != b"\x7fELF" or
            header[4:6] != bytes((2, 1))):
        raise ValueError(f"Plugin is not ELF64 little-endian: {module['path']}")
    elf_type, machine = struct.unpack_from("<HH", header, 16)
    if elf_type != ELF_SHARED_OBJECT or machine != ELF_ARCHITECTURES[
            architecture]:
        raise ValueError(
            f"Plugin ELF type or architecture mismatch: {module['path']}")
    if not reuse:
        destination.chmod(0o444)


def _check_staged_tree(staging, expected):
    """Check the shallow, fixed layout before opening any existing modules."""
    expected_files = set(expected) | {"manifest.json"}
    expected_directories = {PurePosixPath(name).parts[0] for name in expected}
    found = set()
    for child in staging.iterdir():
        if child.is_symlink():
            raise ValueError("Symlink in existing plugin staging directory")
        if child.is_dir():
            if child.name not in expected_directories:
                raise ValueError("Unexpected existing plugin directory")
            for module in child.iterdir():
                name = f"{child.name}/{module.name}"
                if (module.is_symlink() or not module.is_file() or
                        name not in expected_files):
                    raise ValueError("Unexpected existing plugin entry")
                found.add(name)
        elif child.is_file() and child.name == "manifest.json":
            found.add(child.name)
        else:
            raise ValueError("Unexpected existing plugin entry")
    if found != expected_files:
        raise ValueError(
            "Existing plugin staging files do not match the archive")


def _stage_archive(source,
                   staging,
                   delivery,
                   architecture,
                   framework_commit,
                   cuda_version,
                   required,
                   reuse=False):
    # Opening the gzip stream explicitly also validates its trailer when reading
    # the tar padding, instead of silently accepting a truncated gzip member.
    with gzip.GzipFile(fileobj=source, mode="rb") as decompressed:
        with tarfile.open(fileobj=decompressed, mode="r:") as archive:
            first = archive.next()
            if first is None or first.name != "manifest.json":
                raise ValueError("Plugin archive must begin with manifest.json")
            _check_regular_member(first)
            if first.size > MANIFEST_LIMIT_BYTES:
                raise ValueError("Plugin manifest is too large")
            with archive.extractfile(first) as stream:
                manifest = _read_manifest(stream.read(), delivery, architecture,
                                          framework_commit, cuda_version,
                                          required)
            expected = {
                module["path"]: module for module in manifest["modules"]
            }
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) +
                              "\n").encode("utf-8")
            if reuse:
                _check_staged_tree(staging, expected)
                with (staging / "manifest.json").open("rb") as staged_manifest:
                    if staged_manifest.read(len(manifest_bytes) +
                                            1) != manifest_bytes:
                        raise ValueError(
                            "Existing plugin manifest differs from the archive")
            seen = {"manifest.json"}
            for member in archive:
                if member is first:
                    continue
                _check_regular_member(member)
                if member.name in seen or member.name not in expected:
                    raise ValueError(
                        f"Unexpected plugin archive member: {member.name}")
                seen.add(member.name)
                module = expected[member.name]
                if member.size != module["size_bytes"]:
                    raise ValueError(f"Plugin size mismatch: {member.name}")
                destination = staging / member.name
                if not reuse:
                    destination.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as stream:
                    _copy_module(stream, destination, module, architecture,
                                 reuse)
            if seen != set(expected) | {"manifest.json"}:
                raise ValueError("Archive is missing a manifest-listed module")
            decompressed.seek(archive.offset)
            for chunk in iter(lambda: decompressed.read(COPY_CHUNK_BYTES), b""):
                if any(chunk):
                    raise ValueError("Nonzero trailing plugin archive payload")
    if not reuse:
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        manifest_path.chmod(0o444)
    return manifest


def import_plugins(archive_path,
                   output_path,
                   delivery,
                   architecture,
                   framework_commit,
                   cuda_version,
                   required=(),
                   reuse=False):
    """Publish validated copies, leaving no output on validation failure."""
    if delivery not in ("native",
                        "wheel") or architecture not in ELF_ARCHITECTURES:
        raise ValueError("Unsupported delivery or architecture")
    if not _matches(COMMIT_PATTERN, framework_commit):
        raise ValueError(
            "An exact reviewed producer framework commit is required")
    if not _matches(VERSION_PATTERN, cuda_version):
        raise ValueError("A CUDA toolkit version is required")
    archive_path, output_path = Path(archive_path), Path(output_path)
    existing = output_path.exists() or output_path.is_symlink()
    if existing and (not reuse or output_path.is_symlink() or
                     not output_path.is_dir()):
        raise ValueError(
            f"Refusing to replace plugin staging output: {output_path}")
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ValueError("Plugin archive must be a regular file")
    try:
        with archive_path.open("rb") as source:
            _check_archive_checksum(source, archive_path)
            if existing:
                return _stage_archive(source,
                                      output_path,
                                      delivery,
                                      architecture,
                                      framework_commit,
                                      cuda_version,
                                      required,
                                      reuse=True)
            with tempfile.TemporaryDirectory(
                    prefix=".qec-plugin-import-",
                    dir=output_path.parent) as temporary:
                staged = Path(temporary) / "validated"
                staged.mkdir()
                manifest = _stage_archive(source, staged, delivery,
                                          architecture, framework_commit,
                                          cuda_version, required)
                if output_path.exists() or output_path.is_symlink():
                    raise ValueError(
                        "Plugin staging output appeared during validation")
                staged.rename(output_path)
                return manifest
    except (tarfile.TarError, EOFError, gzip.BadGzipFile,
            UnicodeError) as error:
        raise ValueError("Invalid plugin archive") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delivery",
                        choices=("native", "wheel"),
                        required=True)
    parser.add_argument("--architecture",
                        choices=tuple(ELF_ARCHITECTURES),
                        required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument(
        "--framework-commit",
        required=True,
        help="Exact reviewed producer commit; not a host ABI check")
    parser.add_argument("--require", action="append", default=[])
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Accept existing staging only if every byte matches")
    args = parser.parse_args()
    try:
        manifest = import_plugins(args.archive, args.output, args.delivery,
                                  args.architecture, args.framework_commit,
                                  args.cuda_version, args.require, args.reuse)
    except (ValueError, OSError) as error:
        parser.exit(1, f"External plugin import failed: {error}\n")
    print(f"Validated {len(manifest['modules'])} plugin(s): {args.output}")


if __name__ == "__main__":
    main()
