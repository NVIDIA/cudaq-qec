#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack and import checksummed runtime data, never source or executable code.

This companion to binary plugin delivery deliberately does not load engines,
interpret provider model formats, or make qualification claims. The private
producer validates its model profile before calling this shared implementation.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile

SCHEMA_VERSION = 1
MANIFEST_BYTES = 1024 * 1024
CHECKSUM_BYTES = 1024
CHUNK_BYTES = 1024 * 1024
MAX_FILES = 4096
MAX_FILE_BYTES = 4 * 1024**3
MAX_TOTAL_BYTES = 16 * 1024**3
MAX_ARCHIVE_BYTES = MAX_TOTAL_BYTES + MAX_FILES * 1024 + MANIFEST_BYTES
KINDS = {
    "engine": {".engine"},
    "constant": {".bin"},
    "json": {".json"},
    "yaml": {".yaml", ".yml"},
    "schedule": {".schedule"},
    "records": {".txt"},
    "expected": {".csv"}
}
FORBIDDEN_SUFFIXES = {
    ".py", ".pyc", ".sh", ".c", ".cc", ".cpp", ".cu", ".h", ".hpp", ".so", ".a",
    ".o", ".onnx", ".pt", ".pth", ".ckpt", ".safetensors"
}
FIELDS = {
    "schema_version", "asset_id", "provider_id", "provider_commit",
    "framework_abi_version", "compatibility", "model", "bundle_root",
    "examples", "files"
}


def _object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def _matches(pattern, value):
    return isinstance(value, str) and re.fullmatch(pattern, value) is not None


def _path(value):
    if not isinstance(value, str):
        raise ValueError("Asset paths must be strings")
    path = PurePosixPath(value)
    if (path.is_absolute() or str(path) != value or not path.parts or
            any(not _matches(r"[A-Za-z0-9_.+-]+", part) or part in (".", "..")
                for part in path.parts)):
        raise ValueError(f"Noncanonical runtime asset path: {value!r}")
    return path


def _keys(value, names, description):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError(f"Unknown or missing {description} fields")


def _text(value):
    return (isinstance(value, str) and 0 < len(value) <= 256 and
            all(ord(character) >= 32 for character in value))


def read_spec(path):
    """Read bounded, duplicate-aware producer metadata without interpreting models."""
    with Path(path).open("rb") as source:
        data = source.read(MANIFEST_BYTES + 1)
    if len(data) > MANIFEST_BYTES:
        raise ValueError("Runtime asset metadata is too large")
    return json.loads(data, object_pairs_hook=_object)


def _validate(manifest, packed=True, **checks):
    _keys(manifest, FIELDS, "runtime asset manifest")
    if (type(manifest["schema_version"]) is not int or
            manifest["schema_version"] != SCHEMA_VERSION):
        raise ValueError("Unsupported runtime asset schema")
    for key in ("asset_id", "provider_id"):
        if not _matches(r"[A-Za-z0-9_-]{1,100}", manifest[key]):
            raise ValueError(f"Invalid runtime asset {key}")
    if not _matches(r"[0-9a-f]{40}", manifest["provider_commit"]):
        raise ValueError("An exact provider commit is required")
    abi = manifest["framework_abi_version"]
    if abi is not None and (type(abi) is not int or abi <= 0):
        raise ValueError("Invalid runtime asset framework ABI")
    compatibility = manifest["compatibility"]
    _keys(compatibility, ("architecture", "cuda_version", "tensorrt_version",
                          "gpu_compute_capabilities"), "compatibility")
    if compatibility["architecture"] not in ("x86_64", "aarch64"):
        raise ValueError("Unsupported runtime asset architecture")
    if not _matches(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?",
                    compatibility["cuda_version"]):
        raise ValueError("Invalid runtime asset CUDA version")
    if not _matches(r"[0-9]+\.[0-9]+\.[0-9]+(?:\.[0-9]+)?",
                    compatibility["tensorrt_version"]):
        raise ValueError("Invalid runtime asset TensorRT version")
    capabilities = compatibility["gpu_compute_capabilities"]
    if (not isinstance(capabilities, list) or not capabilities or
            len(capabilities) > 32 or any(not _matches(r"[0-9]+\.[0-9]+", value)
                                          for value in capabilities) or
            len(capabilities) != len(set(capabilities))):
        raise ValueError(
            "Explicit unique GPU compute capabilities are required")
    model = manifest["model"]
    _keys(model, ("profile", "variant", "producer_commit", "producer_version",
                  "checkpoint_sha256"), "model provenance")
    if (not all(
            _text(model[key])
            for key in ("profile", "variant", "producer_version")) or
            not _matches(r"[0-9a-f]{40}", model["producer_commit"]) or
            not _matches(r"[0-9a-f]{64}", model["checkpoint_sha256"])):
        raise ValueError("Invalid runtime model provenance")
    bundle = _path(manifest["bundle_root"])
    files = manifest["files"]
    if not isinstance(files, list) or not 0 < len(files) <= MAX_FILES:
        raise ValueError("Invalid runtime asset file count")
    paths, kinds, total = set(), {}, 0
    for entry in files:
        _keys(entry, ("path", "kind", "sha256", "size_bytes") if packed else
              ("path", "kind"), "asset file")
        path = _path(entry["path"])
        kind = entry["kind"]
        if (not isinstance(kind, str) or kind not in KINDS or
                path.suffix not in KINDS[kind] or
                any(suffix.lower() in FORBIDDEN_SUFFIXES
                    for suffix in path.suffixes) or
                str(path) == "manifest.json"):
            raise ValueError(f"Unsupported runtime data file: {path}")
        if str(path) in paths:
            raise ValueError("Duplicate runtime asset file")
        paths.add(str(path))
        kinds[str(path)] = kind
        if packed:
            if (type(entry["size_bytes"]) is not int or
                    not 0 < entry["size_bytes"] <= MAX_FILE_BYTES or
                    not _matches(r"[0-9a-f]{64}", entry["sha256"])):
                raise ValueError("Invalid runtime asset fingerprint")
            total += entry["size_bytes"]
    if total > MAX_TOTAL_BYTES or not any(
            path.startswith(str(bundle) + "/") for path in paths):
        raise ValueError("Runtime bundle is absent or too large")
    examples = manifest["examples"]
    if not isinstance(examples, list) or not 0 < len(examples) <= MAX_FILES:
        raise ValueError("Runtime assets require executable playback examples")
    identities = set()
    for example in examples:
        _keys(example, ("id", "config", "schedule", "expected"), "example")
        if (not _matches(r"[A-Za-z0-9_-]+", example["id"]) or
                example["id"] in identities):
            raise ValueError("Invalid or duplicate runtime example ID")
        identities.add(example["id"])
        for key, kind in (("config", "yaml"), ("schedule", "schedule"),
                          ("expected", "expected")):
            if kinds.get(str(_path(example[key]))) != kind:
                raise ValueError("Example must reference declared typed files")
    for option, key in (("require_asset", "asset_id"), ("require_provider",
                                                        "provider_id"),
                        ("provider_commit", "provider_commit"),
                        ("framework_abi_version", "framework_abi_version")):
        if checks.get(option) is not None and manifest[key] != checks[option]:
            raise ValueError(f"Runtime asset {key} mismatch")
    # Engine serialization can depend on TensorRT's build component, not just
    # its SONAME. A short requirement must not approve an untested build.
    for key, components in (("architecture", None), ("cuda_version", 2),
                            ("tensorrt_version", None)):
        expected = checks.get(key)
        actual = compatibility[key]
        if expected is not None and (actual if components is None else
                                     actual.split(".")[:components]) != (
                                         expected if components is None else
                                         expected.split(".")[:components]):
            raise ValueError(f"Runtime asset {key} mismatch")
    if (checks.get("gpu_compute_capability") is not None and
            checks["gpu_compute_capability"] not in capabilities):
        raise ValueError("Runtime asset GPU compute capability mismatch")


def _sha(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _member(name, size):
    member = tarfile.TarInfo(name)
    member.size = size
    member.mode = 0o444
    return member


def pack(root, spec, output, checksum=True):
    """Export only explicitly declared regular data; never replace an archive."""
    root, output = Path(root), Path(output)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Asset root must be a real directory")
    _validate(spec, packed=False)
    manifest = json.loads(json.dumps(spec))
    for entry in manifest["files"]:
        path = root
        for part in _path(entry["path"]).parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Runtime assets cannot contain symlinks")
        if (not path.is_file() or path.stat().st_mode & 0o111 or
                not 0 < path.stat().st_size <= MAX_FILE_BYTES):
            raise ValueError(
                "Runtime assets must be regular nonexecutable data")
        with path.open("rb") as source:
            entry.update(sha256=_sha(source), size_bytes=path.stat().st_size)
    _validate(manifest)
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if len(payload) > MANIFEST_BYTES:
        raise ValueError("Runtime asset manifest is too large")
    sidecar = output.with_name(output.name + ".sha256")
    if output.exists() or output.is_symlink() or sidecar.exists(
    ) or sidecar.is_symlink():
        raise ValueError("Refusing to replace runtime asset delivery")
    published = False
    with tempfile.TemporaryDirectory(prefix=".qec-runtime-pack-",
                                     dir=output.parent) as directory:
        temporary = Path(directory) / "archive"
        with temporary.open("wb") as target:
            with gzip.GzipFile(filename="", mode="wb", fileobj=target,
                               mtime=0) as compressed:
                with tarfile.open(fileobj=compressed,
                                  mode="w",
                                  format=tarfile.USTAR_FORMAT) as archive:
                    archive.addfile(_member("manifest.json", len(payload)),
                                    io.BytesIO(payload))
                    for entry in sorted(manifest["files"],
                                        key=lambda item: item["path"]):
                        with (root / entry["path"]).open("rb") as source:
                            archive.addfile(
                                _member(entry["path"], entry["size_bytes"]),
                                source)
        # Verify the exported bytes before publication, including a producer
        # input that changed after its fingerprint was collected.
        with temporary.open("rb") as source:
            digest = _sha(source)
        temporary.with_name("archive.sha256").write_text(f"{digest}  archive\n",
                                                         encoding="ascii")
        verify(temporary)
        try:
            os.link(temporary, output)
            published = True
            if checksum:
                checksum_file = Path(directory) / "checksum"
                checksum_file.write_text(f"{digest}  {output.name}\n",
                                         encoding="ascii")
                os.link(checksum_file, sidecar)
        except BaseException:
            if published:
                output.unlink()
            raise
    return manifest


class _BoundedGzip:

    def __init__(self, source):
        self.stream = gzip.GzipFile(fileobj=source)

    def read(self, size=CHUNK_BYTES):
        if size < 0 or size > MANIFEST_BYTES + tarfile.RECORDSIZE:
            raise ValueError("Oversized runtime archive read")
        data = self.stream.read(size)
        if self.tell() > MAX_ARCHIVE_BYTES:
            raise ValueError("Runtime archive exceeds its size bound")
        return data

    def tell(self):
        return self.stream.tell()

    def seek(self, offset, whence=0):
        if whence != 0 or not 0 <= offset <= MAX_ARCHIVE_BYTES:
            raise ValueError("Unsafe runtime archive offset")
        return self.stream.seek(offset)


def _walk(archive_path, destination=None, **checks):
    archive_path = Path(archive_path)
    sidecar = archive_path.with_name(archive_path.name + ".sha256")
    if (archive_path.is_symlink() or not archive_path.is_file() or
            sidecar.is_symlink() or not sidecar.is_file() or
            archive_path.stat().st_size > MAX_ARCHIVE_BYTES):
        raise ValueError("A bounded regular archive and checksum are required")
    with sidecar.open("rb") as checksum_file:
        checksum = checksum_file.read(CHECKSUM_BYTES + 1)
    match = re.fullmatch(
        rb"([0-9a-f]{64})  " + re.escape(os.fsencode(archive_path.name)) +
        rb"\n?", checksum)
    if match is None:
        raise ValueError("Invalid basename-only runtime archive checksum")
    with archive_path.open("rb") as source:
        if _sha(source) != match.group(1).decode("ascii"):
            raise ValueError("Runtime archive checksum mismatch")
        source.seek(0)
        compressed = _BoundedGzip(source)
        with tarfile.open(fileobj=compressed, mode="r:") as archive:
            first = archive.next()
            if first is None or first.name != "manifest.json" or first.size > MANIFEST_BYTES:
                raise ValueError("Runtime archive must begin with its manifest")
            manifest, entries, seen = None, {}, set()
            for member in archive:
                _path(member.name)
                if (member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or
                        member.linkname or member.sparse is not None or
                        member.pax_headers or member.offset_data
                        != member.offset + tarfile.BLOCKSIZE or
                        member.mode & 0o111 or member.name in seen):
                    raise ValueError(
                        "Unsafe or duplicate runtime archive member")
                seen.add(member.name)
                if member is first:
                    with archive.extractfile(member) as data:
                        payload = data.read(MANIFEST_BYTES + 1)
                    manifest = json.loads(payload, object_pairs_hook=_object)
                    _validate(manifest, **checks)
                    entries = {
                        entry["path"]: entry for entry in manifest["files"]
                    }
                    if destination is not None:
                        (destination / "manifest.json").write_bytes(payload)
                        (destination / "manifest.json").chmod(0o444)
                    continue
                entry = entries.get(member.name)
                if entry is None or entry["size_bytes"] != member.size:
                    raise ValueError(
                        "Unlisted or incorrectly sized runtime data")
                target = destination / member.name if destination is not None else None
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                with archive.extractfile(member) as data:
                    with target.open(
                            "xb") if target is not None else io.BytesIO(
                            ) as output:
                        for chunk in iter(lambda: data.read(CHUNK_BYTES), b""):
                            digest.update(chunk)
                            if target is not None:
                                output.write(chunk)
                if digest.hexdigest() != entry["sha256"]:
                    raise ValueError("Runtime data checksum mismatch")
                if target is not None:
                    target.chmod(0o444)
            if manifest is None or seen != set(entries) | {"manifest.json"}:
                raise ValueError("Runtime archive is missing declared data")
            compressed.seek(archive.offset)
            for padding in iter(lambda: compressed.read(CHUNK_BYTES), b""):
                if any(padding):
                    raise ValueError(
                        "Unexpected trailing runtime archive payload")
            return manifest


def verify(archive, **checks):
    """Validate all manifest, compatibility and payload bytes without extraction."""
    return _walk(archive, **checks)


def import_assets(archive, output, **checks):
    """Atomically stage validated data into a new directory, without execution."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Refusing to replace installed runtime assets")
    with tempfile.TemporaryDirectory(prefix=".qec-runtime-import-",
                                     dir=output.parent) as directory:
        staging = Path(directory) / "validated"
        staging.mkdir()
        manifest = _walk(archive, staging, **checks)
        if output.exists() or output.is_symlink():
            raise ValueError("Runtime asset output appeared during validation")
        staging.rename(output)
    return manifest


def verify_install(root, **checks):
    """Recheck the copied customer data and its target runtime compatibility."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Installed asset root must be a real directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Installed runtime asset manifest is absent")
    manifest = read_spec(manifest_path)
    _validate(manifest, **checks)
    expected = {entry["path"] for entry in manifest["files"]
               } | {"manifest.json"}
    found = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Installed runtime assets cannot contain symlinks")
        if not path.is_dir():
            if not path.is_file() or path.stat().st_mode & 0o111:
                raise ValueError(
                    "Installed runtime assets must be regular data")
            found.add(path.relative_to(root).as_posix())
        if len(found) > MAX_FILES + 1:
            raise ValueError("Installed runtime assets exceed the file bound")
    if found != expected:
        raise ValueError("Installed runtime assets differ from their manifest")
    for entry in manifest["files"]:
        path = root / entry["path"]
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError("Installed runtime asset size mismatch")
        with path.open("rb") as source:
            if _sha(source) != entry["sha256"]:
                raise ValueError("Installed runtime asset checksum mismatch")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    packing = commands.add_parser("pack")
    packing.add_argument("--root", type=Path, required=True)
    packing.add_argument("--spec", type=Path, required=True)
    packing.add_argument("--output", type=Path, required=True)
    packing.add_argument("--checksum", action="store_true", default=True)
    for command in ("verify", "import", "verify-installed"):
        checking = commands.add_parser(command)
        if command == "verify-installed":
            checking.add_argument("--root", type=Path, required=True)
        else:
            checking.add_argument("--archive", type=Path, required=True)
        if command == "import":
            checking.add_argument("--output", type=Path, required=True)
        for option in ("architecture", "cuda-version", "tensorrt-version",
                       "provider-commit", "require-provider", "require-asset",
                       "gpu-compute-capability"):
            checking.add_argument("--" + option)
        checking.add_argument("--framework-abi-version", type=int)
    args = vars(parser.parse_args())
    command = args.pop("command")
    try:
        if command == "pack":
            spec = read_spec(args.pop("spec"))
            manifest = pack(spec=spec, **args)
        elif command == "verify":
            manifest = verify(**args)
        elif command == "verify-installed":
            manifest = verify_install(**args)
        else:
            manifest = import_assets(**args)
    except (ValueError, OSError, EOFError, tarfile.TarError) as error:
        parser.exit(1, f"Runtime asset delivery failed: {error}\n")
    print(f"Validated runtime assets: {manifest['asset_id']}")


if __name__ == "__main__":
    main()
