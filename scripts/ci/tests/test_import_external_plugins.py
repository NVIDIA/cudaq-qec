# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Public import contract tests using synthetic, never-loaded ELF modules."""

import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "import_external_plugins.py"
SPEC = importlib.util.spec_from_file_location("external_plugin_import", SCRIPT)
IMPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORTER)
PRODUCER_COMMIT = "1234567890abcdef" * 2 + "12345678"


def synthetic_elf(architecture="x86_64"):
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HH", header, 16, 3, {
        "x86_64": 62,
        "aarch64": 183
    }[architecture])
    return bytes(header) + b"synthetic module, not executable\n"


def module_descriptor(identity, kind, data, architecture="x86_64"):
    directory = {
        "decoder": "decoder-plugins",
        "task_graph": "task-graph-plugins"
    }[kind]
    return {
        "schema_version": 1,
        "id": identity,
        "kind": kind,
        "path": f"{directory}/lib{identity}.so",
        "framework_commit": PRODUCER_COMMIT,
        "source_dirty": False,
        "build_type": "Release",
        "architecture": architecture,
        "compiler": {
            "id": "Clang",
            "version": "20.1.0"
        },
        "cxx_standard": 17,
        "cuda_version": "13.0.88",
        "tensorrt_version": None,
        "glibcxx_use_cxx11_abi": 1,
        "framework_abi_version": 1 if kind == "task_graph" else None,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "needed": ["libcudaq-qec-decoders.so"],
        "rpath": ["$ORIGIN/.."],
        "build_id": "abcdef012345",
        "required_symbol_versions": ["GLIBCXX_3.4.30"]
    }


class ImportExternalPluginsTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "plugins.tar.gz"
        self.output = self.root / "staged"
        self.data = synthetic_elf()
        self.manifest = {
            "schema_version": 1,
            "delivery": "native",
            "modules": [module_descriptor("example", "task_graph", self.data)]
        }

    def write_archive(self, entries=None, manifest_data=None, trailing=b""):
        if manifest_data is None:
            manifest_data = json.dumps(self.manifest).encode()
        if entries is None:
            entries = ([(self.manifest["modules"][0]["path"],
                         self.data)] if self.manifest["modules"] else [])
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload,
                          mode="w",
                          format=tarfile.USTAR_FORMAT) as archive:
            for name, data in [("manifest.json", manifest_data)] + entries:
                if isinstance(name, tarfile.TarInfo):
                    member = name
                else:
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        self.archive.write_bytes(gzip.compress(payload.getvalue() + trailing))
        self.write_checksum()

    def write_checksum(self, basename=None):
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.archive.with_name(self.archive.name + ".sha256").write_text(
            f"{digest}  {basename or self.archive.name}\n", encoding="ascii")

    def run_import(self, **kwargs):
        arguments = {
            "archive_path": self.archive,
            "output_path": self.output,
            "delivery": "native",
            "architecture": "x86_64",
            "framework_commit": PRODUCER_COMMIT,
            "cuda_version": "13.0"
        }
        arguments.update(kwargs)
        return IMPORTER.import_plugins(**arguments)

    def assert_rejected(self, **kwargs):
        with self.assertRaises((ValueError, OSError)):
            self.run_import(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertEqual([], list(self.root.glob(".qec-plugin-import-*")))

    def test_both_deliveries_and_architectures(self):
        for delivery in ("native", "wheel"):
            for architecture in ("x86_64", "aarch64"):
                with self.subTest(delivery=delivery, architecture=architecture):
                    data = synthetic_elf(architecture)
                    self.manifest["delivery"] = delivery
                    self.manifest["modules"] = [
                        module_descriptor("decoder", "decoder", data,
                                          architecture),
                        module_descriptor("graph", "task_graph", data,
                                          architecture)
                    ]
                    entries = [(module["path"], data)
                               for module in self.manifest["modules"]]
                    self.write_archive(entries)
                    output = self.root / f"{delivery}-{architecture}"
                    original = self.archive.read_bytes()
                    result = self.run_import(output_path=output,
                                             delivery=delivery,
                                             architecture=architecture,
                                             required=("decoder", "graph"))
                    self.assertEqual(self.manifest, result)
                    self.assertEqual(
                        self.manifest,
                        json.loads((output / "manifest.json").read_text()))
                    self.assertEqual(
                        {"manifest.json", *(name for name, _ in entries)}, {
                            str(path.relative_to(output))
                            for path in output.rglob("*")
                            if path.is_file()
                        })
                    self.assertEqual(original, self.archive.read_bytes())
                    for name, _ in entries:
                        self.assertEqual(data, (output / name).read_bytes())
                        self.assertEqual(0,
                                         (output / name).stat().st_mode & 0o222)

    def test_cpu_provider_without_cuda_requirement(self):
        self.manifest["modules"][0]["cuda_version"] = None
        self.write_archive()
        self.run_import()

    def test_task_graph_abi_must_match_before_staging(self):
        for value in (None, 0, 2, -1, True, "1"):
            with self.subTest(value=value):
                self.manifest["modules"][0]["framework_abi_version"] = value
                self.write_archive()
                with self.assertRaisesRegex(ValueError, "ABI version mismatch"):
                    self.run_import()
                self.assertFalse(self.output.exists())

    def test_decoder_does_not_claim_task_graph_abi(self):
        self.manifest["modules"] = [
            module_descriptor("decoder", "decoder", self.data)
        ]
        self.manifest["modules"][0]["framework_abi_version"] = 1
        self.write_archive()
        with self.assertRaisesRegex(ValueError, "must not declare"):
            self.run_import()
        self.assertFalse(self.output.exists())

    def test_import_abi_version_matches_public_gateway(self):
        import re
        header = SCRIPT.parents[
            2] / "libs/qec/include/cudaq/qec/task_graph_plugin.h"
        match = re.search(
            r"#define CUDAQ_QEC_TASK_GRAPH_FRAMEWORK_ABI_VERSION ([0-9]+)",
            header.read_text(encoding="utf-8"))
        self.assertIsNotNone(match)
        self.assertEqual(int(match.group(1)),
                         IMPORTER.EXPECTED_TASK_GRAPH_FRAMEWORK_ABI_VERSION)

    def test_checksum_is_required_before_tar_processing(self):
        self.archive.write_bytes(b"not a tar file")
        with mock.patch.object(IMPORTER.gzip, "GzipFile") as decompressor:
            self.assert_rejected()
            decompressor.assert_not_called()

    def test_checksum_mismatch_precedes_tar_processing(self):
        self.write_archive()
        self.archive.write_bytes(self.archive.read_bytes() + b"changed")
        with mock.patch.object(IMPORTER.gzip, "GzipFile") as decompressor:
            self.assert_rejected()
            decompressor.assert_not_called()

    def test_checksum_cannot_name_another_path(self):
        self.write_archive()
        for name in ("../plugins.tar.gz", "/tmp/plugins.tar.gz",
                     "different.tar.gz"):
            with self.subTest(name=name):
                self.write_checksum(name)
                self.assert_rejected()

    def test_checksum_symlink_and_oversize(self):
        self.write_archive()
        sidecar = self.archive.with_name(self.archive.name + ".sha256")
        original = self.root / "checksum"
        sidecar.rename(original)
        sidecar.symlink_to(original)
        self.assert_rejected()
        sidecar.unlink()
        sidecar.write_bytes(b"x" * 1025)
        self.assert_rejected()

    def test_requested_compatibility_is_required(self):
        self.write_archive()
        for arguments in ({
                "delivery": "wheel"
        }, {
                "architecture": "aarch64"
        }, {
                "architecture": "amd64"
        }, {
                "framework_commit": "f" * 40
        }, {
                "framework_commit": "HEAD"
        }, {
                "cuda_version": "12.9"
        }, {
                "cuda_version": "13"
        }, {
                "required": ("missing",)
        }):
            with self.subTest(arguments=arguments):
                self.assert_rejected(**arguments)

    def test_manifest_and_module_schema(self):
        mutations = [
            lambda value: value.update(extra=True),
            lambda value: value.update(schema_version=True),
            lambda value: value.update(modules=[]),
            lambda value: value["modules"][0].update(extra=True),
            lambda value: value["modules"][0].update(build_type="Debug"),
            lambda value: value["modules"][0].update(source_dirty="false"),
            lambda value: value["modules"][0].update(cxx_standard=True), lambda
            value: value["modules"][0].update(glibcxx_use_cxx11_abi=True),
            lambda value: value["modules"][0].update(compiler={"id": "Clang"}),
            lambda value: value["modules"][0].update(needed=["/tmp/libhost.so"]
                                                    ),
            lambda value: value["modules"][0].update(rpath=["/tmp"]),
            lambda value: value["modules"][0].update(build_id="not hex"),
            lambda value: value["modules"][0].update(kind="unknown"),
            lambda value: value["modules"][0].update(size_bytes=True),
            lambda value: value["modules"].append(
                copy.deepcopy(value["modules"][0]))
        ]
        pristine = copy.deepcopy(self.manifest)
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self.manifest = copy.deepcopy(pristine)
                mutate(self.manifest)
                self.write_archive()
                self.assert_rejected()

    def test_duplicate_json_fields_at_all_levels(self):
        for old, new in ((b'"delivery": "native"',
                          b'"delivery": "native", "delivery": "native"'),
                         (b'"id": "example"',
                          b'"id": "example", "id": "example"'),
                         (b'"id": "Clang"', b'"id": "Clang", "id": "Clang"')):
            with self.subTest(old=old):
                data = json.dumps(self.manifest).encode().replace(old, new)
                self.write_archive(manifest_data=data)
                self.assert_rejected()

    def test_noncanonical_and_wrong_kind_paths(self):
        for path in ("../outside.so", "/outside.so",
                     "decoder-plugins/../outside.so",
                     "task-graph-plugins//libexample.so",
                     "task-graph-plugins/./libexample.so",
                     "task-graph-plugins\\libexample.so",
                     "decoder-plugins/libexample.so",
                     "task-graph-plugins/libexample.so/extra", "C:/example.so",
                     "task-graph-plugins/lib;example.so",
                     "task-graph-plugins/lib\nexample.so"):
            with self.subTest(path=path):
                self.manifest["modules"][0]["path"] = path
                self.write_archive()
                self.assert_rejected()
        self.assertFalse((self.root / "outside.so").exists())

    def test_extra_duplicate_missing_and_noncanonical_members(self):
        path = self.manifest["modules"][0]["path"]
        for entries in ([], [(path, self.data), (path, self.data)], [
            (path, self.data), ("extra.txt", b"extra")
        ], [("task-graph-plugins/./libexample.so", self.data)]):
            with self.subTest(entries=[name for name, _ in entries]):
                self.write_archive(entries)
                self.assert_rejected()

    def test_nonregular_members(self):
        path = self.manifest["modules"][0]["path"]
        for member_type in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
                            tarfile.FIFOTYPE, tarfile.CHRTYPE,
                            tarfile.CONTTYPE):
            with self.subTest(member_type=member_type):
                member = tarfile.TarInfo(path)
                member.type = member_type
                member.linkname = "../outside.so" if member_type in (
                    tarfile.SYMTYPE, tarfile.LNKTYPE) else ""
                self.write_archive([(member, b"")])
                self.assert_rejected()

    def test_extended_headers_rejected(self):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w",
                          format=tarfile.PAX_FORMAT) as archive:
            member = tarfile.TarInfo("manifest.json")
            member.pax_headers = {"comment": "not in the export protocol"}
            data = json.dumps(self.manifest).encode()
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        self.archive.write_bytes(gzip.compress(buffer.getvalue()))
        self.write_checksum()
        self.assert_rejected()

    def test_payload_fingerprint_and_elf_header(self):
        path = self.manifest["modules"][0]["path"]
        self.write_archive([(path, self.data + b"extra")])
        self.assert_rejected()
        self.write_archive([(path, self.data[:-1] + b"X")])
        self.assert_rejected()
        for offset, replacement in ((0, b"bad!"), (4, b"\x01"), (5, b"\x02"),
                                    (16, struct.pack("<H", 2)),
                                    (18, struct.pack("<H", 183))):
            with self.subTest(offset=offset):
                data = bytearray(self.data)
                data[offset:offset + len(replacement)] = replacement
                self.manifest["modules"][0] = module_descriptor(
                    "example", "task_graph", data)
                self.write_archive([(path, data)])
                self.assert_rejected()

    def test_trailing_payload_and_truncated_gzip(self):
        self.write_archive(trailing=b"hidden payload")
        self.assert_rejected()
        for extra in (b"hidden compressed bytes",
                      gzip.compress(b"hidden second member")):
            with self.subTest(extra=extra):
                self.write_archive()
                self.archive.write_bytes(self.archive.read_bytes() + extra)
                self.write_checksum()
                self.assert_rejected()
        self.write_archive()
        self.archive.write_bytes(self.archive.read_bytes()[:-5])
        self.write_checksum()
        self.assert_rejected()

    def test_manifest_first_and_size_limit(self):
        self.archive.write_bytes(gzip.compress(b""))
        self.write_checksum()
        self.assert_rejected()
        self.write_archive(manifest_data=b" " * (1024 * 1024 + 1))
        self.assert_rejected()

    def test_existing_output_and_symlink_are_preserved(self):
        self.write_archive()
        self.output.mkdir()
        marker = self.output / "existing"
        marker.write_text("preserve")
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertEqual("preserve", marker.read_text())
        marker.unlink()
        self.output.rmdir()
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(ValueError):
            self.run_import()
        self.assertTrue(self.output.is_symlink())

    def test_reuse_validates_without_mutation(self):
        self.write_archive()
        self.run_import()
        files = [path for path in self.output.rglob("*") if path.is_file()]
        original = {
            path: (path.read_bytes(), path.stat().st_mtime_ns,
                   path.stat().st_mode) for path in files
        }
        self.assertEqual(self.manifest, self.run_import(reuse=True))
        self.assertEqual(
            original, {
                path: (path.read_bytes(), path.stat().st_mtime_ns,
                       path.stat().st_mode) for path in files
            })

    def test_reuse_rejects_tampering_without_repair(self):
        self.write_archive()
        self.run_import()
        module = self.output / self.manifest["modules"][0]["path"]
        module.chmod(0o644)
        module.write_bytes(self.data[:-1] + b"X")
        with self.assertRaises(ValueError):
            self.run_import(reuse=True)
        self.assertEqual(self.data[:-1] + b"X", module.read_bytes())
        module.write_bytes(self.data)
        manifest_path = self.output / "manifest.json"
        manifest_path.chmod(0o644)
        manifest_path.write_text("tampered")
        with self.assertRaises(ValueError):
            self.run_import(reuse=True)
        self.assertEqual("tampered", manifest_path.read_text())

    def test_reuse_rejects_extra_files_directories_and_links(self):
        self.write_archive()
        self.run_import()
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                extra = self.output / "extra"
                if kind == "file":
                    extra.write_bytes(b"unexpected")
                elif kind == "directory":
                    extra.mkdir()
                else:
                    extra.symlink_to(self.root / "missing")
                with self.assertRaises(ValueError):
                    self.run_import(reuse=True)
                if kind == "directory":
                    extra.rmdir()
                else:
                    extra.unlink()
        module = self.output / self.manifest["modules"][0]["path"]
        module.unlink()
        module.symlink_to(self.archive)
        with self.assertRaises(ValueError):
            self.run_import(reuse=True)
        self.assertTrue(module.is_symlink())

    def test_reuse_revalidates_archive_before_accepting_stage(self):
        self.write_archive()
        self.run_import()
        self.write_archive(trailing=b"hidden payload")
        with self.assertRaises(ValueError):
            self.run_import(reuse=True)
        self.assertEqual(self.data,
                         (self.output /
                          self.manifest["modules"][0]["path"]).read_bytes())

    def test_cli_and_required_producer_commit(self):
        self.write_archive()
        command = [
            sys.executable,
            str(SCRIPT), "--archive",
            str(self.archive), "--output",
            str(self.output), "--delivery", "native", "--architecture",
            "x86_64", "--cuda-version", "13.0"
        ]
        rejected = subprocess.run(command,
                                  capture_output=True,
                                  text=True,
                                  check=False)
        self.assertNotEqual(0, rejected.returncode)
        self.assertIn("--framework-commit", rejected.stderr)
        accepted = subprocess.run(
            command +
            ["--framework-commit", PRODUCER_COMMIT, "--require", "example"],
            capture_output=True,
            text=True,
            check=False)
        self.assertEqual(0, accepted.returncode, accepted.stderr)
        self.assertIn("Validated 1 plugin", accepted.stdout)


if __name__ == "__main__":
    unittest.main()
