# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data archive security and provider-independent compatibility contracts."""

import copy
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "runtime_assets.py"
SPEC = importlib.util.spec_from_file_location("runtime_assets", SCRIPT)
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


class RuntimeAssetsTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.data.mkdir()
        payloads = {
            "bundle/model.engine": ("engine", b"engine fixture"),
            "bundle/constants.bin": ("constant", b"constant fixture"),
            "bundle/bundle_manifest.json": ("json", b"{}"),
            "examples/x.yaml": ("yaml", b"artifact_bundle: bundle\n"),
            "examples/x.schedule": ("schedule", b"+0 reset\n"),
            "examples/x.csv": ("expected", b"expected\n0\n")
        }
        for name, (_, contents) in payloads.items():
            path = self.data / name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(contents)
        self.spec = {
            "schema_version":
                1,
            "asset_id":
                "fixture-data",
            "provider_id":
                "fixture-provider",
            "provider_commit":
                "a" * 40,
            "framework_abi_version":
                3,
            "compatibility": {
                "architecture": "x86_64",
                "cuda_version": "13.0.88",
                "tensorrt_version": "10.14.1.48",
                "gpu_compute_capabilities": ["12.0"]
            },
            "model": {
                "profile": "fixture-profile",
                "variant": "fixture-variant",
                "producer_commit": "b" * 40,
                "producer_version": "v1",
                "checkpoint_sha256": "c" * 64
            },
            "bundle_root":
                "bundle",
            "examples": [{
                "id": "x",
                "config": "examples/x.yaml",
                "schedule": "examples/x.schedule",
                "expected": "examples/x.csv"
            }],
            "files": [{
                "path": name,
                "kind": kind
            } for name, (kind, _) in payloads.items()]
        }
        self.archive = self.root / "assets.tar.gz"
        self.output = self.root / "installed"

    def pack(self):
        return ASSETS.pack(self.data, self.spec, self.archive)

    def test_installed_copy_rechecked_with_runtime_compatibility(self):
        self.pack()
        ASSETS.import_assets(self.archive, self.output)
        ASSETS.verify_install(self.output, tensorrt_version="10.14.1.48")
        with self.assertRaisesRegex(ValueError, "TensorRT|tensorrt"):
            ASSETS.verify_install(self.output, tensorrt_version="10.16.0.1")
        target = self.output / "bundle/constants.bin"
        target.chmod(0o644)
        target.write_bytes(b"different data")
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ASSETS.verify_install(self.output)

    def test_installed_copy_rejects_extra_symlink_and_executable(self):
        self.pack()
        ASSETS.import_assets(self.archive, self.output)
        extra = self.output / "extra.py"
        extra.write_text("not delivered")
        with self.assertRaisesRegex(ValueError, "differ"):
            ASSETS.verify_install(self.output)
        extra.unlink()
        extra.symlink_to("bundle/constants.bin")
        with self.assertRaisesRegex(ValueError, "symlink"):
            ASSETS.verify_install(self.output)
        extra.unlink()
        (self.output / "bundle/constants.bin").chmod(0o755)
        with self.assertRaisesRegex(ValueError, "regular data"):
            ASSETS.verify_install(self.output)

    def checksum(self):
        self.archive.with_name(self.archive.name + ".sha256").write_text(
            hashlib.sha256(self.archive.read_bytes()).hexdigest() + "  " +
            self.archive.name + "\n")

    def rewrite(self, transform):
        with tarfile.open(self.archive, "r:gz") as source:
            entries = [
                (member, source.extractfile(member).read()) for member in source
            ]
        entries = transform(entries)
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w",
                          format=tarfile.USTAR_FORMAT) as target:
            for member, data in entries:
                target.addfile(member, io.BytesIO(data))
        self.archive.write_bytes(gzip.compress(buffer.getvalue()))
        self.checksum()

    def test_round_trip_and_compatibility(self):
        manifest = self.pack()
        self.assertEqual(
            manifest,
            ASSETS.verify(self.archive,
                          architecture="x86_64",
                          cuda_version="13.0",
                          tensorrt_version="10.14.1.48",
                          provider_commit="a" * 40,
                          require_provider="fixture-provider",
                          require_asset="fixture-data",
                          gpu_compute_capability="12.0",
                          framework_abi_version=3))
        self.assertEqual(manifest,
                         ASSETS.import_assets(self.archive, self.output))
        self.assertEqual((self.data / "bundle/model.engine").read_bytes(),
                         (self.output / "bundle/model.engine").read_bytes())
        self.assertEqual(
            0, (self.output / "bundle/model.engine").stat().st_mode & 0o222)

    def test_wrong_compatibility_and_identity_fail_without_output(self):
        self.pack()
        for key, value in (("architecture", "aarch64"), ("cuda_version",
                                                         "12.6"),
                           ("tensorrt_version",
                            "10.13.3"), ("gpu_compute_capability",
                                         "10.0"), ("provider_commit", "d" * 40),
                           ("require_provider",
                            "other"), ("require_asset",
                                       "other"), ("framework_abi_version", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ASSETS.import_assets(self.archive, self.output, **{key: value})
            self.assertFalse(self.output.exists())

    def test_tensorrt_engine_build_version_must_match_exactly(self):
        self.pack()
        for version in ("10.14.1", "10.14.1.49", "10.14.2.48"):
            with self.subTest(version=version), self.assertRaisesRegex(
                    ValueError, "tensorrt_version mismatch"):
                ASSETS.verify(self.archive, tensorrt_version=version)

    def test_only_declared_data_is_exported(self):
        for name in ("private.py", "model.onnx", "weights.safetensors",
                     "train.ckpt"):
            (self.data / name).write_text("never exported")
        self.pack()
        with tarfile.open(self.archive, "r:gz") as source:
            self.assertEqual({"manifest.json"} |
                             {entry["path"] for entry in self.spec["files"]},
                             set(source.getnames()))

    def test_source_and_checkpoint_suffixes_rejected(self):
        for name in ("source.py", "source.py.bin", "model.onnx", "weights.pt",
                     "module.so"):
            with self.subTest(name=name):
                spec = copy.deepcopy(self.spec)
                spec["files"].append({"path": name, "kind": "constant"})
                with self.assertRaises(ValueError):
                    ASSETS.pack(self.data, spec, self.archive)
                self.assertFalse(self.archive.exists())

    def test_links_executable_and_traversal_rejected(self):
        engine = self.data / "bundle/model.engine"
        engine.chmod(0o755)
        with self.assertRaises(ValueError):
            self.pack()
        engine.chmod(0o644)
        engine.unlink()
        engine.symlink_to(self.data / "bundle/constants.bin")
        with self.assertRaises(ValueError):
            self.pack()
        self.spec["files"][0]["path"] = "../outside.bin"
        with self.assertRaises(ValueError):
            self.pack()

    def test_modified_payload_and_archive_checksum_rejected(self):
        self.pack()
        original = self.archive.read_bytes()
        self.archive.write_bytes(original + b"corrupt")
        with self.assertRaises(ValueError):
            ASSETS.verify(self.archive)
        self.archive.write_bytes(original)
        self.rewrite(lambda entries: entries[:-1])
        with self.assertRaises(ValueError):
            ASSETS.import_assets(self.archive, self.output)
        self.assertFalse(self.output.exists())

    def test_unlisted_duplicate_and_executable_members_rejected(self):
        for mode in ("unlisted", "duplicate", "executable"):
            with self.subTest(mode=mode):
                self.pack()

                def alter(entries):
                    if mode == "duplicate":
                        return entries + [entries[-1]]
                    if mode == "unlisted":
                        member = ASSETS._member("secret.py", 1)
                        return entries + [(member, b"x")]
                    entries[-1][0].mode = 0o755
                    return entries

                self.rewrite(alter)
                with self.assertRaises(ValueError):
                    ASSETS.import_assets(self.archive, self.output)
                self.assertFalse(self.output.exists())
                self.archive.unlink()
                self.archive.with_name(self.archive.name + ".sha256").unlink()

    def test_output_and_checksum_never_replaced(self):
        self.pack()
        before = self.archive.read_bytes()
        with self.assertRaises(ValueError):
            self.pack()
        self.assertEqual(before, self.archive.read_bytes())
        self.output.mkdir()
        with self.assertRaises(ValueError):
            ASSETS.import_assets(self.archive, self.output)

    def test_strict_metadata_and_duplicate_aware_reader(self):
        self.spec["qualification"] = True
        with self.assertRaises(ValueError):
            self.pack()
        path = self.root / "duplicate.json"
        path.write_text('{"asset_id":"a","asset_id":"b"}')
        with self.assertRaises(ValueError):
            ASSETS.read_spec(path)

    def test_explicit_gpu_support_and_bound_required(self):
        self.spec["compatibility"]["gpu_compute_capabilities"] = []
        with self.assertRaises(ValueError):
            self.pack()
        self.spec["compatibility"]["gpu_compute_capabilities"] = ["12.0"]
        original = ASSETS.MAX_FILE_BYTES
        try:
            ASSETS.MAX_FILE_BYTES = 1
            with self.assertRaises(ValueError):
                self.pack()
        finally:
            ASSETS.MAX_FILE_BYTES = original

    def test_changed_input_is_not_published(self):
        original = tarfile.TarFile.addfile

        def change_input(archive, member, fileobj=None):
            if member.name == "manifest.json":
                (self.data /
                 "bundle/model.engine").write_bytes(b"changed engine")
            return original(archive, member, fileobj)

        with mock.patch.object(tarfile.TarFile, "addfile", change_input):
            with self.assertRaises(ValueError):
                self.pack()
        self.assertFalse(self.archive.exists())
        self.assertFalse(
            self.archive.with_name(self.archive.name + ".sha256").exists())

    def test_pax_and_link_headers_are_rejected(self):
        self.pack()
        with tarfile.open(self.archive, "r:gz") as source:
            entries = [
                (member, source.extractfile(member).read()) for member in source
            ]
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w",
                          format=tarfile.PAX_FORMAT) as target:
            entries[0][0].pax_headers = {"comment": "forbidden"}
            for member, data in entries:
                target.addfile(member, io.BytesIO(data))
        self.archive.write_bytes(gzip.compress(buffer.getvalue()))
        self.checksum()
        with self.assertRaises(ValueError):
            ASSETS.verify(self.archive)
        entries[0][0].pax_headers = {}
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE,
                     tarfile.FIFOTYPE, tarfile.CHRTYPE):
            with self.subTest(kind=kind):
                entries[-1][0].type = kind
                entries[-1][0].size = 0
                entries[-1][0].linkname = "../outside"
                buffer = io.BytesIO()
                with tarfile.open(fileobj=buffer,
                                  mode="w",
                                  format=tarfile.USTAR_FORMAT) as target:
                    for member, data in entries:
                        target.addfile(member, io.BytesIO(data))
                self.archive.write_bytes(gzip.compress(buffer.getvalue()))
                self.checksum()
                with self.assertRaises(ValueError):
                    ASSETS.import_assets(self.archive, self.output)
                self.assertFalse(self.output.exists())

    def test_trailing_payload_is_rejected(self):
        self.pack()
        self.archive.write_bytes(
            gzip.compress(
                gzip.decompress(self.archive.read_bytes()) + b"hidden"))
        self.checksum()
        with self.assertRaises(ValueError):
            ASSETS.verify(self.archive)

    def test_ustar_prefix_and_deterministic_pack(self):
        long_name = "bundle/" + "x" * 80 + "/" + "y" * 70 + ".bin"
        path = self.data / long_name
        path.parent.mkdir()
        path.write_bytes(b"data")
        self.spec["files"].append({"path": long_name, "kind": "constant"})
        self.pack()
        other = self.root / "other.tar.gz"
        ASSETS.pack(self.data, self.spec, other)
        self.assertEqual(self.archive.read_bytes(), other.read_bytes())
        self.assertEqual("fixture-data", ASSETS.verify(other)["asset_id"])

    def test_unsupported_metadata_types_are_rejected(self):
        for field, value in (("framework_abi_version",
                              True), ("schema_version", True), ("examples", []),
                             ("provider_commit", "HEAD")):
            with self.subTest(field=field):
                spec = copy.deepcopy(self.spec)
                spec[field] = value
                with self.assertRaises(ValueError):
                    ASSETS.pack(self.data, spec, self.archive)


if __name__ == "__main__":
    unittest.main()
