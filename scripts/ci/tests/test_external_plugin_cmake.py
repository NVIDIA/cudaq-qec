# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Standalone external-plugin CMake contracts, not GPU qualification.

Tests compile a tiny ordinary shared object and configure the real public CMake
module. A unit-only FindCUDAToolkit fixture supplies version metadata; it does
not supply a CUDA compiler, GPU runtime, or evidence of CUDA compatibility.
The recording patcher exercises staging/arguments; a separate conditional test
uses actual patchelf when it is installed.
"""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[3]
CMAKE_MODULE = REPOSITORY / "libs/qec/cmake/ExternalPlugins.cmake"
PRODUCER_COMMIT = "abcdef0123456789" * 2 + "abcdef01"
PATCHELF_MARKER = b"\nrecording-patchelf\n"


class ExternalPluginCMakeTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cmake = shutil.which("cmake")
        compiler = shutil.which("g++") or shutil.which("clang++")
        cls.architecture = {
            "x86_64": "x86_64",
            "amd64": "x86_64",
            "aarch64": "aarch64",
            "arm64": "aarch64"
        }.get(platform.machine().lower())
        if (platform.system() != "Linux" or not cls.cmake or not compiler or
                not cls.architecture):
            raise unittest.SkipTest(
                "Host CMake, C++ compiler and Linux ELF architecture required")
        version = subprocess.run([cls.cmake, "--version"],
                                 check=True,
                                 capture_output=True,
                                 text=True).stdout.splitlines()[0].split()[-1]
        if tuple(int(part) for part in version.split(".")[:2]) < (3, 28):
            raise unittest.SkipTest("CMake 3.28 or newer required")
        cls.compilation = tempfile.TemporaryDirectory(prefix="qec-plugin-elf-")
        cls.addClassCleanup(cls.compilation.cleanup)
        module = Path(cls.compilation.name) / "libfixture.so"
        result = subprocess.run(
            [compiler, "-shared", "-fPIC", "-x", "c++", "-", "-o",
             str(module)],
            input='extern "C" int qec_fixture() { return 7; }\n',
            capture_output=True,
            text=True,
            check=False)
        if result.returncode:
            raise RuntimeError(
                f"Cannot compile CMake test fixture: {result.stderr}")
        cls.elf_bytes = module.read_bytes()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qec-plugin-cmake-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.build = self.root / "build"
        self.install = self.root / "install"
        self.tools = self.root / "tools"
        self.tools.mkdir()
        self.patcher = self.tools / "patchelf"
        self.log = self.root / "patchelf.jsonl"
        self.patcher.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "arguments = sys.argv[1:]\n"
            "if len(arguments) != 3 or arguments[0] != '--set-rpath':\n"
            "    raise SystemExit('unexpected patcher arguments')\n"
            "target = pathlib.Path(arguments[2])\n"
            "if not target.stat().st_mode & 0o200:\n"
            "    raise SystemExit('staged plugin is not owner-writable')\n"
            "with open(os.environ['QEC_RECORDING_PATCHELF_LOG'], 'a') as stream:\n"
            "    stream.write(json.dumps(arguments) + '\\n')\n"
            f"with target.open('ab') as stream: stream.write({PATCHELF_MARKER!r})\n",
            encoding="utf-8")
        self.patcher.chmod(0o755)
        self.environment = dict(os.environ,
                                QEC_RECORDING_PATCHELF_LOG=str(self.log))
        # Isolate configuration from a developer's optional external delivery.
        for name in ("QEC_EXTERNAL_DECODERS", "QEC_EXTERNAL_PLUGIN_ARCHIVE",
                     "QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT",
                     "QEC_EXTERNAL_PLUGIN_DELIVERY",
                     "QEC_REQUIRED_EXTERNAL_PLUGINS"):
            self.environment.pop(name, None)
        self.modules = self.source / "cmake"
        self.modules.mkdir()
        (self.modules / "FindCUDAToolkit.cmake").write_text(
            "# Unit fixture only: metadata, not a CUDA toolchain.\n"
            "if(FIXTURE_TOOLKIT_UNAVAILABLE)\n"
            "  message(FATAL_ERROR \"Fixture toolkit unavailable\")\n"
            "endif()\n"
            "set(CUDAToolkit_FOUND TRUE)\n"
            "if(NOT DEFINED FIXTURE_CUDA_VERSION)\n"
            "  set(FIXTURE_CUDA_VERSION 13.0.88)\n"
            "endif()\n"
            "set(CUDAToolkit_VERSION \"${FIXTURE_CUDA_VERSION}\")\n",
            encoding="utf-8")
        (self.source / "CMakeLists.txt").write_text(
            "cmake_minimum_required(VERSION 3.28)\n"
            "project(ExternalPluginContract LANGUAGES NONE)\n"
            "set(CMAKE_INSTALL_LIBDIR lib)\n"
            "list(PREPEND CMAKE_MODULE_PATH \"${CMAKE_CURRENT_SOURCE_DIR}/cmake\")\n"
            f"include(\"{CMAKE_MODULE.as_posix()}\")\n",
            encoding="utf-8")
        self.legacy = self.root / "libdecoder_fixture.so"
        self.legacy.write_bytes(self.elf_bytes)
        self.archive = self.root / "plugins.tar.gz"

    def configure(self, *options, patcher=None):
        return subprocess.run([
            self.cmake, "-S",
            str(self.source), "-B",
            str(self.build), f"-DCMAKE_INSTALL_PREFIX={self.install}",
            f"-DQEC_PATCHELF_EXECUTABLE={patcher or self.patcher}", *options
        ],
                              env=self.environment,
                              capture_output=True,
                              text=True,
                              check=False)

    def assert_configures(self, *options, patcher=None):
        result = self.configure(*options, patcher=patcher)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        return result

    def assert_configuration_fails(self, expected, *options):
        result = self.configure(*options)
        self.assertNotEqual(0, result.returncode, result.stdout + result.stderr)
        self.assertIn(" ".join(expected.split()), " ".join(
            (result.stdout + result.stderr).split()))
        return result

    def records(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def write_archive(self, delivery="native"):
        paths = (("decoder_fixture", "decoder",
                  "decoder-plugins/libdecoder_fixture.so"),
                 ("graph_fixture", "task_graph",
                  "task-graph-plugins/libgraph_fixture.so"))
        modules = []
        for identity, kind, path in paths:
            modules.append({
                "schema_version": 1,
                "id": identity,
                "kind": kind,
                "path": path,
                "framework_commit": PRODUCER_COMMIT,
                "source_dirty": False,
                "build_type": "Release",
                "architecture": self.architecture,
                "compiler": {
                    "id": "GNU",
                    "version": "fixture"
                },
                "cxx_standard": 17,
                "cuda_version": "13.0.88",
                "tensorrt_version": None,
                "glibcxx_use_cxx11_abi": 1,
                "sha256": hashlib.sha256(self.elf_bytes).hexdigest(),
                "size_bytes": len(self.elf_bytes),
                "needed": [],
                "rpath": ["$ORIGIN", "$ORIGIN/../nvidia/tensorrt_libs"],
                "build_id": None,
                "required_symbol_versions": []
            })
        self.manifest = {
            "schema_version": 1,
            "delivery": delivery,
            "modules": modules
        }
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w",
                          format=tarfile.USTAR_FORMAT) as archive:
            entries = [("manifest.json", json.dumps(self.manifest).encode())]
            entries.extend(
                (module["path"], self.elf_bytes) for module in modules)
            for name, payload in entries:
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o644
                archive.addfile(member, io.BytesIO(payload))
        self.archive.write_bytes(gzip.compress(buffer.getvalue()))
        digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.archive.with_name(self.archive.name + ".sha256").write_text(
            f"{digest}  {self.archive.name}\n", encoding="ascii")
        self.stage = self.build / "external-plugins" / digest
        return (f"-DQEC_EXTERNAL_PLUGIN_ARCHIVE={self.archive}",
                f"-DQEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT={PRODUCER_COMMIT}")

    def install_plugins(self):
        result = subprocess.run([
            self.cmake, "--install",
            str(self.build), "--component", "qec-lib-plugins"
        ],
                                env=self.environment,
                                capture_output=True,
                                text=True,
                                check=False)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def test_legacy_decoder_copies_and_installs_without_modifying_source(self):
        original = self.legacy.read_bytes()
        self.legacy.chmod(0o444)
        self.assert_configures(f"-DQEC_EXTERNAL_DECODERS={self.legacy}")
        destination = self.build / "lib/decoder-plugins" / self.legacy.name
        self.assertEqual(original, self.legacy.read_bytes())
        self.assertEqual(0, self.legacy.stat().st_mode & 0o222)
        self.assertEqual(original + PATCHELF_MARKER, destination.read_bytes())
        self.assertEqual([["--set-rpath", "$ORIGIN/..",
                           str(destination)]], self.records())
        self.install_plugins()
        self.assertEqual(destination.read_bytes(),
                         (self.install / "lib/decoder-plugins" /
                          self.legacy.name).read_bytes())

    def test_manifest_routes_both_kinds_and_preserves_rpath_list(self):
        options = self.write_archive()
        original = self.archive.read_bytes()
        self.assert_configures(
            *options,
            "-DQEC_REQUIRED_EXTERNAL_PLUGINS=decoder_fixture;graph_fixture")
        expected_rpath = "$ORIGIN/..:$ORIGIN:$ORIGIN/../nvidia/tensorrt_libs"
        self.assertEqual(2, len(self.records()))
        for arguments, module in zip(self.records(), self.manifest["modules"]):
            destination = self.build / "lib" / module["path"]
            self.assertEqual(["--set-rpath", expected_rpath,
                              str(destination)], arguments)
            self.assertEqual(self.elf_bytes + PATCHELF_MARKER,
                             destination.read_bytes())
            self.assertEqual(self.elf_bytes,
                             (self.stage / module["path"]).read_bytes())
        self.assertEqual(original, self.archive.read_bytes())
        self.install_plugins()
        for module in self.manifest["modules"]:
            self.assertEqual(self.elf_bytes + PATCHELF_MARKER,
                             (self.install / "lib" /
                              module["path"]).read_bytes())
        installed_manifest = self.install / "lib/external-plugins" / self.stage.name / "source-manifest.json"
        self.assertEqual(self.manifest,
                         json.loads(installed_manifest.read_text()))

    def test_wheel_delivery_accepts_only_wheel_archive(self):
        options = self.write_archive(delivery="wheel")
        self.assert_configuration_fails("Plugin delivery mismatch", *options)
        self.assert_configures(*options, "-DSKBUILD=ON",
                               "-DQEC_EXTERNAL_PLUGIN_DELIVERY=wheel")

    def test_build_delivery_mismatch_fails_before_import(self):
        options = self.write_archive()
        self.assert_configuration_fails(
            "wheel build cannot import a native", *options, "-DSKBUILD=ON",
            "-DQEC_EXTERNAL_PLUGIN_DELIVERY=native")
        self.assert_configuration_fails("native build cannot import a wheel",
                                        *options, "-DSKBUILD=OFF",
                                        "-DQEC_EXTERNAL_PLUGIN_DELIVERY=wheel")

    def test_required_provider_missing_and_required_without_archive(self):
        self.assert_configuration_fails(
            "without an archive", "-DQEC_REQUIRED_EXTERNAL_PLUGINS=missing")
        options = self.write_archive()
        self.assert_configuration_fails(
            "Required plugins are missing", *options,
            "-DQEC_REQUIRED_EXTERNAL_PLUGINS=missing")

    def test_conflicting_legacy_and_manifest_destination(self):
        options = self.write_archive()
        self.legacy.write_bytes(self.elf_bytes + b"different build")
        self.assert_configuration_fails(
            "Conflicting external plugin destination", *options,
            f"-DQEC_EXTERNAL_DECODERS={self.legacy}")
        self.assertEqual(self.elf_bytes + b"different build",
                         self.legacy.read_bytes())

    def test_identical_legacy_and_manifest_module_is_installed_once(self):
        options = self.write_archive()
        self.assert_configures(*options,
                               f"-DQEC_EXTERNAL_DECODERS={self.legacy}")
        self.assertEqual(3, len(self.records()))
        for rpath in self.manifest["modules"][0]["rpath"]:
            self.assertIn(rpath, self.records()[1][1].split(":"))
        self.assertEqual(self.elf_bytes, self.legacy.read_bytes())
        result = subprocess.run([
            self.cmake, "--install",
            str(self.build), "--component", "qec-lib-plugins"
        ],
                                capture_output=True,
                                text=True,
                                check=False)
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)
        entries = (
            self.build /
            "install_manifest_qec-lib-plugins.txt").read_text().splitlines()
        self.assertEqual(
            1, sum(path.endswith("/libdecoder_fixture.so") for path in entries))

    def test_reconfiguration_revalidates_stage_without_changing_archive(self):
        options = self.write_archive()
        self.assert_configures(*options)
        original_archive = self.archive.read_bytes()
        original_manifest = (self.stage / "manifest.json").read_bytes()
        original_modules = {
            module["path"]: (self.stage / module["path"]).read_bytes()
            for module in self.manifest["modules"]
        }
        self.assert_configures(*options)
        self.assertEqual(4, len(self.records()))
        self.assertEqual(original_archive, self.archive.read_bytes())
        self.assertEqual(original_manifest,
                         (self.stage / "manifest.json").read_bytes())
        for path, data in original_modules.items():
            self.assertEqual(data, (self.stage / path).read_bytes())

    def test_reconfiguration_rejects_tampered_stage_without_repair(self):
        options = self.write_archive()
        self.assert_configures(*options)
        tampered = self.stage / self.manifest["modules"][0]["path"]
        tampered.chmod(0o644)
        tampered.write_bytes(b"tampered")
        self.assert_configuration_fails("Existing plugin differs", *options)
        self.assertEqual(b"tampered", tampered.read_bytes())
        self.assertEqual(2, len(self.records()))

    def test_archive_requires_reviewed_producer_commit(self):
        self.write_archive()
        self.assert_configuration_fails(
            "requires QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT",
            f"-DQEC_EXTERNAL_PLUGIN_ARCHIVE={self.archive}")

    def test_toolkit_mismatch_rejected(self):
        options = self.write_archive()
        self.assert_configuration_fails("Plugin CUDA toolkit mismatch",
                                        *options, "-DFIXTURE_CUDA_VERSION=12.9")

    def test_missing_legacy_library_rejected(self):
        self.assert_configuration_fails(
            "Invalid file path to external plugin",
            f"-DQEC_EXTERNAL_DECODERS={self.root / 'missing.so'}")

    def test_generic_build_does_not_require_plugin_dependencies(self):
        self.assert_configures("-DFIXTURE_TOOLKIT_UNAVAILABLE=ON")
        self.assertFalse(self.log.exists())
        self.assertFalse((self.build / "external-plugins").exists())

    def test_archive_requires_toolkit_metadata(self):
        options = self.write_archive()
        self.assert_configuration_fails("Fixture toolkit unavailable", *options,
                                        "-DFIXTURE_TOOLKIT_UNAVAILABLE=ON")

    def test_patcher_failure_is_reported(self):
        result = self.configure(f"-DQEC_EXTERNAL_DECODERS={self.legacy}",
                                patcher=self.root / "missing-patchelf")
        self.assertNotEqual(0, result.returncode)
        self.assertIn("Failed to patch staged external plugin", " ".join(
            (result.stdout + result.stderr).split()))
        self.assertEqual(self.elf_bytes, self.legacy.read_bytes())

    @unittest.skipUnless(shutil.which("patchelf"),
                         "Actual patchelf is not installed")
    def test_actual_patchelf_updates_copy_only(self):
        options = self.write_archive()
        self.assert_configures(*options, patcher=shutil.which("patchelf"))
        for module in self.manifest["modules"]:
            destination = self.build / "lib" / module["path"]
            rpath = subprocess.run(
                [shutil.which("patchelf"), "--print-rpath",
                 str(destination)],
                check=True,
                capture_output=True,
                text=True).stdout.strip()
            self.assertEqual(
                "$ORIGIN/..:$ORIGIN:$ORIGIN/../nvidia/tensorrt_libs", rpath)
            self.assertEqual(self.elf_bytes,
                             (self.stage / module["path"]).read_bytes())


if __name__ == "__main__":
    unittest.main()
