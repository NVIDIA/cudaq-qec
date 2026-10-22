# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Execute release-asset selection scripts without GitHub, CUDA or networking.

Only the named step's literal Bash block is extracted. These tests deliberately
do not implement a YAML parser or simulate archive validation; the importer and
CMake suites exercise those contracts separately.
"""

import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

REPOSITORY = Path(__file__).resolve().parents[3]
WORKFLOWS = REPOSITORY / ".github/workflows"
PRODUCER_COMMIT = "0123456789abcdef" * 2 + "01234567"


def named_step(filename, name):
    """Read one conventionally indented workflow step, not arbitrary YAML."""
    lines = (WORKFLOWS / filename).read_text(encoding="utf-8").splitlines()
    marker = f"      - name: {name}"
    start = lines.index(marker)
    end = start + 1
    while end < len(lines):
        if lines[end].startswith("      - "):
            break
        end += 1
    return lines[start:end]


def bash_block(filename, name):
    lines = named_step(filename, name)
    start = lines.index("        run: |") + 1
    result = []
    for line in lines[start:]:
        if not line.strip():
            result.append("")
        elif line.startswith("          "):
            result.append(line[10:])
        else:
            break
    if not result:
        raise AssertionError(f"No literal Bash block in {filename}: {name}")
    return "\n".join(result) + "\n"


class ExternalPluginWorkflowTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="qec-plugin-workflow-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment_file = self.root / "github-env"
        self.summary = self.root / "github-summary"

    def environment(self, **overrides):
        environment = dict(os.environ)
        for name in ("BASH_ENV", "QEC_EXTERNAL_DECODERS",
                     "QEC_EXTERNAL_PLUGIN_ARCHIVE",
                     "QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT",
                     "QEC_EXTERNAL_PLUGIN_DELIVERY",
                     "QEC_REQUIRED_EXTERNAL_PLUGINS"):
            environment.pop(name, None)
        environment.update({
            "GITHUB_ENV": str(self.environment_file),
            "GITHUB_STEP_SUMMARY": str(self.summary),
            "PLUGIN_FRAMEWORK_COMMIT": PRODUCER_COMMIT,
            "REQUIRED_PLUGIN_IDS": "",
            "PLUGIN_PLATFORM": "amd64",
            "PLUGIN_CUDA_VERSION": "13.0",
            "PLUGIN_PYTHON_VERSION": "3.11"
        })
        environment.update(overrides)
        return environment

    def execute(self, script, environment):
        self.environment_file.write_text("", encoding="utf-8")
        return subprocess.run(
            ["bash", "--noprofile", "--norc", "-eu", "-c", script],
            cwd=self.root,
            env=environment,
            capture_output=True,
            text=True,
            check=False)

    def select(self, delivery, **overrides):
        filename = "build_wheels.yaml" if delivery == "wheel" else "all_libs_release.yaml"
        script = bash_block(
            filename, f"Select manifest-based {delivery} plugin delivery")
        environment = self.environment(**overrides)
        return self.execute(script, environment)

    def selected_environment(self):
        return dict(
            line.split("=", 1)
            for line in self.environment_file.read_text().splitlines())

    def artifact_name(self,
                      delivery,
                      platform="amd64",
                      cuda="13.0",
                      python="3.11"):
        suffix = f"_py{python}" if delivery == "wheel" else ""
        return f"qec-private-plugins-{platform}_ubuntu24.04_cuda{cuda}_release{suffix}.tar.gz"

    def test_optional_absence_leaves_environment_unchanged(self):
        for delivery in ("native", "wheel"):
            with self.subTest(delivery=delivery):
                result = self.select(delivery)
                self.assertEqual(0, result.returncode, result.stderr)
                self.assertEqual({}, self.selected_environment())

    def test_reused_wheels_cannot_silently_ignore_plugin_requirements(self):
        script = bash_block(
            "build_wheels.yaml",
            "Reject unchecked plugin requirements on reused wheels")
        for reused, commit, providers, expected in (("", PRODUCER_COMMIT,
                                                     "fixture", 0), ("1234", "",
                                                                     "", 0),
                                                    ("1234", PRODUCER_COMMIT,
                                                     "", 1), ("1234", "",
                                                              "fixture", 1)):
            with self.subTest(reused=reused, commit=commit,
                              providers=providers):
                result = self.execute(
                    script,
                    self.environment(REUSED_RUN=reused,
                                     PLUGIN_FRAMEWORK_COMMIT=commit,
                                     REQUIRED_PLUGIN_IDS=providers))
                self.assertEqual(expected, result.returncode, result.stderr)

    def test_missing_required_archive_fails(self):
        for delivery in ("native", "wheel"):
            with self.subTest(delivery=delivery):
                result = self.select(delivery, REQUIRED_PLUGIN_IDS="fixture")
                self.assertNotEqual(0, result.returncode)
                self.assertIn("Required external plugin archive is absent",
                              result.stderr)
                self.assertEqual({}, self.selected_environment())

    def test_native_and_wheel_release_matrix_select_matching_names(self):
        for delivery in ("native", "wheel"):
            for platform in ("amd64", "arm64"):
                for cuda in ("12.6", "13.0"):
                    pythons = ("3.11", "3.12",
                               "3.13") if delivery == "wheel" else ("3.11",)
                    for python in pythons:
                        with self.subTest(delivery=delivery,
                                          platform=platform,
                                          cuda=cuda,
                                          python=python):
                            artifact = self.root / self.artifact_name(
                                delivery, platform, cuda, python)
                            # Selection uses names only; archive verification is
                            # intentionally covered by the separate import suite.
                            artifact.write_bytes(b"selection fixture")
                            result = self.select(
                                delivery,
                                PLUGIN_PLATFORM=platform,
                                PLUGIN_CUDA_VERSION=cuda,
                                PLUGIN_PYTHON_VERSION=python,
                                REQUIRED_PLUGIN_IDS=
                                "decoder_fixture;graph_fixture")
                            self.assertEqual(0, result.returncode,
                                             result.stderr)
                            self.assertEqual(
                                {
                                    "QEC_EXTERNAL_PLUGIN_ARCHIVE":
                                        str(artifact),
                                    "QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT":
                                        PRODUCER_COMMIT,
                                    "QEC_EXTERNAL_PLUGIN_DELIVERY":
                                        delivery,
                                    "QEC_REQUIRED_EXTERNAL_PLUGINS":
                                        "decoder_fixture;graph_fixture"
                                }, self.selected_environment())
                            artifact.unlink()

    def test_invalid_commit_and_provider_inputs_cannot_inject_environment(self):
        for delivery in ("native", "wheel"):
            artifact = self.root / self.artifact_name(delivery)
            artifact.write_bytes(b"selection fixture")
            for value in ("HEAD", "a" * 39, "A" * 40,
                          PRODUCER_COMMIT + "\nINJECTED=yes",
                          "$(touch injected)"):
                with self.subTest(delivery=delivery, commit=value):
                    result = self.select(delivery,
                                         PLUGIN_FRAMEWORK_COMMIT=value)
                    self.assertNotEqual(0, result.returncode)
                    self.assertEqual({}, self.selected_environment())
            for value in ("fixture;;other", "fixture\nINJECTED=yes",
                          "fixture;$(touch injected)", "fixture other",
                          "fixture;"):
                with self.subTest(delivery=delivery, providers=value):
                    result = self.select(delivery, REQUIRED_PLUGIN_IDS=value)
                    self.assertNotEqual(0, result.returncode)
                    self.assertEqual({}, self.selected_environment())
            artifact.unlink()
        self.assertFalse((self.root / "injected").exists())

    def test_wheel_asset_transfer_includes_checksum_sidecars(self):
        step = named_step("build_wheels.yaml", "Upload release assets")
        path = step.index("          path: |")
        patterns = []
        for line in step[path + 1:]:
            if not line.startswith("            "):
                break
            patterns.append(line.strip())
        self.assertIn("*.tar.gz", patterns)
        self.assertIn("*.tar.gz.sha256", patterns)

    def run_legacy_asset_selection(self, delivery):
        replacements = {
            "${{ inputs.assets_repo }}": "fixture/repository",
            "${{ inputs.assets_tag }}": "fixture-tag",
            "${{ matrix.runner.arch }}": "amd64",
            "${{ matrix.platform }}": "amd64",
            "${{ matrix.cuda_version }}": "13.0",
            "${{ matrix.python }}": "3.11"
        }
        if delivery == "native":
            script = bash_block("all_libs_release.yaml",
                                "Fetch assets and set QEC_EXTERNAL_DECODERS")
            installer = self.root / ".github/workflows/scripts/install_git_cli.sh"
            installer.parent.mkdir(parents=True, exist_ok=True)
            installer.write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
            # This shell function shadows gh only inside the test subprocess;
            # downloads are supplied by local fixtures, never by networking.
            script = "gh() { return 0; }\n" + script
        else:
            script = bash_block("build_wheels.yaml", "Build CUDA-QX wheels")
            script = script[:script.index("# This is needed")]
            script += "printf '\\nLEGACY=%s\\n' \"${QEC_EXTERNAL_DECODERS-}\"\n"
        for source, value in replacements.items():
            script = script.replace(source, value)
        self.assertNotIn("${{", script)
        return self.execute(script, self.environment())

    def test_manifest_only_release_does_not_require_legacy_decoder_archive(
            self):
        for delivery in ("native", "wheel"):
            with self.subTest(delivery=delivery):
                artifact = self.root / self.artifact_name(delivery)
                artifact.write_bytes(b"selection fixture")
                result = self.run_legacy_asset_selection(delivery)
                self.assertEqual(0, result.returncode,
                                 result.stdout + result.stderr)
                self.assertNotIn("QEC_EXTERNAL_DECODERS",
                                 self.selected_environment())
                if delivery == "wheel":
                    self.assertIn("LEGACY=\n", result.stdout)
                artifact.unlink()

    def test_requested_release_without_any_supported_archive_fails(self):
        for delivery in ("native", "wheel"):
            with self.subTest(delivery=delivery):
                result = self.run_legacy_asset_selection(delivery)
                self.assertNotEqual(0, result.returncode)
                self.assertIn(f"No supported {delivery} plugin delivery",
                              result.stderr)

    def test_legacy_delivery_remains_supported_without_manifest_archive(self):
        for delivery in ("native", "wheel"):
            with self.subTest(delivery=delivery):
                suffix = "_py3.11" if delivery == "wheel" else ""
                artifact = self.root / f"nv-qldpc-decoder-amd64_ubuntu24.04_cuda13.0{suffix}_release.tar.gz"
                with tarfile.open(artifact, "w:gz") as archive:
                    content = b"legacy selection fixture"
                    member = tarfile.TarInfo("libcudaq-qec-nv-qldpc-decoder.so")
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))
                result = self.run_legacy_asset_selection(delivery)
                self.assertEqual(0, result.returncode,
                                 result.stdout + result.stderr)
                expected = str(self.root /
                               "tmp/libcudaq-qec-nv-qldpc-decoder.so")
                if delivery == "native":
                    self.assertEqual(
                        expected,
                        self.selected_environment()["QEC_EXTERNAL_DECODERS"])
                else:
                    self.assertIn(f"LEGACY={expected}\n", result.stdout)
                artifact.unlink()


if __name__ == "__main__":
    unittest.main()
