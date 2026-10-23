# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute image artifact selection and check the final-runtime smoke wiring."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from test_external_plugin_workflows import bash_block, named_step, REPOSITORY

PRODUCER_COMMIT = "0123456789abcdef" * 2 + "01234567"
WORKFLOW = "build_decoding_server_docker.yaml"


class ImagePluginDeliveryTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qec-image-plugin-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "artifacts").mkdir()
        self.archive = self.root / "qec-private-plugins-amd64_ubuntu24.04_cuda13.0_release.tar.gz"
        self.sidecar = self.archive.with_name(self.archive.name + ".sha256")
        self.envfile = self.root / "github-env"

    def run_selection(self, **overrides):
        environment = dict(os.environ)
        environment.pop("BASH_ENV", None)
        environment.update({
            "GITHUB_ENV": str(self.envfile),
            "PLUGIN_FRAMEWORK_COMMIT": "",
            "REQUIRED_PLUGIN_IDS": "",
            "REQUIRED_TASK_GRAPH_KINDS": "",
            "PLUGIN_PLATFORM": "amd64",
            "PLUGIN_CUDA_VERSION": "13.0"
        })
        environment.update(overrides)
        self.envfile.write_text("", encoding="utf-8")
        return subprocess.run([
            "bash", "--noprofile", "--norc", "-eu", "-c",
            bash_block(WORKFLOW, "Select manifest-based image plugin delivery")
        ],
                              cwd=self.root,
                              env=environment,
                              check=False,
                              capture_output=True,
                              text=True)

    def supply_archive(self):
        # The selection step only transfers bytes. Importer/CMake tests cover
        # checksums, ELF metadata, ABI and archive validation separately.
        self.archive.write_bytes(b"opaque native archive")
        self.sidecar.write_text("checksum fixture", encoding="utf-8")

    def test_generic_image_allows_optional_absence(self):
        result = self.run_selection()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("QEC_EXTERNAL_PLUGIN_ARCHIVE=",
                         self.envfile.read_text())

    def run_asset_selection(self, **overrides):
        environment = dict(os.environ)
        environment.pop("BASH_ENV", None)
        environment.update({
            "GITHUB_ENV": str(self.envfile),
            "DELIVERY_PROFILE": "generic",
            "RUNTIME_ASSET_ID": "",
            "PLUGIN_PLATFORM": "amd64",
            "PLUGIN_CUDA_VERSION": "13.0"
        })
        environment.update(overrides)
        self.envfile.write_text("")
        return subprocess.run([
            "bash", "--noprofile", "--norc", "-eu", "-c",
            bash_block(WORKFLOW, "Select runtime model and example delivery")
        ],
                              cwd=self.root,
                              env=environment,
                              check=False,
                              capture_output=True,
                              text=True)

    def test_ising_profile_requires_binary_provider_and_kinds(self):
        result = self.run_selection(DELIVERY_PROFILE="ising")
        self.assertNotEqual(0, result.returncode)
        self.supply_archive()
        result = self.run_selection(DELIVERY_PROFILE="ising",
                                    PLUGIN_FRAMEWORK_COMMIT=PRODUCER_COMMIT)
        self.assertEqual(0, result.returncode, result.stderr)
        values = self.envfile.read_text()
        self.assertIn(
            "QEC_REQUIRED_EXTERNAL_PLUGINS=cudaq-qec-ising-task-graph-plugin\n",
            values)
        self.assertIn("QEC_REQUIRED_TASK_GRAPH_KINDS=ising\n", values)

    def test_runtime_data_optional_for_generic_required_for_ising(self):
        self.assertEqual(0, self.run_asset_selection().returncode)
        self.assertNotEqual(
            0,
            self.run_asset_selection(DELIVERY_PROFILE="ising").returncode)
        self.assertNotEqual(
            0,
            self.run_asset_selection(RUNTIME_ASSET_ID="fixture").returncode)
        archive = self.root / "qec-runtime-assets-fixture-amd64_ubuntu24.04_cuda13.0_release.tar.gz"
        archive.write_bytes(b"data archive fixture")
        sidecar = archive.with_name(archive.name + ".sha256")
        sidecar.write_text("checksum fixture")
        result = self.run_asset_selection(DELIVERY_PROFILE="ising",
                                          RUNTIME_ASSET_ID="fixture")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(archive.read_bytes(),
                         (self.root / "artifacts" / archive.name).read_bytes())
        self.assertIn(f"QEC_RUNTIME_ASSET_ARCHIVE={archive.name}\n",
                      self.envfile.read_text())
        # The arm64 matrix entry cannot silently use an amd64/SM120 delivery.
        result = self.run_asset_selection(DELIVERY_PROFILE="ising",
                                          RUNTIME_ASSET_ID="fixture",
                                          PLUGIN_PLATFORM="arm64")
        self.assertNotEqual(0, result.returncode)
        sidecar.unlink()
        sidecar.symlink_to(archive)
        self.assertNotEqual(
            0,
            self.run_asset_selection(RUNTIME_ASSET_ID="fixture").returncode)

    def test_runtime_asset_name_cannot_inject_build_arguments(self):
        for value in ("../fixture", "fixture\nINJECTED=yes", "fixture;other",
                      "$(touch injected)"):
            with self.subTest(value=value):
                result = self.run_asset_selection(RUNTIME_ASSET_ID=value)
                self.assertNotEqual(0, result.returncode)
                self.assertEqual("", self.envfile.read_text())
        self.assertFalse((self.root / "injected").exists())

    def test_ising_profile_input_validation_runs_before_download(self):
        script = bash_block(WORKFLOW, "Validate image delivery profile")
        environment = dict(os.environ,
                           DELIVERY_PROFILE="ising",
                           RUNTIME_ASSET_ID="fixture",
                           CUDA_SELECTION="13.0",
                           ASSETS_TAG="docker-files-fixture")
        result = subprocess.run(["bash", "-eu", "-c", script],
                                env=environment,
                                check=False,
                                capture_output=True)
        self.assertEqual(0, result.returncode, result.stderr)
        for field, value in (("CUDA_SELECTION",
                              "all"), ("CUDA_SELECTION",
                                       "12.6"), ("RUNTIME_ASSET_ID", ""),
                             ("ASSETS_TAG", ""), ("DELIVERY_PROFILE", "other")):
            with self.subTest(field=field, value=value):
                result = subprocess.run(["bash", "-eu", "-c", script],
                                        env=dict(environment, **{field: value}),
                                        check=False,
                                        capture_output=True)
                self.assertNotEqual(0, result.returncode)

    def test_provider_kind_or_commit_requirement_needs_archive(self):
        for field, value in (("PLUGIN_FRAMEWORK_COMMIT", PRODUCER_COMMIT),
                             ("REQUIRED_PLUGIN_IDS", "fixture"),
                             ("REQUIRED_TASK_GRAPH_KINDS", "fixture_node")):
            with self.subTest(field=field):
                result = self.run_selection(**{field: value})
                self.assertNotEqual(0, result.returncode)
                self.assertIn("archive is absent", result.stderr)

    def test_native_archive_and_sidecar_pass_unchanged_to_artifact_context(
            self):
        self.supply_archive()
        result = self.run_selection(
            PLUGIN_FRAMEWORK_COMMIT=PRODUCER_COMMIT,
            REQUIRED_PLUGIN_IDS="fixture;another",
            REQUIRED_TASK_GRAPH_KINDS="fixture_node;other_node")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(self.archive.read_bytes(),
                         (self.root / "artifacts" /
                          self.archive.name).read_bytes())
        self.assertEqual(self.sidecar.read_bytes(),
                         (self.root / "artifacts" /
                          self.sidecar.name).read_bytes())
        self.assertIn(f"QEC_EXTERNAL_PLUGIN_ARCHIVE={self.archive.name}\n",
                      self.envfile.read_text())
        self.assertIn("QEC_REQUIRED_TASK_GRAPH_KINDS=fixture_node;other_node\n",
                      self.envfile.read_text())

    def test_required_lists_cannot_inject_build_arguments(self):
        self.supply_archive()
        for field in ("REQUIRED_PLUGIN_IDS", "REQUIRED_TASK_GRAPH_KINDS"):
            for value in ("fixture;;other", "fixture\nINJECTED=yes",
                          "fixture;$(touch injected)", "fixture other"):
                with self.subTest(field=field, value=value):
                    result = self.run_selection(
                        PLUGIN_FRAMEWORK_COMMIT=PRODUCER_COMMIT,
                        **{field: value})
                    self.assertNotEqual(0, result.returncode)
                    self.assertEqual("", self.envfile.read_text())
        self.assertFalse((self.root / "injected").exists())

    def test_archive_requires_reviewed_commit_and_regular_sidecar(self):
        self.supply_archive()
        for value in ("", "HEAD", "A" * 40, "a" * 39,
                      PRODUCER_COMMIT + "\nINJECTED=yes"):
            with self.subTest(commit=value):
                result = self.run_selection(PLUGIN_FRAMEWORK_COMMIT=value)
                self.assertNotEqual(0, result.returncode)
        self.sidecar.unlink()
        self.assertNotEqual(
            0,
            self.run_selection(
                PLUGIN_FRAMEWORK_COMMIT=PRODUCER_COMMIT).returncode)
        self.sidecar.symlink_to(self.archive)
        self.assertNotEqual(
            0,
            self.run_selection(
                PLUGIN_FRAMEWORK_COMMIT=PRODUCER_COMMIT).returncode)

    def test_image_build_passes_all_delivery_requirements(self):
        step = "\n".join(named_step(WORKFLOW, "Build and push image by digest"))
        self.assertIn("file: docker/release/Dockerfile.decoding-server", step)
        self.assertIn("target: runtime", step)
        for name in ("QEC_EXTERNAL_PLUGIN_ARCHIVE",
                     "QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT",
                     "QEC_REQUIRED_EXTERNAL_PLUGINS",
                     "QEC_REQUIRED_TASK_GRAPH_KINDS", "QEC_DELIVERY_PROFILE",
                     "QEC_RUNTIME_ASSET_ID", "QEC_RUNTIME_ASSET_ARCHIVE"):
            self.assertIn(f"{name}=${{{{ env.{name} }}}}", step)

    def test_cuda_selection_is_shared_by_build_publish_and_promotion(self):
        workflow = (REPOSITORY / ".github/workflows" / WORKFLOW).read_text()
        matrices = [
            line.strip()
            for line in workflow.splitlines()
            if line.startswith("        cuda_version:")
        ]
        self.assertEqual(3, len(matrices))
        self.assertEqual(1, len(set(matrices)))
        self.assertIn("inputs.cuda_version == '13.0'", matrices[0])
        self.assertIn("options: [all, '12.6', '13.0']", workflow)

    def image_stages(self):
        dockerfile = (REPOSITORY /
                      "docker/release/Dockerfile.decoding-server").read_text()
        base_start = dockerfile.index("FROM nvidia/cuda:")
        validation_start = dockerfile.index("FROM runtime-base AS validation")
        runtime_start = dockerfile.index("FROM runtime-base AS runtime\n")
        return (dockerfile[:base_start],
                dockerfile[base_start:validation_start],
                dockerfile[validation_start:runtime_start],
                dockerfile[runtime_start:])

    def test_customer_filesystem_has_no_validation_tools_or_reports(self):
        builder, base, _, runtime = self.image_stages()
        self.assertIn("COPY --from=builder /opt/nvidia/cudaqx", base)
        self.assertIn("-o /tmp/qec-plugin-registration-probe", builder)
        self.assertNotIn("bin/qec-plugin-registration-probe", builder)
        for value in ("python3-minimal", "patchelf", "installed_plugin_smoke",
                      "fixtures/", "qualification/", "COPY --from=validation"):
            with self.subTest(value=value):
                self.assertNotIn(value, base)
                self.assertNotIn(value, runtime)
        self.assertNotIn("/private/ising", base)
        self.assertIn('ENTRYPOINT ["bash", "-l"]', base)

    def test_validation_runs_against_installed_customer_artifacts(self):
        builder, _, validation, _ = self.image_stages()
        self.assertIn("QEC_EXTERNAL_PLUGIN_DELIVERY=native", builder)
        self.assertIn("python3-minimal patchelf", validation)
        self.assertIn("COPY --from=builder /tmp/qec-plugin-registration-probe",
                      validation)
        self.assertIn("CUDA_VISIBLE_DEVICES=\"\" python3", validation)
        self.assertIn("--prefix=/opt/nvidia/cudaqx", validation)
        self.assertIn("--require-provider=$provider", validation)
        self.assertIn("--require-node-kind=$kind", validation)
        self.assertIn("--output=/tmp/qec-plugin-validation/results", validation)
        self.assertIn("touch /tmp/qec-plugin-validation/passed", validation)
        self.assertNotIn("/private/ising", validation)

    def test_data_import_and_final_runtime_compatibility_are_both_checked(self):
        builder, base, validation, _ = self.image_stages()
        self.assertIn('ARG QEC_DELIVERY_PROFILE=generic', builder)
        self.assertIn('test "${CUDA_VERSION%%.*}" = 13', builder)
        self.assertIn('Ising delivery requires its binary provider', builder)
        self.assertIn('Ising delivery requires kind $kind', builder)
        self.assertIn('python3 scripts/ci/runtime_assets.py import', builder)
        self.assertIn(
            '--provider-commit="$QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT"',
            builder)
        self.assertIn('--require-asset="$QEC_RUNTIME_ASSET_ID"', builder)
        self.assertIn('dpkg-query -W', builder)
        self.assertIn('runtime_assets.py verify-installed', validation)
        self.assertIn('dpkg-query -W', validation)
        self.assertIn('share/cudaq-qec/runtime-assets/$QEC_RUNTIME_ASSET_ID',
                      validation)
        self.assertIn('run-ising-example.sh', base)
        self.assertNotIn('runtime_assets.py', base)

    def test_direct_docker_profile_fails_before_build_without_requirements(
            self):
        builder, _, _, _ = self.image_stages()
        start = builder.index("# Import data separately")
        start = builder.index("RUN --mount=", start)
        end = builder.index("\n\n", start)
        # Execute the literal preflight RUN body. Negative cases stop before
        # consulting archives or building anything; no container is simulated.
        script = "\n".join(builder[start:end].splitlines()[1:])
        script = script.replace("\\\n", "\n")
        environment = dict(os.environ,
                           CUDA_VERSION="13.0",
                           QEC_DELIVERY_PROFILE="generic",
                           QEC_EXTERNAL_PLUGIN_ARCHIVE="",
                           QEC_EXTERNAL_PLUGIN_FRAMEWORK_COMMIT="",
                           QEC_REQUIRED_EXTERNAL_PLUGINS="",
                           QEC_REQUIRED_TASK_GRAPH_KINDS="",
                           QEC_RUNTIME_ASSET_ARCHIVE="",
                           QEC_RUNTIME_ASSET_ID="")
        result = subprocess.run(["sh", "-c", script],
                                env=environment,
                                capture_output=True,
                                check=False)
        self.assertEqual(0, result.returncode, result.stderr)
        for changes in ({
                "QEC_DELIVERY_PROFILE": "ising"
        }, {
                "QEC_DELIVERY_PROFILE": "unknown"
        }, {
                "QEC_RUNTIME_ASSET_ID": "fixture"
        }, {
                "QEC_DELIVERY_PROFILE": "ising",
                "CUDA_VERSION": "12.6"
        }):
            with self.subTest(changes=changes):
                result = subprocess.run(["sh", "-c", script],
                                        env=dict(environment, **changes),
                                        capture_output=True,
                                        check=False)
                self.assertNotEqual(0, result.returncode)

        # Valid required lists must not mask absent archives through POSIX
        # `set -e` exceptions for intermediate AND-list commands.
        required = dict(
            environment,
            QEC_DELIVERY_PROFILE="ising",
            QEC_REQUIRED_EXTERNAL_PLUGINS="cudaq-qec-ising-task-graph-plugin",
            QEC_REQUIRED_TASK_GRAPH_KINDS="ising",
            QEC_EXTERNAL_PLUGIN_ARCHIVE="plugin.tar.gz",
            QEC_RUNTIME_ASSET_ARCHIVE="data.tar.gz",
            QEC_RUNTIME_ASSET_ID="fixture")
        for missing in ("QEC_EXTERNAL_PLUGIN_ARCHIVE",
                        "QEC_RUNTIME_ASSET_ARCHIVE", "QEC_RUNTIME_ASSET_ID"):
            with self.subTest(missing=missing):
                result = subprocess.run(["sh", "-c", script],
                                        env=dict(required, **{missing: ""}),
                                        capture_output=True,
                                        check=False)
                self.assertNotEqual(0, result.returncode)
                self.assertIn(b"requires binary, model and example artifacts",
                              result.stderr)

    def test_packaging_contracts_run_in_cpu_pr_ci(self):
        workflow = (REPOSITORY /
                    ".github/workflows/pr_sanity_checks.yaml").read_text()
        self.assertIn("check-packaging-contracts:", workflow)
        self.assertIn("binutils patchelf", workflow)
        self.assertIn("python3 -m unittest discover -s scripts/ci/tests",
                      workflow)

    def test_published_runtime_requires_validation_without_copying_it(self):
        _, _, _, runtime = self.image_stages()
        self.assertIn(
            "--mount=type=bind,from=validation,source=/tmp/qec-plugin-validation/passed,"
            "target=/tmp/qec-plugin-validation-passed,readonly", runtime)
        self.assertIn("test -f /tmp/qec-plugin-validation-passed", runtime)
        self.assertNotIn("COPY", runtime)


if __name__ == "__main__":
    unittest.main()
