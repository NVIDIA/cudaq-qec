# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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
                     "QEC_REQUIRED_TASK_GRAPH_KINDS"):
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

    def test_published_runtime_requires_validation_without_copying_it(self):
        _, _, _, runtime = self.image_stages()
        self.assertIn(
            "--mount=type=bind,from=validation,source=/tmp/qec-plugin-validation/passed,"
            "target=/tmp/qec-plugin-validation-passed,readonly", runtime)
        self.assertIn("test -f /tmp/qec-plugin-validation-passed", runtime)
        self.assertNotIn("COPY", runtime)


if __name__ == "__main__":
    unittest.main()
