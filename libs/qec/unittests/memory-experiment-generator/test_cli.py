#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Black-box tests of the draft model generator, not A/B graph execution."""

import argparse
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import yaml
except ImportError:
    requirements = Path(__file__).with_name("requirements.txt")
    print("SKIP: PyYAML is required only for the generator's tests. Install "
          "the test dependencies with: "
          f"{sys.executable} -m pip install -r {requirements}")
    raise SystemExit(77)

GENERATOR = None


class GeneratorCliTest(unittest.TestCase):

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="memory-experiment-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence = 0

    def invoke(self, *args, success=True):
        result = subprocess.run(
            [str(GENERATOR), *map(str, args)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
            check=False,
        )
        detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, detail)
        else:
            self.assertNotEqual(result.returncode, 0, detail)
        return result

    def generate(self,
                 *,
                 basis="Z",
                 rounds=26,
                 logical_qubits=1,
                 batch=3,
                 replay_shots=0,
                 seed=None):
        self.sequence += 1
        output = self.root / f"prepared bundle {self.sequence}"
        options = [
            "--distance=3",
            f"--rounds={rounds}",
            f"--basis={basis}",
            f"--logical-qubits={logical_qubits}",
            f"--batch-rounds={batch}",
            f"--replay-shots={replay_shots}",
            f"--output={output}",
        ]
        if seed is not None:
            options.append(f"--seed={seed}")
        self.invoke(*options)
        manifest = yaml.safe_load((output / "manifest.yaml").read_text())
        return output, manifest

    def assert_manifest(self, output, manifest):
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["kind"], "cudaq-qec-memory-model-bundle")
        self.assertEqual(manifest["status"], "model_prepared_graph_pending")
        validation = manifest["validation"]
        self.assertIs(validation["phase_model_roundtrip"], True)
        for key in ("graph_generated", "graph_constructed", "replayed"):
            self.assertIs(validation[key], False)
        model = manifest["model"]
        self.assertEqual(model["path"], "model.yaml")
        model_bytes = (output / model["path"]).read_bytes()
        self.assertEqual(
            hashlib.sha256(model_bytes).hexdigest(), model["sha256"])
        self.assertNotIn("decoders:", model_bytes.decode())
        self.assertFalse((output / "server.yaml").exists())
        self.assertFalse((output / "schedule.txt").exists())
        self.assertFalse((output / "manifest.json").exists())
        replay = manifest.get("replay")
        expected_files = {"manifest.yaml", "model.yaml"}
        if replay is not None:
            expected_files.update(item["path"] for item in replay["files"])
        self.assertEqual(set(path.name for path in output.iterdir()),
                         expected_files)
        final_width = manifest["layout"]["final_measurements"]
        for row in manifest["final_m2o"]:
            self.assertEqual(row, sorted(set(row)))
            self.assertTrue(all(0 <= item < final_width for item in row))
        self.invoke(f"--validate-bundle={output}")

    def write_manifest(self, output, manifest):
        (output / "manifest.yaml").write_text(
            yaml.safe_dump(manifest, sort_keys=False))

    def assert_raw_replay(self, output, manifest, *, shots, seed):
        self.assert_manifest(output, manifest)
        replay = manifest["replay"]
        self.assertEqual(replay["format"], "playback-static-source/v1")
        self.assertEqual(replay["shots_per_patch"], shots)
        self.assertEqual(replay["base_seed"], seed)
        self.assertIs(replay["decoded"], False)
        patches = manifest["experiment"]["logical_qubits"]
        self.assertEqual(len(replay["files"]), patches * shots)
        delivery = manifest["delivery"]
        widths = [delivery["regular_batch_measurements"]
                 ] * delivery["regular_batches"]
        widths.append(delivery["terminal_batch_measurements"])
        final_width = manifest["layout"]["final_measurements"]
        for index, item in enumerate(replay["files"]):
            patch, shot = divmod(index, shots)
            self.assertEqual(item["patch"], patch)
            self.assertEqual(item["shot"], shot)
            self.assertEqual(item["seed"], seed + index)
            self.assertEqual(item["path"], f"raw-{patch}-{shot}.txt")
            raw = (output / item["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), item["sha256"])
            lines = raw.decode("ascii").splitlines()
            self.assertEqual([len(line) for line in lines], widths)
            self.assertTrue(all(set(line) <= {"0", "1"} for line in lines))
            self.assertEqual(raw, ("\n".join(lines) + "\n").encode("ascii"))
            measurements = "".join(lines)
            self.assertEqual(len(measurements),
                             manifest["layout"]["raw_measurements"])
            readout = measurements[-final_width:]
            raw_logical = "".join(
                str(sum(int(readout[position])
                        for position in row) % 2)
                for row in manifest["final_m2o"])
            self.assertEqual(item["raw_logical"], raw_logical)

    def test_help_is_available_without_model_preparation(self):
        result = self.invoke("--help")
        for option in ("--distance", "--rounds", "--basis", "--output",
                       "--validate-bundle"):
            self.assertIn(option, result.stdout)
        self.assertIn("offline decoder testing", result.stdout)
        self.assertIn("No measurements are simulated by default", result.stdout)
        self.assertIn("YAML generation is not yet supported", result.stdout)

    def test_request_file_flags_are_unknown_and_create_no_output(self):
        for flag in ("request", "check-request"):
            with self.subTest(flag=flag):
                output = self.root / "must-not-create"
                request = self.root / "must-not-read.yaml"
                result = self.invoke("--distance=3",
                                     "--rounds=26",
                                     "--basis=Z",
                                     f"--output={output}",
                                     f"--{flag}={request}",
                                     success=False)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"unknown option: --{flag}", result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertFalse(output.exists())
                self.assertFalse(request.exists())

    def test_justin_reference_in_both_bases_roundtrips_without_graph_claim(
            self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                output, manifest = self.generate(basis=basis)
                self.assert_manifest(output, manifest)
                experiment = manifest["experiment"]
                self.assertEqual(experiment["code_family"], "surface_code")
                self.assertEqual(experiment["distance"], 3)
                self.assertEqual(experiment["rounds"], 26)
                self.assertEqual(experiment["basis"], basis)
                self.assertEqual(experiment["logical_qubits"], 1)
                layout = manifest["layout"]
                self.assertEqual(layout["regular_measurements"], 8)
                self.assertEqual(layout["final_measurements"], 9)
                self.assertEqual(layout["input_rounds"], 27)
                self.assertEqual(layout["raw_measurements"], 217)
                self.assertEqual(layout["physical_qubits_per_patch"], 17)
                self.assertEqual(
                    manifest["delivery"], {
                        "batch_rounds": 3,
                        "regular_batches": 8,
                        "regular_batch_measurements": 24,
                        "terminal_extraction_rounds": 2,
                        "terminal_batch_measurements": 25,
                    })

    def test_long_horizon_does_not_expand_representative_preparation(self):
        small_dir, small = self.generate(rounds=26)
        small_size = (small_dir / "model.yaml").stat().st_size
        for rounds in (10000, 1000000000):
            with self.subTest(rounds=rounds):
                output, manifest = self.generate(rounds=rounds)
                self.assert_manifest(output, manifest)
                self.assertEqual(manifest["experiment"]["rounds"], rounds)
                self.assertEqual(manifest["layout"]["raw_measurements"],
                                 rounds * 8 + 9)
                self.assertEqual(manifest["model"]["characterization_rounds"],
                                 small["model"]["characterization_rounds"])
                self.assertLess((output / "model.yaml").stat().st_size,
                                small_size + 1024)

    def test_independent_patches_share_model_without_implied_core_budget(self):
        one_dir, one = self.generate()
        many_dir, many = self.generate(logical_qubits=4)
        self.assert_manifest(many_dir, many)
        self.assertEqual(many["experiment"]["logical_qubits"], 4)
        self.assertEqual(many["layout"], one["layout"])
        self.assertEqual(many["model"]["sha256"], one["model"]["sha256"])
        self.assertEqual((one_dir / "model.yaml").read_bytes(),
                         (many_dir / "model.yaml").read_bytes())

    def test_exact_batch_and_oversized_batch_preserve_readout(self):
        for rounds, batch, batches, remainder, terminal in (
            (24, 3, 8, 0, 9),
            (26, 27, 0, 26, 217),
        ):
            with self.subTest(rounds=rounds, batch=batch):
                _, manifest = self.generate(rounds=rounds, batch=batch)
                delivery = manifest["delivery"]
                self.assertEqual(delivery["regular_batches"], batches)
                self.assertEqual(delivery["terminal_extraction_rounds"],
                                 remainder)
                self.assertEqual(delivery["terminal_batch_measurements"],
                                 terminal)
                self.assertEqual(delivery["regular_batch_measurements"],
                                 batch * 8 if batches else 0)

    def test_probability_upper_boundary_is_rejected_before_backend_abort(self):
        output = self.root / "probability-one"
        result = self.invoke("--distance=3",
                             "--rounds=26",
                             "--basis=Z",
                             "--error-rate=1",
                             f"--output={output}",
                             success=False)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(output.exists())

    def test_malformed_and_duplicate_options_leave_no_output(self):
        invalid = (
            "--distance=3.5",
            "--distance=-3",
            "--distance=+3",
            "--distance=3x",
            "--distance=18446744073709551616",
            "--rounds=0",
            "--rounds=26.5",
            "--basis=Y",
            "--logical-qubits=0",
            "--batch-rounds=0",
            "--replay-shots=-1",
            "--replay-shots=1.5",
            "--seed=-1",
            "--seed=1.5",
            "--seed=18446744073709551616",
            "--error-rate=nan",
            "--error-rate=inf",
            "--error-rate=-0.1",
            "--error-rate=1.1",
            "--error-rate=0.1junk",
            "--unknown-option=1",
        )
        for option in invalid:
            with self.subTest(option=option):
                output = self.root / "must-not-exist"
                key = option.split("=", 1)[0]
                defaults = ["--distance=3", "--rounds=26", "--basis=Z"]
                args = [
                    arg for arg in defaults if not arg.startswith(key + "=")
                ]
                self.invoke(*args, f"--output={output}", option, success=False)
                self.assertFalse(output.exists())
        output = self.root / "duplicate"
        self.invoke("--distance=3",
                    "--distance=3",
                    "--rounds=26",
                    "--basis=Z",
                    f"--output={output}",
                    success=False)
        self.assertFalse(output.exists())
        self.invoke("--distance=3", "--rounds=26", success=False)
        self.invoke("--distance", "3", "--rounds=26", success=False)

    def test_existing_directory_and_file_are_never_overwritten(self):
        for name, is_directory in (("directory", True), ("file", False)):
            with self.subTest(target=name):
                output = self.root / name
                if is_directory:
                    output.mkdir()
                    sentinel = output / "keep.txt"
                else:
                    sentinel = output
                sentinel.write_text("unrelated user content")
                self.invoke("--distance=3",
                            "--rounds=26",
                            "--basis=Z",
                            f"--output={output}",
                            success=False)
                self.assertEqual(sentinel.read_text(), "unrelated user content")
        output = self.root / "empty"
        output.mkdir()
        self.invoke("--distance=3",
                    "--rounds=26",
                    "--basis=Z",
                    f"--output={output}",
                    success=False)
        self.assertEqual(list(output.iterdir()), [])

    def test_provider_failures_and_pending_graph_leave_no_partial_bundle(self):
        for extra in (("--distance=4",), ("--error-rate=0",),
                      ("--emit-graph",)):
            with self.subTest(extra=extra):
                output = self.root / "failed"
                distance = extra[0] if extra[0].startswith(
                    "--distance=") else "--distance=3"
                rest = () if extra[0].startswith("--distance=") else extra
                result = self.invoke(distance,
                                     "--rounds=26",
                                     "--basis=Z",
                                     f"--output={output}",
                                     *rest,
                                     success=False)
                self.assertFalse(output.exists())
                self.assertTrue(result.stderr.strip())
        blocker = self.root / "parent-is-file"
        blocker.write_text("keep")
        self.invoke("--distance=3",
                    "--rounds=26",
                    "--basis=Z",
                    f"--output={blocker / 'bundle'}",
                    success=False)
        self.assertEqual(blocker.read_text(), "keep")

    def test_validation_detects_model_tampering_and_missing_files(self):
        output, manifest = self.generate()
        model_path = output / manifest["model"]["path"]
        original = model_path.read_bytes()
        model_path.write_bytes(original + b"\n# tampered\n")
        self.invoke(f"--validate-bundle={output}", success=False)
        model_path.unlink()
        self.invoke(f"--validate-bundle={output}", success=False)
        model_path.write_bytes(original)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_schema_layout_and_logical_map_corruption(self):
        output, original = self.generate()
        cases = (
            ("schema_version", 2),
            ("kind", "server-config"),
            ("status", "graph_replayed"),
        )
        for key, value in cases:
            with self.subTest(key=key):
                manifest = copy.deepcopy(original)
                manifest[key] = value
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        manifest = copy.deepcopy(original)
        manifest["layout"]["raw_measurements"] += 1
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}", success=False)
        manifest = copy.deepcopy(original)
        manifest["final_m2o"] = [[manifest["layout"]["final_measurements"]]]
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, original)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_malformed_yaml_manifest(self):
        output, _ = self.generate()
        path = output / "manifest.yaml"
        original = path.read_text()
        for text in ("", "[]\n", "schema_version: [\n", "null\n"):
            with self.subTest(text=text):
                path.write_text(text)
                self.invoke(f"--validate-bundle={output}", success=False)
        path.write_text(original)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_duplicate_yaml_manifest_fields(self):
        output, manifest = self.generate()
        path = output / "manifest.yaml"
        document = yaml.safe_dump(manifest, sort_keys=False)
        repeated_distance = document.replace("  distance: 3\n",
                                             "  distance: 3\n  distance: 3\n",
                                             1)
        self.assertNotEqual(document, repeated_distance)
        for text in (document + "schema_version: 1\n", repeated_distance):
            with self.subTest(text=text):
                path.write_text(text)
                self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_multiple_yaml_documents_including_empty(self):
        output, manifest = self.generate()
        path = output / "manifest.yaml"
        document = yaml.safe_dump(manifest,
                                  sort_keys=False,
                                  explicit_start=True,
                                  explicit_end=True)
        empty_document = "---\n...\n"
        for text in (document + document, empty_document + document,
                     document + empty_document):
            with self.subTest(text=text):
                path.write_text(text)
                self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_geometry_and_provenance_corruption(self):
        output, original = self.generate()
        cases = (
            ("experiment", "distance", 5),
            ("layout", "physical_qubits_per_patch", 99),
            ("model", "characterization_rounds", 0),
            ("model", "validation_rounds", 0),
            ("noise", "kind", "unsupported_noise"),
            ("noise", "probability", -1),
            ("noise", "probability", "not-a-probability"),
            ("normalization", "decompose_errors", "not-a-boolean"),
            ("normalization", "correlations", "unknown"),
            ("provenance", "qec_repository_version", []),
        )
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                manifest = copy.deepcopy(original)
                manifest[section][key] = value
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, original)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_unknown_manifest_fields(self):
        output, original = self.generate(replay_shots=1, seed=19)
        for section in (None, "experiment", "layout", "delivery", "model",
                        "normalization", "noise", "provenance", "validation",
                        "replay"):
            with self.subTest(section=section):
                manifest = copy.deepcopy(original)
                target = manifest if section is None else manifest[section]
                target["unsupported_field"] = True
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        manifest = copy.deepcopy(original)
        manifest["replay"]["files"][0]["expected_correction"] = "0"
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, original)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_model_paths_outside_bundle(self):
        output, manifest = self.generate()
        external = self.root / "external.yaml"
        external.write_bytes((output / "model.yaml").read_bytes())
        for path in ("../external.yaml", str(external)):
            with self.subTest(path=path):
                manifest["model"]["path"] = path
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        manifest["model"]["path"] = "model.yaml"
        self.write_manifest(output, manifest)
        model = output / "model.yaml"
        model.unlink()
        model.symlink_to(external)
        self.invoke(f"--validate-bundle={output}", success=False)

    def test_raw_replay_is_seeded_repeatable_in_both_bases(self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                first, manifest = self.generate(basis=basis,
                                                logical_qubits=2,
                                                replay_shots=2,
                                                seed=19)
                self.assert_raw_replay(first, manifest, shots=2, seed=19)
                repeated, other = self.generate(basis=basis,
                                                logical_qubits=2,
                                                replay_shots=2,
                                                seed=19)
                self.assert_raw_replay(repeated, other, shots=2, seed=19)
                self.assertEqual(manifest, other)
                for item in manifest["replay"]["files"]:
                    self.assertEqual((first / item["path"]).read_bytes(),
                                     (repeated / item["path"]).read_bytes())

    def test_raw_replay_preserves_aligned_and_oversized_delivery(self):
        for rounds, batch in ((24, 3), (26, 27)):
            with self.subTest(rounds=rounds, batch=batch):
                output, manifest = self.generate(rounds=rounds,
                                                 batch=batch,
                                                 replay_shots=1,
                                                 seed=7)
                self.assert_raw_replay(output, manifest, shots=1, seed=7)

    def test_raw_replay_zero_shots_and_maximum_single_seed(self):
        maximum = (1 << 64) - 1
        output, manifest = self.generate(replay_shots=0, seed=maximum)
        self.assert_manifest(output, manifest)
        self.assertIsNone(manifest.get("replay"))
        output, manifest = self.generate(replay_shots=1, seed=maximum)
        self.assert_raw_replay(output, manifest, shots=1, seed=maximum)

    def test_replay_seed_or_count_overflow_removes_partial_output(self):
        maximum = (1 << 64) - 1
        for patches, shots, seed in ((2, 2, maximum - 2), (2, maximum, 1),
                                     (maximum, 2, 1), (1, 1, 0)):
            with self.subTest(patches=patches, shots=shots, seed=seed):
                output = self.root / "overflow-must-not-remain"
                self.invoke("--distance=3",
                            "--rounds=26",
                            "--basis=Z",
                            f"--logical-qubits={patches}",
                            f"--replay-shots={shots}",
                            f"--seed={seed}",
                            f"--output={output}",
                            success=False)
                self.assertFalse(output.exists())
                self.assertEqual(list(self.root.iterdir()), [])

    def test_validation_rejects_tampered_truncated_and_missing_raw_files(self):
        output, original = self.generate(replay_shots=1, seed=23)
        item = original["replay"]["files"][0]
        path = output / item["path"]
        raw = path.read_bytes()
        path.write_bytes(raw + b"0\n")
        self.invoke(f"--validate-bundle={output}", success=False)
        path.unlink()
        self.invoke(f"--validate-bundle={output}", success=False)
        lines = raw.splitlines()
        for replacement in (
                b"\n".join(lines[:-1] + [lines[-1][:-1]]) + b"\n",
                b"\n".join(lines[:-1]) + b"\n",
                raw + b"0\n",
                raw[:1] + b"x" + raw[2:],
        ):
            with self.subTest(replacement=replacement):
                manifest = copy.deepcopy(original)
                manifest["replay"]["files"][0]["sha256"] = hashlib.sha256(
                    replacement).hexdigest()
                self.write_manifest(output, manifest)
                path.write_bytes(replacement)
                self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, original)
        path.write_bytes(raw)
        self.invoke(f"--validate-bundle={output}")
        external = self.root / "external-raw.txt"
        external.write_bytes(raw)
        path.unlink()
        path.symlink_to(external)
        self.invoke(f"--validate-bundle={output}", success=False)

    def test_validation_recomputes_raw_logical_from_readout_support(self):
        output, manifest = self.generate(replay_shots=1, seed=29)
        item = manifest["replay"]["files"][0]
        path = output / item["path"]
        lines = path.read_text().splitlines()
        terminal = list(lines[-1])
        readout_start = len(terminal) - manifest["layout"]["final_measurements"]
        position = readout_start + manifest["final_m2o"][0][0]
        terminal[position] = "1" if terminal[position] == "0" else "0"
        lines[-1] = "".join(terminal)
        replacement = ("\n".join(lines) + "\n").encode("ascii")
        path.write_bytes(replacement)
        item["sha256"] = hashlib.sha256(replacement).hexdigest()
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}", success=False)
        # Validating raw inputs is not decoding: once the declared raw parity
        # matches these changed readout bits, no correction oracle is claimed.
        item["raw_logical"] = "1" if item["raw_logical"] == "0" else "0"
        self.write_manifest(output, manifest)
        self.invoke(f"--validate-bundle={output}")

    def test_validation_rejects_bad_replay_manifest(self):
        output, original = self.generate(logical_qubits=2,
                                         replay_shots=2,
                                         seed=31)
        cases = (
            ("format", "unsupported/v1"),
            ("shots_per_patch", 0),
            ("shots_per_patch", -1),
            ("shots_per_patch", "not-a-count"),
            ("shots_per_patch", (1 << 64) - 1),
            ("base_seed", (1 << 64) - 1),
            ("base_seed", -1),
            ("files", []),
            ("files", [None] * 4),
            ("files", [None] * 5),
            ("decoded", True),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                manifest = copy.deepcopy(original)
                manifest["replay"][key] = value
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        for replay in ([], "raw-inputs", 1):
            with self.subTest(replay=replay):
                manifest = copy.deepcopy(original)
                manifest["replay"] = replay
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        cases = (("patch", 1), ("shot", 1), ("seed", 999), ("path",
                                                            "../raw-0-0.txt"),
                 ("sha256", "0" * 64), ("raw_logical", "wrong"))
        for key, value in cases:
            with self.subTest(key=key, value=value):
                manifest = copy.deepcopy(original)
                manifest["replay"]["files"][0][key] = value
                self.write_manifest(output, manifest)
                self.invoke(f"--validate-bundle={output}", success=False)
        self.write_manifest(output, original)
        self.invoke(f"--validate-bundle={output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    GENERATOR = arguments.generator.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
