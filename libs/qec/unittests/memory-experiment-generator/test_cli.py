#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Black-box server-YAML generation tests; no task graph is executed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

try:
    import yaml
except ImportError:
    requirements = Path(__file__).with_name("requirements.txt")
    in_ci = os.environ.get("CI", "").lower() not in ("", "0", "false")
    prefix = "ERROR" if in_ci else "SKIP"
    print(
        f"{prefix}: PyYAML is required only for the generator's tests. Install "
        "the test dependencies with: "
        f"{sys.executable} -m pip install -r {requirements}")
    raise SystemExit(1 if in_ci else 77)

GENERATOR = None
OUTPUT_FILENAME = "experiment.yaml"


def generation_options(**overrides):
    values = {
        "distance": 3,
        "rounds": 26,
        "basis": "Z",
        "decoder": "single_error_lut",
        "output": OUTPUT_FILENAME,
    }
    values.update(overrides)
    return [f"--{key}={value}" for key, value in values.items()]


def run_program(program,
                *arguments,
                cwd,
                environment_overrides=None,
                timeout=90,
                preexec_fn=None,
                umask=-1):
    environment = os.environ.copy()
    if environment_overrides:
        environment.update(environment_overrides)
    return subprocess.run(
        [str(program), *map(str, arguments)],
        cwd=cwd,
        env=environment,
        preexec_fn=preexec_fn,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        umask=umask,
    )


class GeneratorCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(
                prefix="memory-experiment-api-probe-") as temporary:
            working = Path(temporary)
            result = run_program(GENERATOR, *generation_options(), cwd=working)
            if result.returncode != 0:
                raise AssertionError(
                    f"public configuration generation failed:\n{result.stderr}")
            if {path.name for path in working.iterdir()} != {OUTPUT_FILENAME}:
                raise AssertionError(
                    "capability probe created unexpected files")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="memory-experiment-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.sequence = 0

    def case_directory(self):
        self.sequence += 1
        working = self.root / f"case {self.sequence}"
        working.mkdir()
        return working

    def invoke(self, *arguments, success=True, cwd=None):
        result = run_program(GENERATOR, *arguments, cwd=cwd or self.root)
        detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, detail)
        else:
            self.assertNotEqual(result.returncode, 0, detail)
        return result

    def generate(self, *, extra=(), output_path=OUTPUT_FILENAME, **overrides):
        working = self.case_directory()
        output = Path(output_path)
        if not output.is_absolute():
            output = working / output
        result = self.invoke(*generation_options(output=output_path,
                                                 **overrides),
                             *extra,
                             cwd=working)
        self.assertIn(f"Generated server configuration: {output}",
                      result.stdout.splitlines())
        expected = {output.name} if output.parent == working else set()
        self.assertEqual({path.name for path in working.iterdir()}, expected)
        document = yaml.safe_load(output.read_text())
        return output, document

    def assert_server(self,
                      document,
                      *,
                      distance=3,
                      rounds=26,
                      patches=1,
                      commit=None,
                      buffer=None,
                      gap=None,
                      workers=None,
                      output="corrections"):
        self.assertEqual(len(document["data_artifacts"]), 1)
        model_name, artifact = next(iter(document["data_artifacts"].items()))
        model = artifact["dem_chunks"]
        self.assertEqual(model["num_rounds"], rounds + 1)
        self.assertEqual(model["measurements_per_round"], distance**2 - 1)
        mapping = document["syndrome_mapping"]
        self.assertEqual(
            [(row["id"], row.get("type", "bulk")) for row in mapping],
            [(0, "bulk"), (1, "final")])
        for row in mapping:
            self.assertEqual(row.get("vps", [[]]), [[]])
        decoders = document["decoders"]
        self.assertEqual([decoder["id"] for decoder in decoders],
                         list(range(patches)))
        for decoder in decoders:
            self.assertEqual(decoder["dem_chunks"], model_name)
            graph = decoder["task_graph"]
            if workers is not None:
                self.assertIs(graph["async_streaming"], True)
                self.assertEqual(graph["async_num_workers"], workers)
            # C++ round-trip tests check typed edges and framework defaults;
            # do not pin their YAML spelling or omitted optional fields here.
            self.assertEqual(len(graph["nodes"]), 1)
            node = graph["nodes"][0]
            self.assertEqual(node["type"], "parallel_window")
            self.assertEqual(node["output"], output)
            self.assertEqual(node["commit_rounds"],
                             distance if commit is None else commit)
            self.assertEqual(node["buffer_rounds"],
                             distance if buffer is None else buffer)
            if gap is not None:
                self.assertEqual(node["gap_rounds"], gap)
            elif "gap_rounds" in node:
                self.assertEqual(
                    node["gap_rounds"],
                    node["commit_rounds"] + 2 * node["buffer_rounds"])
        connections = model["connections"]
        connected = {connection["to"] for connection in connections}
        outgoing = {connection["from"] for connection in connections}
        terminal_names = connected - outgoing
        self.assertEqual(len(terminal_names), 1)
        terminal = next(phase["spec"]
                        for phase in model["phases"]
                        if phase["name"] in terminal_names)
        self.assertEqual(terminal["num_measurements"], distance**2)
        readout = terminal["m2o_sparse"]
        self.assertEqual(readout[-1], -1)
        self.assertEqual(readout.count(-1), 1)
        self.assertEqual(readout[:-1], sorted(set(readout[:-1])))
        self.assertTrue(all(0 <= index < distance**2 for index in readout[:-1]))
        return model

    def test_help_lists_standalone_inputs_and_output(self):
        for arguments in (("--help",), ("-h",), ("--distance=3", "--help"),
                          ("-h", "--basis=Z")):
            with self.subTest(arguments=arguments):
                result = self.invoke(*arguments)
                for option in ("--code", "--distance", "--rounds", "--basis",
                               "--decoder", "--commit-rounds",
                               "--buffer-rounds", "--decoder-arg",
                               "--pattern-arg", "--workers-per-graph",
                               "--output"):
                    self.assertIn(option, result.stdout)
                for removed in ("--batch-rounds", "--replay-shots", "--seed",
                                "--validate-bundle"):
                    self.assertNotIn(removed, result.stdout)
                self.assertIn("at least 3", result.stdout)
                self.assertIn("at least 4", result.stdout)
                self.assertIn("--output=PATH", result.stdout)
                self.assertNotIn("--output defaults", result.stdout)
                self.assertNotIn("CUDAQ_QEC_DECODER_CONFIG", result.stdout)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_removed_modes_and_file_inputs_are_rejected_without_output(self):
        for name in ("batch-rounds", "replay-shots", "seed", "validate-bundle",
                     "model-only", "request", "check-request", "config",
                     "parameters", "input"):
            with self.subTest(option=name):
                working = self.case_directory()
                unopened = working / "must-not-read.json"
                result = self.invoke(*generation_options(),
                                     f"--{name}={unopened}",
                                     cwd=working,
                                     success=False)
                self.assertIn(f"unknown option: --{name}", result.stderr)
                self.assertEqual(list(working.iterdir()), [])

    def test_experiment_decoder_and_output_inputs_are_required(self):
        for required in ("distance", "rounds", "basis", "decoder", "output"):
            with self.subTest(option=required):
                working = self.case_directory()
                args = [
                    argument for argument in generation_options()
                    if not argument.startswith(f"--{required}=")
                ]
                result = self.invoke(*args, cwd=working, success=False)
                self.assertIn(f"missing --{required}", result.stderr)
                self.assertEqual(list(working.iterdir()), [])

    def test_malformed_duplicate_and_invalid_inputs_leave_no_output(self):
        invalid = {
            "distance": ("3.5", "-3", "+3", "3x", "4", "18446744073709551616"),
            "rounds": ("0", "1", "26.5", "unknown", "18446744073709551616"),
            "logical-qubits": ("0", "-1", "1.5"),
            "error-rate": ("0", "1", "nan", "inf", "-0.1", "0.1junk"),
            "commit-rounds": ("0", "-1", "1.5"),
            "buffer-rounds": ("0", "-1", "1.5"),
            "gap-rounds": ("0", "-1", "1.5"),
            "workers-per-graph":
                ("0", "-1", "+1", "1.5", "4junk", "18446744073709551616"),
        }
        for option, values in invalid.items():
            for value in values:
                with self.subTest(option=option, value=value):
                    working = self.case_directory()
                    self.invoke(*generation_options(**{option: value}),
                                cwd=working,
                                success=False)
                    self.assertEqual(list(working.iterdir()), [])
        for extra in (("--distance=3",), ("--decoder=single_error_lut",),
                      ("--distance", "3"), ("--workers-per-graph=2",
                                            "--workers-per-graph=3"),
                      ("--output=one.yaml",
                       "--output=two.yaml"), ("--output=",)):
            with self.subTest(extra=extra):
                working = self.case_directory()
                self.invoke(*generation_options(),
                            *extra,
                            cwd=working,
                            success=False)
                self.assertEqual(list(working.iterdir()), [])

    def test_explicit_output_never_overwrites_any_existing_entry(self):
        for kind in ("file", "directory", "file symlink", "directory symlink",
                     "dangling symlink"):
            with self.subTest(kind=kind):
                working = self.case_directory()
                output = working / OUTPUT_FILENAME
                if kind == "file":
                    output.write_text("unrelated user content")
                elif kind == "directory":
                    output.mkdir()
                    (output / "keep.txt").write_text("unrelated user content")
                else:
                    target = working / "target"
                    if kind == "file symlink":
                        target.write_text("unrelated user content")
                    elif kind == "directory symlink":
                        target.mkdir()
                        (target /
                         "keep.txt").write_text("unrelated user content")
                    output.symlink_to(
                        target, target_is_directory=kind == "directory symlink")
                before = {entry.name for entry in working.iterdir()}
                self.invoke(*generation_options(), cwd=working, success=False)
                self.assertEqual({entry.name for entry in working.iterdir()},
                                 before)
                if "symlink" in kind:
                    self.assertTrue(output.is_symlink())
                if kind == "dangling symlink":
                    self.assertFalse(target.exists())
                elif kind == "directory" or kind == "directory symlink":
                    self.assertEqual((output / "keep.txt").read_text(),
                                     "unrelated user content")
                else:
                    self.assertEqual(output.read_text(),
                                     "unrelated user content")

    def test_relative_and_absolute_output_paths_require_existing_parent(self):
        deployment = self.root / "deployment % directory"
        deployment.mkdir()
        destination = deployment / "config %.yaml"
        output, document = self.generate(output_path=destination)
        self.assertEqual(output, destination)
        self.assert_server(document)
        relative, document = self.generate(output_path="relative config.yaml")
        self.assertEqual(relative.name, "relative config.yaml")
        self.assert_server(document)
        working = self.case_directory()
        unrelated = working / "unrelated.yaml"
        unrelated.write_text("unrelated configuration")
        self.invoke(*generation_options(output="another.yaml"), cwd=working)
        self.assertEqual(unrelated.read_text(), "unrelated configuration")
        self.assert_server(
            yaml.safe_load((working / "another.yaml").read_text()))
        self.assertEqual({entry.name for entry in working.iterdir()},
                         {"unrelated.yaml", "another.yaml"})
        working = self.case_directory()
        missing = self.root / "missing parent" / OUTPUT_FILENAME
        self.invoke(*generation_options(output=missing),
                    cwd=working,
                    success=False)
        self.assertEqual(list(working.iterdir()), [])
        self.assertFalse(missing.parent.exists())

    def test_plugin_configuration_environment_does_not_select_output(self):
        deployment = self.root / "deployment.yaml"
        deployment.write_text("unrelated deployment configuration")
        for environment_path in (deployment,
                                 self.root / "missing" / "deployment.yaml"):
            with self.subTest(environment_path=environment_path):
                working = self.case_directory()
                result = run_program(GENERATOR,
                                     *generation_options(),
                                     cwd=working,
                                     environment_overrides={
                                         "CUDAQ_QEC_DECODER_CONFIG":
                                             str(environment_path)
                                     })
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({path.name for path in working.iterdir()},
                                 {OUTPUT_FILENAME})
                self.assert_server(
                    yaml.safe_load((working / OUTPUT_FILENAME).read_text()))
        self.assertEqual(deployment.read_text(),
                         "unrelated deployment configuration")
        self.assertFalse((self.root / "missing").exists())
        working = self.case_directory()
        arguments = [
            argument for argument in generation_options()
            if not argument.startswith("--output=")
        ]
        result = run_program(
            GENERATOR,
            *arguments,
            cwd=working,
            environment_overrides={"CUDAQ_QEC_DECODER_CONFIG": str(deployment)})
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("missing --output", result.stderr)
        self.assertEqual(list(working.iterdir()), [])

    @unittest.skipUnless(os.name == "posix", "requires POSIX file permissions")
    def test_published_permissions_follow_deployment_umask(self):
        for mask in (0o022, 0o027, 0o077):
            with self.subTest(umask=oct(mask)):
                working = self.case_directory()
                result = run_program(GENERATOR,
                                     *generation_options(),
                                     cwd=working,
                                     umask=mask)
                self.assertEqual(result.returncode, 0, result.stderr)
                output = working / OUTPUT_FILENAME
                self.assertEqual(stat.S_IMODE(output.stat().st_mode),
                                 0o666 & ~mask)

    def test_concurrent_publication_has_one_winner_without_temporary_residue(
            self):
        working = self.case_directory()
        with ThreadPoolExecutor(max_workers=2) as workers:
            futures = [
                workers.submit(run_program,
                               GENERATOR,
                               *generation_options(),
                               cwd=working) for _ in range(2)
            ]
            results = [future.result() for future in futures]
        self.assertEqual(sum(result.returncode == 0 for result in results), 1,
                         [result.stderr for result in results])
        self.assertEqual({path.name for path in working.iterdir()},
                         {OUTPUT_FILENAME})
        self.assert_server(
            yaml.safe_load((working / OUTPUT_FILENAME).read_text()))

    @unittest.skipUnless(os.name == "posix", "requires POSIX file-size limits")
    def test_failed_write_removes_its_output_and_allows_retry(self):
        import resource
        import signal

        def limit_child_file_size():
            # Fault injection is confined to this child; normal generation and
            # the test runner retain their original resource limits.
            signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
            file_size_limit_bytes = 64
            resource.setrlimit(resource.RLIMIT_FSIZE,
                               (file_size_limit_bytes, file_size_limit_bytes))

        working = self.case_directory()
        result = run_program(GENERATOR,
                             *generation_options(),
                             cwd=working,
                             preexec_fn=limit_child_file_size)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("cannot write server configuration", result.stderr)
        self.assertEqual(list(working.iterdir()), [])
        self.invoke(*generation_options(), cwd=working)
        self.assert_server(
            yaml.safe_load((working / OUTPUT_FILENAME).read_text()))

    def test_reference_layout_in_both_bases_emits_only_requested_yaml(self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                output, document = self.generate(basis=basis)
                self.assertEqual(output.name, OUTPUT_FILENAME)
                model = self.assert_server(document)
                self.assertEqual(26 * model["measurements_per_round"] + 9, 217)

    def test_model_factorization_follows_backend(self):

        def maximum_fault_degree(model):
            maximum = 0
            for phase in model["phases"]:
                spec = phase["spec"]
                degree = [0] * spec["num_faults"]
                for seam in spec["seam_specs"]:
                    for column in seam["spec"]["H_sparse"]:
                        if column >= 0:
                            degree[column] += 1
                maximum = max(maximum, *degree)
            return maximum

        _, lut = self.generate(decoder="multi_error_lut", rounds=6)
        self.assertGreater(
            maximum_fault_degree(self.assert_server(lut, rounds=6)), 2)
        working = self.case_directory()
        result = run_program(GENERATOR,
                             *generation_options(decoder="pymatching",
                                                 rounds=6),
                             cwd=working)
        if result.returncode != 0:
            self.assertIn(
                "decoder is not registered in this QEC build: "
                "pymatching", result.stderr)
            return
        matching = yaml.safe_load((working / OUTPUT_FILENAME).read_text())
        self.assertLessEqual(
            maximum_fault_degree(self.assert_server(matching, rounds=6)), 2)

    def test_basis_case_is_normalized_and_other_basis_rejected(self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis.lower()):
                _, upper = self.generate(basis=basis, rounds=4)
                _, lower = self.generate(basis=basis.lower(), rounds=4)
                self.assertEqual(lower, upper)
        working = self.case_directory()
        result = self.invoke(*generation_options(basis="Y", rounds=4),
                             cwd=working,
                             success=False)
        self.assertIn("--basis must be X or Z", result.stderr)
        self.assertEqual(list(working.iterdir()), [])

    def test_default_geometry_tracks_distance_and_partial_overrides(self):
        for distance in (3, 5):
            with self.subTest(distance=distance):
                _, document = self.generate(distance=distance)
                self.assert_server(document, distance=distance)
        for flag, parameter, value in (("commit-rounds", "commit",
                                        2), ("buffer-rounds", "buffer", 4),
                                       ("gap-rounds", "gap", 9)):
            with self.subTest(flag=flag):
                _, document = self.generate(**{flag: value})
                self.assert_server(document, **{parameter: value})

    def test_schema_typed_pattern_arguments_apply_to_every_patch(self):
        parameters = {
            "commit_rounds": 2,
            "buffer_rounds": 4,
            "gap_rounds": 7,
            "boundary_prior": 1e-6,
            "input_boundary": "columns",
            "merge_identical": True,
            "conflicting_parallel": "most_likely",
            "verify_residual": "report",
            "output": "corrected_logical",
        }
        pattern_arguments = (
            "--pattern-arg=commit_rounds=2",
            "--pattern-arg=buffer_rounds=4",
            "--pattern-arg=gap_rounds=7",
            "--pattern-arg=boundary_prior=1e-6",
            "--pattern-arg=input_boundary=columns",
            "--pattern-arg=merge_identical=true",
            "--pattern-arg=conflicting_parallel=most_likely",
            "--pattern-arg=verify_residual=report",
            "--pattern-arg=output=corrected_logical",
            "--decoder-arg=lut_error_depth=2",
        )
        for patches in (1, 3):
            with self.subTest(patches=patches):
                _, document = self.generate(decoder="multi_error_lut",
                                            extra=pattern_arguments,
                                            **{"logical-qubits": patches})
                self.assert_server(document,
                                   patches=patches,
                                   commit=2,
                                   buffer=4,
                                   gap=7,
                                   output="corrected_logical")
                for decoder in document["decoders"]:
                    node = decoder["task_graph"]["nodes"][0]
                    for name, expected in parameters.items():
                        self.assertIs(type(node[name]), type(expected))
                        self.assertEqual(node[name], expected)
                    self.assertEqual(node["inner"]["type"], "multi_error_lut")
                    self.assertEqual(node["inner"]["lut_error_depth"], 2)

    def test_pattern_geometry_overrides_only_unspecified_profile_defaults(self):
        for key, parameter, value in (("commit_rounds", "commit",
                                       2), ("buffer_rounds", "buffer", 4),
                                      ("gap_rounds", "gap", 7)):
            with self.subTest(key=key):
                _, document = self.generate(
                    extra=(f"--pattern-arg={key}={value}",))
                self.assert_server(document, **{parameter: value})

    def test_truncated_boundary_and_residual_policies_roundtrip(self):
        for verification in ("none", "fail", "report"):
            with self.subTest(verification=verification):
                _, document = self.generate(extra=(
                    "--pattern-arg=input_boundary=truncate",
                    "--pattern-arg=merge_identical=false",
                    "--pattern-arg=conflicting_parallel=reject",
                    f"--pattern-arg=verify_residual={verification}",
                ))
                self.assert_server(document)
                node = document["decoders"][0]["task_graph"]["nodes"][0]
                self.assertEqual(node["input_boundary"], "truncate")
                self.assertIs(node["merge_identical"], False)
                self.assertEqual(node["conflicting_parallel"], "reject")
                self.assertEqual(node["verify_residual"], verification)

    def test_pattern_geometry_aliases_reject_explicit_dedicated_flags(self):
        for flag, key, value in (("commit-rounds", "commit_rounds",
                                  2), ("buffer-rounds", "buffer_rounds", 4),
                                 ("gap-rounds", "gap_rounds", 7)):
            for pattern_value in (value, value + 1):
                for pattern_first in (False, True):
                    with self.subTest(flag=flag,
                                      value=pattern_value,
                                      pattern_first=pattern_first):
                        working = self.case_directory()
                        arguments = generation_options(**{flag: value})
                        pattern = f"--pattern-arg={key}={pattern_value}"
                        if pattern_first:
                            arguments.insert(0, pattern)
                        else:
                            arguments.append(pattern)
                        result = self.invoke(*arguments,
                                             cwd=working,
                                             success=False)
                        self.assertIn(f"--{flag}", result.stderr)
                        self.assertIn(f"--pattern-arg={key}", result.stderr)
                        self.assertEqual(list(working.iterdir()), [])

    def test_invalid_pattern_arguments_are_rejected_without_output(self):
        invalid = (
            ("unknown=1",),
            ("commit_rounds=0",),
            ("buffer_rounds=0",),
            ("gap_rounds=0",),
            ("boundary_prior=0",),
            ("boundary_prior=1",),
            ("boundary_prior=nan",),
            ("boundary_prior=inf",),
            ("boundary_prior=1junk",),
            ("merge_identical=1",),
            ("gap_rounds=-1",),
            ("gap_rounds=1.5",),
            ("output=unsupported",),
            ("input_boundary=unsupported",),
            ("conflicting_parallel=unsupported",),
            ("verify_residual=unsupported",),
            ("output=corrections", "output=corrections"),
            ("output=corrections", "output=corrected_logical"),
            ("input_boundary=truncate", "boundary_prior=1e-6"),
            ("boundary_prior=1e-6", "input_boundary=truncate"),
            (".output=corrections",),
            ("output.child=corrections",),
            ("missing_equals",),
            ("=value",),
            ("output=",),
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                working = self.case_directory()
                self.invoke(
                    *generation_options(),
                    *(f"--pattern-arg={argument}" for argument in arguments),
                    cwd=working,
                    success=False)
                self.assertEqual(list(working.iterdir()), [])

    def test_long_and_awkward_fixed_horizons_are_not_rounded(self):
        self.assertGreater(96, 10 * 3)
        self.assertEqual(96 % 3, 0)
        for rounds in (96, 25):
            with self.subTest(rounds=rounds):
                _, document = self.generate(rounds=rounds)
                self.assert_server(document, rounds=rounds)

    def test_billion_round_configuration_stays_compact(self):
        small_path, small = self.generate()
        large_path, large = self.generate(rounds=1000000000)
        small_model = self.assert_server(small)
        large_model = self.assert_server(large, rounds=1000000000)
        self.assertEqual(large_model["phases"], small_model["phases"])
        self.assertEqual(large_model["connections"], small_model["connections"])
        self.assertLess(large_path.stat().st_size,
                        small_path.stat().st_size + 1024)

    def test_independent_patches_share_one_model_without_core_allocations(self):
        _, single = self.generate()
        _, multiple = self.generate(**{"logical-qubits": 4})
        self.assertEqual(self.assert_server(single),
                         self.assert_server(multiple, patches=4))

    def test_worker_summary_does_not_invent_a_framework_selected_total(self):
        working = self.case_directory()
        result = self.invoke(*generation_options(**{"logical-qubits": 3}),
                             cwd=working)
        self.assertIn(
            "Workers per graph: framework-selected (total not specified).",
            result.stdout)
        document = yaml.safe_load((working / OUTPUT_FILENAME).read_text())
        self.assert_server(document, patches=3)
        for decoder in document["decoders"]:
            self.assertNotIn("max_in_flight", decoder["task_graph"]["nodes"][0])

    def test_workers_per_graph_apply_to_every_patch_not_inner_decoders(self):
        for workers in (1, 4):
            with self.subTest(workers=workers):
                working = self.case_directory()
                result = self.invoke(
                    *generation_options(**{
                        "logical-qubits": 3,
                        "workers-per-graph": workers,
                    }),
                    cwd=working)
                self.assertIn(
                    f"Workers per graph: {workers}; graphs: 3; total "
                    f"configured workers: {workers * 3} (excluding coordinators).",
                    result.stdout)
                document = yaml.safe_load(
                    (working / OUTPUT_FILENAME).read_text())
                self.assert_server(document, patches=3, workers=workers)
                for decoder in document["decoders"]:
                    node = decoder["task_graph"]["nodes"][0]
                    self.assertEqual(node.get("max_in_flight", 1), workers)
                    self.assertNotIn("workers_per_graph", node)
                    self.assertNotIn("async_num_workers", node)
                    self.assertEqual(node["inner"]["type"], "single_error_lut")

    def test_total_worker_overflow_is_rejected_without_output(self):
        working = self.case_directory()
        result = self.invoke(
            *generation_options(**{
                "logical-qubits": 2,
                "workers-per-graph": "18446744073709551615",
            }),
            cwd=working,
            success=False)
        self.assertIn("worker", result.stderr.lower())
        self.assertEqual(list(working.iterdir()), [])

    def test_pymatching_profile_default_and_explicit_policies_for_every_patch(
            self):
        for requested, expected in ((None, "smallest_weight"),
                                    ("smallest_weight", "smallest_weight"),
                                    ("independent",
                                     "independent"), ("disallow", "disallow")):
            with self.subTest(policy=requested):
                working = self.case_directory()
                arguments = generation_options(decoder="pymatching",
                                               **{"logical-qubits": 2})
                if requested is not None:
                    arguments.append(
                        f"--decoder-arg=merge_strategy={requested}")
                result = run_program(GENERATOR, *arguments, cwd=working)
                if (result.returncode != 0 and
                        "decoder is not registered in this QEC build: pymatching"
                        in result.stderr):
                    self.assertEqual(list(working.iterdir()), [])
                    self.skipTest("PyMatching plugin is not installed")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual({path.name for path in working.iterdir()},
                                 {OUTPUT_FILENAME})
                document = yaml.safe_load(
                    (working / OUTPUT_FILENAME).read_text())
                self.assert_server(document, patches=2)
                for decoder in document["decoders"]:
                    inner = decoder["task_graph"]["nodes"][0]["inner"]
                    self.assertEqual(inner["type"], "pymatching")
                    self.assertEqual(inner["merge_strategy"], expected)

    def test_bad_decoder_arguments_and_backend_name_leave_no_output(self):
        for arguments in (
            ("--decoder-arg=lut_error_depth=1junk",),
            ("--decoder-arg=unknown=2",),
            ("--decoder-arg=lut_error_depth=2",
             "--decoder-arg=lut_error_depth=3"),
            ("--decoder-arg=lut_error_depth..child=2",),
            ("--decoder-arg=missing_equals",),
        ):
            with self.subTest(arguments=arguments):
                working = self.case_directory()
                self.invoke(*generation_options(decoder="multi_error_lut"),
                            *arguments,
                            cwd=working,
                            success=False)
                self.assertEqual(list(working.iterdir()), [])
        working = self.case_directory()
        self.invoke(*generation_options(decoder="not_a_registered_decoder"),
                    cwd=working,
                    success=False)
        self.assertEqual(list(working.iterdir()), [])

    def test_incompatible_window_backend_fails_before_model_generation(self):
        working = self.case_directory()
        result = self.invoke(*generation_options(decoder="sliding_window"),
                             "--decoder-arg=inner_decoder_name=multi_error_lut",
                             cwd=working,
                             success=False)
        self.assertIn("no verified Surface Code parallel-window model profile",
                      result.stderr)
        self.assertEqual(list(working.iterdir()), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    GENERATOR = arguments.generator.resolve()
    unittest.main(argv=[sys.argv[0], *remaining])
