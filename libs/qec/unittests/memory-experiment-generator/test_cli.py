#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Black-box server-YAML generation tests; no task graph is executed."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
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
DECODING_SERVER = None
EXPECT_PUBLIC_API = "auto"
CONFIG_ENV = "CUDAQ_QEC_DECODER_CONFIG"
SERVER_UNAVAILABLE = (
    "server YAML generation requires the public parallel_window configuration "
    "and named-model/readout support; this QEC build does not provide it")


def generation_options(**overrides):
    values = {
        "distance": 3,
        "rounds": 26,
        "basis": "Z",
        "decoder": "single_error_lut",
    }
    values.update(overrides)
    return [f"--{key}={value}" for key, value in values.items()]


def run_program(program,
                *arguments,
                cwd,
                config_path=None,
                timeout=90,
                preexec_fn=None):
    environment = os.environ.copy()
    environment.pop(CONFIG_ENV, None)
    environment.pop("QEC_DECODING_SERVER_CPU_DMA_LATENCY_US", None)
    if config_path is not None:
        environment[CONFIG_ENV] = str(config_path)
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
    )


class GeneratorCliTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory(
                prefix="memory-experiment-api-probe-") as temporary:
            working = Path(temporary)
            result = run_program(GENERATOR, *generation_options(), cwd=working)
            cls.public_api = result.returncode == 0
            cls.probe_stderr = result.stderr
            expected = {"server.yaml"} if cls.public_api else set()
            if {path.name for path in working.iterdir()} != expected:
                raise AssertionError(
                    "capability probe created unexpected files")
            if not cls.public_api and SERVER_UNAVAILABLE not in result.stderr:
                raise AssertionError(
                    f"unexpected capability probe failure:\n{result.stderr}")
            if EXPECT_PUBLIC_API == "available" and not cls.public_api:
                raise AssertionError("this test run requires the public API")
            if EXPECT_PUBLIC_API == "unavailable" and cls.public_api:
                raise AssertionError("this test run expects the old public API")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="memory-experiment-cli-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sequence = 0

    def case_directory(self):
        self.sequence += 1
        working = self.root / f"case {self.sequence}"
        working.mkdir()
        return working

    def invoke(self, *arguments, success=True, cwd=None, config_path=None):
        result = run_program(GENERATOR,
                             *arguments,
                             cwd=cwd or self.root,
                             config_path=config_path)
        detail = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        if success:
            self.assertEqual(result.returncode, 0, detail)
        else:
            self.assertNotEqual(result.returncode, 0, detail)
        return result

    def require_public_api(self):
        if not self.public_api:
            self.skipTest("linked public parallel-window API is unavailable")

    def generate(self, *, extra=(), config_path=None, **overrides):
        self.require_public_api()
        working = self.case_directory()
        output = Path(config_path) if config_path else Path("server.yaml")
        if not output.is_absolute():
            output = working / output
        self.invoke(*generation_options(**overrides),
                    *extra,
                    cwd=working,
                    config_path=config_path)
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
                      gap=None):
        self.assertEqual(set(document),
                         {"data_artifacts", "decoders", "syndrome_mapping"})
        self.assertEqual(set(document["data_artifacts"]), {"memory_model"})
        model = document["data_artifacts"]["memory_model"]["dem_chunks"]
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
            self.assertEqual(decoder["dem_chunks"], "memory_model")
            graph = decoder["task_graph"]
            self.assertEqual(set(graph), {"nodes", "edges"})
            self.assertEqual(graph["edges"], [
                {
                    "from": "measurements",
                    "to": "parallel_memory"
                },
                {
                    "from": "parallel_memory",
                    "to": "output"
                },
            ])
            self.assertEqual(len(graph["nodes"]), 1)
            node = graph["nodes"][0]
            self.assertEqual(node["id"], "parallel_memory")
            self.assertEqual(node["type"], "parallel_window")
            self.assertEqual(node["output"], "corrections")
            self.assertEqual(node["commit_rounds"],
                             distance if commit is None else commit)
            self.assertEqual(node["buffer_rounds"],
                             distance if buffer is None else buffer)
            if gap is None:
                self.assertNotIn("gap_rounds", node)
            else:
                self.assertEqual(node["gap_rounds"], gap)
            self.assertNotIn("async_num_workers", graph)
            self.assertNotIn("async_worker_cpus", graph)
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

    def test_help_lists_inputs_without_an_output_or_config_flag(self):
        result = self.invoke("--help")
        for option in ("--distance", "--rounds", "--basis", "--decoder",
                       "--commit-rounds", "--buffer-rounds", "--decoder-arg"):
            self.assertIn(option, result.stdout)
        for removed in ("--output", "--config", "--batch-rounds",
                        "--replay-shots", "--seed", "--validate-bundle"):
            self.assertNotIn(removed, result.stdout)
        self.assertIn("server.yaml", result.stdout)
        self.assertIn(CONFIG_ENV, result.stdout)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_public_api_failure_is_explicit_and_leaves_no_output(self):
        if self.public_api:
            self.assertNotIn(SERVER_UNAVAILABLE, self.probe_stderr)
        else:
            self.assertIn(SERVER_UNAVAILABLE, self.probe_stderr)

    def test_removed_modes_and_file_inputs_are_rejected_without_output(self):
        for name in ("output", "batch-rounds", "replay-shots", "seed",
                     "validate-bundle", "model-only", "request",
                     "check-request", "config", "parameters", "input"):
            with self.subTest(option=name):
                working = self.case_directory()
                unopened = working / "must-not-read.json"
                result = self.invoke(*generation_options(),
                                     f"--{name}={unopened}",
                                     cwd=working,
                                     success=False)
                self.assertIn(f"unknown option: --{name}", result.stderr)
                self.assertEqual(list(working.iterdir()), [])

    def test_only_experiment_and_decoder_inputs_are_required(self):
        for required in ("distance", "rounds", "basis", "decoder"):
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
            "basis": ("Y",),
            "logical-qubits": ("0", "-1", "1.5"),
            "error-rate": ("0", "1", "nan", "inf", "-0.1", "0.1junk"),
            "commit-rounds": ("0", "-1", "1.5"),
            "buffer-rounds": ("0", "-1", "1.5"),
            "gap-rounds": ("0", "-1", "1.5"),
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
                      ("--distance", "3")):
            with self.subTest(extra=extra):
                working = self.case_directory()
                self.invoke(*generation_options(),
                            *extra,
                            cwd=working,
                            success=False)
                self.assertEqual(list(working.iterdir()), [])

    def test_default_path_never_overwrites_any_existing_entry(self):
        self.require_public_api()
        for kind in ("file", "directory", "file symlink", "directory symlink",
                     "dangling symlink"):
            with self.subTest(kind=kind):
                working = self.case_directory()
                output = working / "server.yaml"
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

    def test_shared_environment_overrides_default_and_requires_existing_parent(
            self):
        self.require_public_api()
        destination = self.root / "deployment config.yaml"
        output, document = self.generate(config_path=destination)
        self.assertEqual(output, destination)
        self.assert_server(document)
        relative, document = self.generate(config_path="relative config.yaml")
        self.assertEqual(relative.name, "relative config.yaml")
        self.assert_server(document)
        _, document = self.generate(config_path="")
        self.assert_server(document)
        working = self.case_directory()
        missing = self.root / "missing parent" / "server.yaml"
        self.invoke(*generation_options(),
                    cwd=working,
                    config_path=missing,
                    success=False)
        self.assertEqual(list(working.iterdir()), [])
        self.assertFalse(missing.parent.exists())

    def test_concurrent_publication_has_one_winner_without_temporary_residue(
            self):
        self.require_public_api()
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
                         {"server.yaml"})
        self.assert_server(yaml.safe_load(
            (working / "server.yaml").read_text()))

    def test_justin_reference_in_both_bases_emits_only_default_server_yaml(
            self):
        for basis in ("X", "Z"):
            with self.subTest(basis=basis):
                output, document = self.generate(basis=basis)
                self.assertEqual(output.name, "server.yaml")
                model = self.assert_server(document)
                self.assertEqual(26 * model["measurements_per_round"] + 9, 217)

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

    def test_schema_typed_nested_decoder_arguments_are_preserved(self):
        _, document = self.generate(
            decoder="sliding_window",
            extra=(
                "--decoder-arg=inner_decoder_params.lut_error_depth=2",
                "--decoder-arg=window_size=6",
                "--decoder-arg=step_size=3",
                "--decoder-arg=num_syndromes_per_round=8",
                "--decoder-arg=num_boundary_syndromes=0",
                "--decoder-arg=straddle_start_round=true",
                "--decoder-arg=straddle_end_round=false",
                "--decoder-arg=inner_decoder_name=multi_error_lut",
            ))
        self.assert_server(document)
        inner = document["decoders"][0]["task_graph"]["nodes"][0]["inner"]
        self.assertEqual(inner["type"], "sliding_window")
        self.assertIs(type(inner["window_size"]), int)
        self.assertIs(inner["straddle_start_round"], True)
        self.assertIs(inner["straddle_end_round"], False)
        self.assertEqual(inner["inner_decoder_name"], "multi_error_lut")
        self.assertEqual(inner["inner_decoder_params"], {"lut_error_depth": 2})

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


class DecodingServerConfigPathTest(unittest.TestCase):

    def setUp(self):
        if DECODING_SERVER is None:
            self.skipTest("decoding_server target was not provided")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="decoding-server-config-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def invoke(self, *arguments, config_path=None):
        result = run_program(DECODING_SERVER,
                             *arguments,
                             cwd=self.root,
                             config_path=config_path,
                             timeout=15)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result.stderr

    def test_missing_and_malformed_default_stop_before_transport(self):
        output = self.root / "server.yaml"
        error = self.invoke()
        self.assertIn("cannot open config file", error)
        self.assertIn(str(output), error)
        self.assertEqual(list(self.root.iterdir()), [])
        output.write_text("decoders: [\n")
        error = self.invoke()
        self.assertIn("failed to parse", error)
        self.assertIn(str(output), error)
        self.assertEqual(output.read_text(), "decoders: [\n")

    def test_shared_environment_and_explicit_config_precedence(self):
        default = self.root / "server.yaml"
        default.write_text("decoders: [\n")
        environment_path = self.root / "environment.yaml"
        error = self.invoke(config_path=environment_path)
        self.assertIn("cannot open config file", error)
        self.assertIn(str(environment_path), error)
        environment_path.write_text("decoders: [\n")
        error = self.invoke(config_path=environment_path)
        self.assertIn("failed to parse", error)
        self.assertIn(str(environment_path), error)
        explicit = self.root / "explicit.yaml"
        error = self.invoke(f"--config={explicit}",
                            config_path=environment_path)
        self.assertIn("cannot open config file", error)
        self.assertIn(str(explicit), error)
        explicit.write_text("decoders: [\n")
        error = self.invoke(f"--config={explicit}",
                            config_path=self.root / "missing.yaml")
        self.assertIn("failed to parse", error)
        self.assertIn(str(explicit), error)
        self.assertEqual({path.name for path in self.root.iterdir()},
                         {"server.yaml", "environment.yaml", "explicit.yaml"})

    def test_explicit_empty_config_is_rejected_before_file_loading(self):
        error = self.invoke("--config=")
        self.assertIn("--config", error)
        self.assertNotIn("cannot open config file", error)
        self.assertEqual(list(self.root.iterdir()), [])

    @unittest.skipUnless(os.name == "posix", "requires POSIX directory removal")
    def test_removed_working_directory_does_not_break_help_or_explicit_config(
            self):
        explicit = self.root / "explicit.yaml"
        explicit.write_text("decoders: [\n")
        cases = ((["--help"], "Usage:"), ([f"--config={explicit}"],
                                          "failed to parse"),
                 ([], "cannot resolve default configuration"))
        for index, (arguments, diagnostic) in enumerate(cases):
            with self.subTest(arguments=arguments):
                working = self.root / f"removed {index}"
                working.mkdir()
                # The single child removes only its own empty CWD after
                # subprocess has entered it, before exec of the absolute tool.
                result = run_program(DECODING_SERVER,
                                     *arguments,
                                     cwd=working,
                                     timeout=15,
                                     preexec_fn=lambda: os.rmdir(working))
                self.assertGreaterEqual(result.returncode, 0, result.stderr)
                self.assertIn(diagnostic, result.stdout + result.stderr)
                self.assertFalse(working.exists())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generator", type=Path, required=True)
    parser.add_argument("--decoding-server", type=Path)
    parser.add_argument("--expect-public-api",
                        choices=("auto", "available", "unavailable"),
                        default="auto")
    arguments, remaining = parser.parse_known_args()
    GENERATOR = arguments.generator.resolve()
    DECODING_SERVER = (arguments.decoding_server.resolve()
                       if arguments.decoding_server else None)
    EXPECT_PUBLIC_API = arguments.expect_public_api
    unittest.main(argv=[sys.argv[0], *remaining])
