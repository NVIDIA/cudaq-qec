#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Prepare the Ising bundle used by surface_code-4-yaml.

The default mode exports the gated Fast model as a d7/T7/Z/XV CUDA-QX bundle.
The ``d-sparse`` mode generates the measurement mapping for custom exports.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import types

ISING_REPOSITORY = "https://github.com/NVIDIA/Ising-Decoding.git"
ISING_COMMIT = "214839eb190447b4d8d5ed950d912b12d076771b"
HF_REPOSITORY = "nvidia/Ising-Decoder-SurfaceCode-1-Fast"
HF_REVISION = "824f9020ce97e16a88b87a7202c19758fb4723d4"
HF_FILENAME = "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors"

DISTANCE = 7
N_ROUNDS = 7
BASIS = "Z"
CODE_ROTATION = "XV"
P_SPAM = 0.01
REQUIRED_ARTIFACTS = (
    "model.onnx",
    "H_csr.bin",
    "O_csr.bin",
    "priors.bin",
    "metadata.txt",
    "D_sparse.txt",
)


def default_artifacts_dir() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "cudaqx" / "ising" / "fast" / "d7_t7_z_xv"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="download and prepare the built-in Fast model bundle")
    prepare.add_argument(
        "--app",
        required=True,
        type=Path,
        help="path to the built surface_code-4-yaml executable",
    )
    prepare.add_argument(
        "--artifacts-dir",
        type=Path,
        default=default_artifacts_dir(),
        help="installation directory (default: %(default)s)",
    )
    prepare.add_argument("--yes",
                         action="store_true",
                         help="skip the download confirmation")
    prepare.add_argument(
        "--force",
        action="store_true",
        help="replace an existing bundle after preserving it as .backup",
    )
    d_sparse = subparsers.add_parser(
        "d-sparse", help="generate D_sparse.txt for a custom Ising export")
    d_sparse.add_argument("distance", type=int)
    d_sparse.add_argument("n_rounds", type=int)
    d_sparse.add_argument("basis")
    d_sparse.add_argument("code_rotation")
    d_sparse.add_argument("sched",
                          type=Path,
                          help="app output containing cnot_schedX/Z_flat")
    d_sparse.add_argument("out", type=Path)
    d_sparse.add_argument(
        "--ising-code",
        required=True,
        type=Path,
        help="path containing qec/surface_code in an Ising checkout",
    )

    if not argv or argv[0] not in ("prepare", "d-sparse", "-h", "--help"):
        argv = ["prepare", *argv]
    return parser.parse_args(argv)


def display_command(command: list[str | Path]) -> str:
    return shlex.join(str(value) for value in command)


def run(
    command: list[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdout=None,
) -> None:
    print(f"+ {display_command(command)}", flush=True)
    subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=env,
        stdout=stdout,
        check=True,
    )


def confirm_download(artifacts_dir: Path) -> bool:
    print(f"Model: https://huggingface.co/{HF_REPOSITORY}@{HF_REVISION}")
    print(f"Ising source: {ISING_REPOSITORY}@{ISING_COMMIT}")
    print(f"Preset: d={DISTANCE}, T={N_ROUNDS}, basis={BASIS}, "
          f"rotation={CODE_ROTATION}, p_spam={P_SPAM}")
    print(f"Install directory: {artifacts_dir}")
    return input("Download the pinned source and gated weights? [y/N] ").strip(
    ).lower() in ("y", "yes")


def require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required command is not on PATH: {name}")


def require_python_packages() -> None:
    modules = ("beliefmatching", "hydra", "ldpc", "matplotlib", "numpy",
               "omegaconf", "onnx", "pymatching", "safetensors", "scipy",
               "stim", "torch")
    missing = [
        name for name in modules if importlib.util.find_spec(name) is None
    ]
    if missing:
        raise RuntimeError(
            "missing Python packages required by the Ising exporter: " +
            ", ".join(missing))


def sparse_checkout(checkout: Path) -> None:
    commands = (
        ["git", "init", checkout],
        ["git", "-C", checkout, "remote", "add", "origin", ISING_REPOSITORY],
        ["git", "-C", checkout, "sparse-checkout", "init", "--cone"],
        ["git", "-C", checkout, "sparse-checkout", "set", "code", "conf"],
        [
            "git", "-C", checkout, "fetch", "--depth", "1",
            "--filter=blob:none", "origin", ISING_COMMIT
        ],
        ["git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD"],
    )
    for command in commands:
        run(command)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if revision != ISING_COMMIT:
        raise RuntimeError(
            f"checked out Ising revision {revision}, expected {ISING_COMMIT}")


def parse_schedule(
        path: Path) -> tuple[list[frozenset[int]], list[frozenset[int]]]:

    def parse_line(line: str) -> list[frozenset[int]]:
        numbers = [
            int(value)
            for value in line.split(":", 1)[1].replace(",", " ").split()
        ]
        supports: dict[int, set[int]] = {}
        for index in range(0, len(numbers), 2):
            supports.setdefault(numbers[index], set()).add(numbers[index + 1])
        return [frozenset(supports[key]) for key in sorted(supports)]

    x_supports = z_supports = None
    with path.open(encoding="utf-8") as schedule:
        for line in schedule:
            if line.startswith("cnot_schedX_flat:"):
                x_supports = parse_line(line)
            elif line.startswith("cnot_schedZ_flat:"):
                z_supports = parse_line(line)
    if not x_supports or not z_supports:
        raise RuntimeError(
            f"{path} must contain cnot_schedX_flat and cnot_schedZ_flat")
    return x_supports, z_supports


def generate_d_sparse(
    distance: int,
    n_rounds: int,
    basis: str,
    code_rotation: str,
    schedule: Path,
    output: Path,
    ising_code: Path,
) -> None:
    basis = basis.upper()
    code_rotation = code_rotation.upper()
    if len(code_rotation) != 2:
        raise RuntimeError("code_rotation must contain two characters, e.g. XV")
    x_supports, z_supports = parse_schedule(schedule)
    num_x, num_z = len(x_supports), len(z_supports)

    if not (ising_code / "qec" / "surface_code").is_dir():
        raise RuntimeError(f"{ising_code} does not contain qec/surface_code")
    sys.path.insert(0, str(ising_code))
    surface_code_package = types.ModuleType("qec.surface_code")
    surface_code_package.__path__ = [str(ising_code / "qec" / "surface_code")]
    surface_code_package.__package__ = "qec.surface_code"
    sys.modules.setdefault("qec.surface_code", surface_code_package)
    from qec.noise_model import NoiseModel
    from qec.surface_code.memory_circuit import MemoryCircuit, SurfaceCode

    surface_code = SurfaceCode(
        distance,
        first_bulk_syndrome_type=code_rotation[0],
        rotated_type=code_rotation[1],
    )
    ising_z = [
        frozenset(index
                  for index in range(surface_code.hz.shape[1])
                  if surface_code.hz[row, index] == 1)
        for row in range(surface_code.hz.shape[0])
    ]
    ising_x = [
        frozenset(index
                  for index in range(surface_code.hx.shape[1])
                  if surface_code.hx[row, index] == 1)
        for row in range(surface_code.hx.shape[0])
    ]

    def permutation(cudaqx, ising, label):
        result = [None] * len(cudaqx)
        used = set()
        for cudaqx_index, support in enumerate(cudaqx):
            matches = [
                row for row, ising_support in enumerate(ising)
                if ising_support == support and row not in used
            ]
            if len(matches) != 1:
                raise RuntimeError(f"{label} ancilla {cudaqx_index} support "
                                   f"{sorted(support)} matches {matches}")
            result[cudaqx_index] = matches[0]
            used.add(matches[0])
        return result

    z_permutation = permutation(z_supports, ising_z, "Z")
    x_permutation = permutation(x_supports, ising_x, "X")
    if x_permutation != list(range(num_x)):
        raise RuntimeError("expected an identity X-ancilla mapping")

    inverse_z = [None] * num_z
    for cudaqx_index, ising_index in enumerate(z_permutation):
        inverse_z[ising_index] = cudaqx_index
    measurements_per_round = num_x + num_z
    ancilla_measurements = measurements_per_round * n_rounds

    def translate(measurement: int) -> int:
        if measurement >= ancilla_measurements:
            return measurement
        round_index, offset = divmod(measurement, measurements_per_round)
        if offset < num_x:
            return measurements_per_round * round_index + offset
        return (measurements_per_round * round_index + num_x +
                inverse_z[offset - num_x])

    default_noise = {
        "p_prep_X": 0.002,
        "p_prep_Z": 0.002,
        "p_meas_X": 0.002,
        "p_meas_Z": 0.002,
        "p_idle_cnot_X": 0.001,
        "p_idle_cnot_Y": 0.001,
        "p_idle_cnot_Z": 0.001,
        "p_idle_spam_X": 0.001996,
        "p_idle_spam_Y": 0.001996,
        "p_idle_spam_Z": 0.001996,
        "p_cnot_IX": 0.0002,
        "p_cnot_IY": 0.0002,
        "p_cnot_IZ": 0.0002,
        "p_cnot_XI": 0.0002,
        "p_cnot_XX": 0.0002,
        "p_cnot_XY": 0.0002,
        "p_cnot_XZ": 0.0002,
        "p_cnot_YI": 0.0002,
        "p_cnot_YX": 0.0002,
        "p_cnot_YY": 0.0002,
        "p_cnot_YZ": 0.0002,
        "p_cnot_ZI": 0.0002,
        "p_cnot_ZX": 0.0002,
        "p_cnot_ZY": 0.0002,
        "p_cnot_ZZ": 0.0002,
    }
    noise_model = NoiseModel(**default_noise)
    max_probability = float(noise_model.get_max_probability())
    circuit = MemoryCircuit(
        distance=distance,
        idle_error=max_probability,
        sqgate_error=max_probability,
        tqgate_error=max_probability,
        spam_error=(2.0 / 3.0) * max_probability,
        n_rounds=n_rounds,
        basis=basis,
        code_rotation=code_rotation,
        noise_model=noise_model,
        add_boundary_detectors=True,
    )
    circuit.set_error_rates()
    detector_rows = []
    measurement_count = 0
    for instruction in circuit.stim_circuit.flattened():
        if instruction.name in ("M", "MR", "MX", "MZ", "MRX", "MRZ"):
            measurement_count += len(instruction.targets_copy())
        elif instruction.name == "DETECTOR":
            detector_rows.append(
                sorted(measurement_count + target.value
                       for target in instruction.targets_copy()))

    flat_mapping = []
    for row in detector_rows:
        flat_mapping.extend(translate(measurement) for measurement in row)
        flat_mapping.append(-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        " ".join(str(value) for value in flat_mapping) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}: {len(detector_rows)} detectors, "
          f"nmeas={measurement_count}, Z-ancilla permutation "
          f"nontrivial={z_permutation != list(range(num_z))}")


def validate_bundle(staging: Path) -> None:
    for filename in REQUIRED_ARTIFACTS:
        path = staging / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"export did not produce a nonempty {filename}")
    metadata = {}
    for line in (staging /
                 "metadata.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            metadata[key] = value
    expected = {
        "distance": str(DISTANCE),
        "n_rounds": str(N_ROUNDS),
        "basis": BASIS,
        "code_rotation": CODE_ROTATION,
    }
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if metadata.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("exported metadata mismatch: " +
                           ", ".join(mismatches))


def install_bundle(staging: Path, destination: Path, force: bool) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(destination.name + ".backup")
    if destination.exists():
        if not force:
            raise RuntimeError(
                f"artifact directory already exists: {destination}\n"
                "Pass --force to preserve it as .backup and install a new bundle."
            )
        if backup.exists():
            raise RuntimeError(
                f"refusing to replace existing backup directory: {backup}")

    # Copy beside the destination so the final rename is atomic.
    with tempfile.TemporaryDirectory(prefix=f".{destination.name}-",
                                     dir=destination.parent) as temporary:
        prepared = Path(temporary)
        for filename in REQUIRED_ARTIFACTS:
            shutil.copy2(staging / filename, prepared / filename)
        if destination.exists():
            destination.rename(backup)
            print(f"Preserved previous bundle at {backup}")
        prepared.rename(destination)


def prepare(args: argparse.Namespace) -> None:
    app = args.app.expanduser().resolve()
    destination = args.artifacts_dir.expanduser().resolve()
    if not app.is_file() or not os.access(app, os.X_OK):
        raise RuntimeError(f"--app is not an executable file: {app}")
    for command in ("bash", "git", "hf"):
        require_executable(command)
    require_python_packages()
    if destination.exists() and not args.force:
        raise RuntimeError(
            f"artifact directory already exists: {destination}\n"
            "Pass --force to preserve it as .backup and install a new bundle.")
    backup = destination.with_name(destination.name + ".backup")
    if destination.exists() and args.force and backup.exists():
        raise RuntimeError(
            f"refusing to replace existing backup directory: {backup}")

    if not args.yes and not confirm_download(destination):
        print("Cancelled.")
        return

    with tempfile.TemporaryDirectory(prefix="cudaqx-ising-") as temporary:
        work = Path(temporary)
        checkout = work / "Ising-Decoding"
        weights = work / "weights"
        staging = work / "bundle"
        sparse_checkout(checkout)
        weights.mkdir()
        run([
            "hf", "download", HF_REPOSITORY, HF_FILENAME, "--revision",
            HF_REVISION, "--local-dir", weights
        ])
        checkpoint = weights / HF_FILENAME
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError(f"Hugging Face did not produce {checkpoint}")

        export_environment = os.environ.copy()
        export_environment.update({
            "PREDECODER_BASE_OUTPUT_DIR": str(work / "outputs"),
            "PREDECODER_LOG_BASE_DIR": str(work / "logs"),
            "PREDECODER_SAFETENSORS_CHECKPOINT": str(checkpoint),
            "PREDECODER_INFERENCE_MEAS_BASIS": BASIS,
            "PREDECODER_INFERENCE_NUM_WORKERS": "0",
            "PREDECODER_INFERENCE_NUM_SAMPLES": "32",
            "PREDECODER_INFERENCE_LATENCY_SAMPLES": "0",
            "PREDECODER_PYTHON": sys.executable,
            "PYTHON_BIN": sys.executable,
            "ONNX_WORKFLOW": "1",
            "WORKFLOW": "inference",
            "DISTANCE": str(DISTANCE),
            "N_ROUNDS": str(N_ROUNDS),
            "GPUS": "1",
        })
        run(
            ["bash", "code/scripts/local_run.sh"],
            cwd=checkout,
            env=export_environment,
        )
        exported_onnx = checkout / (
            f"predecoder_memory_d{DISTANCE}_T{N_ROUNDS}_{BASIS}.onnx")
        if not exported_onnx.is_file() or exported_onnx.stat().st_size == 0:
            raise RuntimeError(
                "the pinned Ising workflow completed without producing "
                f"{exported_onnx.name}")

        run([
            sys.executable,
            checkout / "code/export/generate_test_data.py",
            "--distance",
            str(DISTANCE),
            "--n-rounds",
            str(N_ROUNDS),
            "--basis",
            BASIS,
            "--code-rotation",
            CODE_ROTATION,
            "--num-samples",
            "1",
            "--output-dir",
            staging,
        ],
            cwd=checkout)
        shutil.copy2(exported_onnx, staging / "model.onnx")

        schedule = work / "schedule.txt"
        app_environment = os.environ.copy()
        app_environment["CUDAQ_DEFAULT_SIMULATOR"] = "stim"
        with schedule.open("w", encoding="utf-8") as output:
            run([
                app,
                "--distance",
                str(DISTANCE),
                "--num_rounds",
                str(N_ROUNDS),
                "--num_shots",
                "1",
                "--p_spam",
                str(P_SPAM),
                "--decoder_type",
                "pymatching",
                "--save_dem",
                work / "decoder.yml",
            ],
                env=app_environment,
                stdout=output)
        generate_d_sparse(
            DISTANCE,
            N_ROUNDS,
            BASIS,
            CODE_ROTATION,
            schedule,
            staging / "D_sparse.txt",
            checkout / "code",
        )
        validate_bundle(staging)
        install_bundle(staging, destination, args.force)
    print(f"Installed the Ising bundle at {destination}")
    print("Run the example with --use-ising.")


def main(argv: list[str]) -> int:
    try:
        args = parse_args(argv)
        if args.command == "d-sparse":
            generate_d_sparse(
                args.distance,
                args.n_rounds,
                args.basis,
                args.code_rotation,
                args.sched,
                args.out,
                args.ising_code,
            )
        else:
            prepare(args)
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
