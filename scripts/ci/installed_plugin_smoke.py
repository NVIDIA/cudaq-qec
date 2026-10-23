# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check final installed binary plugins and run a CPU decoding-server replay.

Run this against the copied customer install, not against a build tree. The
release image uses a separate validation stage so QA tools are not shipped.
Model/engine execution is deliberately a separate GPU qualification.
"""

import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys

import replay_decoding_server as playback

MISSING_NODE_KIND = "qec_packaging_intentionally_missing_kind"
COMMAND_TIMEOUT_SECONDS = 60
PLUGIN_DIRECTORIES = {
    "decoder": "decoder-plugins",
    "task_graph": "task-graph-plugins"
}


def run_command(command):
    result = subprocess.run(command,
                            check=False,
                            capture_output=True,
                            text=True,
                            timeout=COMMAND_TIMEOUT_SECONDS)
    return {
        "command": [str(item) for item in command],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr
    }


def installed_modules(prefix, manifests, required):
    """Resolve only manifest-named modules below the final install's libdir."""
    modules, providers, destinations = [], set(), set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get(
                "delivery") not in ("native", "wheel"):
            raise ValueError("unsupported installed source manifest")
        entries = manifest.get("modules")
        if not isinstance(entries, list) or not entries:
            raise ValueError("installed source manifest contains no modules")
        for entry in entries:
            kind, relative = entry.get("kind"), entry.get("path")
            if kind not in PLUGIN_DIRECTORIES or not isinstance(relative, str):
                raise ValueError("invalid installed plugin kind/path")
            path = PurePosixPath(relative)
            if (len(path.parts) != 2 or str(path) != relative or
                    path.parts[0] != PLUGIN_DIRECTORIES[kind] or
                    path.name in (".", "..") or path.suffix != ".so"):
                raise ValueError(
                    "installed plugin path escapes its destination")
            provider = entry.get("id")
            if not isinstance(provider, str) or not provider:
                raise ValueError("installed plugin provider ID is missing")
            installed = prefix / "lib" / relative
            if installed.is_symlink() or not installed.is_file():
                raise ValueError(
                    f"installed module is not a regular file: {installed}")
            if installed.resolve().parent != (
                    prefix / "lib" / PLUGIN_DIRECTORIES[kind]).resolve():
                raise ValueError(
                    "installed plugin destination escapes its libdir")
            if relative not in destinations:
                modules.append({
                    "id": provider,
                    "kind": kind,
                    "file": installed,
                    "source_manifest": playback.fingerprint(manifest_path)
                })
                destinations.add(relative)
            providers.add(provider)
    missing = set(required) - providers
    if missing:
        raise ValueError(
            f"required installed providers are missing: {sorted(missing)}")
    return modules


def check_module(module):
    result = {key: value for key, value in module.items() if key != "file"}
    result["installed_file"] = playback.fingerprint(module["file"])
    result["rpath"] = run_command(["patchelf", "--print-rpath", module["file"]])
    rpath = result["rpath"]
    if rpath["returncode"]:
        raise ValueError(
            f"cannot inspect installed module RPATH: {module['file']}")
    entries = rpath["stdout"].strip().split(":")
    if "$ORIGIN/.." not in entries or any(
            entry != "$ORIGIN" and not entry.startswith("$ORIGIN/")
            for entry in entries):
        raise ValueError(
            f"installed module has nonrelative/missing RPATH: {entries}")
    result["dependencies"] = run_command(["ldd", module["file"]])
    dependencies = result["dependencies"]
    if dependencies["returncode"] or "not found" in dependencies["stdout"]:
        raise ValueError(
            f"unresolved installed module dependency: {module['file']}")
    return result


def smoke(args):
    manifests = args.manifest or sorted(
        (args.prefix / "lib/external-plugins").glob("*/source-manifest.json"))
    modules = installed_modules(args.prefix, manifests, args.require_provider)
    task_graph = any(module["kind"] == "task_graph" for module in modules)
    if task_graph and not args.require_node_kind:
        raise ValueError(
            "task-graph deliveries require explicit expected node kinds")
    if args.require_node_kind and (args.registration_probe is None or
                                   not args.registration_probe.is_file()):
        raise ValueError(
            "expected node kinds require an installed registration probe")
    args.output.mkdir(exist_ok=False)
    report = {"status": "failed", "scope": "installed artifact, no-GPU smoke"}
    try:
        report["modules"] = [check_module(module) for module in modules]
        if args.require_node_kind:
            positive = run_command([args.registration_probe] +
                                   args.require_node_kind)
            report["registration"] = positive
            if positive["returncode"]:
                raise ValueError(
                    "installed node registration/schema check failed")
            negative = run_command([args.registration_probe, MISSING_NODE_KIND])
            report["missing_kind"] = negative
            if negative["returncode"] != 1:
                raise ValueError(
                    "missing-kind registration probe did not reject the kind")
        replay_args = playback.arguments([
            f"--server={args.prefix / 'bin/decoding_server'}",
            f"--playback={args.prefix / 'bin/playback-emulator'}",
            f"--config={args.fixtures / 'pymatching-smoke.yaml'}",
            f"--schedule={args.fixtures / 'pymatching-smoke.schedule'}",
            f"--expected={args.fixtures / 'pymatching-smoke.expected.csv'}",
            f"--output={args.output / 'pymatching-replay'}",
            f"--transport={args.transport}", "--server-arg=--host-loop=threaded"
        ])
        report["pymatching"] = playback.replay(replay_args)
        report["status"] = "passed"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) +
                                                  "\n",
                                                  encoding="utf-8")
    return report


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prefix", "output"):
        parser.add_argument(f"--{name}",
                            type=lambda value: Path(value).resolve(),
                            required=True)
    parser.add_argument("--manifest",
                        type=lambda value: Path(value).resolve(),
                        action="append",
                        default=[])
    parser.add_argument("--require-provider", action="append", default=[])
    parser.add_argument("--require-node-kind", action="append", default=[])
    parser.add_argument("--registration-probe",
                        type=lambda value: Path(value).resolve())
    parser.add_argument("--fixtures",
                        type=lambda value: Path(value).resolve(),
                        default=Path(__file__).resolve().parent / "fixtures")
    parser.add_argument("--transport", default="udp")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        result = smoke(arguments())
        print(
            f"Passed installed smoke: {len(result['modules'])} external modules, "
            f"{result['pymatching']['result']['shots']} CPU shots")
    except (OSError, ValueError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"Installed smoke failed: {error}", file=sys.stderr)
        sys.exit(1)
