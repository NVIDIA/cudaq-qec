# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Replay completed shots through installed server/client processes over UDP.

Fixtures and model bundles are external inputs. This correctness smoke test does
not generate circuits, implement a decoder, or qualify performance or accuracy.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from playback_result_checks import load_expected, validate_records

READY_MARKER = "QEC_DECODING_SERVER_READY"
STARTUP_POLL_SECONDS = 0.05
SHUTDOWN_TIMEOUT_SECONDS = 10
OWNED_SERVER_OPTIONS = {"--config", "--port", "--transport"}
RECORDED_ENVIRONMENT = ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
                        "CUDAQ_REALTIME_BRIDGE_LIB")


def positive_seconds(value):
    result = float(value)
    if not 0 < result < float("inf"):
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return result


def source_argument(value):
    identity, separator, filename = value.partition(":")
    if not separator or re.fullmatch(r"[0-9]+",
                                     identity) is None or not filename:
        raise argparse.ArgumentTypeError("source must be ID:PATH")
    return int(identity), Path(filename).resolve()


def parse_ready(log, decoder_ids):
    """Return per-decoder ports only after the server publishes all rings."""
    lines = [line for line in log.splitlines() if line.startswith(READY_MARKER)]
    if not lines:
        return None
    if len(lines) != 1:
        raise ValueError("server published more than one READY record")
    ports = {}
    for decoder, port in re.findall(r"\bring(\d+)=(\d+)\b", lines[0]):
        decoder, port = int(decoder), int(port)
        if decoder in ports or not 0 < port <= 65535:
            raise ValueError("invalid or duplicate server ring endpoint")
        ports[decoder] = port
    if set(ports) != set(decoder_ids):
        raise ValueError(
            "server ring IDs do not match the expected decoder IDs")
    if len(set(ports.values())) != len(ports):
        raise ValueError("server rings share a UDP port")
    return ports


def validate_server_counts(log, requests_per_decoder):
    """Cross-check client RPC counts against the server's shutdown counters."""
    rings = {}
    for decoder, count in re.findall(
            r"^QEC_DECODING_SERVER_RING decoder=(\d+) dispatched=(\d+)$", log,
            re.MULTILINE):
        if decoder in rings:
            raise ValueError("duplicate server ring counter")
        rings[decoder] = int(count)
    expected = {str(key): value for key, value in requests_per_decoder.items()}
    totals = re.findall(r"^QEC_DECODING_SERVER_DISPATCHED count=(\d+)$", log,
                        re.MULTILINE)
    if rings != expected or len(totals) != 1 or int(totals[0]) != sum(
            expected.values()):
        raise ValueError("server/client dispatched request counts disagree")
    return {"requests_per_decoder": rings, "requests": int(totals[0])}


def wait_ready(process, log_path, decoder_ids, timeout):
    deadline = time.monotonic() + timeout
    while True:
        ports = parse_ready(log_path.read_text(encoding="utf-8"), decoder_ids)
        if ports is not None:
            if process.poll() is not None:
                raise RuntimeError("server exited immediately after READY")
            return ports
        if process.poll() is not None:
            raise RuntimeError("server exited before READY; see server.log")
        if time.monotonic() >= deadline:
            raise TimeoutError("server readiness timed out; see server.log")
        time.sleep(STARTUP_POLL_SECONDS)


def stop_server(process):
    """Terminate only the child owned by this replay; never kill by name."""
    if process.poll() is None:
        process.terminate()
    try:
        return process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise TimeoutError(
            "server did not shut down gracefully; see server.log")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest()}


def replay(args):
    for name in ("server", "playback", "config", "schedule", "expected"):
        path = getattr(args, name)
        if not path.is_file():
            raise ValueError(f"{name} is not a regular file: {path}")
    for name in ("server", "playback"):
        if not os.access(getattr(args, name), os.X_OK):
            raise ValueError(f"{name} is not executable")
    for option in args.server_arg:
        if not option.startswith("--") or option.split(
                "=", 1)[0] in OWNED_SERVER_OPTIONS:
            raise ValueError(
                "server arguments must not override config/port/transport")
    if len({identity for identity, _ in args.source}) != len(args.source):
        raise ValueError("source IDs must be unique")
    for _, path in args.source:
        if not path.is_file():
            raise ValueError(f"source is not a regular file: {path}")
    expected = load_expected(args.expected)
    args.output.mkdir(exist_ok=False)
    report = {
        "scope": "installed server/client UDP correctness replay",
        "status": "failed",
        "inputs": {
            name: fingerprint(getattr(args, name))
            for name in ("server", "playback", "config", "schedule", "expected")
        },
        "environment": {
            name: os.environ[name]
            for name in RECORDED_ENVIRONMENT
            if name in os.environ
        },
        "require_final": args.require_final
    }
    report["sources"] = {
        str(identity): fingerprint(path) for identity, path in args.source
    }
    server_log = args.output / "server.log"
    # Keep the server alive for the bounded startup and client run. A caller
    # can deliberately override this lifetime through --server-arg=--timeout=N.
    lifetime = math.ceil(args.startup_timeout + args.client_timeout +
                         SHUTDOWN_TIMEOUT_SECONDS)
    server_command = [
        str(args.server), f"--config={args.config}", "--port=0",
        f"--timeout={lifetime}"
    ]
    if args.transport is not None:
        server_command.append(f"--transport={args.transport}")
    server_command.extend(args.server_arg)
    report["server_command"] = server_command
    environment = dict(os.environ, QEC_DECODING_SERVER_STATS="1")
    try:
        with server_log.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(server_command,
                                       stdout=log,
                                       stderr=subprocess.STDOUT,
                                       env=environment)
            try:
                ports = wait_ready(process, server_log, expected,
                                   args.startup_timeout)
                client_command = [
                    str(args.playback), f"--config={args.config}",
                    f"--schedule={args.schedule}", "--backend=udp",
                    f"--tick={args.tick}", f"--udp-timeout={args.udp_timeout}",
                    f"--out={args.output / 'events.csv'}"
                ]
                client_command.extend(
                    f"--udp-endpoint={decoder}:127.0.0.1:{port}"
                    for decoder, port in sorted(ports.items()))
                client_command.extend(f"--source={identity}:{path}"
                                      for identity, path in args.source)
                report["client_command"] = client_command
                with (args.output / "client.log").open(
                        "w", encoding="utf-8") as client_log:
                    subprocess.run(client_command,
                                   stdout=client_log,
                                   stderr=subprocess.STDOUT,
                                   check=True,
                                   timeout=args.client_timeout)
                with (args.output / "events.csv").open(
                        newline="", encoding="utf-8") as data:
                    report["result"] = validate_records(
                        list(csv.DictReader(data)), expected,
                        args.require_final)
            finally:
                report["server_returncode"] = stop_server(process)
        if report["server_returncode"] != 0:
            raise RuntimeError("server exited unsuccessfully; see server.log")
        report["server_counts"] = validate_server_counts(
            server_log.read_text(encoding="utf-8"),
            report["result"]["requests_per_decoder"])
        report["events"] = fingerprint(args.output / "events.csv")
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
    for name in ("server", "playback", "config", "schedule", "expected",
                 "output"):
        parser.add_argument(f"--{name}",
                            type=lambda value: Path(value).resolve(),
                            required=True)
    parser.add_argument(
        "--transport",
        help="UDP bridge override; otherwise use the server/YAML default")
    parser.add_argument("--server-arg",
                        action="append",
                        default=[],
                        help="repeat as --server-arg=--option=value")
    parser.add_argument("--source",
                        type=source_argument,
                        action="append",
                        default=[],
                        help="repeat ID:PATH")
    parser.add_argument("--udp-timeout", default="30s")
    parser.add_argument("--tick", default="1ns")
    parser.add_argument("--startup-timeout",
                        type=positive_seconds,
                        default=90.0)
    parser.add_argument("--client-timeout",
                        type=positive_seconds,
                        default=600.0)
    parser.add_argument("--require-final", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        result = replay(arguments())
        print(f"Passed: {result['result']['shots']} shots, "
              f"{result['result']['requests']} RPCs")
    except (OSError, ValueError, RuntimeError,
            subprocess.SubprocessError) as error:
        print(f"Replay failed: {error}", file=sys.stderr)
        sys.exit(1)
