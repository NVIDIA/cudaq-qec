# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
# SPDX-License-Identifier: Apache-2.0
"""Run one decoder configuration against the prebuilt Docker server.

Use either --schedule for packaged playback or --client followed by a built
CUDA-Q client and its arguments. The client runs on the Docker host.
"""

import argparse
import csv
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid


def docker(*args, check=True):
    result = subprocess.run(["docker", *args],
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if check and result.returncode:
        raise RuntimeError("docker %s failed:\n%s%s" %
                           (args[0], result.stdout, result.stderr))
    return result


def wait_for_server(name, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        log = docker("logs", name)
        text = log.stdout + log.stderr
        ready = re.search(r"QEC_DECODING_SERVER_READY[^\n]*", text)
        if ready:
            rings = re.findall(r"\bring(\d+)=(\d+)", ready.group())
            if len(rings) != 1 or rings[0][0] != "0":
                raise RuntimeError(
                    "This example requires exactly one host-dispatch "
                    "decoder with id 0; got: " + ready.group())
            return rings[0][1]
        running = docker("inspect", "--format", "{{.State.Running}}", name)
        if running.stdout.strip() != "true":
            raise RuntimeError("Server exited before readiness:\n" + text)
        time.sleep(0.25)
    raise RuntimeError(
        "Server readiness timed out. Inspect the saved server log; "
        "model initialization may require a longer --startup-timeout.")


def check_playback(path, expected_shots, expect_no_errors):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError("Playback returned no records")
    for row in rows:
        expected_status = "100" if row["op"] == "stream" else "0"
        if row["status"] != expected_status or row["dispatched"] != "1":
            raise RuntimeError("Incomplete or failed playback event: " +
                               str(row))
    reads = [row for row in rows if row["op"] == "get_corrections"]
    if len(reads) != expected_shots or any(
            row["read_completed"] != "1" or not row["correction_bits"]
            for row in reads):
        raise RuntimeError("Expected %d completed corrections, received %d" %
                           (expected_shots,
                            sum(row["read_completed"] == "1" for row in reads)))
    errors = sum(row["correction_mismatch"] == "1" for row in reads)
    if expect_no_errors and errors:
        raise RuntimeError(
            "Unexpected correction mismatch in connectivity check")
    print("Completed shots: %d; logical mismatches: %d; fraction: %.6g" %
          (len(reads), errors, errors / len(reads)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--writable", action="store_true")
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument("--client-timeout", type=int, default=600)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--expect-no-errors", action="store_true")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--schedule", type=Path)
    inputs.add_argument("--client", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Run this example on a Linux Docker host")
    if min(args.startup_timeout, args.client_timeout, args.shots) <= 0:
        parser.error("Timeouts and shots must be positive")
    work = args.work_dir.resolve(strict=True)
    config = args.config.resolve(strict=True)
    if not config.is_relative_to(work):
        parser.error("Configuration must be inside --work-dir")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    server_log = output.with_suffix(output.suffix + ".server.log")
    if output.exists() or server_log.exists():
        parser.error(
            "Output or server log already exists; choose a new --output")
    if args.schedule:
        schedule = args.schedule.resolve(strict=True)
        if not schedule.is_relative_to(work):
            parser.error("Schedule must be inside --work-dir")
    elif not args.client:
        parser.error("--client requires an executable and its arguments")

    # Preserve absolute artifact paths generated by the SDK. All inputs must
    # be in this directory tree on the same Linux Docker host.
    mount = "type=bind,src=%s,dst=%s" % (work, work)
    if not args.writable:
        mount += ",readonly"
    name = "qec-guide-" + uuid.uuid4().hex[:12]
    command = [
        "run", "-d", "--name", name, "--network", "host", "--mount", mount,
        "--workdir",
        str(work), "--env", "QEC_DECODING_SERVER_STATS=1"
    ]
    if args.gpu:
        command += ["--gpus", "all"]
    command += [
        "--entrypoint", "decoding_server", args.image,
        "--config=" + str(config), "--transport=udp", "--port=0",
        "--timeout=" + str(args.startup_timeout + args.client_timeout)
    ]
    try:
        docker(*command)
    except RuntimeError:
        docker("rm", "--force", name, check=False)
        raise
    try:
        print("Waiting for server readiness...", flush=True)
        port = wait_for_server(name, args.startup_timeout)
        print("Server ready on UDP port " + port, flush=True)
        if args.schedule:
            # Only the server needs a GPU. Playback consumes immutable inputs.
            client = [
                "docker", "run", "--rm", "--name", name + "-playback",
                "--network", "host", "--mount",
                "type=bind,src=%s,dst=%s,readonly" % (work, work), "--workdir",
                str(work), "--entrypoint", "playback-emulator", args.image,
                "--schedule=" + str(schedule), "--backend=udp",
                "--udp-endpoint=0:127.0.0.1:" + port, "--tick=1ms"
            ]
        else:
            client = args.client
        env = dict(os.environ,
                   QEC_DECODING_SERVER_PORT=port,
                   QEC_DECODING_SERVER_TRANSPORT="udp")
        with output.open("x") as stream:
            subprocess.run(client,
                           cwd=work,
                           env=env,
                           stdout=stream,
                           check=True,
                           text=True,
                           timeout=args.client_timeout)
        if args.schedule:
            check_playback(output, args.shots, args.expect_no_errors)
        else:
            text = output.read_text()
            if not re.search(r"^CQR service dispatch count: 0\s*$", text,
                             re.MULTILINE):
                raise RuntimeError(
                    "Client did not confirm external server routing")
            print("Client completed with external server routing confirmed")
    finally:
        # Keep the stopped container until its final dispatch statistics have
        # been saved, then remove only the container created by this run.
        if args.schedule:
            docker("rm", "--force", name + "-playback", check=False)
        docker("stop", "--time", "10", name, check=False)
        log = docker("logs", name, check=False)
        server_text = log.stdout + log.stderr
        server_log.write_text(server_text)
        docker("rm", "--force", name, check=False)
        print("Saved server log: " + str(server_log), flush=True)
    dispatched = re.search(r"QEC_DECODING_SERVER_DISPATCHED count=(\d+)",
                           server_text)
    if not dispatched or int(dispatched.group(1)) == 0:
        raise RuntimeError("Server did not report dispatched requests")
    stats = re.search(
        r"QEC_DECODING_SERVER_DECODER_STATS id=0 "
        r"decodes=(\d+) enqueues=\d+ corrections=(\d+) resets=\d+ errors=(\d+)",
        server_text)
    if not stats or int(stats.group(1)) != args.shots or int(
            stats.group(2)) < args.shots or int(stats.group(3)) != 0:
        raise RuntimeError(
            "Server statistics did not confirm %d decodes and correction requests "
            "and zero errors; inspect %s" % (args.shots, server_log))
    print("Server dispatched %s requests; saved client output: %s" %
          (dispatched.group(1), output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        sys.exit(1)
