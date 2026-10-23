#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run one shipped, single-decoder example without a compiler or Python.
set -euo pipefail

# Bounded launch settings for the shipped single-session playback profile.
readonly RPC_SLOT_BYTES=8192
readonly RPC_SLOT_COUNT=8
readonly SERVER_LIFETIME_SECONDS=600
readonly STARTUP_ATTEMPTS=900
readonly STARTUP_POLL_SECONDS=0.1
readonly UDP_TIMEOUT=30s

example=${1:-x-global}
case "$example" in
    x-global|z-global|x-partitioned|z-partitioned) ;;
    *) echo 'Choose x-global, z-global, x-partitioned or z-partitioned.' >&2; exit 2 ;;
esac
asset_id=${QEC_RUNTIME_ASSET_ID:-}
case "$asset_id" in
    ''|*[!A-Za-z0-9_-]*) echo 'This image has no configured Ising runtime assets.' >&2; exit 1 ;;
esac
prefix=${CUDAQX_INSTALL_DIR:-/opt/nvidia/cudaqx}
asset_root="$prefix/share/cudaq-qec/runtime-assets/$asset_id"
config="examples/$example.yaml"
schedule="examples/$example.schedule"
expected="examples/$example.csv"
cd "$asset_root"
for input in manifest.json "$config" "$schedule" "$expected"; do
    test -f "$input" || { echo "Missing installed example data: $input" >&2; exit 1; }
done
output=$(mktemp -d "${TMPDIR:-/tmp}/ising-example-XXXXXX")
printf 'Example: %s; results: %s\n' "$example" "$output"
# Configuration deliberately uses artifact_bundle: bundle relative to this cwd.
# Do not allow an inherited diagnostic override to change the selected example.
unset CUDAQ_QEC_ISING_MATCHING
QEC_DECODING_SERVER_STATS=1 "$prefix/bin/decoding_server" \
    --config="$config" --port=0 --timeout="$SERVER_LIFETIME_SECONDS" --transport=udp \
    --host-loop=threaded --slot-size="$RPC_SLOT_BYTES" --num-slots="$RPC_SLOT_COUNT" \
    > "$output/server.log" 2>&1 &
server_pid=$!
server_status=0
stop_server() {
    if kill -0 "$server_pid" 2>/dev/null; then kill -TERM "$server_pid"; fi
    wait "$server_pid" || server_status=$?
}
trap stop_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
port=''
for ((attempt = 0; attempt < STARTUP_ATTEMPTS; ++attempt)); do
    port=$(sed -n 's/^QEC_DECODING_SERVER_READY .* ring0=\([0-9][0-9]*\).*$/\1/p' "$output/server.log")
    if [[ "$port" =~ ^[0-9]+$ ]]; then break; fi
    kill -0 "$server_pid" 2>/dev/null || { cat "$output/server.log" >&2; exit 1; }
    sleep "$STARTUP_POLL_SECONDS"
done
[[ "$port" =~ ^[0-9]+$ ]] || { echo "Server startup timed out; see $output/server.log" >&2; exit 1; }
"$prefix/bin/playback-emulator" --config="$config" --schedule="$schedule" \
    --backend=udp --udp-endpoint="0:127.0.0.1:$port" \
    --tick=1ns --udp-timeout="$UDP_TIMEOUT" --out="$output/events.csv" \
    > "$output/client.log" 2>&1
stop_server
trap - EXIT HUP INT TERM
test "$server_status" -eq 0 || { echo "Server failed; see $output/server.log" >&2; exit 1; }

# The shipped example profile uses decoder 0 and unquoted binary CSV cells.
# The playback writer's request-vector cells are space-separated, never commas.
# Check the independent oracle as well as individual RPCs: client exit status
# alone does not indicate whether decoding succeeded. Full CI uses the public
# replay harness for timing, vector identity and multi-session validation.
awk -F, '
function fail(message) { print "Example failed: " message > "/dev/stderr"; failed=1; exit 1 }
FILENAME == ARGV[1] {
    sub(/\r$/, "")
    if (FNR == 1) {
        for (i=1; i<=NF; ++i) { if ($i in e) fail("duplicate oracle header"); e[$i]=i }
        if (!("decoder_id" in e) || !("expected" in e)) fail("invalid oracle header")
    } else {
        if ($(e["decoder_id"]) != "0" || $(e["expected"]) !~ /^[01]+$/) fail("invalid oracle row")
        oracle[++shots]=$(e["expected"])
    }
    next
}
FILENAME == ARGV[2] {
    if (FNR == 1) {
        for (i=1; i<=NF; ++i) { if ($i in h) fail("duplicate event header"); h[$i]=i }
        required="decoder_id op status dispatched request_status correction_mismatch read_completed correction_bits"
        count=split(required, names, " ")
        for (i=1; i<=count; ++i) if (!(names[i] in h)) fail("missing event field")
        next
    }
    operation=$(h["op"])
    if ($(h["decoder_id"]) != "0" || $(h["dispatched"]) != "1" || $(h["correction_mismatch"]) != "0") fail("unsuccessful event")
    status=(operation == "stream" || operation == "enqueue_data") ? "100" : "0"
    if ("s" $(h["status"]) != "s" status) fail("unsuccessful event status")
    count=split($(h["request_status"]), statuses, " ")
    if (!count) fail("missing request status")
    for (i=1; i<=count; ++i) if ("s" statuses[i] != "s0") fail("unsuccessful RPC")
    requests+=count
    if (operation == "reset" && state == 0) state=1
    else if (operation == "stream" && state == 1) {}
    else if (operation == "enqueue_data" && state == 1) state=2
    else if (operation == "get_corrections" && state == 2) {
        if ($(h["read_completed"]) != "1" || ++reads > shots || "b" $(h["correction_bits"]) != "b" oracle[reads]) fail("correction differs from oracle")
        state=0
    } else fail("unexpected shot operation")
    next
}
/^QEC_DECODING_SERVER_RING decoder=0 dispatched=/ {
    split($0, fields, "="); ring_requests=fields[3]+0; ++rings
}
/^QEC_DECODING_SERVER_DISPATCHED count=/ {
    split($0, fields, "="); total_requests=fields[2]+0; ++totals
}
END {
    if (failed) exit 1
    if (!shots || reads != shots || state != 0 || rings != 1 || totals != 1 || ring_requests != requests || total_requests != requests) fail("incomplete shots or dispatch counts")
    printf "Validated %d completed shots (%d requests).\n", reads, requests
}' "$expected" "$output/events.csv" "$output/server.log"
