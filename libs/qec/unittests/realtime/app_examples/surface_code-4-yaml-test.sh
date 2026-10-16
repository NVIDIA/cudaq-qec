# ============================================================================ #
# Copyright (c) 2024 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Driver for the surface_code-4-yaml realtime example. It exercises the two
# phases of the app:
#   Phase 1 (generation): --save_dem <cfg> --decoder_type <type> writes a YAML
#                         decoder config.
#   Phase 2 (realtime):   --yaml <cfg> loads that config and decodes; the
#                         decoder is read FROM the file (so --decoder_type is
#                         NOT passed here -- the app rejects --yaml +
#                         --decoder_type).
#
# The example decodes ONE volume of num_rounds rounds (no sliding windows).
#
# For trt_decoder, select exactly one model source explicitly:
#   --generate-identity-onnx
#   --onnx-path <file>
#   --use-ising [--ising-artifacts-dir <dir>]
# The Ising path is deliberately opt-in and is not exercised by CI.
#
# The Ising path is opt-in. To prepare the default d7/T7/Z/XV Fast bundle,
# accept the gated model terms and authenticate with `hf auth login`:
#   https://huggingface.co/nvidia/Ising-Decoder-SurfaceCode-1-Fast
# Then, from this directory, run:
#   python prepare_ising_artifacts.py --app <path>/surface_code-4-yaml
#   bash surface_code-4-yaml-test.sh <path>/surface_code-4-yaml 7 7 \
#       trt_decoder 200 --use-ising
#
# The utility installs under
# ${XDG_CACHE_HOME:-$HOME/.cache}/cudaqx/ising/fast/d7_t7_z_xv. Compatible
# custom exports may instead use --ising-artifacts-dir; use the utility's
# d-sparse subcommand to generate their CUDA-QX measurement mapping.

set -euo pipefail

# Expected args:
#  $1 exe            Path to the surface_code-4-yaml executable
#  $2 distance       Surface code distance (D)
#  $3 num_rounds     Number of measurement rounds (R >= 1; R < D is decodable
#                    but not fault-tolerant -- no multiple-of-D constraint)
#  $4 decoder_type   Decoder(s) to generate: a single type, or a comma list
#                    with one entry per patch (pass --num_logical N in the
#                    extra args). Optional, defaults to pymatching.
#  $5 num_shots      Number of shots (optional, defaults to 200)
#  $6... options     One explicit TRT model-source option, when needed, plus
#                    extra app args for both generation and realtime phases

if [[ $# -lt 3 ]]; then
  echo "Error: Expected at least 3 arguments (got $#)"
  echo "Usage: $0 <exe> <distance> <num_rounds> [decoder_type=pymatching] [num_shots=200]"
  exit 1
fi

EXE=$1
DISTANCE=$2
NUM_ROUNDS=$3
DECODER_TYPE=${4:-pymatching}
NUM_SHOTS=${5:-200}
ONNX_PATH=""
MODEL_SOURCE=""
ISING_ARTIFACTS_DIR=""
EXTRA_APP_ARGS=()

select_model_source() {
  local requested=$1
  if [[ -n "$MODEL_SOURCE" ]]; then
    echo "Error: model sources are mutually exclusive; already selected '$MODEL_SOURCE', got '$requested'"
    exit 1
  fi
  MODEL_SOURCE=$requested
}

if [[ $# -ge 5 ]]; then
  shift 5
else
  shift "$#"
fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate-identity-onnx)
      select_model_source identity
      shift
      ;;
    --onnx-path|--onnx_path)
      if [[ $# -lt 2 ]]; then
        echo "Error: $1 requires a value"
        exit 1
      fi
      select_model_source onnx
      ONNX_PATH=$2
      shift 2
      ;;
    --use-ising)
      select_model_source ising
      shift
      ;;
    --ising-artifacts-dir)
      if [[ $# -lt 2 ]]; then
        echo "Error: --ising-artifacts-dir requires a value"
        exit 1
      fi
      ISING_ARTIFACTS_DIR=$2
      shift 2
      ;;
    *)
      EXTRA_APP_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$ISING_ARTIFACTS_DIR" && "$MODEL_SOURCE" != "ising" ]]; then
  echo "Error: --ising-artifacts-dir does not enable Ising; pass --use-ising explicitly"
  exit 1
fi
if [[ ",$DECODER_TYPE," == *",trt_decoder,"* ]]; then
  if [[ -z "$MODEL_SOURCE" ]]; then
    echo "Error: trt_decoder requires one model source: --generate-identity-onnx, --onnx-path <file>, or --use-ising"
    exit 1
  fi
elif [[ -n "$MODEL_SOURCE" ]]; then
  echo "Error: '$MODEL_SOURCE' is a TensorRT model source, but decoder_type has no trt_decoder entry"
  exit 1
fi

PYTHON_BIN=${PYTHON:-python3}

# The app defaults to one logical patch. The aggregate result sums residual
# errors across patches, so retain the per-patch ceiling when --num_logical
# requests multiple patches.
NUM_LOGICAL=1
for ((i = 0; i < ${#EXTRA_APP_ARGS[@]}; i++)); do
  if [[ "${EXTRA_APP_ARGS[$i]}" == "--num_logical" ]] &&
    ((i + 1 < ${#EXTRA_APP_ARGS[@]})); then
    i=$((i + 1))
    NUM_LOGICAL=${EXTRA_APP_ARGS[$i]}
  fi
done

export CUDAQ_DEFAULT_SIMULATOR=stim

P_SPAM=0.01

# Residual logical-error ceiling: a PREDECLARED correctness bound, set at
# half the d3 UNCORRECTED logical-flip rate (~4% at p_spam=0.01; ~15% at
# d5/T6). A decoder that loads but never corrects reliably exceeds it; any
# working decoder sits well below it. This is a wiring/correctness check, not
# a performance target -- it must never be tightened toward a particular
# decoder's measured rate. Floored at 1 per logical patch so small shot counts
# do not truncate the ceiling to 0, then scaled for the aggregate result.
MAX_NON_ZERO_PER_LOGICAL=$((NUM_SHOTS / 50))
if [[ $MAX_NON_ZERO_PER_LOGICAL -lt 1 ]]; then MAX_NON_ZERO_PER_LOGICAL=1; fi
MAX_NON_ZERO=$((MAX_NON_ZERO_PER_LOGICAL * NUM_LOGICAL))

# Multi-type mode: a comma list binds one decoder type per patch and the
# per-decoder ceilings below replace the aggregate ceiling (which is
# calibrated for ONE patch and would silently tighten N-fold). The same
# predeclared bound applies per decoder; cases must run enough shots that
# working-vs-broken is unambiguous for every entry (>= 1000 for BP-family
# entries, whose working residual sits closest to the bound).
IFS=',' read -r -a DECODER_TYPES <<< "$DECODER_TYPE"
MULTI_TYPE=0
if [[ ${#DECODER_TYPES[@]} -gt 1 ]]; then MULTI_TYPE=1; fi

# Create an isolated working directory and (by default) clean it up on exit.
WORKDIR=$(mktemp -d)
SERVER_PID=""
SERVER_LOG=$WORKDIR/server.log
MISSING_PORT_LOG=$WORKDIR/missing-port.log

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
  fi
}

cleanup() {
  stop_server
  if [[ -z "${KEEP_LOG_FILES:-}" ]]; then
    rm -rf "$WORKDIR"
  else
    echo "KEEP_LOG_FILES set; leaving work dir: $WORKDIR"
  fi
}
trap cleanup EXIT

CONFIG_FILE=$WORKDIR/config.yml
REALTIME_LOG=$WORKDIR/realtime.log

if [[ "$MODEL_SOURCE" == "identity" ]]; then
  ONNX_PATH=$WORKDIR/trt_identity_predecoder.onnx
  SYNDROME_SIZE=$(((DISTANCE * DISTANCE - 1) * NUM_ROUNDS))
  "$PYTHON_BIN" - "$ONNX_PATH" "$SYNDROME_SIZE" <<'PY'
import sys

import onnx
from onnx import TensorProto, helper

output_path = sys.argv[1]
syndrome_size = int(sys.argv[2])

input_info = helper.make_tensor_value_info(
    "input", TensorProto.FLOAT, [1, syndrome_size])
output_info = helper.make_tensor_value_info(
    "output", TensorProto.FLOAT, [1, syndrome_size + 1])
zero = helper.make_node(
    "Constant",
    [],
    ["pre_l"],
    value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 1], [0.0]),
)
concat = helper.make_node("Concat", ["pre_l", "input"], ["output"], axis=1)
graph = helper.make_graph(
    [zero, concat], "trt_identity_predecoder", [input_info], [output_info])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
# IR 9 is sufficient for opset 19 and remains readable by the ONNX checker in
# the CUDA-QX development image.
model.ir_version = 9
onnx.checker.check_model(model)
onnx.save(model, output_path)
PY
fi

echo "=============================================================="
echo "surface_code-4-yaml test"
echo "  exe            = $EXE"
echo "  distance       = $DISTANCE"
echo "  num_rounds     = $NUM_ROUNDS"
echo "  decoder_type   = $DECODER_TYPE"
echo "  num_shots      = $NUM_SHOTS"
echo "  num_logical    = $NUM_LOGICAL"
if [[ -n "$ONNX_PATH" ]]; then
  echo "  onnx_path      = $ONNX_PATH"
fi
if [[ "$MODEL_SOURCE" == "ising" ]]; then
  echo "  model source   = Ising"
  if [[ -n "$ISING_ARTIFACTS_DIR" ]]; then
    echo "  artifacts dir  = $ISING_ARTIFACTS_DIR"
  else
    echo "  artifacts dir  = default CUDA-QX cache"
  fi
fi
if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  echo "  extra args     = ${EXTRA_APP_ARGS[*]}"
fi
echo "  max non-zero   = $MAX_NON_ZERO"
echo "=============================================================="

return_code=0

# -------------------------------------------------------------------------- #
# Phase 1: generation -- write the YAML decoder config.
# -------------------------------------------------------------------------- #
echo ""
echo "=== Phase 1: generate config (--save_dem, --decoder_type $DECODER_TYPE) ==="
GEN_ARGS=(
  --distance "$DISTANCE" \
  --num_rounds "$NUM_ROUNDS" \
  --num_shots "$NUM_SHOTS" \
  --p_spam "$P_SPAM" \
  --decoder_type "$DECODER_TYPE" \
  --save_dem "$CONFIG_FILE"
)
if [[ -n "$ONNX_PATH" ]]; then
  GEN_ARGS+=(--onnx-path "$ONNX_PATH")
elif [[ "$MODEL_SOURCE" == "ising" ]]; then
  GEN_ARGS+=(--use-ising)
  if [[ -n "$ISING_ARTIFACTS_DIR" ]]; then
    GEN_ARGS+=(--ising-artifacts-dir "$ISING_ARTIFACTS_DIR")
  fi
fi
if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  GEN_ARGS+=("${EXTRA_APP_ARGS[@]}")
fi
"$EXE" "${GEN_ARGS[@]}"

# Assert the config file was created and is non-empty.
if [[ ! -s "$CONFIG_FILE" ]]; then
  echo "FAIL: config file '$CONFIG_FILE' was not created or is empty"
  exit 1
fi
echo "Config file generated: $CONFIG_FILE ($(stat -c %s "$CONFIG_FILE") bytes)"

# Dual-parse structural proof (mixed BP + matching lists): BP entries carry
# the undecomposed hyperedge H, so their block_size must be strictly LESS
# than the matching entries' decomposed block_size. Deterministic guard that
# the per-family factorization actually happened.
if [[ "$MULTI_TYPE" -eq 1 && ",$DECODER_TYPE," == *",nv-qldpc-decoder,"* ]] \
   && [[ ",$DECODER_TYPE," == *",pymatching,"* || ",$DECODER_TYPE," == *",trt_decoder,"* ]]; then
  nv_bs=""
  match_bs=""
  while read -r typ bs; do
    if [[ "$typ" == "nv-qldpc-decoder" ]]; then nv_bs=$bs; else match_bs=$bs; fi
  done < <(awk '/- id:/{n++} /type:/{t[n]=$2} /block_size:/{b[n]=$2} END{for(i=1;i<=n;i++) print t[i], b[i]}' "$CONFIG_FILE")
  if [[ -z "$nv_bs" || -z "$match_bs" ]] || [[ ! "$nv_bs" -lt "$match_bs" ]]; then
    echo "FAIL: dual-parse structural check: nv-qldpc block_size ('$nv_bs') must be < matching block_size ('$match_bs')"
    exit 1
  fi
  echo "Dual-parse structural check: nv block_size $nv_bs < matching $match_bs -- OK"
fi

# The hard-patch experiment must carry distinct decoder priors, not merely
# inject different runtime noise into three identically configured decoders.
if [[ -n "${CHECK_HARD_PATCH_MODELS:-}" ]]; then
  "$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import re
import sys

text = open(sys.argv[1], encoding="utf-8").read()
blocks = re.findall(r"(?ms)^  - id:\s+(-?\d+)\n(.*?)(?=^  - id:|\Z)", text)
priors = {}
for decoder_id, block in blocks:
    match = re.search(r"error_rate_vec:\s*\[([^\]]*)\]", block, re.S)
    if match:
        priors[int(decoder_id)] = tuple(
            round(float(value.strip()), 12)
            for value in match.group(1).split(",") if value.strip())

if set(priors) != {0, 1, 2}:
    raise SystemExit(f"expected priors for decoder ids 0,1,2; got {sorted(priors)}")
if priors[0] != priors[2]:
    raise SystemExit("easy-patch decoder priors differ")
if priors[1] == priors[0]:
    raise SystemExit("hard-patch decoder priors are identical to easy patches")
print("Hard-patch model check: decoder 1 priors differ; decoders 0 and 2 match -- OK")
PY
fi

if [[ -n "${CHECK_MISSING_SERVER_PORT:-}" ]]; then
  MISSING_PORT_ARGS=(
    --distance "$DISTANCE"
    --num_rounds "$NUM_ROUNDS"
    --num_shots 1
    --p_spam "$P_SPAM"
    --yaml "$CONFIG_FILE"
  )
  if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
    MISSING_PORT_ARGS+=("${EXTRA_APP_ARGS[@]}")
  fi

  set +e
  env -u QEC_DECODING_SERVER_PORT \
    "$EXE" "${MISSING_PORT_ARGS[@]}" >"$MISSING_PORT_LOG" 2>&1
  missing_port_status=$?
  set -e

  if [[ "$missing_port_status" -ne 1 ]] || ! grep -Fq \
    "Error: QEC_DECODING_SERVER_PORT is required for external decoding" \
    "$MISSING_PORT_LOG"; then
    echo "FAIL: missing server port did not produce a clean configuration error"
    cat "$MISSING_PORT_LOG"
    exit 1
  fi
  echo "Missing external-server port rejected cleanly -- OK"
fi

# An external-server test uses the generated YAML as the server's authoritative
# decoder configuration. The application still reloads and validates that YAML,
# but its CQR build deliberately does not construct local decoders.
SERVER_PORT=""
if [[ -n "${QEC_DECODING_SERVER:-}" ]]; then
  if [[ -n "${REQUIRE_SERVER_DECODE_COUNTS:-}" ]]; then
    QEC_DECODING_SERVER_STATS=1 \
      "$QEC_DECODING_SERVER" --config="$CONFIG_FILE" --transport=udp --port=0 \
        --timeout=300 >"$SERVER_LOG" 2>&1 &
  else
    "$QEC_DECODING_SERVER" --config="$CONFIG_FILE" --transport=udp --port=0 \
      --timeout=300 >"$SERVER_LOG" 2>&1 &
  fi
  SERVER_PID=$!

  for _ in $(seq 1 1200); do
    SERVER_PORT=$(grep -m1 "QEC_DECODING_SERVER_READY" "$SERVER_LOG" \
      2>/dev/null | sed -n 's/.*port=\([0-9]\+\).*/\1/p' || true)
    [[ -n "$SERVER_PORT" ]] && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      break
    fi
    sleep 0.1
  done

  if [[ -z "$SERVER_PORT" ]]; then
    echo "FAIL: decoding server did not become ready"
    cat "$SERVER_LOG"
    exit 1
  fi
  echo "External decoding server ready on UDP port $SERVER_PORT"

  # The server serves each decoder on its own ring and the READY line carries
  # one ring<id>=<port> token per decoder.  Export ring 1..N-1 as
  # QEC_DECODING_SERVER_PORT_<id> so the app wires each logical qubit's
  # device_call session to its decoder's ring (same pattern as the
  # surface_code-1 two-process driver).  Requests execute inline on the
  # server's per-ring dispatcher threads, so decoder concurrency requires the
  # per-decoder-ring topology — a single shared ring serializes decodes.
  # Every expected ring token is REQUIRED: a missing token would silently
  # route that decoder over the shared default ring, and the test would no
  # longer exercise per-ring dispatch.
  READY_LINE=$(grep -m1 "QEC_DECODING_SERVER_READY" "$SERVER_LOG")
  for ((id = 1; id < NUM_LOGICAL; id++)); do
    RING_PORT=$(echo "$READY_LINE" | sed -n "s/.*ring${id}=\([0-9]\+\).*/\1/p")
    if [[ -z "$RING_PORT" ]]; then
      echo "FAIL: READY line has no ring${id}= token for decoder ${id}"
      echo "$READY_LINE"
      exit 1
    fi
    export "QEC_DECODING_SERVER_PORT_${id}=${RING_PORT}"
    echo "Decoder ${id} ring on port $RING_PORT"
  done
fi

# -------------------------------------------------------------------------- #
# Phase 2: realtime -- load the YAML config and decode.
# The decoder is read from the file, so do NOT pass --decoder_type here.
# -------------------------------------------------------------------------- #
echo ""
echo "=== Phase 2: realtime decode (--yaml $CONFIG_FILE) ==="
# Use a pipefail-safe tee so a crash in the app still surfaces a non-zero status
# while we keep the full log for assertions below.
set +e
REALTIME_ARGS=(
  --distance "$DISTANCE" \
  --num_rounds "$NUM_ROUNDS" \
  --num_shots "$NUM_SHOTS" \
  --p_spam "$P_SPAM" \
  --yaml "$CONFIG_FILE"
)
if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  REALTIME_ARGS+=("${EXTRA_APP_ARGS[@]}")
fi
if [[ -n "$SERVER_PORT" ]]; then
  QEC_DECODING_SERVER_PORT="$SERVER_PORT" \
    "$EXE" "${REALTIME_ARGS[@]}" 2>&1 | tee "$REALTIME_LOG"
  app_status=${PIPESTATUS[0]}
else
  "$EXE" "${REALTIME_ARGS[@]}" 2>&1 | tee "$REALTIME_LOG"
  app_status=${PIPESTATUS[0]}
fi
set -e

if [[ -n "$SERVER_PORT" ]]; then
  stop_server
fi

if [[ "$app_status" -ne 0 ]]; then
  echo "FAIL: realtime phase exited with non-zero status ($app_status)"
  return_code=1
fi

# -------------------------------------------------------------------------- #
# Assertions on the realtime log.
# -------------------------------------------------------------------------- #
echo ""
echo "=== Checking realtime output ==="

# A non-graphlike DEM handed to pymatching surfaces as "Invalid column in H".
if grep -q "Invalid column in H" "$REALTIME_LOG"; then
  echo "FAIL: found 'Invalid column in H' (decoder received a non-graphlike DEM)"
  return_code=1
fi

# Hard decoder-init / dispatch failures.
if grep -q "terminate called" "$REALTIME_LOG"; then
  echo "FAIL: found 'terminate called' (the app aborted)"
  return_code=1
fi
if grep -q "Decoder 0 not found" "$REALTIME_LOG"; then
  echo "FAIL: found 'Decoder 0 not found' (decoder was not registered)"
  return_code=1
fi
if grep -q "Error initializing decoders" "$REALTIME_LOG"; then
  echo "FAIL: found 'Error initializing decoders'"
  return_code=1
fi

# A "Number of corrections decoder found:" line MUST be present -- it proves the
# realtime decoding path actually ran to completion.
if ! grep -q "Number of corrections decoder found:" "$REALTIME_LOG"; then
  echo "FAIL: missing 'Number of corrections decoder found:' line (decoding did not complete)"
  return_code=1
fi

# Pull out the residual logical-error count and sanity check it.
num_non_zero_values=$(grep "Number of non-zero values measured :" "$REALTIME_LOG" | awk -F': ' '{print $2}' | tr -d '[:space:]')

if ! [[ "$num_non_zero_values" =~ ^[0-9]+$ ]]; then
  echo "FAIL: 'Number of non-zero values measured' is not a number (got '$num_non_zero_values')"
  return_code=1
elif [[ -n "${SKIP_LOGICAL_ERROR_CEILING:-}" ]]; then
  echo "Logical-error ceiling skipped for deterministic concurrency test"
elif [[ "$MULTI_TYPE" -eq 1 ]]; then
  echo "Multi-type mode: aggregate ceiling replaced by per-decoder ceilings below"
elif [[ "$num_non_zero_values" -gt "$MAX_NON_ZERO" ]]; then
  echo "FAIL: residual logical errors ($num_non_zero_values) exceed ceiling ($MAX_NON_ZERO) -- decoder appears wired-but-wrong"
  return_code=1
else
  echo "Residual logical errors: $num_non_zero_values (ceiling $MAX_NON_ZERO) -- OK"
fi

# Multi-type mode: one report line per patch, matched literally (grep -F --
# the bracketed label is a regex trap), with a per-type residual ceiling.
if [[ "$MULTI_TYPE" -eq 1 ]]; then
  for i in "${!DECODER_TYPES[@]}"; do
    t=${DECODER_TYPES[$i]}
    line=$(grep -F "decoder[$i] ($t):" "$REALTIME_LOG" || true)
    if [[ -z "$line" ]]; then
      echo "FAIL: missing per-decoder report line 'decoder[$i] ($t):'"
      return_code=1
      continue
    fi
    errs=$(printf '%s\n' "$line" | sed -n 's/.*logical_errors=\([0-9][0-9]*\)\/[0-9][0-9]*.*/\1/p')
    ceil=$MAX_NON_ZERO
    if ! [[ "$errs" =~ ^[0-9]+$ ]]; then
      echo "FAIL: could not parse logical_errors from: $line"
      return_code=1
    elif [[ -n "${SKIP_LOGICAL_ERROR_CEILING:-}" ]]; then
      echo "decoder[$i] ($t): logical-error ceiling skipped"
    elif [[ "$errs" -gt "$ceil" ]]; then
      echo "FAIL: decoder[$i] ($t) residual logical errors ($errs) exceed ceiling ($ceil)"
      return_code=1
    else
      echo "decoder[$i] ($t): residual logical errors $errs (ceiling $ceil) -- OK"
    fi
  done
fi

if [[ -n "${EXPECTED_DECODER_CORRECTIONS:-}" ]]; then
  IFS=',' read -r -a expected_corrections <<< \
    "$EXPECTED_DECODER_CORRECTIONS"
  if [[ ${#expected_corrections[@]} -ne ${#DECODER_TYPES[@]} ]]; then
    echo "FAIL: EXPECTED_DECODER_CORRECTIONS has ${#expected_corrections[@]} entries; expected ${#DECODER_TYPES[@]}"
    return_code=1
  else
    for i in "${!expected_corrections[@]}"; do
      line=$(grep -F "decoder[$i] (${DECODER_TYPES[$i]}):" \
        "$REALTIME_LOG" || true)
      got=$(printf '%s\n' "$line" | \
        sed -n 's/.*corrections=\([0-9][0-9]*\),.*/\1/p')
      if [[ "$got" != "${expected_corrections[$i]}" ]]; then
        echo "FAIL: decoder[$i] corrections='$got'; expected ${expected_corrections[$i]}"
        return_code=1
      fi
    done
  fi
fi

if [[ -n "$SERVER_PORT" ]]; then
  server_dispatches=$(sed -n \
    's/^QEC_DECODING_SERVER_DISPATCHED count=\([0-9][0-9]*\)$/\1/p' \
    "$SERVER_LOG" | tail -n1)
  server_max_concurrent=$(sed -n \
    's/^QEC_DECODING_SERVER_MAX_CONCURRENT_DECODERS count=\([0-9][0-9]*\)$/\1/p' \
    "$SERVER_LOG" | tail -n1)
  minimum_dispatches=$((NUM_SHOTS * ${#DECODER_TYPES[@]} * (NUM_ROUNDS + 3)))

  if ! [[ "$server_dispatches" =~ ^[0-9]+$ ]] || \
     [[ "$server_dispatches" -lt "$minimum_dispatches" ]]; then
    echo "FAIL: server dispatch count '$server_dispatches' is below $minimum_dispatches"
    return_code=1
  fi
  if ! grep -q \
    "External decoding server owns all configured decoder instances" \
    "$REALTIME_LOG"; then
    echo "FAIL: external application did not report server-owned decoders"
    return_code=1
  fi
  if [[ -n "${REQUIRE_DECODER_CONCURRENCY:-}" ]] && \
     { ! [[ "$server_max_concurrent" =~ ^[0-9]+$ ]] || \
       [[ "$server_max_concurrent" -lt "$REQUIRE_DECODER_CONCURRENCY" ]]; }; then
    echo "FAIL: server max concurrency '$server_max_concurrent' is below $REQUIRE_DECODER_CONCURRENCY"
    return_code=1
  fi
  if [[ -n "${EXPECTED_BARRIER_COMPLETIONS:-}" ]]; then
    barrier_completions=$(grep -c \
      '^QEC_CONCURRENCY_TEST_BARRIER generation=' "$SERVER_LOG" || true)
    if [[ "$barrier_completions" -ne "$EXPECTED_BARRIER_COMPLETIONS" ]]; then
      echo "FAIL: barrier completed $barrier_completions times; expected $EXPECTED_BARRIER_COMPLETIONS"
      return_code=1
    fi
  fi
  if [[ -n "${EXPECTED_SERVER_DECODER_CONSTRUCTIONS:-}" ]]; then
    server_constructions=$(grep -c \
      '^QEC_CONCURRENCY_TEST_DECODER_CONSTRUCTED$' "$SERVER_LOG" || true)
    client_constructions=$(grep -c \
      '^QEC_CONCURRENCY_TEST_DECODER_CONSTRUCTED$' "$REALTIME_LOG" || true)
    if [[ "$server_constructions" != \
          "$EXPECTED_SERVER_DECODER_CONSTRUCTIONS" ]]; then
      echo "FAIL: server constructed $server_constructions test decoders; expected $EXPECTED_SERVER_DECODER_CONSTRUCTIONS"
      return_code=1
    fi
    if [[ "$client_constructions" -ne 0 ]]; then
      echo "FAIL: external application constructed $client_constructions local decoder instances"
      return_code=1
    fi
  fi
  # Decoder-agnostic execution evidence: the server reports per-session
  # counters (QEC_DECODING_SERVER_STATS=1); every decoder must have completed
  # exactly one decode per shot. This replaces the removed trt-internal
  # QEC_TRT_INFERENCE_EXECUTIONS counter and covers closed-source decoders.
  if [[ -n "${REQUIRE_SERVER_DECODE_COUNTS:-}" ]]; then
    stats_lines=$(grep -c '^QEC_DECODING_SERVER_DECODER_STATS ' \
      "$SERVER_LOG" || true)
    if [[ "$stats_lines" -ne "${#DECODER_TYPES[@]}" ]]; then
      echo "FAIL: server reported $stats_lines decoder stats lines; expected ${#DECODER_TYPES[@]}"
      return_code=1
    fi
    for ((i = 0; i < ${#DECODER_TYPES[@]}; i++)); do
      got_decodes=$(sed -n \
        "s/^QEC_DECODING_SERVER_DECODER_STATS id=$i decodes=\([0-9][0-9]*\) .*/\1/p" \
        "$SERVER_LOG" | tail -n1)
      if [[ "$got_decodes" != "$NUM_SHOTS" ]]; then
        echo "FAIL: decoder[$i] (${DECODER_TYPES[$i]}) completed '$got_decodes' decodes; expected $NUM_SHOTS"
        return_code=1
      fi
    done
  fi
  echo "Server evidence: dispatches=$server_dispatches, max_concurrent=$server_max_concurrent"
fi

echo ""
if [[ "$return_code" -eq 0 ]]; then
  echo "PASS: surface_code-4-yaml ($DECODER_TYPE, d=$DISTANCE) realtime decode succeeded"
else
  echo "FAIL: surface_code-4-yaml ($DECODER_TYPE, d=$DISTANCE) test failed"
fi

exit $return_code
