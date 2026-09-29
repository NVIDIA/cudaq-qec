# SPDX-FileCopyrightText: Copyright (c) 2024 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Abort on failure
set -e

return_code=0

# Expected args:
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-1-local{-quantinuum-emulate}
#  distance
#  number_of_non_zero_values_threshold
#  number_of_corrections_decoder_threshold
#  Path to server executable
#  num_rounds
#  Path to libcudaq-qec-realtime-decoding-quantinuum-private.so
#  decoder_type (optional, defaults to multi_error_lut)
#  sw_window_size (optional, for sliding_window decoder)
#  sw_step_size (optional, for sliding_window decoder, defaults to 1)

# Check that at least 8 arguments are provided.
if [[ $# -lt 8 ]]; then
  echo "Error: Expected at least 8 arguments (got $#)"
  exit 1
fi

EXE_PATH1=$1
EXE_PATH2=$2
DISTANCE=$3
number_of_non_zero_values_threshold=$4
number_of_corrections_decoder_threshold=$5
SERVER_EXECUTABLE=$6
NUM_ROUNDS=$7
LIB_DIR=$8
DECODER_TYPE=${9:-multi_error_lut}
SW_WINDOW_SIZE=${10:-5}
SW_STEP_SIZE=${11:-1}
EXTRA_CLI_ARGS=${EXTRA_CLI_ARGS:-}

export CUDAQ_DEFAULT_SIMULATOR=stim

NUM_SHOTS=1000

# Get timestamp suffix in YYYY-MM-DD-HH-MM-SS, with random number appended using /dev/urandom.
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
RNG_SUFFIX=$(od -An -N4 -i /dev/urandom | tr -d ' ')
# Remove any negative sign.
RNG_SUFFIX=$(echo $RNG_SUFFIX | sed 's/-//g')
FULL_SUFFIX=$timestamp-$RNG_SUFFIX

export CUDAQ_DUMP_JIT_IR=${CUDAQ_DUMP_JIT_IR:-0}

# Run the full experiment (DEM characterization + shots) with the target
# executable. Both steps happen in the same process.
echo "Running $EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --num_rounds $NUM_ROUNDS --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS"
$EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS \
  --num_rounds $NUM_ROUNDS \
  --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE \
  --sw_step_size $SW_STEP_SIZE $EXTRA_CLI_ARGS |& tee run-$FULL_SUFFIX.log

# If CUDAQ_DUMP_JIT_IR is "1", extract the QIR from the log file.
if [[ "${CUDAQ_DUMP_JIT_IR}" == "1" ]]; then
  QIR=$(sed -n '/ModuleID/,/backwards_branching/p' run-$FULL_SUFFIX.log)
  echo "Writing QIR to qir-$FULL_SUFFIX.ll"
  echo "$QIR" > qir-$FULL_SUFFIX.ll
fi

# Look for results like this in the output:
# Number of non-zero values measured : 2
# Number of corrections decoder found: 48

num_non_zero_values=$(grep "Number of non-zero values measured :" run-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
num_corrections_decoder=$(grep "Number of corrections decoder found:" run-$FULL_SUFFIX.log | awk -F': ' '{print $2}')

if ! [[ "$num_non_zero_values" =~ ^[0-9]+$ ]]; then
  echo "Error: Number of non-zero values measured is not a number"
  return_code=1
fi

if ! [[ "$num_corrections_decoder" =~ ^[0-9]+$ ]]; then
  echo "Error: Number of corrections decoder found is not a number"
  return_code=1
fi

if [[ "$num_non_zero_values" -gt $number_of_non_zero_values_threshold ]]; then
  echo "Error: Number of non-zero values measured is greater than $number_of_non_zero_values_threshold (unexpected)"
  return_code=1
fi

if [[ "$num_corrections_decoder" -lt $number_of_corrections_decoder_threshold ]]; then
  echo "Error: Number of corrections decoder found is less than $number_of_corrections_decoder_threshold (unexpected)"
  return_code=1
fi

# For the cqr host-dispatch variant, verify the device_calls actually crossed
# the cudaq-realtime ring to the in-process decoding server (the count is 0 if
# they silently bypassed to a direct trampoline).
if [[ "${CUDAQ_DEVICE_CALL_CHANNEL:-}" == "host_dispatch" ]]; then
  cqr_dispatch_count=$(grep "CQR service dispatch count:" run-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
  if ! [[ "$cqr_dispatch_count" =~ ^[0-9]+$ ]] || [[ "$cqr_dispatch_count" -eq 0 ]]; then
    echo "Error: CQR service dispatch count is missing or zero; device_calls did not traverse host dispatch"
    return_code=1
  else
    echo "CQR service dispatch count check passed ($cqr_dispatch_count dispatches)"
  fi
fi

echo "Test completed for distance $DISTANCE with return code $return_code"

# ============================================================================ #
# Test --save_syndrome and --load_syndrome functionality
# ============================================================================ #
echo ""
echo "=== Testing syndrome save/load functionality ==="

SYNDROME_FILE=syndromes-${FULL_SUFFIX}.txt
SYNDROME_NUM_SHOTS=10

# Step 1: Run simulation with --save_syndrome to capture syndrome data.
# Use the local executable (syndrome capture is a host-side feature).
echo "Step 1: Saving syndromes to $SYNDROME_FILE"
$EXE_PATH1 --distance $DISTANCE --num_shots $SYNDROME_NUM_SHOTS \
  --num_rounds $NUM_ROUNDS \
  --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE \
  --sw_step_size $SW_STEP_SIZE --save_syndrome $SYNDROME_FILE \
  $EXTRA_CLI_ARGS |& tee save_syndrome-$FULL_SUFFIX.log

if [[ ! -f "$SYNDROME_FILE" ]]; then
  echo "Error: Syndrome file was not created"
  return_code=1
else
  echo "Syndrome file created successfully"

  if grep -q "SHOT_START" $SYNDROME_FILE && grep -q "CORRECTIONS_START" $SYNDROME_FILE; then
    echo "Syndrome file contains expected markers"
  else
    echo "Error: Syndrome file missing expected markers"
    return_code=1
  fi

  # Step 2: Replay syndromes with --load_syndrome.
  echo "Step 2: Replaying syndromes from $SYNDROME_FILE"
  $EXE_PATH1 --distance $DISTANCE --num_shots $SYNDROME_NUM_SHOTS \
    --num_rounds $NUM_ROUNDS \
    --decoder_type $DECODER_TYPE --sw_window_size $SW_WINDOW_SIZE \
    --sw_step_size $SW_STEP_SIZE --load_syndrome $SYNDROME_FILE \
    $EXTRA_CLI_ARGS |& tee load_syndrome-$FULL_SUFFIX.log

  if grep -q "Replay complete" load_syndrome-$FULL_SUFFIX.log; then
    echo "Syndrome replay completed successfully"

    if grep -q "SUCCESS: All corrections match" load_syndrome-$FULL_SUFFIX.log; then
      echo "Correction verification: PASSED"
    elif grep -q "mismatched" load_syndrome-$FULL_SUFFIX.log; then
      echo "Error: Corrections did not match during replay"
      return_code=1
    fi
  else
    echo "Error: Syndrome replay did not complete"
    return_code=1
  fi
fi

echo "Syndrome save/load test completed"

# Remove the log files, unless an environment variable is set.
if [[ -z "${KEEP_LOG_FILES}" ]]; then
  rm -f run-$FULL_SUFFIX.log
  rm -f save_syndrome-$FULL_SUFFIX.log load_syndrome-$FULL_SUFFIX.log $SYNDROME_FILE
fi

exit $return_code
