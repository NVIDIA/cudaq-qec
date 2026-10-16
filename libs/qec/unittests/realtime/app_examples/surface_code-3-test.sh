# ============================================================================ #
# Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Abort on failure
set -e

return_code=0

# Expected args:
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-3-local{-quantinuum-emulate}
#  distance
#  Path to server executable
#  Error rate
#  Initial state
#  Path to libcudaq-qec-realtime-decoding-quantinuum-private.so

# Check that all 6 arguments are provided.
if [[ $# -ne 7 ]]; then
  echo "Error: Expected 7 arguments"
  exit 1
fi

EXE_PATH1=$1
EXE_PATH2=$2
DISTANCE=$3
# number_of_non_zero_values_threshold=$4 # Legacy
# number_of_corrections_decoder_threshold=$5 #Legacy 
SERVER_EXECUTABLE=$4
ERROR_RATE=$5
STATE_PREP=$6
LIB_DIR=$7

export CUDAQ_DEFAULT_SIMULATOR=stim

NUM_SHOTS=1000

# Get timestamp suffix in YYYY-MM-DD-HH-MM-SS, with random number appended using /dev/urandom.
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
RNG_SUFFIX=$(od -An -N4 -i /dev/urandom | tr -d ' ')
# Remove any negative sign.
RNG_SUFFIX=$(echo $RNG_SUFFIX | sed 's/-//g')
FULL_SUFFIX=$timestamp-$RNG_SUFFIX

CONFIG_FILE=config-${FULL_SUFFIX}.yml

# Generate the config file using the first executable.
$EXE_PATH1 --distance $DISTANCE --num_shots $NUM_SHOTS --save_dem $CONFIG_FILE --p_spam $ERROR_RATE --state_prep $STATE_PREP | tee save_dem-$FULL_SUFFIX.log

export CUDAQ_DUMP_JIT_IR=${CUDAQ_DUMP_JIT_IR:-0}


# Use the config file using the second executable.
echo Running $EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE
$EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE  --p_spam $ERROR_RATE --state_prep $STATE_PREP  |& tee load_dem-$FULL_SUFFIX.log

# If CUDAQ_DUMP_JIT_IR is "1", then extract the QIR from the
# load_dem-$FULL_SUFFIX.log file and place it in qir-$FULL_SUFFIX.ll.
if [[ "${CUDAQ_DUMP_JIT_IR}" == "1" ]]; then
  # The QIR starts with "ModuleID" and ends with "backwards_branching"
  QIR=$(sed -n '/ModuleID/,/backwards_branching/p' load_dem-$FULL_SUFFIX.log)
  echo "Writing QIR to qir-$FULL_SUFFIX.ll"
  echo "$QIR" > qir-$FULL_SUFFIX.ll
fi

# ------------------------------------------------------------------------------
# Validation of decoder results using mismatch percentage:
#
# The mismatch percentage measures the absolute difference between the number of
# logical errors and decoder corrections, normalized by the total number of shots:
#
#     mismatch % = |logical_errors - corrections| / shots * 100%
#
# This metric reflects how closely the decoder corrections align with the actual
# logical errors observed in the simulation. A high mismatch indicates poor decoder
# performance, and the test flags an error if mismatch exceeds 50%.
#
# This replaces previous threshold-based checks with a normalized, intuitive metric
# for evaluating decoder accuracy.
# ------------------------------------------------------------------------------


# --- X CHECKS ---
if [[ "$STATE_PREP" == "prep0" ]]; then
  echo "--- Validating X basis results ---"
  x_error_rate=$(grep "X final error rate:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
  x_safety=$(grep "X Unsafety:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')

  if ! [[ "$x_error_rate" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error (X): Final error rate is not a number."
    return_code=1
  fi
  if ! [[ "$x_safety" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error (X): Unsafety is not a number."
    return_code=1
  fi

  # Threshold checks
  if (( $(awk "BEGIN {print ($x_error_rate > 0.1)}") )); then
    echo "Error (X): Final error rate exceeds 10%"
    return_code=1
  fi

  if (( $(awk "BEGIN {print ($x_safety > 0.05)}") )); then
    echo "Warning (X): Unsafety exceeds 5%"
  fi
fi


# --- Z CHECKS ---
if [[ "$STATE_PREP" == "prepp" ]]; then
  echo "--- Validating Z basis results ---"
  z_error_rate=$(grep "Z final error rate:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
  z_safety=$(grep "Z Unsafety:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')

  if ! [[ "$z_error_rate" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error (Z): Final error rate is not a number."
    return_code=1
  fi
  if ! [[ "$z_safety" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "Error (Z): Unsafety is not a number."
    return_code=1
  fi

  # Threshold checks
  if (( $(awk "BEGIN {print ($z_error_rate > 0.1)}") )); then
    echo "Error (Z): Final error rate exceeds 10%"
    return_code=1
  fi

  if (( $(awk "BEGIN {print ($z_safety > 0.05)}") )); then
    echo "Warning (Z): Unsafety exceeds 5%"
  fi
fi


echo "Test completed for distance $DISTANCE with return code $return_code"

KEEP_LOG_FILES=true
if [[ -z "${KEEP_LOG_FILES}" ]]; then
  rm -f load_dem-$FULL_SUFFIX.log save_dem-$FULL_SUFFIX.log server-$FULL_SUFFIX.log $CONFIG_FILE
fi

exit $return_code