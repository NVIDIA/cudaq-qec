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
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local
#  ${CMAKE_CURRENT_BINARY_DIR}/surface_code-2-local{-quantinuum-emulate}
#  distance
#  number_of_non_zero_values_threshold
#  number_of_corrections_decoder_threshold
#  Path to server executable
#  Path to libcudaq-qec-realtime-decoding-quantinuum-private.so

# Check that all 7 arguments are provided.
if [[ $# -ne 7 ]]; then
  echo "Error: Expected 7 arguments"
  exit 1
fi

EXE_PATH1=$1
EXE_PATH2=$2
DISTANCE=$3
number_of_non_zero_values_threshold=$4
number_of_corrections_decoder_threshold=$5
SERVER_EXECUTABLE=$6
LIB_DIR=$7

export CUDAQ_DEFAULT_SIMULATOR=stim

ERROR_RATE=0.01
NUM_SHOTS=1000

# Get timestamp suffix in YYYY-MM-DD-HH-MM-SS, with random number appended using /dev/urandom.
timestamp=$(date +%Y-%m-%d-%H-%M-%S)
RNG_SUFFIX=$(od -An -N4 -i /dev/urandom | tr -d ' ')
# Remove any negative sign.
RNG_SUFFIX=$(echo $RNG_SUFFIX | sed 's/-//g')
FULL_SUFFIX=$timestamp-$RNG_SUFFIX

CONFIG_FILE=config-${FULL_SUFFIX}.yml

# Generate the config file using the first executable.
$EXE_PATH1 --distance $DISTANCE --num_shots $NUM_SHOTS --save_dem $CONFIG_FILE | tee save_dem-$FULL_SUFFIX.log

export CUDAQ_DUMP_JIT_IR=${CUDAQ_DUMP_JIT_IR:-0}


# Use the config file using the second executable.
echo Running $EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE
$EXE_PATH2 --distance $DISTANCE --num_shots $NUM_SHOTS --load_dem $CONFIG_FILE |& tee load_dem-$FULL_SUFFIX.log

# If CUDAQ_DUMP_JIT_IR is "1", then extract the QIR from the
# load_dem-$FULL_SUFFIX.log file and place it in qir-$FULL_SUFFIX.ll.
if [[ "${CUDAQ_DUMP_JIT_IR}" == "1" ]]; then
  # The QIR starts with "ModuleID" and ends with "backwards_branching"
  QIR=$(sed -n '/ModuleID/,/backwards_branching/p' load_dem-$FULL_SUFFIX.log)
  echo "Writing QIR to qir-$FULL_SUFFIX.ll"
  echo "$QIR" > qir-$FULL_SUFFIX.ll
fi

# Look for results like this in the output:
# Number of non-zero values measured : 2
# Number of corrections decoder found: 48

# Make sure that the first value is < 10, and the second value is > 10.

num_non_zero_values=$(grep "Number of non-zero values measured :" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')
num_corrections_decoder=$(grep "Number of corrections decoder found:" load_dem-$FULL_SUFFIX.log | awk -F': ' '{print $2}')

# Make sure that the value is a number.
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

echo "Test completed for distance $DISTANCE with return code $return_code"

# Remove the log files and the config file, unless an environment variable is set.
if [[ -z "${KEEP_LOG_FILES}" ]]; then
  rm -f load_dem-$FULL_SUFFIX.log save_dem-$FULL_SUFFIX.log server-$FULL_SUFFIX.log $CONFIG_FILE
fi

exit $return_code
