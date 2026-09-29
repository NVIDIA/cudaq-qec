# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script allows a user to replay a real-time decoder log file (assuming the
# right instrumentation is enabled). It can be used to compare the online results
# to the offline results and/or replay the data with a different config file.

# How to use this script:
# python3 replay_decoder_logs.py --config config.yml --decoder-log decoder.log

# Capturing logs for replay:
#   * Recommended (forwarder-proof): emit the [DecoderStats] lines via a direct
#     printf to stdout, bypassing the logger (and any forwarder) entirely:
#         CUDAQ_QEC_DEBUG_DECODER=1 ./your_app > decoder.log 2>&1
#   * Alternative: CUDAQ_LOG_LEVEL=info also emits the lines, but only via the
#     logger, so a cudaq::qec log forwarder would truncate long messages (to
#     message_capacity, appending " ...[truncated]"). Capture without a
#     forwarder if you use this path.
#   * This parser skips "...[truncated]" lines and warns.

import sys
import argparse
import os
import numpy
import yaml
import cudaq_qec as qec

# Suffix the cudaq::qec log forwarder appends to messages it truncates to fit
# its bounded message_capacity (see realtime_truncation_suffix in logger.h).
# Lines containing this marker are incomplete and unsafe to parse.
TRUNCATION_MARKER = "...[truncated]"


# ---------------------------------------------------------------------------- #
# Helper function to convert a sparse list to a dense matrix. -1 is the row delimiter.
def sparse_to_dense(sparse_list, num_rows, num_cols, dtype=numpy.uint8):
    mat = numpy.zeros((num_rows, num_cols), dtype=dtype)
    row = 0
    for idx in sparse_list:
        if idx == -1:
            row += 1
        else:
            mat[row, idx] = 1
    return mat


# ---------------------------------------------------------------------------- #
# Traverse the decoder log file looking for decode calls. Note that when a
# decoder is created, a dummy decode call is made to "warm up" the decoder, so
# you may see more decode calls than shots.
def parse_decoder_log(decoder_log_file, log_detectors_sparse, log_errors_sparse,
                      log_observables_sparse, log_observables_dense,
                      log_result_types, decoder_id_list):
    # The decoder id is read directly from each [DecoderStats] line's
    # "DecoderId:" field. last_decoder_id is only a fallback for older logs
    # whose stats lines predate that field (it is tracked from the enqueue
    # message, which is emitted at the info level).
    last_decoder_id = -1
    truncated_lines = 0  # count of skipped "...[truncated]" lines
    print(f'Parsing decoder log file {decoder_log_file}...')
    enqueue_msg = "Entering enqueue_syndromes_ui64 for decoder id: "  # fallback decoder id
    with open(decoder_log_file, 'r') as f:
        for line in f:
            line = line.strip()
            # A forwarder-truncated line has incomplete fields (e.g. a clipped
            # InputDetectors list), which would silently desync the parallel
            # result lists. Skip it entirely so nothing partial is appended.
            if TRUNCATION_MARKER in line:
                truncated_lines += 1
                if truncated_lines == 1:
                    print(f"WARNING: skipping '{TRUNCATION_MARKER}' lines "
                          "(forwarder message_capacity too small). Re-capture "
                          "without a log forwarder. See header comment.")
                continue
            if enqueue_msg in line:
                # Needed for last_decoder_id.
                last_decoder_id = int(line.split(enqueue_msg)[1].split(" ")[0])
            if "[DecoderStats]" in line:
                line = line.split("[DecoderStats]")[1]
                # print(line)
                if "InputDetectors:" in line:  # this is a decode call
                    fields = {}
                    for elem in line.split(" "):
                        if ":" in elem:
                            key, value = elem.split(":", 1)
                            fields[key] = value

                    value = fields["InputDetectors"]
                    if value == "":
                        log_detectors_sparse.append([])
                    else:
                        log_detectors_sparse.append(
                            [int(x) for x in value.split(",")])
                    # Prefer the DecoderId carried in the stats line; fall back
                    # to the enqueue-derived id for older logs that lack it.
                    if "DecoderId" in fields:
                        decoder_id = int(fields["DecoderId"])
                    elif last_decoder_id != -1:
                        decoder_id = last_decoder_id
                    else:
                        print("Error: could not determine decoder id (no "
                              "DecoderId field and no enqueue message). This "
                              "is a fatal error processing the log file.")
                        exit(1)
                    decoder_id_list.append(decoder_id)

                    value = fields.get("Errors", "")
                    if value == "":
                        log_errors_sparse.append([])
                    else:
                        log_errors_sparse.append(
                            [int(x) for x in value.split(",")])

                    value = fields.get("Observables", "")
                    if value == "":
                        log_observables_sparse.append([])
                    else:
                        log_observables_sparse.append(
                            [int(x) for x in value.split(",")])

                    value = fields["ObservableCorrectionsThisCall"]
                    log_observables_dense.append(
                        [int(x) for x in value.split(",")])

                    value = fields.get("ResultType", "errs")
                    result_types = {x for x in value.split(",") if x}
                    if not result_types:
                        result_types = {"errs"}
                    log_result_types.append(result_types)
    if truncated_lines > 0:
        print(f"WARNING: skipped {truncated_lines} truncated line(s); "
              "replay coverage is incomplete.")


# ---------------------------------------------------------------------------- #
# Parse the decoder config file and create the decoders from it.
def decoder_outputs_from_log(log_result_types, decoder_id_list):
    """Return the one fixed output basis observed for each decoder id."""
    outputs = {}
    for decoder_id, result_types in zip(decoder_id_list, log_result_types):
        if result_types == {"errs"}:
            output = "errors"
        elif result_types == {"obs"}:
            output = "observables"
        else:
            raise RuntimeError(
                f"unsupported ResultType set {sorted(result_types)} for "
                f"decoder {decoder_id}")

        previous = outputs.setdefault(decoder_id, output)
        if previous != output:
            raise RuntimeError(
                f"decoder {decoder_id} logged both {previous} and {output} "
                "results; a decoder instance must have one fixed output basis")
    return outputs


def normalize_error_rate_args(decoder, decoder_id):
    """Promote the one accepted legacy rate vector into factory kwargs."""
    locations = []

    if 'error_rate_vec' in decoder:
        locations.append(('error_rate_vec', decoder['error_rate_vec']))

    def strip_nested_rates(mapping, path):
        normalized = {}
        for key, value in mapping.items():
            key_path = f'{path}.{key}'
            if key == 'error_rate_vec':
                locations.append((key_path, value))
            elif isinstance(value, dict):
                normalized[key] = strip_nested_rates(value, key_path)
            else:
                normalized[key] = value
        return normalized

    decoder_custom_args = strip_nested_rates(
        decoder.get('decoder_custom_args', {}), 'decoder_custom_args')
    if len(locations) > 1:
        paths = " and ".join(f"'{path}'" for path, _ in locations)
        raise RuntimeError(
            f"decoder {decoder_id} supplies error_rate_vec more than once at "
            f"{paths}; supply it in exactly one location")
    if locations:
        decoder_custom_args['error_rate_vec'] = locations[0][1]
    return decoder_custom_args


def parse_decoder_config(config_file, decoders, O_per_decoder,
                         output_per_decoder):
    print(f'Creating decoders from config file {args.config}...')
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for config_index, decoder in enumerate(config['decoders']):
            # Loop through each decoder in the config.
            decoder_id = decoder.get('id', config_index)
            decoder_custom_args = normalize_error_rate_args(decoder, decoder_id)

            # Chunk-form YAML deliberately omits the flat matrix fields. Use
            # the same bound expansion routine as the realtime construction
            # path so replay sees the identical closed model and priors.
            expanded_chunks = None
            if decoder.get(
                    'dem_chunks') is not None and not decoder.get('H_sparse'):
                expanded_chunks = qec.decoder_config.from_yaml_str(
                    yaml.safe_dump(decoder))
                qec.expand_dem_chunks(expanded_chunks)

            # Change these to primitive types. This is annoying to have to do, but I
            # don't know of a better way to do this.
            decoder_custom_args = dict(decoder_custom_args)
            for key, value in decoder_custom_args.items():
                if type(value) == list:
                    # TODO - update this to be more general if some decoder parameters need
                    # to be a different type.
                    decoder_custom_args[key] = numpy.array(value,
                                                           dtype=numpy.float64)
                elif type(value) == int:
                    decoder_custom_args[key] = int(value)
                elif type(value) == float:
                    decoder_custom_args[key] = float(value)
                elif type(value) == bool:
                    decoder_custom_args[key] = bool(value)
            output = output_per_decoder.get(decoder_id, "observables")
            stim_dem_path = decoder.get('stim_dem_path', '')
            if stim_dem_path:
                if (decoder.get('H_sparse') or decoder.get('O_sparse') or
                        'error_rate_vec' in decoder_custom_args):
                    raise RuntimeError(
                        f"decoder {decoder_id} supplies stim_dem_path together "
                        "with H_sparse, O_sparse, or error_rate_vec; supply "
                        "exactly one model source")
                if not os.path.isabs(stim_dem_path):
                    stim_dem_path = os.path.join(
                        os.path.dirname(os.path.abspath(config_file)),
                        stim_dem_path)
                with open(stim_dem_path, 'r') as dem_file:
                    dem_text = dem_file.read()
                dem = qec.dem_from_stim_text(dem_text)
                O = numpy.asarray(dem.observables_flips_matrix,
                                  dtype=numpy.uint8)
                decoder_instance = qec.get_decoder(decoder['type'],
                                                   dem_text,
                                                   output=output,
                                                   **decoder_custom_args)
            else:
                if expanded_chunks is None:
                    num_rows = decoder['syndrome_size']
                    num_cols = decoder['block_size']
                    H_sparse = decoder['H_sparse']
                    O_sparse = decoder['O_sparse']
                else:
                    num_rows = expanded_chunks.syndrome_size
                    num_cols = expanded_chunks.block_size
                    H_sparse = expanded_chunks.H_sparse
                    O_sparse = expanded_chunks.O_sparse
                    if "error_rate_vec" not in decoder_custom_args:
                        decoder_custom_args["error_rate_vec"] = numpy.array(
                            expanded_chunks.error_rate_vec, dtype=numpy.float64)
                num_observables = O_sparse.count(-1)
                H = sparse_to_dense(H_sparse, num_rows, num_cols)
                O = sparse_to_dense(O_sparse, num_observables, num_cols)
                decoder_instance = qec.get_decoder(decoder['type'],
                                                   H,
                                                   O=O,
                                                   output=output,
                                                   **decoder_custom_args)

            # Replaying an obs-frame decoder reconstructs the full composite
            # decoder here, so trt_decoder replay needs TensorRT, a GPU, and
            # the referenced ONNX/engine artifacts. LUT replay is lighter.
            # The online realtime path fixes every decoder's result basis at
            # construction. Reconstruct that same basis explicitly: O is
            # model data and no longer acts as an output-selection switch.
            decoders[decoder_id] = decoder_instance
            O_per_decoder[decoder_id] = O
            print(f"Decoder {decoder_id} created.")


# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #

parser = argparse.ArgumentParser(description='Replay decoder logs.')
parser.add_argument('--config', type=str, required=True, help='Config file.')
parser.add_argument('--decoder-log',
                    type=str,
                    required=True,
                    help='Decoder log file.')
parser.add_argument('--verbose-on-mismatch',
                    action='store_true',
                    help='Verbose output on mismatches.')
args = parser.parse_args()

# Check if the files exist.
if not os.path.exists(args.config):
    print(f"Config file does not exist: {args.config}")
    exit(1)
if not os.path.exists(args.decoder_log):
    print(f"Decoder log file does not exist: {args.decoder_log}")
    exit(1)

# The length of these lists should be the same as the number of decode calls in
# the log file.
decoder_id_list = []  # Decoder ID of each decode call.
log_detectors_sparse = []  # Detection events seen in the log file.
log_errors_sparse = []  # Errors seen in the log file.
replay_errors_sparse = []  # Errors seen in the replay.
log_observables_sparse = []  # Observable results seen in the log file.
log_observables_dense = []  # Observable flips seen in the log file.
replay_observables_dense = []  # Observable flips calculated in the replay.
log_result_types = []  # ResultType tokens seen in each log decode call.

decoders = {}
O_per_decoder = {}

parse_decoder_log(args.decoder_log, log_detectors_sparse, log_errors_sparse,
                  log_observables_sparse, log_observables_dense,
                  log_result_types, decoder_id_list)
output_per_decoder = decoder_outputs_from_log(log_result_types, decoder_id_list)
parse_decoder_config(args.config, decoders, O_per_decoder, output_per_decoder)

# Basic error checking
missing_decoder_ids = sorted(set(decoder_id_list) - decoders.keys())
if missing_decoder_ids:
    print(f"Error: no decoder config for logged decoder ids "
          f"{missing_decoder_ids}.")
    exit(1)

# ---------------------------------------------------------------------------- #
# Now loop through the syndromes and compare the results.
decode_call_idx = 0
replay_error_mismatch = 0
replay_observable_result_mismatch = 0
replay_observable_mismatch = 0
print(f'Processing {len(log_detectors_sparse)} decode calls.')
for s, o in zip(log_detectors_sparse, log_observables_dense):
    # Create a 1D array of length syndrome size.
    syndrome = numpy.zeros(
        decoders[decoder_id_list[decode_call_idx]].get_syndrome_size(),
        dtype=numpy.uint8)
    for idx in s:
        syndrome[idx] = 1
    result = decoders[decoder_id_list[decode_call_idx]].decode(syndrome)
    result_types = log_result_types[decode_call_idx]

    mismatch_flag = False
    decoded_sparse = [
        i for i in range(len(result.result)) if result.result[i] > 0.5
    ]
    if "errs" in result_types:
        dec_err_sparse = decoded_sparse
        replay_errors_sparse.append(dec_err_sparse)
        if dec_err_sparse != log_errors_sparse[decode_call_idx]:
            replay_error_mismatch += 1
            mismatch_flag = True
            if args.verbose_on_mismatch:
                print(
                    f"Replay mismatch in error in decode_call_idx {decode_call_idx}"
                )
                print(f"Decoded errors : {dec_err_sparse}")
                print(f"Expected errors: {log_errors_sparse[decode_call_idx]}")
        dec_err_dense = numpy.array(result.result, dtype=numpy.uint8)
        O_replay = (
            O_per_decoder[decoder_id_list[decode_call_idx]] @ dec_err_dense %
            2).astype(numpy.uint8)
        decoded_observables_sparse = [
            i for i in range(len(O_replay)) if O_replay[i]
        ]
    elif "obs" in result_types:
        replay_errors_sparse.append([])
        O_replay = numpy.array([1 if x > 0.5 else 0 for x in result.result],
                               dtype=numpy.uint8)
        decoded_observables_sparse = decoded_sparse
    else:
        print(f"Error: unsupported ResultType set {sorted(result_types)} "
              f"in decode_call_idx {decode_call_idx}.")
        exit(1)

    if "obs" in result_types:
        expected_observables_sparse = log_observables_sparse[decode_call_idx]
        if decoded_observables_sparse != expected_observables_sparse:
            replay_observable_result_mismatch += 1
            mismatch_flag = True
            if args.verbose_on_mismatch:
                print(
                    f"Replay mismatch in observable result in decode_call_idx {decode_call_idx}"
                )
                print(
                    f"Decoded observable result : {decoded_observables_sparse}")
                print(
                    f"Expected observable result: {expected_observables_sparse}"
                )

    replay_observables_dense.append(O_replay)
    O_log = numpy.array(log_observables_dense[decode_call_idx],
                        dtype=numpy.uint8)
    if O_replay.shape != O_log.shape or (O_replay != O_log).any():
        replay_observable_mismatch += 1
        mismatch_flag = True
        if args.verbose_on_mismatch:
            print(
                f"Replay mismatch in observables in decode_call_idx {decode_call_idx}"
            )
            print(f"Decoded observables : {O_replay}")
            print(f"Expected observables: {O_log}")
    if not args.verbose_on_mismatch:
        if mismatch_flag:
            print('x', end='', flush=True)
        else:
            print('.', end='', flush=True)
    decode_call_idx += 1

print()
print(f"Number of error mismatches during replay: {replay_error_mismatch}")
print(
    f"Number of observable result mismatches during replay: {replay_observable_result_mismatch}"
)
print(
    f"Number of observable mismatches during replay: {replay_observable_mismatch}"
)
