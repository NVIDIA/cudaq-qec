# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This demo shows the number of stabilizer extraction rounds in a Streaming
# Interleaved Feed-forward Latency (SIFL) experiment executed in real-time:
#
#   * Stim's rotated surface-code memory circuit supplies the syndromes, which
#     the playback emulator streams one stabilizer round every T us.
#   * Two decoders alternate shots. After an initial shot with a fixed number of
#     rounds streamed to decoder A, decoder A begins to asynchronously decode while
#     syndromes continue to stream to decoder B. Once decoder A returns corrections,
#     decoder B begins to decode while syndromes stream to decoder A.
#   * per_round_decoder hands each shot to a sub-decoder built from the full
#     DEM of a circuit with that many rounds.

# [Begin Documentation]
"""Streaming Interleaved Feed-forward Latency (SIFL) on the playback emulator.

Usage: ./run_sifl_demo.sh [options]
"""
import argparse, ctypes, os, tempfile
import numpy as np
import stim
import cudaq_qec as qec

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--distance", type=int, default=5, help="code distance")
parser.add_argument("--p", type=float, default=0.01, help="physical error rate")
parser.add_argument(
    "--max-rounds",
    type=int,
    default=60,
    help="longest shot, in stabilizer rounds, the decoder accepts")
parser.add_argument("--shots", type=int, default=30, help="shots per cadence")
parser.add_argument("--init-rounds",
                    type=int,
                    default=10,
                    help="stabilizer rounds in the first shot")
parser.add_argument("--cadences",
                    type=float,
                    nargs="+",
                    default=[2, 5, 10, 20, 50],
                    help="round periods T to run, in us")
parser.add_argument("--seed", type=int, default=1, help="syndrome source seed")
parser.add_argument("--decoder",
                    default="pymatching",
                    help="decoder used for each round count")
args = parser.parse_args()
if not 0 < args.init_rounds < args.max_rounds:
    parser.error("--init-rounds must be between 1 and --max-rounds - 1")
STREAM_CAP = args.max_rounds - args.init_rounds
# Longest shot the schedule can produce.
MAX_SHOT = max(args.init_rounds, STREAM_CAP)

# Loading the plugin registers `per_round_decoder` with CUDA-Q QEC.
ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "libper_round_decoder.so"),
            mode=ctypes.RTLD_GLOBAL)


def circuit(rounds):
    return stim.Circuit.generated("surface_code:rotated_memory_z",
                                  rounds=rounds,
                                  distance=args.distance,
                                  before_measure_flip_probability=args.p,
                                  after_clifford_depolarization=args.p)


round_width = circuit(2).num_measurements - circuit(1).num_measurements
terminal_width = circuit(1).num_measurements - round_width


def write_dem(rounds, path):
    """Writes the full DEM of an r-round circuit as H, O, D and rates lines."""
    c = circuit(rounds)
    text = str(c.detector_error_model(decompose_errors=True))
    dem = qec.dem_from_stim_text(text, use_decomp_suggestions=True)
    dem.canonicalize_for_rounds(round_width, remove_zero_syndrome_errors=True)
    # D: flipping measurement i alone fires exactly the detectors in column i.
    flips = np.eye(c.num_measurements, dtype=np.bool_)
    D = c.compile_m2d_converter().convert(measurements=flips,
                                          append_observables=False).T
    lines = [
        qec.pcm_to_sparse_vec(np.ascontiguousarray(M, dtype=np.uint8))
        for M in (dem.detector_error_matrix, dem.observables_flips_matrix, D)
    ] + [dem.error_rates]
    with open(path, "w") as f:
        f.write("\n".join(" ".join(map(str, line)) for line in lines) + "\n")


def ring(decoder_id, dem_dir):
    config = qec.decoder_config()
    config.id, config.type = decoder_id, "per_round_decoder"
    # Placeholders: the decoder builds its sub-decoders from dem_dir. The large
    # D_sparse index lets a shot carry any number of measurements.
    config.block_size, config.syndrome_size = 1, 1
    config.H_sparse, config.O_sparse = [0, -1], [0, -1]
    config.D_sparse = [9_999_999, -1]
    config.decoder_custom_args = dict(dem_dir=dem_dir,
                                      round_width=round_width,
                                      terminal_width=terminal_width,
                                      max_rounds=MAX_SHOT,
                                      delegate_type=args.decoder)
    return config


# Shot i streams to ring i % 2 until shot i-1's correction lands.
lines = [
    f"0 stream source=0 rounds={args.init_rounds}", "- enqueue_data source=0",
    "- get_corrections return_size=1 signal=shot0"
]
for i in range(1, args.shots):
    lines += [
        f"- stream session={i % 2} source=0 every=1 min_rounds=1 "
        f"max_rounds={STREAM_CAP} until=shot{i - 1}",
        f"- enqueue_data session={i % 2} source=0",
        f"- get_corrections session={i % 2} return_size=1 signal=shot{i}"
    ]
schedule = "\n".join(lines) + "\n"
source = dict(type="stim_memory",
              seed=args.seed,
              code="surface_code",
              task="rotated_memory_z",
              distance=args.distance,
              rounds=10_000,
              before_measure_flip_probability=args.p,
              after_clifford_depolarization=args.p)

with tempfile.TemporaryDirectory() as dem_dir:
    for r in range(1, MAX_SHOT + 1):
        write_dem(r, f"{dem_dir}/r{r}.txt")
    decoders = qec.multi_decoder_config()
    decoders.decoders = [ring(0, dem_dir), ring(1, dem_dir)]

    pb = qec.playback
    print(f"Decode time per round and rounds streamed per shot "
          f"(stream cap {STREAM_CAP}):")
    for period_us in args.cadences:
        result = pb.run(schedule,
                        tick_ns=round(period_us * 1000),
                        decoders=decoders,
                        sources={0: source})
        rounds = [
            r.rounds_streamed
            for r in result.records
            if r.op == pb.operation.stream
        ]
        readouts = [
            r for r in result.records if r.op == pb.operation.enqueue_data
        ]
        reads = [
            r for r in result.records if r.op == pb.operation.get_corrections
        ]
        assert all(r.read_completed for r in reads)
        # Decode time runs from dispatching a shot's data readout to its
        # correction, excluding the source's time preparing the readout.
        decode_us = sum(
            (g.return_ns - result.request_timings(e.event_index)[0][1]) / 1e3
            for e, g in zip(readouts, reads))
        print(f"  T = {period_us:>4g} us  decode time per round "
              f"{decode_us / sum(rounds):5.2f} us  {rounds}")
# [End Documentation]
