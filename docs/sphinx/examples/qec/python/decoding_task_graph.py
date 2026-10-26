# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cudaq_qec as qec

# ---------------------------------------------------------------------------
# Helper: build a rep-5 dem_chunks_spec (init / bulk / final phases)
# ---------------------------------------------------------------------------


def _make_phase(name, num_faults, H_sparse, O_sparse, error_rates):
    entry = qec.PhaseSpecEntry()
    entry.id = qec.SeamId(name)
    s = qec.DemChunkSpec()
    s.num_faults = num_faults
    s.H_sparse = H_sparse
    s.O_sparse = O_sparse
    s.error_rates = error_rates
    entry.spec = s
    return entry


def _make_connection(from_name, to_name):
    c = qec.PhaseConnection()
    c.from_phase = qec.SeamId(from_name)
    c.to_phase = qec.SeamId(to_name)
    return c


def rep5_spec(num_rounds=5):
    seam = qec.SeamConnection()
    seam.from_seam = qec.seam_name.next_round
    seam.to_seam = qec.seam_name.prev_round

    spec = qec.DemChunksSpec()
    spec.seam = seam
    spec.phases = [
        _make_phase("init", 9,
                    [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1],
                    [0, -1], [0.02] * 9),
        _make_phase("bulk", 9,
                    [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1],
                    [0, -1], [0.02] * 9),
        _make_phase("final", 5, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1],
                    [0, -1], [0.02] * 5),
    ]
    spec.connections = [
        _make_connection("init", "bulk"),
        _make_connection("bulk", "bulk"),
        _make_connection("bulk", "final"),
    ]
    spec.num_rounds = num_rounds
    return spec


# ---------------------------------------------------------------------------
# [Begin from_spec]
# Build a single-decoder task graph from a dem_chunks_spec.
#
# When num_rounds is set the graph uses a fixed-T window that processes one
# complete shot per run() call.  When num_rounds is absent and a repeating
# (bulk self-loop) phase exists, the graph uses a streaming sliding window.
# ---------------------------------------------------------------------------

# Fixed-T graph: processes one 5-round shot at a time.
fixed_spec = rep5_spec(num_rounds=5)
fixed_graph = qec.DecodingTaskGraph.from_spec(fixed_spec, "single_error_lut")

print(f"fixed-T  msyn/decode: {fixed_graph.get_num_msyn_per_decode()}")
print(f"fixed-T  observables: {fixed_graph.get_num_observables()}")

# Streaming graph: num_rounds absent — window fires once per round.
streaming_spec = rep5_spec()
streaming_spec.num_rounds = None  # remove fixed round count
streaming_graph = qec.DecodingTaskGraph.from_spec(streaming_spec,
                                                  "single_error_lut")

print(f"streaming msyn/round: {streaming_graph.get_num_msyn_per_decode()}")
print(f"streaming observables: {streaming_graph.get_num_observables()}")
# [End from_spec]

# ---------------------------------------------------------------------------
# [Begin one-shot]
# One-shot execution: run() processes one complete shot and returns a
# LogicalOutcome.  Does not touch the streaming accumulation buffer.
# ---------------------------------------------------------------------------

n = fixed_graph.get_num_msyn_per_decode()

m = qec.MeasurementResults()
m.bits = [0] * n  # zero syndrome → no predicted errors

lo = fixed_graph.run(m)
print(f"converged: {lo.converged}, corrections: {list(lo.bits)}")
# [End one-shot]

# ---------------------------------------------------------------------------
# [Begin batch]
# Batch execution: run_batch() processes a collection of complete shots.
# ---------------------------------------------------------------------------

batch_shot = qec.MeasurementResults()
batch_shot.bits = [0] * fixed_graph.get_num_msyn_per_decode()
shots = [batch_shot] * 8
results = fixed_graph.run_batch(shots)
print(f"batch results: {len(results)} shots processed")
# [End batch]

# ---------------------------------------------------------------------------
# [Begin streaming]
# Streaming interface: enqueue_syndrome accumulates syndrome bytes one round
# at a time.  Returns True when the window fires and corrections are ready.
# Call get_obs_corrections() immediately after to read the result, then
# clear_corrections() to reset the XOR accumulator before the next window.
# Call reset_decoder() between shots to clear accumulation buffers.
# ---------------------------------------------------------------------------

graph = qec.DecodingTaskGraph.from_spec(streaming_spec, "single_error_lut")
# A slid window has room for one round per fire, so deliver a round per call
# rather than the whole window.
per_round = graph.get_num_msyn_per_round()
k = graph.get_num_observables()

# Simulate 10 rounds of zero syndromes (no errors).  The window fires once it
# holds get_num_msyn_per_decode() bytes, then on every round after.
num_rounds = 10
for round_idx in range(num_rounds):
    zero_syndrome = bytes(per_round)
    fired = graph.enqueue_syndrome(zero_syndrome)
    if fired:
        corrections = graph.get_obs_corrections()
        print(f"round {round_idx}: fired, corrections = {list(corrections)}")
        graph.clear_corrections()

# Full reset between shots.
graph.reset_decoder()
print("decoder reset for next shot")
# [End streaming]

# ---------------------------------------------------------------------------
# [Begin combine]
# combine() runs two independent decoders and XORs their logical outcomes.
# Each branch applies its own D matrix; they may decode the same DEM with
# different algorithms.  Both input graphs are consumed by the call.
# In streaming mode, enqueue_syndrome routes the syndrome to both branches
# simultaneously.
# ---------------------------------------------------------------------------

left = qec.DecodingTaskGraph.from_spec(rep5_spec(5), "single_error_lut")
right = qec.DecodingTaskGraph.from_spec(rep5_spec(5), "single_error_lut")

combined = qec.DecodingTaskGraph.combine(left, right)
print(f"combined observables: {combined.get_num_observables()}")

# Batch: run() walks both branches and XORs the results.
n = combined.get_num_msyn_per_decode()
m = qec.MeasurementResults()
m.bits = [0] * n
lo = combined.run(m)
print(f"combined corrections: {list(lo.bits)}")  # XOR of identical results = 0

# Streaming: both branches receive the same syndrome bytes.
fired = combined.enqueue_syndrome(bytes(n))
if fired:
    corr = combined.get_obs_corrections()
    print(f"combined streaming corrections: {list(corr)}")
# [End combine]
