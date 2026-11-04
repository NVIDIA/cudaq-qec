# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallel-window decoding of a distance-5 repetition-code memory experiment,
# the model of yaml/task_graph_parallel_window.yaml built in Python.

import cudaq_qec as qec


def phase(name,
          num_faults,
          seams,
          D_sparse,
          num_measurements=None,
          m2o_sparse=None,
          rate=0.02):
    """A phase whose faults flip the rows each seam lists, and whose detectors
    read the measurements D_sparse lists."""
    spec = qec.DemChunkSpec()
    spec.num_faults = num_faults
    entries = []
    for seam, rows in seams:
        rows_spec = qec.DemSeamSpec()
        rows_spec.H_sparse = rows
        entry = qec.SeamSpecEntry()
        entry.id = seam
        entry.spec = rows_spec
        entries.append(entry)
    spec.seam_specs = entries
    spec.D_sparse = D_sparse
    if num_measurements is not None:
        spec.num_measurements = num_measurements
    if m2o_sparse is not None:
        spec.m2o_sparse = m2o_sparse
    spec.O_sparse = [0, -1]  # observable 0 reads data qubit 0
    spec.error_rates = [rate] * num_faults
    entry = qec.PhaseSpecEntry()
    entry.id = qec.SeamId(name)
    entry.spec = spec
    return entry


def connection(from_phase, to_phase):
    c = qec.PhaseConnection()
    c.from_phase = qec.SeamId(from_phase)
    c.to_phase = qec.SeamId(to_phase)
    return c


# Faults 0-4 flip data qubits 0-4; faults 5-8 are errors in measuring checks
# 0-3. Check i reads data qubits i and i + 1. The final round reads the data
# qubits.
checks_seams = [
    (qec.seam_name.prev_round,
     [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1]),
    (qec.seam_name.next_round, [5, -1, 6, -1, 7, -1, 8, -1]),
]
init = phase("init", 9, checks_seams, [0, -1, 1, -1, 2, -1, 3, -1])
bulk = phase("bulk", 9, checks_seams, [0, 4, -1, 1, 5, -1, 2, 6, -1, 3, 7, -1])
final = phase(
    "final",
    5,
    [(qec.seam_name.prev_round, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1])],
    [0, 4, 5, -1, 1, 5, 6, -1, 2, 6, 7, -1, 3, 7, 8, -1],
    num_measurements=5,  # data qubits 0-4
    m2o_sparse=[0, -1])  # observable 0 reads data qubit 0

seam = qec.SeamConnection()
seam.from_seam = qec.seam_name.next_round
seam.to_seam = qec.seam_name.prev_round
model = qec.DemChunksSpec()
model.seam = seam
model.measurements_per_round = 4  # one per check
model.phases = [init, bulk, final]
model.connections = [
    connection("init", "bulk"),
    connection("bulk", "bulk"),
    connection("bulk", "final"),
]

# [Begin parallel_window]
node = qec.TaskGraphNodeConfig()
node.id = "pw"
node.kind = "parallel_window"
node.dem = model
node.node_params = {
    "commit_rounds": 3,
    "buffer_rounds": 3,
    "gap_rounds": 9,
    "output": "corrections_and_logical",
}
node.max_in_flight = 4
node.decoder_type = "pymatching"
node.params = {"merge_strategy": "smallest_weight"}

options = qec.TaskGraphOptions()
options.async_streaming = True
# One node needs no edges: it is wired measurements -> pw -> output.
graph = qec.DecodingTaskGraph.from_topology([node], options=options)
# [End parallel_window]

# [Begin stream]
# A 40-round shot in which data qubit 0 flips in round 10: from then on,
# check 0 reads 1. Stream one round per call; the last, FINAL, reads the data
# qubits.
rounds = 40
checks = graph.get_num_msyn_per_round()
graph.reset_decoder()
for r in range(rounds - 1):
    syndrome = bytes(int(r >= 10 and i == 0) for i in range(checks))
    graph.enqueue_syndrome(syndrome)
    if r == 30:
        # Mid-shot: the corrections cover rounds [0, frontier).
        progress = graph.streaming_result().outputs[0]
        print(f"after round {r}: rounds [{progress.coverage_begin_round}, "
              f"{progress.coverage_end_round}) committed")
graph.enqueue_syndrome(bytes([1, 0, 0, 0, 0]), qec.round_type.FINAL)

corrections = list(graph.get_obs_corrections())
logical = list(graph.get_logical_measurements())
print(f"corrections: {corrections}")
print(f"logical measurements: {logical}")
graph.clear_corrections()  # start the next shot
# [End stream]
assert corrections == [1] and logical == [0]
