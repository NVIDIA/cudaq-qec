# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# A dynamic DEM of a distance-5 repetition-code memory experiment,
# the model of yaml/task_graph_parallel_window.yaml built in Python.

import cudaq_qec as qec

# [Begin Documentation]


def phase(name, num_faults, seams):
    """A phase whose faults flip the rows each seam lists."""
    spec = qec.DemChunkSpec()
    spec.num_faults = num_faults
    entries = []
    for seam, rows in seams:
        entry = qec.SeamSpecEntry()
        entry.id = seam
        entry.spec.H_sparse = rows
        entries.append(entry)
    spec.seam_specs = entries  # assign the list; appending edits a copy
    spec.O_sparse = [0, -1]  # observable 0 reads data qubit 0
    spec.error_rates = [0.02] * num_faults
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
# 0-3. Check i reads data qubits i and i + 1.
prev_round, next_round = qec.seam_name.prev_round, qec.seam_name.next_round
seam = qec.SeamConnection()
seam.from_seam = next_round  # this round's next_round rows...
seam.to_seam = prev_round  # ...are the next round's prev_round rows
model = qec.DemChunksSpec()
model.seam = seam
model.phases = [
    phase("bulk", 9,
          [(prev_round, [0, 1, 5, -1, 1, 2, 6, -1, 2, 3, 7, -1, 3, 4, 8, -1]),
           (next_round, [5, -1, 6, -1, 7, -1, 8, -1])]),
    phase("final", 5, [(prev_round, [0, 1, -1, 1, 2, -1, 2, 3, -1, 3, 4, -1])]),
]
model.connections = [connection("bulk", "bulk"), connection("bulk", "final")]

# Fix the shot length to decode complete shots: 5 bulk rounds and the final
# readout.
model.num_rounds = 6
graph = qec.DecodingTaskGraph.from_spec(model, "pymatching")

# Data qubit 0 flips in round 2: from then on, check 0 reads 1.
checks = graph.get_num_msyn_per_round()
shot = qec.MeasurementResults()
shot.bits = [int(r >= 2 and i == 0) for r in range(6) for i in range(checks)]
corrections = list(graph.run(shot).bits)
print(f"corrections: {corrections}")
# [End Documentation]

assert corrections == [1]
