# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# [Begin Documentation]
import cudaq
import cudaq_qec as qec

code, rounds = qec.get_code("repetition", distance=3), 3
noise = cudaq.NoiseModel()
noise.add_all_qubit_channel("x", cudaq.Depolarization2(0.01), 1)
# Configure one lookup-table decoder (id 0) for this memory experiment.
ctx = qec.decoder_context_from_memory_circuit(code, qec.operation.prep0, rounds,
                                              noise)
dem, m2d, _ = ctx.full_component()
config = qec.decoder_config()
config.id, config.type = 0, "multi_error_lut"
config.syndrome_size, config.block_size = dem.detector_error_matrix.shape
config.H_sparse = qec.pcm_to_sparse_vec(dem.detector_error_matrix)
config.O_sparse = qec.pcm_to_sparse_vec(dem.observables_flips_matrix)
config.D_sparse = qec.d_sparse(m2d)
decoders = qec.multi_decoder_config()
decoders.decoders = [config]

# One shot: reset, 3 stabilizer rounds, data readout, then decode. Each shot
# starts 100 ticks (100 us) after the previous one finished sending.
shot = """\
+100 reset
-    stream source=0 rounds=3
-    enqueue_data source=0
-    get_corrections return_size=1
"""
schedule = shot * 10
# Raw measurements for one shot: 3 rounds of 2 ancilla bits, then 3 data bits.
# Shots alternate between no error and a flip of data qubit 0.
clean = [[0, 0]] * 3 + [[0, 0, 0]]
flipped = [[1, 0]] * 3 + [[1, 0, 0]]
source = dict(type="static", rounds=(clean + flipped) * 5)
result = qec.playback.run(schedule,
                          tick_ns=1000,
                          sources={0: source},
                          decoders=decoders)
for r in result.records[3::4]:  # The get_corrections records
    print(f"status={r['status']} correction={r['correction_bits']} "
          f"latency={(r['return_ns'] - r['call_ns'])/1000:4.1f} us")
# [End Documentation]
