Generating a Multi-Round Parity Check Matrix
============================================

A single-round DEM captures one measurement cycle. Under circuit-level noise, errors accumulate across many rounds, and the DEM expands into a multi-round parity check matrix. The following example constructs one for an error correction code in Python:

.. tab:: Python

   .. literalinclude:: ../../examples/qec/python/repetition_code_pcm.py
      :language: python
      :start-after: [Begin Documentation]

This example illustrates how to:

* Retrieve and configure an error correction code
  Load a repetition code using ``qec.get_code(...)`` from the CUDA-Q QEC library, and define a custom circuit-level noise model using ``.add_all_qubit_channel(...)``.

* Generate a multi-round parity check matrix
  Extend a single-round detector error model (DEM) across multiple rounds using ``qec.dem_from_memory_circuit(...)``. This captures syndrome evolution over time, including measurement noise, and provides:

  * ``detector_error_matrix`` – the multi-round parity check matrix
  * ``observables_flips_matrix`` – used to identify logical flips due to physical errors

* Simulate circuit-level noise and collect data
  Run multiple shots of the memory experiment using ``qec.sample_memory_circuit(...)`` to sample both the data and syndrome measurements from noisy executions. The resulting bitstrings can be used for decoding and performance evaluation of the error correction scheme.
