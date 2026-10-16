Realtime Decoding
=================

This page covers how realtime decoding works, its workflow, and terminology. For runnable applications — a complete walkthrough, the AI predecoder, and Relay BP — see the :doc:`Realtime Decoding examples </examples_rst/qec/realtime_decoding>`.

CUDA-Q QEC provides realtime decoding for quantum error correction on real quantum hardware: decoders process syndromes and compute corrections within qubit coherence times, making active error correction practical for real quantum computers. The framework supports two primary deployment scenarios:

1. **Hardware Integration**: Decoders running on classical computers connected to real quantum processing units (QPUs) — such as `Quantinuum's Helios QPU <https://www.quantinuum.com/products-solutions/quantinuum-systems/helios>`_ — via low-latency networks.
2. **Simulation Mode**: Decoders operating in simulated environments for testing and development on local systems.

.. note::
   The realtime decoding interfaces are experimental and subject to change. Realtime decoding on Quantinuum's Helios-1 device is currently available only to partners and collaborators. Please email QCSupport@quantinuum.com for more information.

Workflow
^^^^^^^^

Realtime decoding integrates into quantum error correction pipelines through a carefully designed four-stage workflow. This workflow separates the computationally intensive characterization phase from the latency-critical runtime phase, ensuring that decoders can operate efficiently during quantum circuit execution.

1. **Detector Error Model (DEM) Generation**: Before running a quantum program, the user first characterizes how errors propagate through the quantum circuit. The library internally uses Memory Syndrome Matrix (MSM) representations to track error propagation, but this complexity is abstracted through helper functions like ``z_dem_from_memory_circuit``. The user simply provides a quantum code, noise model, and circuit parameters, and receives a complete detector error model that maps error mechanisms to syndrome patterns. This step is performed once during development.

2. **Decoder Configuration and Saving**: Using the DEM, the user configures decoder instances with the specific error model data. This includes converting parity check matrices to sparse format, setting decoder-specific parameters (like lookup table depth or BP iterations), and assigning unique IDs to each logical qubit's decoder. The configuration is then saved to a YAML file, capturing all the information decoders need to interpret syndrome measurements correctly. This creates a portable, reusable configuration that separates characterization from execution.

3. **Decoder Loading and Initialization**: Just before circuit execution, the user loads the saved YAML configuration file. The library parses the configuration, instantiates the appropriate decoder implementations, initializes internal data structures, and registers the decoders with the CUDA-Q runtime. For GPU-based decoders, matrices are transferred to device memory; for lookup table decoders, syndrome-to-correction mappings are constructed. This initialization takes milliseconds to seconds depending on code size and happens before quantum operations begin.

4. **Realtime Decoding**: During quantum circuit execution, the decoding API is used within quantum kernels to interact with decoders. As the circuit measures stabilizers, syndromes are enqueued to the decoder, which processes them concurrently. When corrections are needed, the decoder is queried and the suggested operations are applied to the logical qubits. This entire process happens within the coherence time constraints of the quantum hardware.

Terminology and Data Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^

The realtime decoding workflow involves configuring a decoder (or many) before CUDA-Q kernel launch, and communicating to the decoders with special in-kernel functions.
A decoder is a single software instance of a decoding algorithm, and all its relevant inputs (parity-check matrices, error rates, etc.) which will remain static for the execution of the quantum program.
A decoder config may contain many decoders, each with different algorithms and input parameters.

In a quantum kernel, a user interacts with the decoders via the `enqueue_syndromes` and `get_corrections` interfaces.
The behavior of these functions depends on their configuration and their usage.

The realtime decoding workflow can be described with respect to the offline decoding workflow.
The non-realtime decoders require a detector error model which is specified via a detector error matrix which is the parity check matrix `H` of the decoding problem, and a vector of weights (error rates).
This matrix has dimensions of `[numDetectors, numErrors]`, where each row is a detector, and each column is a possible error.
For realtime decoding, we first need to convert the circuit measurements into detectors.
This is specified via the detector matrix `D`, which has dimensions `[numDetectors, numMeasurements]`.
Each column of the detector matrix defines which detectors a measurement participates in by including an entry of `1`.
Thus, once all `numMeasurements` measurements are enqueued, a matrix-vector multiply can convert this buffer of raw measurements into detectors which are then passed into the decoding algorithm.

Similarly, an observables flips matrix `O` of size `[numObs, numErrors]` must be provided.
Each column of the observables flips matrix describes for each error, which observables are flipped by that error by including an entry of `1`.
Once the decoding algorithm has processed the detectors it provides a vector of predicted errors of length `numErrors`.
This vector then executes a matrix-vector multiply with the observables flips matrix to yield a new vector of length `numObs` which contains an entry of `1` if the observable is predicted to have flipped.

Thus once a decoder is configured, we can view the realtime decoder as a transformation of data starting from a vector of raw measurements, then transformed into detectors via `D`, then error predictions via `H`, then observable flip predictions via `O`. This last step is what is returned via `get_corrections`. The user configures how many bits of information are returned, and what they represent via the `O` matrix in the decoder config.

Similarly, the user determines how many measurements are needed for the decoder via the `D` matrix in the decoder config, and they are sent to the decoder via `enqueue_syndromes`.
For flexibility, the user can choose to send all measurements with a single `enqueue_syndromes` call, or send them over several calls.
However they are split up, the decoder will not begin decoding until all `numMeasurements` have been enqueued, and will throw an error if too many are sent.
Thus it is the final `enqueue_syndromes` call which kicks off the decoder, and is an asynchronous function.
Additional quantum gates can be applied, and only when `get_corrections` is called does the kernel sync and wait for the corrections.

See Also
^^^^^^^^

* :ref:`Pre-built QEC Decoders <prebuilt_qec_decoders>` — decoders available for realtime use
* :doc:`Realtime Decoding examples </examples_rst/qec/realtime_decoding>` — runnable end-to-end examples
* :ref:`realtime_pipeline_api` — Realtime Pipeline C++ API (experimental)
* :ref:`C++ realtime decoding API <cpp_realtime_decoding_api>` and :ref:`Python realtime decoding API <python_realtime_decoding_api>`
