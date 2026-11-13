Realtime Decoding
=================

Realtime decoding processes measurements during quantum circuit execution so
that correction results are available when the circuit or control system
needs them. CUDA-Q QEC provides interfaces to configure decoders, send
measurements, and retrieve corrections. You can develop and test the decoding
pipeline in simulation before connecting it to quantum hardware.

The framework supports two deployment scenarios:

1. **Hardware Integration**: Decoders running on classical computers connected to real quantum processing units (QPUs) — such as `Quantinuum's Helios QPU <https://www.quantinuum.com/products-solutions/quantinuum-systems/helios>`_ — via low-latency networks.
2. **Simulation Mode**: Decoders operating in simulated environments for testing and development on local systems.

.. note::
   The realtime decoding interfaces are experimental and subject to change. Realtime decoding on Quantinuum's Helios-1 device is currently available only to partners and collaborators. Please email QCSupport@quantinuum.com for more information.

The :doc:`Workflow and Data Flow <realtime_decoding_workflow>` guide explains
how to prepare a decoder configuration before circuit execution and how
measurements are converted into detectors, error predictions, and logical
corrections during a shot.

When a computation waits for a correction while further measurement rounds
continue, decoder response time affects how much work arrives next. See
:doc:`Feedback Latency and Decoder Backlog <feedback_latency>` for this
dependency and the SIFL experiment used to study it.

For pipelines that combine a neural network with a conventional decoder,
see :doc:`AI Predecoding <ai_predecoding>`. It explains how the two stages
share the decoding work and links to surface-code and color-code examples.

In a local simulation, the decoders can run in the same process as the quantum
simulator. When the application or control system needs to send measurements
to decoders in a separate process, use a :doc:`Decoding Server <decoding_server>`.
The server hosts the configured pipeline and returns corrections to its
clients. Its guide covers deployment and links to examples using a quantum
kernel, an FPGA, or the software playback emulator.

To test the realtime path without a QPU, use the
:doc:`Playback Emulator <playback_emulator>`. It replays measurements and
correction requests on a schedule, either against decoders in the same process
or through a server. Its guide explains the inputs, timing records, and session
types, with examples for both deployment modes.

For surface-code memory experiments, the
:doc:`Memory Experiment Configuration Generator <memory_experiment_generator>`
prepares a dynamic detector error model and a parallel-window task graph,
then saves the YAML configuration for the server.

.. toctree::
   :hidden:
   :maxdepth: 2

   Workflow and Data Flow <realtime_decoding_workflow>
   Feedback Latency and Decoder Backlog <feedback_latency>
   AI Predecoding <ai_predecoding>
   Decoding Server <decoding_server>
   Playback Emulator <playback_emulator>
   Memory Experiment Configuration Generator <memory_experiment_generator>

.. rubric:: Related Guides and Examples

* :ref:`Pre-built QEC Decoders <prebuilt_qec_decoders>` — decoders available for realtime use
* :doc:`Realtime Decoding examples </examples_rst/qec/realtime_decoding>` — runnable end-to-end examples
* :ref:`realtime_pipeline_api` — Realtime Pipeline C++ API (experimental)
* :ref:`C++ realtime decoding API <cpp_realtime_decoding_api>` and :ref:`Python realtime decoding API <python_realtime_decoding_api>`
