Realtime Decoding
==================

Realtime decoding runs CUDA-Q QEC decoders concurrently with quantum execution, applying corrections within qubit coherence times. For how it works, the four-stage workflow, and terminology, see :doc:`Realtime decoding </components/qec/realtime_decoding>`.

Start with :doc:`Getting Started with Realtime Decoding
<getting_started_realtime_decoding>` to configure and use decoders in a local
simulation. The following examples show circuit-driven and scheduled input:

* :doc:`Realtime Decoding from a Quantum Kernel or FPGA <realtime_decoding_demo>`
  streams measurements from an executing circuit or a real FPGA.
* :doc:`Realtime Decoding with the Playback Emulator <decoding_server_playback>`
  replays scheduled measurement and correction requests in software, first
  with decoders in process and then with a separate server.

For the server routes, see :doc:`Decoding Server </components/qec/decoding_server>`
for deployment and configuration. The
:doc:`Playback Emulator </components/qec/playback_emulator>` guide describes
schedules, measurement sources, and session types.

.. toctree::
   :maxdepth: 2

   Getting Started with Realtime Decoding <getting_started_realtime_decoding>
   Realtime Decoding from a Quantum Kernel or FPGA <realtime_decoding_demo>
   Realtime Decoding with the Playback Emulator <decoding_server_playback>
   AI Predecoder with CUDA-Q Realtime <realtime_predecoder_pymatching>
   AI Predecoder with CUDA-Q Realtime (with FPGA Data Injection) <realtime_predecoder_fpga>
   Comparing Ising and Chromobius for Color-Code Decoding <color_code_demo>
   Measuring Streaming Interleaved Feed-forward Latency (SIFL) <sifl_demo>

See Also
--------

* Example source code: `libs/qec/unittests/realtime/app_examples <https://github.com/NVIDIA/cudaq-qec/tree/main/libs/qec/unittests/realtime/app_examples>`_
* :ref:`Realtime decoding C++ API <cpp_realtime_decoding_api>`
* :ref:`Realtime decoding Python API <python_realtime_decoding_api>`
