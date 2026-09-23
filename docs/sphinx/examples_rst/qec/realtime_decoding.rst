Realtime Decoding
==================

Realtime decoding runs CUDA-Q QEC decoders concurrently with quantum execution, applying corrections within qubit coherence times. For how it works, the four-stage workflow, and terminology, see :doc:`Realtime Decoding </components/qec/realtime_decoding>`.

The examples below cover realtime decoding end to end — start with Getting Started, then explore the specialized predecoding and decoding workloads:

.. toctree::
   :maxdepth: 2

   Getting Started with Realtime Decoding <getting_started_realtime_decoding>
   Realtime Decoding with CUDA-Q Decoding Server <realtime_decoding_demo>
   AI Predecoder with CUDA-Q Realtime <realtime_predecoder_pymatching>
   AI Predecoder with CUDA-Q Realtime (with FPGA Data Injection) <realtime_predecoder_fpga>
   Relay BP Decoding with CUDA-Q Realtime <realtime_relay_bp>
   Dynamic DEM Construction <dyn_dem>

See Also
--------

* Example source code: `libs/qec/unittests/realtime/app_examples <https://github.com/NVIDIA/cudaq-qec/tree/main/libs/qec/unittests/realtime/app_examples>`_
* :ref:`Realtime Decoding C++ API <cpp_realtime_decoding_api>`
* :ref:`Realtime Decoding Python API <python_realtime_decoding_api>`
