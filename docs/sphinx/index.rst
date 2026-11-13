CUDA-Q QEC - The CUDA-Q Quantum Error Correction Library
=========================================================

CUDA-Q QEC is a library that builds upon the CUDA-Q programming model
to enable the rapid development of hybrid quantum-classical application code leveraging
state-of-the-art CPUs, GPUs, and QPUs. It provides a C++ library and Python
package that enable research, development, and application creation for use
cases in quantum error correction.

.. note::

   If you're looking for CUDA-Q Algorithms (formerly known as CUDA-Q Solvers),
   see the `CUDA-Q Algorithms documentation
   <https://nvidia.github.io/cudaq-algorithms/>`__.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   quickstart/installation
   quickstart/quick_start

.. toctree::
   :maxdepth: 1
   :includehidden:
   :caption: Library Guide

   Overview <components/qec/index>
   QEC Decoders <components/qec/decoders>
   Detector Error Models <components/qec/detector_error_models>
   Decoding Task Graphs <components/qec/decoding_task_graph>
   Realtime Decoding <components/qec/realtime_decoding>
   Experiments and Noise Modeling <components/qec/numerical_experiments>
   QEC Codes <components/qec/codes>
   Conventions <components/qec/conventions>

.. toctree::
   :maxdepth: 1
   :caption: Examples

   Overview <examples_rst/qec/examples>
   Decoders <examples_rst/qec/decoders>
   Working with Detector Error Models <examples_rst/qec/detector_error_models>
   Decoding Task Graph <examples_rst/qec/decoding_task_graph>
   Realtime Decoding <examples_rst/qec/realtime_decoding>
   Experiments and Noise Modeling <examples_rst/qec/modeling_noise>
   Creating New QEC Codes <examples_rst/qec/creating_qec_codes>

.. toctree::
   :maxdepth: 1
   :includehidden:
   :caption: Performance Studies

   Overview <performance/index>
   Improving Relay BP Decoding With Gamma Ensembles <performance/nv_qldpc_gamma_ensemble_user_guide>
   Sweeping Relay BP Stopping Criteria From a Single Run <performance/nv_qldpc_relay_solutions_user_guide>
   Improving BP+OSD Decoding With Min-LLR OSD Initialization <performance/nv_qldpc_minllr_osd_user_guide>
   Bounding realtime decode latency with NV-Fusion <performance/nv_fusion_latency_user_guide>

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/core/cpp_api
   api/qec/cpp_api
   api/qec/python_api

.. toctree::
   :maxdepth: 1
   :caption: Releases

   Release History <releases>

Key Features
-------------

* :doc:`QEC Codes <components/qec/codes>`
* :doc:`QEC Decoders <components/qec/decoders>`
* :doc:`Decoding task graphs <components/qec/decoding_task_graph>`
* :doc:`Realtime decoding <components/qec/realtime_decoding>`
* :doc:`Decoding server <components/qec/decoding_server>` and
  :doc:`playback emulator <components/qec/playback_emulator>`
* :doc:`Experiments and noise modeling <components/qec/numerical_experiments>`

Indices
-------

* :ref:`genindex`
* :ref:`search`
